"""Per-ROI readout heads.

A readout maps an ROI's query token (after the cross-attention decoder) to a
voxel-level prediction. RABBiT supports four readout families:

- ``direct``        : ``Linear(hidden_dim -> n_voxels)`` per ROI.
- ``parametric``    : ``Linear(hidden_dim -> K) @ frozen PCA bases``.
- ``shared_deviation`` (headline) : ``coeff_shared(h) @ bases_shared +
                        coeff_dev(h) @ bases_dev[subject]``; both bases are
                        learnable, anchored to PCA init.
- ``dataset_shared_deviation`` : per-dataset ``bases_shared``, routed via a
                        ``subject_to_dataset`` buffer; per-subject ``bases_dev``
                        unchanged.

See ``docs/architecture.md`` for the equations and how the avg-dev trick is
applied at zero-shot inference time.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Direct readout (no bases)
# ─────────────────────────────────────────────────────────────────────────────


def build_direct_readouts(
    hidden_dim: int,
    roi_dims: "OrderedDict[str, int]",
    bias: bool = True,
) -> nn.ModuleDict:
    """``Linear(hidden_dim -> n_voxels)`` per ROI."""
    return nn.ModuleDict(
        OrderedDict(
            (name, nn.Linear(hidden_dim, dim, bias=bias))
            for name, dim in roi_dims.items()
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parametric readout (frozen PCA bases)
# ─────────────────────────────────────────────────────────────────────────────


class ParametricROIReadout(nn.Module):
    """One ROI: predict ``K`` scalar coefficients, multiply by frozen PCA bases.

    ``pred = Linear(hidden_dim -> K)(h) @ bases``,  bases ∈ R^{K × V} (frozen).
    """

    def __init__(
        self,
        hidden_dim: int,
        bases: Tensor,
        bias: bool = True,
        init_uniform: bool = True,
    ) -> None:
        super().__init__()
        K, n_voxels = bases.shape
        self.K = K
        self.n_voxels = n_voxels
        self.coeff_head = nn.Linear(hidden_dim, K, bias=bias)
        if init_uniform and bias:
            nn.init.constant_(self.coeff_head.bias, 1.0 / K)
        # Frozen — registered as buffer so it travels with .to(device)
        # and serialises with state_dict but does not receive gradients.
        self.register_buffer("bases", bases)

    def forward(self, roi_token: Tensor) -> Tensor:
        return self.coeff_head(roi_token) @ self.bases  # (B, V)


def build_parametric_readouts(
    hidden_dim: int,
    roi_bases: dict[str, Tensor],
    bias: bool = True,
) -> nn.ModuleDict:
    return nn.ModuleDict(
        OrderedDict(
            (name, ParametricROIReadout(hidden_dim, bases, bias=bias))
            for name, bases in roi_bases.items()
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared + low-rank deviation readout (the headline factorisation)
# ─────────────────────────────────────────────────────────────────────────────


class SharedDeviationROIReadout(nn.Module):
    """``pred = coeff_shared(h) @ bases_shared + coeff_dev(h) @ bases_dev[subj]``.

    Both ``bases_shared`` (K, V) and ``bases_dev`` (n_subjects, R, V) are
    learnable ``nn.Parameter`` tensors. Anchor buffers keep them near their
    PCA-derived initialisation; anchor and ortho losses are computed by the
    enclosing encoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        shared_bases: Tensor,
        dev_bases_list: list[Tensor],
        bias: bool = True,
        dev_bias_uniform: bool = False,
    ) -> None:
        super().__init__()
        K, V = shared_bases.shape
        R = dev_bases_list[0].shape[0]
        self.K = K
        self.R = R
        self.n_voxels = V
        self.n_subjects = len(dev_bases_list)

        # Shared pathway
        self.bases_shared = nn.Parameter(shared_bases.clone())
        self.coeff_shared = nn.Linear(hidden_dim, K, bias=bias)
        if bias:
            nn.init.constant_(self.coeff_shared.bias, 1.0 / K)

        # Deviation pathway
        self.coeff_dev = nn.Linear(hidden_dim, R, bias=bias)
        if bias:
            if dev_bias_uniform:
                # ROIs where subject identity matters most (AG, MFG, IFG, SM).
                # Mirrors the shared pathway's 1/K bias so the deviation
                # pathway starts with a uniform contribution.
                nn.init.constant_(self.coeff_dev.bias, 1.0 / R)
            else:
                nn.init.zeros_(self.coeff_dev.bias)

        # Per-subject deviation bases stacked: (n_subjects, R, V)
        self.bases_dev = nn.Parameter(torch.stack(dev_bases_list))

        # Anchor buffers for regularisation. Reads only — never overwritten
        # after construction so the anchor loss measures drift from PCA init.
        self.register_buffer("_shared_init", shared_bases.clone())
        self.register_buffer("_dev_init", torch.stack(dev_bases_list).clone())

    def forward(self, roi_token: Tensor, subject_indices: Tensor) -> Tensor:
        shared = self.coeff_shared(roi_token) @ self.bases_shared          # (B, V)
        dev_coeffs = self.coeff_dev(roi_token)                              # (B, R)
        selected_bases = self.bases_dev[subject_indices]                    # (B, R, V)
        dev = torch.bmm(dev_coeffs.unsqueeze(1), selected_bases).squeeze(1)
        return shared + dev

    def anchor_loss(self) -> Tensor:
        shared_drift = (self.bases_shared - self._shared_init).pow(2).mean()
        dev_drift = (self.bases_dev - self._dev_init).pow(2).mean()
        return shared_drift + dev_drift

    def ortho_loss(self) -> Tensor:
        G = self.bases_shared @ self.bases_shared.T
        I = torch.eye(self.K, device=G.device, dtype=G.dtype)
        return (G - I).pow(2).mean()

    @torch.no_grad()
    def replace_subject_with_average(self, slot: int = 0) -> None:
        """In-place avg-dev: overwrite ``bases_dev[slot]`` with the mean across
        **all** subject slots. Used at zero-shot inference time so a held-out
        subject can be served via ``subject_indices=slot``.

        Note: averages across all ``n_subjects`` slots (including ``slot``).
        This matches the convention in the original training code so legacy
        research checkpoints reproduce identically here.
        """
        if self.n_subjects < 1:
            return
        avg = self.bases_dev.mean(dim=0)
        self.bases_dev[slot].copy_(avg)


def build_shared_deviation_readouts(
    hidden_dim: int,
    shared_roi_bases: dict[str, Tensor],
    dev_roi_bases: dict[str, list[Tensor]],
    bias: bool = True,
    dev_bias_uniform_rois: Optional[set[str]] = None,
) -> nn.ModuleDict:
    dev_bias_uniform_rois = set(dev_bias_uniform_rois or [])
    return nn.ModuleDict(
        OrderedDict(
            (
                name,
                SharedDeviationROIReadout(
                    hidden_dim,
                    shared,
                    dev_roi_bases[name],
                    bias=bias,
                    dev_bias_uniform=(name in dev_bias_uniform_rois),
                ),
            )
            for name, shared in shared_roi_bases.items()
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-routed shared + deviation
# ─────────────────────────────────────────────────────────────────────────────


class DatasetSharedDeviationROIReadout(nn.Module):
    """Same factorisation as ``SharedDeviationROIReadout`` but the shared
    basis is **per dataset**: ``bases_shared`` is ``(n_datasets, K, V)`` and
    each batch element picks its dataset via ``subject_to_dataset[subj]``.

    Used by combined moth+friends checkpoints where each dataset keeps its
    own shared subspace while sharing the decoder and per-subject deviation.
    """

    def __init__(
        self,
        hidden_dim: int,
        shared_bases_list: list[Tensor],
        dev_bases_list: list[Tensor],
        subject_to_dataset: list[int],
        bias: bool = True,
        dev_bias_uniform: bool = False,
    ) -> None:
        super().__init__()
        if not shared_bases_list:
            raise ValueError("shared_bases_list must be non-empty.")
        K, V = shared_bases_list[0].shape
        for sb in shared_bases_list:
            if sb.shape != (K, V):
                raise ValueError(
                    f"All shared bases must share shape (K={K}, V={V}); got {tuple(sb.shape)}."
                )
        R = dev_bases_list[0].shape[0]
        n_datasets = len(shared_bases_list)
        n_subjects = len(dev_bases_list)
        if len(subject_to_dataset) != n_subjects:
            raise ValueError(
                f"subject_to_dataset length {len(subject_to_dataset)} != n_subjects {n_subjects}."
            )
        if any(d < 0 or d >= n_datasets for d in subject_to_dataset):
            raise ValueError(
                f"subject_to_dataset has out-of-range entries; valid range [0, {n_datasets - 1}]."
            )

        self.K = K
        self.R = R
        self.n_voxels = V
        self.n_datasets = n_datasets
        self.n_subjects = n_subjects

        self.bases_shared = nn.Parameter(torch.stack(shared_bases_list))
        self.coeff_shared = nn.Linear(hidden_dim, K, bias=bias)
        if bias:
            nn.init.constant_(self.coeff_shared.bias, 1.0 / K)

        self.coeff_dev = nn.Linear(hidden_dim, R, bias=bias)
        if bias:
            if dev_bias_uniform:
                nn.init.constant_(self.coeff_dev.bias, 1.0 / R)
            else:
                nn.init.zeros_(self.coeff_dev.bias)
        self.bases_dev = nn.Parameter(torch.stack(dev_bases_list))

        self.register_buffer(
            "subject_to_dataset", torch.as_tensor(subject_to_dataset, dtype=torch.long)
        )
        self.register_buffer("_shared_init", self.bases_shared.detach().clone())
        self.register_buffer("_dev_init", self.bases_dev.detach().clone())

    def forward(self, roi_token: Tensor, subject_indices: Tensor) -> Tensor:
        ds_idx = self.subject_to_dataset[subject_indices]                 # (B,)
        selected_shared = self.bases_shared[ds_idx]                       # (B, K, V)
        coeffs_s = self.coeff_shared(roi_token).unsqueeze(1)              # (B, 1, K)
        shared = torch.bmm(coeffs_s, selected_shared).squeeze(1)          # (B, V)

        coeffs_d = self.coeff_dev(roi_token).unsqueeze(1)                 # (B, 1, R)
        selected_dev = self.bases_dev[subject_indices]                    # (B, R, V)
        dev = torch.bmm(coeffs_d, selected_dev).squeeze(1)                # (B, V)
        return shared + dev

    def anchor_loss(self) -> Tensor:
        shared_drift = (self.bases_shared - self._shared_init).pow(2).mean()
        dev_drift = (self.bases_dev - self._dev_init).pow(2).mean()
        return shared_drift + dev_drift

    def ortho_loss(self) -> Tensor:
        loss = self.bases_shared.new_zeros(())
        eye = torch.eye(self.K, device=self.bases_shared.device, dtype=self.bases_shared.dtype)
        for d in range(self.n_datasets):
            G = self.bases_shared[d] @ self.bases_shared[d].T
            loss = loss + (G - eye).pow(2).mean()
        return loss / self.n_datasets


def build_dataset_shared_deviation_readouts(
    hidden_dim: int,
    shared_roi_bases_per_dataset: list[dict[str, Tensor]],
    dev_roi_bases: dict[str, list[Tensor]],
    subject_to_dataset: list[int],
    bias: bool = True,
    dev_bias_uniform_rois: Optional[set[str]] = None,
) -> nn.ModuleDict:
    if not shared_roi_bases_per_dataset:
        raise ValueError("shared_roi_bases_per_dataset must contain ≥ 1 dataset.")
    dev_bias_uniform_rois = set(dev_bias_uniform_rois or [])
    readouts = OrderedDict()
    for name in shared_roi_bases_per_dataset[0]:
        readouts[name] = DatasetSharedDeviationROIReadout(
            hidden_dim=hidden_dim,
            shared_bases_list=[d[name] for d in shared_roi_bases_per_dataset],
            dev_bases_list=dev_roi_bases[name],
            subject_to_dataset=subject_to_dataset,
            bias=bias,
            dev_bias_uniform=(name in dev_bias_uniform_rois),
        )
    return nn.ModuleDict(readouts)
