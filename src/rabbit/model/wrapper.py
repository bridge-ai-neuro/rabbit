"""``RABBiT`` — the full speech → ROI-vertex predictor.

Composition:

    audio
      ↓  SpeechBackbone (wav2vec2 / WavLM + LoRA, feature encoder frozen)
    hidden_states (B, T, hidden_size)
      ↓  input_proj: Linear(hidden_size → hidden_dim)
      ↓  + temporal position encoding
      ↓  TemporalROIDecoder (cross-attention with ROI queries)
    roi_tokens (B, n_rois, hidden_dim)
      ↓  per-ROI readout (direct / parametric / shared_dev / dataset_shared_dev)
    flat_predictions (B, flat_dim)
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .backbone import SpeechBackbone
from .encoder import TemporalROIDecoder, build_temporal_position_encoding
from .readouts import (
    build_dataset_shared_deviation_readouts,
    build_direct_readouts,
    build_parametric_readouts,
    build_shared_deviation_readouts,
)
from .roi_layout import ROILayout, build_flat_roi_layout, initialise_roi_queries, flatten_roi_blocks


READOUT_TYPES = {
    "direct",
    "parametric",
    "shared_deviation",
    "dataset_shared_deviation",
}

_SUBJECT_DEPENDENT = {"shared_deviation", "dataset_shared_deviation"}


class RABBiT(nn.Module):
    """Speech → 30-ROI fsaverage6 fMRI predictor.

    Args:
        readout_type: one of ``direct``, ``parametric``, ``shared_deviation``,
            ``dataset_shared_deviation``. The headline RABBiT is
            ``shared_deviation``.
        backbone: configured ``SpeechBackbone`` (wav2vec2 or WavLM with LoRA).
        roi_layout: 30-ROI fs6 layout — see ``rabbit.model.roi_layout``.
        hidden_dim: decoder hidden size (paper: 256).
        roi_bases: ``{roi_name: (K, V) Tensor}`` PCA bases. Required for the
            ``parametric`` and ``shared_deviation`` readouts (used as the
            initial shared subspace).
        dev_roi_bases: ``{roi_name: [Tensor(R, V) for each subject]}``,
            required for ``shared_deviation`` and ``dataset_shared_deviation``.
        shared_roi_bases_per_dataset: list of ``roi_bases`` dicts, one per
            dataset. Required for ``dataset_shared_deviation`` only.
        subject_to_dataset: list of dataset indices for each subject slot.
            Required for ``dataset_shared_deviation`` only.

    Forward:
        ``model(input_wav, subject_indices=None)`` returns a dict with keys:
            ``flat_predictions``     (B, V_total)
            ``predictions_by_roi``   OrderedDict[roi_name -> (B, V_roi)]
            ``roi_tokens``           (B, n_rois, hidden_dim)
            ``attention_weights``    optional, last layer
            ``attention_weights_all``optional, every layer
    """

    def __init__(
        self,
        *,
        readout_type: str,
        backbone: SpeechBackbone,
        roi_layout: ROILayout,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        position_embedding: str = "sine",
        max_positions: int = 4096,
        query_init_std: float = 0.05,
        use_self_attention: bool = True,
        readout_bias: bool = True,
        # Readout-specific
        roi_bases: Optional[dict[str, Tensor]] = None,
        dev_roi_bases: Optional[dict[str, list[Tensor]]] = None,
        shared_roi_bases_per_dataset: Optional[list[dict[str, Tensor]]] = None,
        subject_to_dataset: Optional[list[int]] = None,
        uniform_dev_bias_rois: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        if readout_type not in READOUT_TYPES:
            raise ValueError(
                f"Unknown readout_type={readout_type!r}; expected one of {sorted(READOUT_TYPES)}."
            )
        if hidden_dim % nhead != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by nhead={nhead}.")

        self.readout_type = readout_type
        self.backbone = backbone
        self.roi_layout = roi_layout
        self.hidden_dim = hidden_dim
        self.needs_subject_indices = readout_type in _SUBJECT_DEPENDENT

        # ── Speech → hidden_dim projection ────────────────────────────────────
        in_dim = backbone.hidden_size
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.position_encoder = build_temporal_position_encoding(
            position_embedding_type=position_embedding,
            hidden_dim=hidden_dim,
            max_positions=max_positions,
        )

        # ── ROI queries + transformer decoder ─────────────────────────────────
        self.query_embed = initialise_roi_queries(
            hidden_dim=hidden_dim,
            roi_layout=roi_layout,
            init_std=query_init_std,
        )
        self.decoder = TemporalROIDecoder(
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_self_attention=use_self_attention,
        )

        # ── Per-ROI readout heads ─────────────────────────────────────────────
        uniform_set = set(uniform_dev_bias_rois or [])
        if readout_type == "direct":
            self.roi_readouts = build_direct_readouts(
                hidden_dim=hidden_dim,
                roi_dims=roi_layout.roi_dims_dict,
                bias=readout_bias,
            )
        elif readout_type == "parametric":
            if roi_bases is None:
                raise ValueError("readout_type='parametric' requires roi_bases.")
            self._check_bases_match_layout(roi_bases)
            self.roi_readouts = build_parametric_readouts(
                hidden_dim=hidden_dim, roi_bases=roi_bases, bias=readout_bias,
            )
        elif readout_type == "shared_deviation":
            if roi_bases is None or dev_roi_bases is None:
                raise ValueError(
                    "readout_type='shared_deviation' requires roi_bases (shared) "
                    "and dev_roi_bases (per-subject)."
                )
            self._check_bases_match_layout(roi_bases)
            self.roi_readouts = build_shared_deviation_readouts(
                hidden_dim=hidden_dim,
                shared_roi_bases=roi_bases,
                dev_roi_bases=dev_roi_bases,
                bias=readout_bias,
                dev_bias_uniform_rois=uniform_set,
            )
        elif readout_type == "dataset_shared_deviation":
            if (
                shared_roi_bases_per_dataset is None
                or dev_roi_bases is None
                or subject_to_dataset is None
            ):
                raise ValueError(
                    "readout_type='dataset_shared_deviation' requires "
                    "shared_roi_bases_per_dataset, dev_roi_bases, and subject_to_dataset."
                )
            self._check_bases_match_layout(shared_roi_bases_per_dataset[0])
            self.roi_readouts = build_dataset_shared_deviation_readouts(
                hidden_dim=hidden_dim,
                shared_roi_bases_per_dataset=shared_roi_bases_per_dataset,
                dev_roi_bases=dev_roi_bases,
                subject_to_dataset=subject_to_dataset,
                bias=readout_bias,
                dev_bias_uniform_rois=uniform_set,
            )
        else:  # pragma: no cover — guarded above
            raise ValueError(readout_type)

        self.output_dim = roi_layout.flat_dim

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_wav: Tensor,
        subject_indices: Optional[Tensor] = None,
        *,
        attention_mask: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> dict:
        if input_wav.ndim != 2:
            raise ValueError(f"Expected input_wav of shape (B, T_audio), got {tuple(input_wav.shape)}.")
        if self.needs_subject_indices and subject_indices is None:
            raise ValueError(
                f"readout_type={self.readout_type!r} requires `subject_indices` "
                f"of shape (B,) at forward time."
            )

        hidden_states, feature_mask = self.backbone(input_wav, attention_mask=attention_mask)
        memory = self.input_norm(self.input_proj(hidden_states))
        memory = self.input_dropout(memory)
        memory_pos = self.position_encoder(memory, feature_mask)

        roi_tokens, attn_maps = self.decoder(
            self.query_embed.weight,
            memory,
            memory_pos=memory_pos,
            memory_valid_mask=feature_mask,
            return_attn=return_attn,
        )

        preds: "OrderedDict[str, Tensor]" = OrderedDict()
        for i, name in enumerate(self.roi_layout.roi_names):
            token = roi_tokens[:, i, :]
            head = self.roi_readouts[name]
            if self.needs_subject_indices:
                preds[name] = head(token, subject_indices)
            else:
                preds[name] = head(token)

        flat = flatten_roi_blocks(preds, self.roi_layout)
        return {
            "flat_predictions": flat,
            "predictions_by_roi": preds,
            "roi_tokens": roi_tokens,
            "attention_weights": attn_maps[-1] if attn_maps else None,
            "attention_weights_all": attn_maps,
        }

    # ── Zero-shot avg-dev helper ─────────────────────────────────────────────

    def use_avg_deviation(self, slot: int = 0) -> None:
        """Apply the avg-dev trick: replace ``bases_dev[slot]`` with the mean
        across the remaining subject slots in every ROI's shared-deviation
        readout. After this, call forward with ``subject_indices=torch.full((B,), slot)``
        to predict for a held-out subject.

        No-op for ``direct`` / ``parametric`` readouts.
        """
        if self.readout_type not in _SUBJECT_DEPENDENT:
            return
        for head in self.roi_readouts.values():
            # SharedDeviationROIReadout has the in-place helper.
            if hasattr(head, "replace_subject_with_average"):
                head.replace_subject_with_average(slot=slot)
                continue
            # DatasetSharedDeviationROIReadout: same operation inline.
            with torch.no_grad():
                if head.bases_dev.shape[0] < 1:
                    continue
                head.bases_dev[slot].copy_(head.bases_dev.mean(dim=0))

    # ── Loss hooks (only meaningful for shared_dev variants) ─────────────────

    def compute_anchor_loss(self) -> Tensor:
        if not any(hasattr(h, "anchor_loss") for h in self.roi_readouts.values()):
            return torch.zeros((), device=next(self.parameters()).device)
        return sum(
            h.anchor_loss() for h in self.roi_readouts.values() if hasattr(h, "anchor_loss")
        )

    def compute_ortho_loss(self) -> Tensor:
        if not any(hasattr(h, "ortho_loss") for h in self.roi_readouts.values()):
            return torch.zeros((), device=next(self.parameters()).device)
        return sum(
            h.ortho_loss() for h in self.roi_readouts.values() if hasattr(h, "ortho_loss")
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _check_bases_match_layout(self, roi_bases: dict[str, Tensor]) -> None:
        missing = [n for n in self.roi_layout.roi_names if n not in roi_bases]
        if missing:
            raise ValueError(
                f"roi_bases is missing {len(missing)} ROI keys from the layout, e.g. {missing[:3]}."
            )
        for name in self.roi_layout.roi_names:
            expected = self.roi_layout.roi_dims_dict[name]
            actual = roi_bases[name].shape[1]
            if actual != expected:
                raise ValueError(
                    f"roi_bases['{name}'] has shape (..., {actual}) but layout expects {expected} voxels."
                )

    # ── Convenience: load weights from a legacy-format checkpoint dict ──

    def load_legacy_state_dict(self, state_dict: dict, strict: bool = False) -> tuple[list, list]:
        """Load a legacy-format ``state_dict`` (one produced by the original
        training code) into this RABBiT model. The legacy keys use the old
        module names; we remap them here.

        Returns ``(missing_keys, unexpected_keys)`` from the underlying
        ``load_state_dict`` call.
        """
        remapped = _remap_legacy_keys(state_dict)
        result = self.load_state_dict(remapped, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy-format checkpoint remapping
#
# Old class names in the original training code:
#   Wav2VecLoRATransFriendsSSL.lora_wav2vec.{model.*}   → backbone.backbone.{model.*}
#   Wav2VecLoRATransFriendsSSL.ssl_model.*              → (drop — SSL head)
#   Wav2VecLoRATransFriendsSSL.temporal_brain_encoder.* → top-level (after rename)
#     .input_proj → input_proj
#     .input_norm → input_norm
#     .position_encoder → position_encoder
#     .query_embed → query_embed
#     .decoder → decoder
#     .roi_readouts → roi_readouts
# ─────────────────────────────────────────────────────────────────────────────


def _remap_legacy_keys(state_dict: dict) -> dict:
    """Translate legacy module path prefixes to RABBiT module paths."""
    out = {}
    for k, v in state_dict.items():
        new_k = k

        # Strip the SSL head — RABBiT does not contain it.
        if k.startswith("ssl_model.") and not k.startswith("ssl_model.wav2vec2."):
            continue

        # The legacy `ssl_model.wav2vec2.*` is identical weights to `lora_wav2vec.*`
        # (wav2vec2 backbone case), so prefer the lora_wav2vec path and skip
        # this duplicate prefix.
        if k.startswith("ssl_model.wav2vec2."):
            continue

        # Backbone path.
        if k.startswith("lora_wav2vec."):
            new_k = "backbone.backbone." + k[len("lora_wav2vec.") :]
        elif k.startswith("ssl_model."):
            # WavLM case: the legacy checkpoint stores it directly under ssl_model
            # (no separate lora_wav2vec). Map onto backbone.backbone.
            new_k = "backbone.backbone." + k[len("ssl_model.") :]

        # Brain-encoder modules — strip the temporal_brain_encoder prefix.
        elif k.startswith("temporal_brain_encoder."):
            new_k = k[len("temporal_brain_encoder.") :]

        out[new_k] = v
    return out
