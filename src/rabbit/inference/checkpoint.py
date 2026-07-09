"""Build a ``RABBiT`` model and load a checkpoint from a config + path.

Supports two checkpoint conventions:

  * **Legacy-style** — ``{'model': state_dict}`` produced by the original
    training code (old module names like ``temporal_brain_encoder.*`` and
    ``lora_wav2vec.*``). Remapped to RABBiT module paths on load.

  * **RABBiT-native** — ``{'model': state_dict, 'config': dict,
    'roi_names': list, 'subject_ids': list, 'rabbit_version': str}`` produced
    by future RABBiT training.

Both are detected automatically.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..model import RABBiT, SpeechBackbone, build_fs6_layout, build_flat_roi_layout
from ..utils.config import load_config


# ─────────────────────────────────────────────────────────────────────────────
# PCA basis loading
# ─────────────────────────────────────────────────────────────────────────────


def load_pca_bases(npz_path: str, roi_names: list[str]) -> "OrderedDict[str, torch.Tensor]":
    """Load an ``OrderedDict[roi_name -> (K, V) tensor]`` of PCA bases.

    Expected NPZ layout: keys ``{roi}_bases`` of shape ``(K, V_roi)``.
    """
    data = np.load(npz_path, allow_pickle=True)
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name in roi_names:
        key = f"{name}_bases"
        if key not in data.files:
            raise KeyError(
                f"PCA bases NPZ {npz_path!r} is missing key {key!r}. "
                f"Available keys: {list(data.files)[:5]}..."
            )
        out[name] = torch.from_numpy(data[key]).float()
    return out


def compute_deviation_init(
    avg_bases: "OrderedDict[str, torch.Tensor]",
    subj_bases: "OrderedDict[str, torch.Tensor]",
    R: int,
) -> "OrderedDict[str, torch.Tensor]":
    """For each ROI, return the top-R right-singular-vector directions of
    (subject_PCA − average_PCA) — the per-subject deviation init used by
    the shared+deviation readout.
    """
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for roi in avg_bases:
        avg_b = avg_bases[roi]
        sub_b = subj_bases[roi]
        K_min = min(avg_b.shape[0], sub_b.shape[0])
        residual = sub_b[:K_min] - avg_b[:K_min]
        R_eff = min(R, K_min)
        _, _, Vt = torch.linalg.svd(residual, full_matrices=False)
        out[roi] = Vt[:R_eff]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def load_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: Optional[str | Path] = None,
    overrides: Optional[list[str]] = None,
    device: str | torch.device = "cpu",
    use_avg_dev: bool = True,
) -> RABBiT:
    """Build a ``RABBiT`` model from a config and load weights from a checkpoint.

    Args:
        checkpoint_path: ``.pt`` file. Both legacy-style and RABBiT-native
            checkpoints are accepted; legacy keys are remapped automatically.
        config_path: YAML config matching the checkpoint. Required for
            legacy-style checkpoints (they store no config inline). For
            RABBiT-native checkpoints the embedded config is used and this
            argument is optional.
        overrides: optional ``["key.path=value", ...]`` to apply on top of
            the loaded config.
        device: target device. Model is moved with ``.to(device)`` before
            return.
        use_avg_dev: when ``True`` and the readout supports it, applies the
            avg-dev trick after load (replaces ``bases_dev[0]`` with the mean
            across the other subject slots). Standard zero-shot inference
            convention.

    Returns:
        ``RABBiT`` model in eval mode, on the target device.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict, embedded_config = _split_checkpoint(ckpt)

    if config_path is None and embedded_config is None:
        raise ValueError(
            "config_path is required for legacy-style checkpoints "
            "(this checkpoint does not embed its config)."
        )
    if embedded_config is not None and config_path is None:
        config = embedded_config
    else:
        config = load_config(str(config_path), overrides=overrides)

    model = build_rabbit_from_config(config)
    missing, unexpected = model.load_legacy_state_dict(state_dict, strict=False)

    # Surface anything load-bearing that didn't match — finite warning list,
    # not an error, so a checkpoint with extra SSL keys still loads.
    if missing:
        head = ", ".join(missing[:3]) + ("..." if len(missing) > 3 else "")
        print(f"[rabbit] load_from_checkpoint: {len(missing)} missing keys (e.g. {head})")
    if unexpected:
        head = ", ".join(unexpected[:3]) + ("..." if len(unexpected) > 3 else "")
        print(f"[rabbit] load_from_checkpoint: {len(unexpected)} unexpected keys (e.g. {head})")

    if use_avg_dev:
        model.use_avg_deviation(slot=0)

    return model.to(device).eval()


def _split_checkpoint(ckpt) -> tuple[dict, Optional[dict]]:
    """Return ``(state_dict, embedded_config_or_None)``."""
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt.get("config")
    if isinstance(ckpt, dict) and any(k for k in ckpt.keys() if not isinstance(k, str)):
        # Looks like a raw state_dict already.
        return dict(ckpt), None
    return ckpt, None


# ─────────────────────────────────────────────────────────────────────────────
# Build a RABBiT model from a config dict
# ─────────────────────────────────────────────────────────────────────────────


def build_rabbit_from_config(config: dict) -> RABBiT:
    """Instantiate a ``RABBiT`` (and its bases / per-subject dev init) from a
    YAML config dict produced by ``load_config``.
    """
    backbone_cfg = config.get("backbone") or config.get("wav2vec") or {}
    lora_cfg = config.get("lora", {})
    enc_cfg = config.get("encoder") or config.get("brain_encoder", {})
    data_cfg = config.get("data", {})
    param_cfg = config.get("parametric", {})
    sd_cfg = config.get("shared_deviation", {})

    readout_type = config.get("readout_type") or config.get("encoder_type", "shared_deviation")

    # ── ROI layout ────────────────────────────────────────────────────────────
    roi_names = data_cfg.get("roi_names")
    if roi_names:
        # Build a layout matching the order in the config (needed because
        # checkpoints store readouts ordered by config).
        from ..model.roi_layout import FS6_ROI_DIMS, build_flat_roi_layout as _bld
        layout_dims = OrderedDict((n, FS6_ROI_DIMS[n]) for n in roi_names)
        roi_layout = _bld(layout_dims)
    else:
        roi_layout = build_fs6_layout()

    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone = SpeechBackbone(
        model_name=backbone_cfg.get("model_name", "facebook/wav2vec2-base-960h"),
        lora_rank=int(lora_cfg.get("rank", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.1)),
        sampling_rate=int(backbone_cfg.get("sampling_rate", 16_000)),
    )

    # ── Bases (if needed) ─────────────────────────────────────────────────────
    roi_bases = None
    dev_roi_bases = None
    shared_per_dataset = None
    subject_to_dataset = None

    R = int(sd_cfg.get("R", 15))
    subjects = data_cfg.get("subjects", [])

    if readout_type in ("parametric", "shared_deviation"):
        bases_dir = param_cfg.get("bases_dir")
        if not bases_dir:
            raise ValueError(f"readout_type={readout_type!r} requires parametric.bases_dir in config.")
        avg_path = os.path.join(bases_dir, "average", "pca_bases.npz")
        roi_bases = load_pca_bases(avg_path, list(roi_layout.roi_names))

    if readout_type == "shared_deviation":
        dev_roi_bases = _load_dev_bases(bases_dir, subjects, roi_bases, list(roi_layout.roi_names), R)

    if readout_type == "dataset_shared_deviation":
        ds_dirs = param_cfg.get("dataset_bases_dirs")
        if not isinstance(ds_dirs, dict) or not ds_dirs:
            raise ValueError(
                "readout_type='dataset_shared_deviation' requires "
                "parametric.dataset_bases_dirs as a mapping {dataset_name: bases_dir}."
            )
        # Subject-to-dataset routing: int → moth, str → friends (legacy training convention).
        # If you train multi-dataset models with a different routing, override here.
        def _ds(s):
            return "moth" if isinstance(s, int) else "friends"
        present = [n for n in ("moth", "friends") if any(_ds(s) == n for s in subjects)]
        ds_to_idx = {n: i for i, n in enumerate(present)}
        subject_to_dataset = [ds_to_idx[_ds(s)] for s in subjects]
        shared_per_dataset = []
        for name in present:
            avg = os.path.join(ds_dirs[name], "average", "pca_bases.npz")
            shared_per_dataset.append(load_pca_bases(avg, list(roi_layout.roi_names)))
        # Deviation: each subject's bases anchored against its own dataset average.
        dev_roi_bases = {roi: [] for roi in roi_layout.roi_names}
        for s in subjects:
            ds_name = _ds(s)
            ref = shared_per_dataset[ds_to_idx[ds_name]]
            sub_key = f"UTS0{s}" if ds_name == "moth" else f"sub-{s}"
            subj_path = os.path.join(ds_dirs[ds_name], sub_key, "pca_bases.npz")
            subj_b = load_pca_bases(subj_path, list(roi_layout.roi_names))
            dev = compute_deviation_init(ref, subj_b, R)
            for roi in roi_layout.roi_names:
                dev_roi_bases[roi].append(dev[roi])

    # ── Assemble model ────────────────────────────────────────────────────────
    return RABBiT(
        readout_type=readout_type,
        backbone=backbone,
        roi_layout=roi_layout,
        hidden_dim=int(enc_cfg.get("hidden_dim", 256)),
        nhead=int(enc_cfg.get("nhead", 8)),
        num_layers=int(enc_cfg.get("num_layers", 2)),
        dim_feedforward=int(enc_cfg.get("dim_feedforward", 1024)),
        dropout=float(enc_cfg.get("dropout", 0.1)),
        position_embedding=enc_cfg.get("position_embedding", "sine"),
        max_positions=int(enc_cfg.get("max_positions", 4096)),
        query_init_std=float(enc_cfg.get("query_init_std", 0.05)),
        use_self_attention=bool(enc_cfg.get("use_self_attention", True)),
        readout_bias=bool(enc_cfg.get("readout_bias", True)),
        roi_bases=roi_bases,
        dev_roi_bases=dev_roi_bases,
        shared_roi_bases_per_dataset=shared_per_dataset,
        subject_to_dataset=subject_to_dataset,
        uniform_dev_bias_rois=sd_cfg.get("uniform_dev_bias_rois"),
    )


def _load_dev_bases(
    bases_dir: str,
    subjects: list,
    avg_bases: "OrderedDict[str, torch.Tensor]",
    roi_names: list[str],
    R: int,
) -> dict[str, list[torch.Tensor]]:
    dev_roi_bases = {roi: [] for roi in roi_names}
    for s in subjects:
        sub_key = f"UTS0{s}" if isinstance(s, int) else f"sub-{s}"
        path = os.path.join(bases_dir, sub_key, "pca_bases.npz")
        subj_b = load_pca_bases(path, roi_names)
        dev = compute_deviation_init(avg_bases, subj_b, R)
        for roi in roi_names:
            dev_roi_bases[roi].append(dev[roi])
    return dev_roi_bases
