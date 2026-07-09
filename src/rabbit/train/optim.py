"""Optimiser parameter-group construction for RABBiT training.

Three-way split with separate learning rates:

  1. **backbone** — wav2vec2 / WavLM transformer parameters (LoRA adapters
     when ``lora.rank > 0``, otherwise the un-frozen attention layers). Slow
     LR, weight-decay on.
  2. **brain encoder** — input projection, position encoder, ROI queries,
     transformer decoder, plus the *coefficient heads* of the per-ROI readout
     (``coeff_shared``, ``coeff_dev``). Fast LR, weight-decay on.
  3. **bases** — the learnable ``bases_shared`` and ``bases_dev`` tensors in
     shared-deviation readouts. Slow LR, **no** weight decay (the anchor
     regulariser controls drift).
"""
from __future__ import annotations

from typing import Iterable

import torch.nn as nn

from ..model import RABBiT


__all__ = ["build_param_groups"]


def _backbone_params(model: RABBiT) -> list[nn.Parameter]:
    return [p for p in model.backbone.parameters() if p.requires_grad]


def _bases_params(model: RABBiT) -> list[nn.Parameter]:
    out: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "roi_readouts" in name and (
            name.endswith("bases_shared") or name.endswith("bases_dev")
        ):
            out.append(p)
    return out


def _brain_encoder_params(
    model: RABBiT, exclude_ids: Iterable[int],
) -> list[nn.Parameter]:
    """Everything trainable that's not in the backbone or bases groups."""
    exclude = set(exclude_ids)
    backbone_ids = {id(p) for p in _backbone_params(model)}
    out: list[nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in exclude or id(p) in backbone_ids:
            continue
        out.append(p)
    return out


def build_param_groups(
    model: RABBiT,
    *,
    base_lr: float = 1.0e-4,
    readout_lr: float = 1.0e-3,
    bases_lr: float = 1.0e-4,
    weight_decay: float = 1.0e-2,
) -> list[dict]:
    """Return the optimiser ``param_groups`` list.

    Pass directly into ``torch.optim.AdamW(build_param_groups(model, ...))``.
    The three groups in returned order are *backbone*, *brain_encoder*,
    *bases*. ``bases`` gets ``weight_decay=0`` regardless of the
    ``weight_decay`` argument.
    """
    bases = _bases_params(model)
    bases_ids = {id(p) for p in bases}
    brain = _brain_encoder_params(model, exclude_ids=bases_ids)
    backbone = _backbone_params(model)

    return [
        {"params": backbone, "lr": base_lr,    "weight_decay": weight_decay, "_group": "backbone"},
        {"params": brain,    "lr": readout_lr, "weight_decay": weight_decay, "_group": "brain_encoder"},
        {"params": bases,    "lr": bases_lr,   "weight_decay": 0.0,           "_group": "bases"},
    ]


def count_group_params(param_groups: list[dict]) -> dict[str, int]:
    """Param-count by group name. Useful for logging."""
    out: dict[str, int] = {}
    for g in param_groups:
        name = g.get("_group", "unnamed")
        out[name] = sum(p.numel() for p in g["params"])
    return out
