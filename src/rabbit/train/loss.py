"""Training loss for RABBiT: per-ROI correlation + MSE.

The trainer minimises::

    L = lambda_corr * mean_over_ROIs( 1 - pearson(pred_roi, target_roi) )
      + lambda_l2   * MSE(pred, target)

with optional shared-deviation anchor and orthonormality regularisers
added at the trainer level (computed from ``model.compute_anchor_loss()`` and
``model.compute_ortho_loss()`` respectively).

The correlation term is computed in fp32 even when the forward pass is in
bf16/fp16, since per-vertex correlation on a high-dim output is numerically
sensitive to low-precision arithmetic.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


__all__ = ["UnscaledCorrelationLoss"]


class UnscaledCorrelationLoss(nn.Module):
    """Per-ROI Pearson-correlation loss with an additive MSE term.

    Args:
        roi_slices: ``OrderedDict[roi_name -> slice]`` mapping each ROI to its
            voxel block in the flat prediction tensor (use
            ``RABBiT.roi_layout.roi_slices``).
        lambda_corr: weight on the (1 - mean per-ROI corr) term.
        lambda_l2: weight on the MSE term.

    Returns a scalar loss.
    """

    def __init__(
        self,
        roi_slices: "OrderedDict[str, slice]",
        lambda_corr: float = 1.0,
        lambda_l2: float = 1.0,
    ) -> None:
        super().__init__()
        self.roi_slices = roi_slices
        self.lambda_corr = float(lambda_corr)
        self.lambda_l2 = float(lambda_l2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = pred.float()
        target_f = target.float()

        per_roi_corr: list[torch.Tensor] = []
        for _, slc in self.roi_slices.items():
            p = pred_f[:, slc]
            t = target_f[:, slc]
            p_c = p - p.mean(dim=-1, keepdim=True)
            t_c = t - t.mean(dim=-1, keepdim=True)
            p_std = p_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            t_std = t_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            corr = (p_c * t_c).sum(dim=-1) / (p_std.squeeze(-1) * t_std.squeeze(-1))
            per_roi_corr.append((1.0 - corr).mean())

        corr_loss = torch.stack(per_roi_corr).mean()
        l2_loss = torch.nn.functional.mse_loss(pred_f, target_f)
        return self.lambda_corr * corr_loss + self.lambda_l2 * l2_loss
