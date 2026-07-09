"""Per-vertex correlation metrics.

The headline metric in the paper is per-vertex Pearson `r` between predicted
and held-out group-mean fMRI, averaged over an ROI for the `r_group` summary
table. Both pieces — the raw per-vertex r and the 4-fold z-scored variant —
live here.
"""
from __future__ import annotations

import numpy as np


__all__ = ["pearson_r_per_voxel", "correlation_4fold", "zscore"]


def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score along the given axis. Vertices with zero variance pass through."""
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return (x - m) / s


def pearson_r_per_voxel(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-vertex Pearson correlation along axis 0.

    Args:
        pred, target: ``(n_TRs, n_vertices)`` float arrays of the same shape.

    Returns:
        ``(n_vertices,)`` float64 array. Vertices with zero variance in either
        ``pred`` or ``target`` return 0.0.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} != target {target.shape}")
    pred_m = pred - pred.mean(axis=0, keepdims=True)
    targ_m = target - target.mean(axis=0, keepdims=True)
    num = (pred_m * targ_m).sum(axis=0)
    den = np.sqrt((pred_m ** 2).sum(axis=0) * (targ_m ** 2).sum(axis=0))
    r = np.zeros(pred.shape[1], dtype=np.float64)
    valid = den > 1e-12
    r[valid] = num[valid] / den[valid]
    return r


def correlation_4fold(
    preds: np.ndarray, targets: np.ndarray, n_folds: int = 4,
) -> np.ndarray:
    """N-fold z-scored per-vertex Pearson r.

    The full time-series is split into ``n_folds`` contiguous segments. Within
    each segment, predictions and targets are z-scored independently per
    vertex. Then Pearson r is computed once over the full concatenated array.

    Paper convention. Removes slow drifts and per-fold mean offsets so the
    correlation reflects within-window temporal pattern matching.
    """
    n_trs = preds.shape[0]
    fold_size = n_trs // n_folds
    zs_preds = np.zeros_like(preds)
    zs_targets = np.zeros_like(targets)
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n_trs
        zs_preds[start:end] = zscore(preds[start:end], axis=0)
        zs_targets[start:end] = zscore(targets[start:end], axis=0)
    return pearson_r_per_voxel(zs_preds, zs_targets)
