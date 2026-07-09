"""End-to-end held-out evaluation on narratives-style stories.

Given a trained RABBiT model, an audio stimulus, a held-out subjects' fMRI,
and the .report file with TR triggers, produce per-vertex correlations
(``vertex_corr``) plus optional per-ROI summary and predictions.

A from-scratch reimplementation of the zero-shot narratives evaluation used in
the paper, split into composable pieces (audio alignment, forward pass,
correlation).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from ..inference.audio import TARGET_SAMPLE_RATE, make_delayed
from ..model import RABBiT
from .audio_align import align_wav_to_trs_eval, parse_report, resample_fmri_to_target_tr
from .fs6 import build_fs6_roi_vertex_indices, extract_roi_from_fs6
from .metrics import correlation_4fold, pearson_r_per_voxel


__all__ = [
    "NarrativesStory",
    "NarrativesResult",
    "load_narratives_audio",
    "load_narratives_mean_fmri",
    "evaluate_on_narratives",
]


# ─────────────────────────────────────────────────────────────────────────────
# Story bundle + result containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NarrativesStory:
    """All inputs needed to evaluate one held-out narratives story."""

    name:        str                    # human-readable story id, e.g. "21styear"
    audio_path:  Path                   # 16 kHz mono .npy waveform
    report_path: Path                   # `.report` file with TR triggers + sound-start
    fmri_paths:  list[Path]             # per-subject fs6 fMRI .npy files (full 81924-dim)


@dataclass
class NarrativesResult:
    """One model on one story: per-vertex correlation + provenance."""

    vertex_corr:    np.ndarray                            # (flat_dim,) float64
    roi_corrs:      "OrderedDict[str, float]"             # roi_name -> mean r
    applied_shift:  int
    n_folds:        int
    n_trs:          int
    roi_names:      tuple[str, ...]
    shift_sweep:    "OrderedDict[int, float]" = field(default_factory=OrderedDict)
    preds:          Optional[np.ndarray] = None           # (n_trs, flat_dim) if requested
    fmri_target:    Optional[np.ndarray] = None           # (n_trs, flat_dim) if requested


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────


def load_narratives_audio(audio_path: str | Path) -> torch.Tensor:
    """Load a narratives .npy waveform at 16 kHz. NaN-safe."""
    wav = np.load(audio_path, allow_pickle=False)
    return torch.from_numpy(np.nan_to_num(np.asarray(wav, dtype=np.float32))).squeeze()


def load_narratives_mean_fmri(fmri_paths: Iterable[str | Path]) -> np.ndarray:
    """Load N subjects' fs6 fMRI ``.npy`` files and return their mean.

    Returns ``(n_TRs, 81924)`` float64.
    """
    paths = [Path(p) for p in fmri_paths]
    if not paths:
        raise ValueError("fmri_paths is empty.")
    stack = np.stack([np.load(p) for p in paths], axis=0)   # (n_subj, n_TRs, 81924)
    return stack.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end eval
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_on_narratives(
    model: RABBiT,
    story: NarrativesStory,
    *,
    tr_length: float = 1.49,
    hrf_delay: int = 6,
    trim_start: int = 10,
    trim_end: int = 9,
    n_folds: int = 4,
    batch_size: int = 8,
    shift: Optional[int] = None,
    shift_search: Iterable[int] = (0, 1, 2, 3),
    subjects_dir: Optional[str] = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    apply_processor: bool = True,
    return_predictions: bool = False,
    verbose: bool = False,
) -> NarrativesResult:
    """Zero-shot evaluation of a RABBiT model on a narratives story.

    Pipeline:
        1. Load audio + parse report (TR times, sound_start).
        2. Load 20 subjects' fs6 fMRI, mean, resample to ``tr_length``.
        3. Extract the 30 ROIs from fs6 (z-scored per ROI).
        4. Align audio to TR grid (friends convention for non-2.0 s TR, moth for 2.0 s).
        5. Apply HRF-delay stack, trim ``trim_start`` / ``trim_end`` edge TRs.
        6. Forward through ``model`` per TR (batched), optionally via the HF
           processor.
        7. Pick the best forward shift from ``shift_search`` (or use the
           explicit ``shift`` argument if given).
        8. 4-fold z-scored per-vertex Pearson r → ``vertex_corr``.

    Args:
        model: a ``RABBiT`` in eval mode (avg-dev should already be applied
            for held-out-subject zero-shot eval).
        story: ``NarrativesStory`` bundle.
        tr_length: target TR in seconds. 1.49 for friends-aligned models, 2.0
            for moth-aligned models.
        hrf_delay: number of HRF lag TRs; the model sees stacked windows
            ``range(0, hrf_delay+1)``.
        trim_start, trim_end: number of leading / trailing TRs to drop.
        n_folds: number of segments for the per-vertex z-scored correlation.
        batch_size: forward-pass batch size in TRs.
        shift: if not None, force this shift instead of searching.
        shift_search: candidate forward shifts (in TRs) to sweep.
        subjects_dir: optional FreeSurfer subjects directory for the HCPMMP1
            atlas; defaults to ``mne``'s sample data.
        device: torch device for the model and forward pass.
        apply_processor: when True, run audio through the backbone HF
            processor (matches training).
        return_predictions: include the (n_trs, flat_dim) predictions and
            fMRI targets in the result. Doubles memory; off by default.
        verbose: print progress every 10 batches.

    Returns:
        ``NarrativesResult``.
    """
    device = torch.device(device)
    model.to(device).eval()

    wav_convention = "friends" if abs(tr_length - 2.0) > 1e-3 else "moth"
    delays = list(range(0, hrf_delay + 1))

    # ── 1. audio + report ────────────────────────────────────────────────────
    wav = load_narratives_audio(story.audio_path)
    tr_times_report, sound_start = parse_report(story.report_path)

    # ── 2. mean fMRI + resample ──────────────────────────────────────────────
    mean_fmri = load_narratives_mean_fmri(story.fmri_paths)
    fmri_resamp, new_tr_times = resample_fmri_to_target_tr(
        mean_fmri, tr_times_report, sound_start,
        target_tr=tr_length, convention=wav_convention,
    )

    # ── 3. 30-ROI fs6 extraction ─────────────────────────────────────────────
    roi_verts = build_fs6_roi_vertex_indices(subjects_dir=subjects_dir)
    fmri_roi = extract_roi_from_fs6(fmri_resamp, roi_verts, z_score=True)

    # ── 4. audio TR alignment ────────────────────────────────────────────────
    aligned_wav = align_wav_to_trs_eval(
        wav, new_tr_times, tr_len=tr_length,
        convention=wav_convention, sample_rate=TARGET_SAMPLE_RATE,
    )

    # ── 5. HRF stack + trim ──────────────────────────────────────────────────
    delayed = make_delayed(aligned_wav, delays).float()
    delayed = delayed[trim_start:-trim_end] if trim_end > 0 else delayed[trim_start:]
    fmri_trimmed = fmri_roi[trim_start:-trim_end] if trim_end > 0 else fmri_roi[trim_start:]
    n_trs = min(delayed.shape[0], fmri_trimmed.shape[0])
    delayed = delayed[:n_trs]
    fmri_trimmed = fmri_trimmed[:n_trs]
    if n_trs == 0:
        raise ValueError(
            f"After trim no TRs remain (audio duration "
            f"{wav.shape[0]/TARGET_SAMPLE_RATE:.1f}s, tr_length={tr_length})."
        )

    # ── 6. forward pass ──────────────────────────────────────────────────────
    if apply_processor and model.backbone.processor is not None:
        proc = model.backbone.processor
        sr = model.backbone.sampling_rate
        delayed = torch.stack(
            [
                proc(
                    torch.nan_to_num(delayed[i], nan=0.0, posinf=0.0, neginf=0.0),
                    return_tensors="pt",
                    sampling_rate=sr,
                ).input_values[0]
                for i in range(delayed.shape[0])
            ],
            dim=0,
        )

    preds_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, n_trs, batch_size):
            x = delayed[s : s + batch_size].to(device)
            sid = None
            if model.needs_subject_indices:
                sid = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            preds_chunks.append(model(x, sid)["flat_predictions"].cpu().numpy())
            if verbose and (s // batch_size) % 10 == 9:
                done = min(s + batch_size, n_trs)
                print(f"  inference {done}/{n_trs} TRs")
    preds = np.concatenate(preds_chunks, axis=0)

    # ── 7. shift search ──────────────────────────────────────────────────────
    sweep: "OrderedDict[int, float]" = OrderedDict()
    best_shift, best_r = 0, -np.inf
    candidates = [shift] if shift is not None else list(shift_search)
    for sh in candidates:
        if sh > 0:
            r = pearson_r_per_voxel(preds[:-sh], fmri_trimmed[sh:])
        else:
            r = pearson_r_per_voxel(preds, fmri_trimmed)
        mr = float(r.mean())
        sweep[int(sh)] = mr
        if mr > best_r:
            best_shift, best_r = int(sh), mr

    applied_shift = int(shift) if shift is not None else best_shift
    if applied_shift > 0:
        p_aligned = preds[:-applied_shift]
        t_aligned = fmri_trimmed[applied_shift:]
    else:
        p_aligned, t_aligned = preds, fmri_trimmed

    # ── 8. 4-fold z-scored correlation ───────────────────────────────────────
    vertex_corr = correlation_4fold(p_aligned, t_aligned, n_folds=n_folds).astype(np.float64)

    # Per-ROI summary (mean over each ROI's vertex block).
    roi_corrs: "OrderedDict[str, float]" = OrderedDict()
    offset = 0
    for roi_key, verts in roi_verts.items():
        n_v = len(verts)
        roi_corrs[roi_key] = float(vertex_corr[offset : offset + n_v].mean())
        offset += n_v

    return NarrativesResult(
        vertex_corr=vertex_corr,
        roi_corrs=roi_corrs,
        applied_shift=applied_shift,
        n_folds=n_folds,
        n_trs=n_trs,
        roi_names=tuple(roi_verts.keys()),
        shift_sweep=sweep,
        preds=p_aligned if return_predictions else None,
        fmri_target=t_aligned if return_predictions else None,
    )
