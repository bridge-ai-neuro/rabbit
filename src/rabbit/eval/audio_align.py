"""Eval-time audio + fMRI alignment for narratives-style stories.

These helpers handle the difference between training-time and eval-time
alignment conventions. Two conventions are supported:

  * **friends**: the convention used for the Friends dataset and any model
    trained at TR ≠ 2.0 s. TR ``k`` (1-indexed; `tr_times[k] = sound_start +
    (k+1)*tr_len`) covers audio from ``sound_start + k*tr_len`` to
    ``sound_start + (k+1)*tr_len``. The first TR (k=0) covers audio from
    sample 0 (silence before ``sound_start``) up to ``sound_start + tr_len``.

  * **moth**: the convention used for the moth radio hour at TR=2.0 s. The
    first TR gets a +1 s pre-pad to handle the dataset's 1-second silence
    offset; subsequent TRs cover ``[tr_times[k-1], tr_times[k]]``.

Picking the right convention is required for numerical reproduction of the
saved reference eval outputs. The high-level wrapper in ``rabbit.eval.narratives``
selects automatically based on the target TR.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.interpolate import interp1d


__all__ = [
    "parse_report",
    "resample_fmri_to_target_tr",
    "align_wav_to_trs_eval",
]


# ─────────────────────────────────────────────────────────────────────────────
# Stimulus report parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_report(report_path: str | Path) -> tuple[np.ndarray, float]:
    """Parse a narratives ``.report`` file.

    The format is `<time> <label>` per line; `init-trigger` / `trigger` lines
    mark TR triggers and `sound-start` marks the audio onset.

    Returns:
        ``tr_times``: (n_triggers,) array of TR trigger times in seconds.
        ``sound_start``: scalar audio onset time in seconds.
    """
    tr_times: list[float] = []
    sound_start: float | None = None
    for line in open(report_path):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        t = float(parts[0])
        label = " ".join(parts[1:])
        if label in ("init-trigger", "trigger"):
            tr_times.append(t)
        elif label == "sound-start":
            sound_start = t
    if sound_start is None:
        raise ValueError(f"No 'sound-start' line in {report_path!r}")
    return np.array(tr_times), float(sound_start)


# ─────────────────────────────────────────────────────────────────────────────
# fMRI temporal resampling
# ─────────────────────────────────────────────────────────────────────────────


def resample_fmri_to_target_tr(
    fmri: np.ndarray,
    src_tr_times: np.ndarray,
    sound_start: float,
    target_tr: float,
    convention: str = "friends",
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate fMRI onto a new TR grid.

    Args:
        fmri: ``(n_TRs_src, n_vertices)`` original fMRI sampled at the source
            TR grid implied by ``src_tr_times``.
        src_tr_times: trigger times in seconds — output of ``parse_report``.
        sound_start: audio onset time in seconds.
        target_tr: desired TR period in seconds.
        convention: 'friends' uses ``tr_times = sound_start + (k+1)*target_tr``
            (TR end times). 'moth' uses ``tr_times = sound_start + 1.0 +
            k*target_tr`` (1-second offset baked in).

    Returns:
        ``resampled_fmri``: ``(n_TRs_new, n_vertices)`` float32.
        ``new_tr_times``: ``(n_TRs_new,)`` new TR times in seconds.
    """
    n_fmri = fmri.shape[0]
    src_times = src_tr_times[-n_fmri:] if len(src_tr_times) > n_fmri else src_tr_times[:n_fmri]
    audio_duration = src_times[-1] - sound_start
    n_new = int(audio_duration / target_tr)
    if convention == "moth":
        new_tr_times = sound_start + 1.0 + np.arange(n_new) * target_tr
    elif convention == "friends":
        new_tr_times = sound_start + (np.arange(n_new) + 1) * target_tr
    else:
        raise ValueError(f"Unknown convention: {convention!r}; expected 'moth' or 'friends'.")
    valid = (new_tr_times >= src_times[0]) & (new_tr_times <= src_times[-1])
    new_tr_times = new_tr_times[valid]
    interp_fn = interp1d(src_times, fmri, axis=0, kind="linear", bounds_error=True)
    resampled = interp_fn(new_tr_times).astype(np.float32)
    return resampled, new_tr_times


# ─────────────────────────────────────────────────────────────────────────────
# Audio → TR-grid alignment
# ─────────────────────────────────────────────────────────────────────────────


def align_wav_to_trs_eval(
    wav: torch.Tensor,
    tr_times: Sequence[float],
    tr_len: float,
    convention: str = "friends",
    sample_rate: int = 16_000,
) -> torch.Tensor:
    """Slice a waveform into per-TR windows for eval.

    Differs from training-time alignment in two ways:

      1. Two conventions ('moth', 'friends') for how the first TR's window
         is constructed — the first TR's window depends on dataset-specific
         offsets.
      2. ``tr_times`` is absolute (seconds since recording start) rather
         than zero-anchored.

    Args:
        wav: 1-D float waveform at ``sample_rate``.
        tr_times: per-TR end times in seconds.
        tr_len: TR period in seconds. Also the per-TR audio-window length.
        convention: 'moth' or 'friends'.
        sample_rate: input audio sample rate.

    Returns:
        ``(n_TRs, tr_len * sample_rate)`` float tensor.
    """
    expected = int(tr_len * sample_rate)
    aligned: list[torch.Tensor] = []
    for t, trtime in enumerate(tr_times):
        if convention == "moth":
            if trtime < tr_len:
                pad_n = int((tr_len - 1.0) * sample_rate) if tr_len >= 1.0 else int(tr_len * sample_rate // 2)
                aligned.append(torch.cat([
                    torch.zeros(pad_n, dtype=wav.dtype),
                    wav[: expected - pad_n],
                ]))
                continue
            sidx = int(sample_rate * tr_times[t - 1])
            eidx = int(sample_rate * trtime)
        elif convention == "friends":
            sidx = int(sample_rate * tr_times[t - 1]) if t > 0 else 0
            eidx = int(sample_rate * trtime)
        else:
            raise ValueError(f"Unknown convention: {convention!r}.")

        chunk = wav[sidx:eidx]
        if chunk.shape[0] < expected:
            chunk = torch.cat([
                torch.zeros(expected - chunk.shape[0], dtype=wav.dtype),
                chunk,
            ])
        elif chunk.shape[0] > expected:
            chunk = chunk[:expected]
        aligned.append(chunk)
    return torch.vstack(aligned)
