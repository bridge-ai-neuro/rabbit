"""Visualisation helpers for RABBiT predictions.

Three plot helpers, all matplotlib-only (no pycortex / brain-surface deps):

  * :func:`plot_audio_and_responses` — waveform envelope on top, per-ROI mean
    response traces below, time-aligned.
  * :func:`plot_roi_grid` — small-multiples grid of per-ROI mean response
    traces.
  * :func:`plot_narratives_summary` — bar chart of per-ROI mean r_group from
    a ``rabbit.eval.NarrativesResult``.

Style follows the paper convention (serif/STIX math, tab-style or Tol-Muted
palette) so figures generated from the demo notebook are paper-ready.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .inference import RABBiTPrediction
    from .eval import NarrativesResult


__all__ = [
    "PAPER_RC",
    "plot_audio_and_responses",
    "plot_roi_grid",
    "plot_narratives_summary",
    "audio_envelope",
    "DEFAULT_LANGUAGE_ROIS",
]


# ─────────────────────────────────────────────────────────────────────────────
# Style + colours
# ─────────────────────────────────────────────────────────────────────────────


PAPER_RC = {
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset":  "stix",
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "axes.linewidth":    1.0,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
}

# Paul Tol "Muted" stage palette — warm-to-cool gradient mirroring the
# auditory-to-frontal language hierarchy. Avoids tab10 (used elsewhere in
# the paper for model identity).
_STAGE_COLOR = {
    "aud_primary":     "#882255",   # wine
    "aud_belt":        "#CC6677",   # rose
    "stg_sts":         "#AA4499",   # purple
    "posterior_temp":  "#6699CC",   # steel blue
    "temporal_pole":   "#44AA99",   # teal
    "angular_gyrus":   "#117733",   # dark green
    "supramarginal":   "#999933",   # olive
    "ifg":             "#332288",   # indigo
    "mfg_dlpfc":       "#DDCC77",   # sand
}


DEFAULT_LANGUAGE_ROIS = (
    "aud_primary", "aud_belt", "stg_sts", "ifg",
)


# ─────────────────────────────────────────────────────────────────────────────
# Audio envelope
# ─────────────────────────────────────────────────────────────────────────────


def audio_envelope(
    audio: np.ndarray,
    sample_rate: int = 16_000,
    target_hz: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Coarse RMS envelope of a waveform for display alongside fMRI traces.

    Args:
        audio: 1-D waveform.
        sample_rate: input audio sample rate.
        target_hz: output envelope sampling rate (default 100 Hz). Each
            envelope sample is the RMS energy in a contiguous window of
            ``sample_rate / target_hz`` audio samples.

    Returns:
        ``(times, envelope)`` — both 1-D arrays. ``times`` in seconds.
    """
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    win = max(1, int(round(sample_rate / target_hz)))
    n_keep = (audio.shape[0] // win) * win
    if n_keep == 0:
        return np.zeros(0), np.zeros(0)
    audio = audio[:n_keep]
    blocks = audio.reshape(-1, win)
    env = np.sqrt((blocks ** 2).mean(axis=1) + 1e-12)
    times = (np.arange(env.shape[0]) + 0.5) * (win / sample_rate)
    return times, env


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _roi_mean_trace(
    result: "RABBiTPrediction", roi_base: str,
) -> np.ndarray:
    """Mean predicted response across an ROI's vertices, averaging LH+RH."""
    blocks = []
    for hemi in ("lh", "rh"):
        key = f"{roi_base}_{hemi}"
        if key in result.by_roi:
            blocks.append(result.by_roi[key].mean(axis=1))
    if not blocks:
        raise KeyError(
            f"No ROI matching base {roi_base!r} in result.by_roi (saw: "
            f"{list(result.by_roi)[:3]}...)."
        )
    return np.mean(blocks, axis=0)


def _crop_to_window(
    times: np.ndarray, signal: np.ndarray, time_window: Optional[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if time_window is None:
        return times, signal
    lo, hi = time_window
    keep = (times >= lo) & (times <= hi)
    return times[keep], signal[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def plot_audio_and_responses(
    audio: np.ndarray,
    result: "RABBiTPrediction",
    *,
    sample_rate: int = 16_000,
    audio_onset: float = 0.0,
    roi_bases: Sequence[str] = DEFAULT_LANGUAGE_ROIS,
    time_window: Optional[tuple[float, float]] = None,
    envelope_hz: float = 100.0,
    title: Optional[str] = None,
) -> "tuple[Figure, list[Axes]]":
    """Top panel: audio envelope. Bottom panels: per-ROI mean response traces.

    Args:
        audio: 1-D waveform at ``sample_rate``.
        result: :class:`RABBiTPrediction` from
            ``ROIPredictor.predict(...)`` (or ``rabbit.predict``).
        sample_rate: audio sample rate (16 kHz).
        audio_onset: where the stimulus starts in the audio file (seconds).
            The envelope x-axis is offset by this so the audio and the
            response live on the same time clock as ``result.tr_times``.
        roi_bases: ordered list of ROI base names (no hemisphere suffix) to
            display, one row per ROI. Defaults to A1 / Belt / STG / IFG.
        time_window: optional ``(lo, hi)`` in seconds. When set, both audio
            and responses are cropped to this window.
        envelope_hz: downsampled rate for the audio envelope display.
        title: figure suptitle.

    Returns:
        ``(fig, axes)``. ``axes[0]`` is the audio panel; ``axes[1:]`` are the
        per-ROI panels in the order of ``roi_bases``.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(PAPER_RC):
        n_rows = 1 + len(roi_bases)
        height_per_row = 1.2
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(11.0, 1.6 + n_rows * height_per_row),
            sharex=True, gridspec_kw={"hspace": 0.15,
                                      "height_ratios": [1.4] + [1.0] * len(roi_bases)},
        )
        axes = list(np.atleast_1d(axes))

        # ── Audio envelope ────────────────────────────────────────────────────
        env_t, env = audio_envelope(audio, sample_rate=sample_rate, target_hz=envelope_hz)
        env_t = env_t + audio_onset
        env_t_c, env_c = _crop_to_window(env_t, env, time_window)
        ax0 = axes[0]
        ax0.fill_between(env_t_c, 0, env_c, color="0.45", linewidth=0, alpha=0.85)
        ax0.set_ylabel("audio\nenvelope")
        ax0.set_ylim(0, max(env_c.max() * 1.05, 1e-9))
        ax0.grid(False)
        ax0.spines["top"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        if title:
            ax0.set_title(title, fontsize=12, pad=6)

        # ── Per-ROI responses ─────────────────────────────────────────────────
        times = np.asarray(result.tr_times)
        for ax, roi_base in zip(axes[1:], roi_bases):
            trace = _roi_mean_trace(result, roi_base)
            t_c, y_c = _crop_to_window(times, trace, time_window)
            color = _STAGE_COLOR.get(roi_base, "0.25")
            ax.plot(t_c, y_c, color=color, linewidth=1.5)
            ax.axhline(0, color="0.7", linewidth=0.5, linestyle="--", zorder=0)
            ax.set_ylabel(_pretty_roi_label(roi_base), rotation=0,
                          ha="right", va="center", labelpad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", alpha=0.25)

        axes[-1].set_xlabel("time (s)")
        fig.align_ylabels(axes)
        if time_window is not None:
            axes[-1].set_xlim(time_window)
        fig.tight_layout()
        return fig, axes


def plot_roi_grid(
    result: "RABBiTPrediction",
    *,
    roi_bases: Optional[Sequence[str]] = None,
    time_window: Optional[tuple[float, float]] = None,
    ncols: int = 3,
    title: Optional[str] = None,
) -> "tuple[Figure, np.ndarray]":
    """Small-multiples grid of per-ROI mean response traces.

    Args:
        result: prediction object.
        roi_bases: which ROIs to plot. Defaults to the 9 language ROIs.
        time_window: optional ``(lo, hi)`` time crop in seconds.
        ncols: number of grid columns.
        title: figure suptitle.
    """
    import matplotlib.pyplot as plt

    if roi_bases is None:
        roi_bases = list(_STAGE_COLOR.keys())  # 9 language ROIs in canonical order
    n = len(roi_bases)
    nrows = (n + ncols - 1) // ncols

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.6 * ncols, 1.7 * nrows),
            sharex=True, sharey=False,
        )
        axes = np.atleast_2d(axes).reshape(nrows, ncols)
        times = np.asarray(result.tr_times)
        for i, roi_base in enumerate(roi_bases):
            ax = axes[i // ncols, i % ncols]
            trace = _roi_mean_trace(result, roi_base)
            t_c, y_c = _crop_to_window(times, trace, time_window)
            ax.plot(t_c, y_c, color=_STAGE_COLOR.get(roi_base, "0.25"), linewidth=1.3)
            ax.axhline(0, color="0.7", linewidth=0.5, linestyle="--", zorder=0)
            ax.set_title(_pretty_roi_label(roi_base), fontsize=11, pad=2)
            ax.grid(axis="x", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        # Hide unused axes
        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)
        # Outer labels
        for ax in axes[-1, :]:
            ax.set_xlabel("time (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel("mean response")
        if title:
            fig.suptitle(title, fontsize=13, y=1.0)
        fig.tight_layout()
        return fig, axes


def plot_narratives_summary(
    result: "NarrativesResult",
    *,
    title: Optional[str] = None,
    sort: bool = True,
) -> "tuple[Figure, Axes]":
    """Bar chart of per-ROI mean r_group from a held-out narratives eval.

    Args:
        result: :class:`rabbit.eval.NarrativesResult` from
            ``evaluate_on_narratives(...)``.
        title: figure suptitle.
        sort: when True, sort bars by ROI-mean r descending. When False, keep
            the result's natural ordering.
    """
    import matplotlib.pyplot as plt

    rois = list(result.roi_corrs.items())
    if sort:
        rois.sort(key=lambda kv: -kv[1])
    names = [r[0] for r in rois]
    vals = [r[1] for r in rois]

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(max(9.0, 0.32 * len(rois)), 4.2))
        colors = []
        for n in names:
            base = n[:-3] if n.endswith(("_lh", "_rh")) else n
            colors.append(_STAGE_COLOR.get(base, "0.45"))
        ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="0.5", linewidth=0.6)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("mean per-vertex Pearson r (4-fold z-scored)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        if title:
            ax.set_title(title, fontsize=12, pad=6)
        fig.tight_layout()
        return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────


_PRETTY = {
    "aud_primary":    "A1",
    "aud_belt":       "Belt",
    "stg_sts":        "STG/STS",
    "temporal_pole":  "TP",
    "posterior_temp": "PT",
    "angular_gyrus":  "AG",
    "supramarginal":  "SM",
    "ifg":            "IFG",
    "mfg_dlpfc":      "MFG",
    "mpfc_tom":       "mPFC",
    "precuneus_pcc":  "PCC",
    "motor":          "M1",
    "insula_fop":     "Ins",
    "visual_early":   "V1",
    "visual_higher":  "V2+",
}


def _pretty_roi_label(roi_base: str) -> str:
    return _PRETTY.get(roi_base, roi_base)
