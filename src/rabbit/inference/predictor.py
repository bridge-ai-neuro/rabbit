"""High-level inference API: audio file → fMRI predictions.

This is the entry point most users will reach for. It wraps the low-level
``RABBiT.forward`` with audio loading, TR alignment, HRF-delay stacking,
batching, and (optionally) the avg-dev trick for held-out-subject prediction.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from ..model import RABBiT
from .audio import TARGET_SAMPLE_RATE, align_wav_to_trs, load_audio, make_delayed
from .checkpoint import load_from_checkpoint


@dataclass
class RABBiTPrediction:
    """Container for one inference run."""

    fmri: np.ndarray
    """(n_TRs, flat_dim) predicted z-scored fMRI across all ROIs."""

    by_roi: "OrderedDict[str, np.ndarray]"
    """{roi_name: (n_TRs, V_roi)} per-ROI predictions."""

    tr_times: np.ndarray
    """(n_TRs,) audio center times in seconds."""

    attention: Optional[np.ndarray] = None
    """(n_TRs, n_heads, n_rois, n_tokens) cross-attention weights, if requested."""

    roi_names: tuple[str, ...] = ()
    """Ordering of ROIs in ``by_roi`` and along the second axis of ``fmri``."""


class ROIPredictor:
    """Audio → ROI fMRI predictor.

    Two ways to construct:
      * ``ROIPredictor.from_checkpoint(...)`` — load a trained model and wrap it.
      * ``ROIPredictor(model)`` — wrap an already-loaded ``RABBiT``.
    """

    def __init__(
        self,
        model: RABBiT,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.roi_names: tuple[str, ...] = tuple(model.roi_layout.roi_names)
        self.roi_layout = model.roi_layout

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        use_avg_dev: bool = True,
        overrides: Optional[list[str]] = None,
    ) -> "ROIPredictor":
        model = load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            overrides=overrides,
            device=device,
            use_avg_dev=use_avg_dev,
        )
        return cls(model, device=device)

    # ── Predict ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        audio_path: Union[str, Path, torch.Tensor, np.ndarray],
        *,
        tr_length: float = 1.49,
        hrf_delay: int = 6,
        trim_start: int = 10,
        trim_end: int = 9,
        audio_onset: float = 0.0,
        shift: int = 0,
        window_seconds: Optional[float] = None,
        batch_size: int = 32,
        subject_slot: int = 0,
        return_attention: bool = False,
        tr_times: Optional[np.ndarray] = None,
        apply_processor: bool = True,
    ) -> RABBiTPrediction:
        """Predict fMRI for a stimulus.

        Args:
            audio_path: a filesystem path, or an already-loaded 1-D tensor /
                array at 16 kHz.
            tr_length: TR period in seconds (Friends = 1.49, moth = 2.0).
            hrf_delay: number of TRs of HRF lag; the model sees windows
                ``range(0, hrf_delay + 1)`` shifts.
            trim_start, trim_end: number of leading / trailing TRs to drop
                from the output. Matches training-time edge-trimming.
            window_seconds: per-TR audio window (paper: 2.0 s).
            batch_size: TR batches sent through the GPU at once.
            subject_slot: which subject index to attend to in the readout.
                For zero-shot inference with ``use_avg_dev=True`` (the
                default), use slot 0 — already overwritten with the avg-dev.
            return_attention: when True, populate
                ``RABBiTPrediction.attention``.
            tr_times: optional explicit (n_TRs,) array of TR center times in
                seconds. If None, generated from ``audio_onset`` + the audio
                duration as ``audio_onset + (arange(n_TRs)+1) * tr_length``.
            audio_onset: offset in seconds from the start of the audio file
                to the start of the stimulus. Use this when the audio has a
                leading silence (e.g. narratives ``sound_start ≈ 6 s``).
                Default 0.0 — assumes the stimulus begins at sample 0.
            shift: integer TR offset added to the returned ``tr_times``. Use
                this to declare that "the prediction at frame k describes the
                brain state at time t_k + shift·tr_length". Does not change
                the predictions themselves; only the time stamps. Default 0.
            apply_processor: when True, run each per-TR audio window through
                the backbone's HF processor (a no-op for ``wav2vec2-base-960h``
                since ``do_normalize=False``, but applies per-sample mean/var
                normalization for WavLM). Defaults to True to match training
                behaviour.
        """
        # ── Audio ─────────────────────────────────────────────────────────────
        if isinstance(audio_path, (str, Path)):
            wav = load_audio(audio_path)
        elif isinstance(audio_path, np.ndarray):
            wav = torch.from_numpy(np.asarray(audio_path, dtype=np.float32))
        else:
            wav = audio_path.detach().to(torch.float32).cpu()

        # ── Window length follows tr_length unless explicitly overridden ────
        if window_seconds is None:
            window_seconds = float(tr_length)

        # ── TR grid ───────────────────────────────────────────────────────────
        if tr_times is None:
            duration_s = wav.shape[0] / TARGET_SAMPLE_RATE
            usable_s = max(0.0, duration_s - audio_onset)
            n_TRs_total = int(usable_s // tr_length)
            tr_times = audio_onset + (np.arange(n_TRs_total) + 1) * tr_length
        else:
            tr_times = np.asarray(tr_times, dtype=np.float64)

        # ── TR-aligned audio windows ──────────────────────────────────────────
        aligned = align_wav_to_trs(
            wav, tr_times, sample_rate=TARGET_SAMPLE_RATE, window_seconds=window_seconds,
        )  # (n_TRs, window_samples)

        # ── HRF-delay stacking, then trim ─────────────────────────────────────
        delays = range(0, hrf_delay + 1)
        delayed = make_delayed(aligned, delays).float()  # (n_TRs, (hrf+1) * window_samples)
        if trim_end > 0:
            delayed = delayed[trim_start:-trim_end]
            kept_times = tr_times[trim_start:-trim_end] if trim_end > 0 else tr_times[trim_start:]
        else:
            delayed = delayed[trim_start:]
            kept_times = tr_times[trim_start:]

        # Each TR's input is the concatenated ``(hrf+1) * window`` audio as
        # input_values — the same shape the training loop feeds.
        n_used = delayed.shape[0]
        if n_used == 0:
            raise ValueError(
                f"After trim_start={trim_start}, trim_end={trim_end} no TRs remained "
                f"(audio duration {wav.shape[0]/TARGET_SAMPLE_RATE:.1f}s, tr_length={tr_length})."
            )

        # ── HF processor (mean/var normalization for WavLM; no-op for w2v-base-960h) ──
        if apply_processor and self.model.backbone.processor is not None:
            processed = []
            proc = self.model.backbone.processor
            sr = self.model.backbone.sampling_rate
            for i in range(delayed.shape[0]):
                wav_tr = torch.nan_to_num(delayed[i], nan=0.0, posinf=0.0, neginf=0.0)
                out = proc(wav_tr, return_tensors="pt", sampling_rate=sr)
                # wav2vec / WavLM → input_values; Whisper → input_features (unused here).
                processed.append(out.input_values[0])
            delayed = torch.stack(processed, dim=0)

        # ── Forward pass batched over TRs ─────────────────────────────────────
        flat_preds: list[torch.Tensor] = []
        attn_chunks: list[torch.Tensor] = []
        for s in range(0, n_used, batch_size):
            x = delayed[s : s + batch_size].to(self.device)
            sid = None
            if self.model.needs_subject_indices:
                sid = torch.full((x.shape[0],), subject_slot, dtype=torch.long, device=self.device)
            out = self.model(x, sid, return_attn=return_attention)
            flat_preds.append(out["flat_predictions"].cpu())
            if return_attention and out["attention_weights"] is not None:
                attn_chunks.append(out["attention_weights"].cpu())

        flat = torch.cat(flat_preds, dim=0).numpy()  # (n_TRs, flat_dim)
        by_roi = OrderedDict()
        for name, slc in self.roi_layout.roi_slices.items():
            by_roi[name] = flat[:, slc]

        attention = torch.cat(attn_chunks, dim=0).numpy() if attn_chunks else None

        if shift != 0:
            kept_times = np.asarray(kept_times) + float(shift) * float(tr_length)

        return RABBiTPrediction(
            fmri=flat,
            by_roi=by_roi,
            tr_times=np.asarray(kept_times),
            attention=attention,
            roi_names=self.roi_names,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Directory / multi-file inference
    # ─────────────────────────────────────────────────────────────────────────

    def predict_many(
        self,
        source: "Union[str, Path, list[Union[str, Path]]]",
        *,
        extensions: tuple[str, ...] = (".wav", ".flac", ".npy", ".mp3"),
        progress: bool = True,
        **predict_kwargs,
    ) -> "OrderedDict[str, RABBiTPrediction]":
        """Predict on a directory of audio files (or an explicit file list).

        Args:
            source: directory path (non-recursive listing) or a list of file
                paths. Filenames sorted lexicographically.
            extensions: audio file extensions to include when scanning a
                directory. Ignored when ``source`` is a list.
            progress: print one line per file with name + n_TRs + finite-frac.
            **predict_kwargs: forwarded to ``predict()``.

        Returns:
            ``OrderedDict[stem -> RABBiTPrediction]`` keyed by each file's
            stem (filename minus extension), in iteration order.
        """
        if isinstance(source, (str, Path)):
            src = Path(source)
            if src.is_dir():
                paths = sorted(
                    p for p in src.iterdir()
                    if p.is_file() and p.suffix.lower() in extensions
                )
            elif src.is_file():
                paths = [src]
            else:
                raise FileNotFoundError(source)
        else:
            paths = [Path(p) for p in source]

        if not paths:
            raise ValueError(f"No audio files matched {source!r} (extensions={extensions}).")

        results: "OrderedDict[str, RABBiTPrediction]" = OrderedDict()
        for path in paths:
            if progress:
                print(f"[predict_many] {path.name} ...", flush=True)
            try:
                res = self.predict(path, **predict_kwargs)
            except ValueError as exc:
                if progress:
                    print(f"  skipped: {exc}", flush=True)
                continue
            results[path.stem] = res
            if progress:
                fmri = res.fmri
                finite = float(np.isfinite(fmri).mean())
                print(
                    f"  {path.stem}: n_TRs={fmri.shape[0]}  "
                    f"finite={finite:.4f}  duration~{res.tr_times[-1]:.1f}s",
                    flush=True,
                )
        return results
