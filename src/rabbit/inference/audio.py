"""Audio loading and TR-alignment helpers.

RABBiT trains and predicts on a per-TR basis: the audio is sliced into
fixed-length windows that match each fMRI repetition time (TR). For each TR,
the network sees a 2-second window of audio ending at the TR boundary, and
``hrf_delay + 1`` such windows are stacked along the batch dimension so the
prediction at TR t can attend to acoustic content from TR ``t-hrf_delay``
through ``t``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
import torch


# Wav2vec2 and WavLM both natively operate at 16 kHz.
TARGET_SAMPLE_RATE = 16_000


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────


def load_audio(path: Union[str, Path], target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Load an audio file as a mono float32 tensor at ``target_sr``.

    Accepts:
      * ``.wav`` / ``.flac`` / anything ``soundfile`` reads.
      * ``.npy`` containing a 1-D float array assumed to already be at
        ``target_sr``.

    Returns:
        1-D ``torch.float32`` tensor of length ``n_samples``.
    """
    path = Path(path)
    if path.suffix == ".npy":
        wav = np.load(path, allow_pickle=False)
        wav = np.nan_to_num(np.asarray(wav, dtype=np.float32).squeeze())
        return torch.from_numpy(wav)

    import soundfile as sf  # local import keeps top-level import light
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)  # mono mix
    data = np.nan_to_num(data, copy=False)
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return torch.from_numpy(data.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# TR alignment
# ─────────────────────────────────────────────────────────────────────────────


def align_wav_to_trs(
    wav: torch.Tensor,
    tr_times: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    window_seconds: float = 2.0,
) -> torch.Tensor:
    """Stack a ``window_seconds`` window of audio ending at each TR boundary.

    Args:
        wav: 1-D waveform at ``sample_rate``.
        tr_times: (n_TRs,) array of TR center times in seconds. Must be
            monotonically increasing.
        sample_rate: input sample rate (must match the wav).
        window_seconds: window length per TR (default 2.0 s — matches what
            wav2vec2 sees per audio chunk in RABBiT training).

    Returns:
        ``(n_TRs, window_seconds * sample_rate)`` float32 tensor. TRs whose
        windows would extend before t=0 are zero-padded on the left.
    """
    expected = int(window_seconds * sample_rate)
    aligned = []
    for tridx, trtime in enumerate(tr_times):
        if trtime < window_seconds:
            head = max(0.0, trtime)
            n_head = int(head * sample_rate)
            chunk = torch.cat([
                torch.zeros(expected - n_head, dtype=wav.dtype),
                wav[: n_head],
            ])
        else:
            sidx = sample_rate * int(round(trtime - window_seconds))
            eidx = sample_rate * int(round(trtime))
            chunk = wav[sidx:eidx]
            if chunk.shape[0] < expected:
                # Edge / rounding pad
                chunk = torch.cat([
                    torch.zeros(expected - chunk.shape[0], dtype=wav.dtype),
                    chunk,
                ])
            elif chunk.shape[0] > expected:
                chunk = chunk[-expected:]
        aligned.append(chunk)
    return torch.vstack(aligned)


# ─────────────────────────────────────────────────────────────────────────────
# HRF-delay stacking
# ─────────────────────────────────────────────────────────────────────────────


def make_delayed(stim: torch.Tensor, delays: Iterable[int]) -> torch.Tensor:
    """Stack time-shifted copies of ``stim`` along the feature dimension.

    Replicates the trainer's ``range(0, hrf_delay+1)`` recipe so the eval-time
    audio matches the training-time audio exactly.

    Args:
        stim: (n_TRs, n_features) tensor.
        delays: iterable of integer shifts; positive = shift later in time.

    Returns:
        ``(n_TRs, len(delays) * n_features)`` tensor. Edges are zero-padded.
    """
    nt, ndim = stim.shape
    dstims = []
    for d in reversed(list(delays)):
        dstim = torch.zeros((nt, ndim), dtype=stim.dtype)
        if d < 0:
            dstim[:d, :] = stim[-d:, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
        else:
            dstim = stim.clone()
        dstims.append(dstim)
    return torch.hstack(dstims)
