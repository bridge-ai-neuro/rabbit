"""Friends fs6 30-ROI dataset adapter.

Data layout (set via env vars or config — see ``configs/friends_shared_dev.yaml``):

  <roi_dir>/sub-{01..06}/sXX_eYY_{a,b}.npz       fMRI: one .npz per clip per subject
  <audio_dir>/s{1..6}/friends_sXXeXX{a,b}.npy     audio: 16 kHz mono waveform per clip

Each fMRI .npz holds 30 keys (one per ROI in the canonical order), each a
``(n_TRs, V_roi)`` float32 array, already z-scored per ROI at preprocessing
time. Each clip is short (~400 TRs ≈ 10 minutes at TR=1.49 s).

The dataset design mirrors the original training code: one ``FriendsClipDataset`` per clip,
containing that clip's audio shared across subjects and a ``(n_subjects *
n_TRs, V_total)`` stacked fMRI block, indexed by ``subj_indices`` for the
shared-deviation readout.

The training loop iterates over clips and rebuilds the per-clip dataset each
epoch (or uses one pass through a ``ConcatDataset``, depending on the
trainer). This keeps per-worker memory bounded.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..inference.audio import TARGET_SAMPLE_RATE, load_audio, make_delayed
from ..model.roi_layout import FS6_ROI_DIMS


__all__ = [
    "FriendsClipManifest",
    "FriendsClipDataset",
    "parse_clip_name",
    "build_clip_audio_path",
    "build_clip_fmri_path",
]


_CLIP_RE = re.compile(r"^s(\d{2})_e(\d{2})_([ab])$")


def parse_clip_name(clip_name: str) -> Optional[tuple[int, int, str]]:
    """``'s01_e02_a'`` → ``(1, 2, 'a')``."""
    m = _CLIP_RE.match(clip_name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3)


def build_clip_audio_path(audio_dir: str | Path, clip_name: str) -> Path:
    """Resolve the .npy waveform path for one clip.

    Layout: ``<audio_dir>/s{season}/friends_s{SS}e{EE}{part}.npy``.
    """
    parsed = parse_clip_name(clip_name)
    if parsed is None:
        raise ValueError(f"Cannot parse clip name: {clip_name!r}")
    season, episode, part = parsed
    return Path(audio_dir) / f"s{season}" / f"friends_s{season:02d}e{episode:02d}{part}.npy"


def build_clip_fmri_path(roi_dir: str | Path, sub_id: str, clip_name: str) -> Path:
    """Resolve the .npz fMRI path for one (subject, clip)."""
    return Path(roi_dir) / f"sub-{sub_id}" / f"{clip_name}.npz"


# ─────────────────────────────────────────────────────────────────────────────
# Manifest: enumerate all (subject, clip) pairs once
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FriendsClipManifest:
    """Index of all clips available for a configured set of subjects.

    A clip is included only if **every** configured subject has an fMRI .npz
    for it AND the audio .npy exists. This avoids per-batch mid-training
    surprises.
    """

    subjects:        list[str]
    subject_to_idx:  dict[str, int]
    roi_dir:         Path
    audio_dir:       Path
    roi_names:       list[str]
    clip_names:      list[str] = field(default_factory=list)

    @classmethod
    def discover(
        cls,
        roi_dir: str | Path,
        audio_dir: str | Path,
        subjects: Sequence[str],
        roi_names: Optional[Sequence[str]] = None,
    ) -> "FriendsClipManifest":
        roi_dir, audio_dir = Path(roi_dir), Path(audio_dir)
        subjects = [str(s) for s in subjects]
        subject_to_idx = {s: i for i, s in enumerate(subjects)}
        roi_names = list(roi_names) if roi_names is not None else list(FS6_ROI_DIMS.keys())

        # Union of clip names across configured subjects
        per_subject_clips: dict[str, set[str]] = {}
        for s in subjects:
            sub_dir = roi_dir / f"sub-{s}"
            if not sub_dir.is_dir():
                raise FileNotFoundError(f"Friends fMRI subject dir not found: {sub_dir}")
            per_subject_clips[s] = {
                fn[:-4] for fn in os.listdir(sub_dir) if fn.endswith(".npz")
            }
        common = set.intersection(*per_subject_clips.values()) if per_subject_clips else set()

        # Drop clips missing audio
        valid: list[str] = []
        for clip_name in sorted(common):
            ap = build_clip_audio_path(audio_dir, clip_name)
            if ap.exists():
                valid.append(clip_name)

        return cls(
            subjects=subjects,
            subject_to_idx=subject_to_idx,
            roi_dir=roi_dir,
            audio_dir=audio_dir,
            roi_names=roi_names,
            clip_names=valid,
        )

    def split(
        self,
        val_clips: Iterable[str] = (),
        test_clips: Iterable[str] = (),
    ) -> tuple[list[str], list[str], list[str]]:
        """Split clips into (train, val, test) with set semantics."""
        val_set, test_set = set(val_clips), set(test_clips)
        train = [c for c in self.clip_names if c not in val_set and c not in test_set]
        return train, sorted(val_set & set(self.clip_names)), sorted(test_set & set(self.clip_names))

    def __len__(self) -> int:
        return len(self.clip_names)

    def __repr__(self) -> str:
        return (
            f"FriendsClipManifest(n_clips={len(self.clip_names)}, "
            f"subjects={self.subjects}, roi_dir={self.roi_dir})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip dataset: one clip across N subjects, audio shared
# ─────────────────────────────────────────────────────────────────────────────


class FriendsClipDataset(Dataset):
    """All (TR × subject) pairs for one Friends clip.

    Audio is aligned once at construction time and re-used across subjects;
    fMRI is stacked across subjects along the time axis. Each ``__getitem__``
    returns ``(audio_TR_window, fmri_TR_vector, subject_idx)``.

    Args:
        clip_name: e.g. ``"s01_e02_a"``.
        manifest: discovered Friends manifest.
        hrf_delay: number of HRF-lag TRs to stack along the feature axis
            (``range(0, hrf_delay+1)``).
        tr_length: TR period in seconds (Friends = 1.49).
        trim_start, trim_end: edge-TR trimming applied to both audio and fMRI.
        processor: optional HF processor — applied per-TR in ``__getitem__``.
            Defaults to None (the trainer's DataLoader collate or the
            backbone's pre-pass typically handles this).
        sampling_rate: input audio sample rate (16 kHz).
    """

    def __init__(
        self,
        clip_name: str,
        manifest: FriendsClipManifest,
        *,
        hrf_delay: int = 6,
        tr_length: float = 1.49,
        trim_start: int = 10,
        trim_end: int = 9,
        processor=None,
        sampling_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        if clip_name not in manifest.clip_names:
            raise ValueError(f"Clip {clip_name!r} not in manifest.")
        self.clip_name = clip_name
        self.manifest = manifest
        self.tr_length = float(tr_length)
        self.hrf_delay = int(hrf_delay)
        self.trim_start = int(trim_start)
        self.trim_end = int(trim_end)
        self.processor = processor
        self.sampling_rate = int(sampling_rate)

        self.delayed_wav: torch.Tensor = torch.empty(0)
        self.fmri_data:   torch.Tensor = torch.empty(0)
        self.subj_idx:    torch.Tensor = torch.empty(0, dtype=torch.long)
        self.n_TRs_per_subject: int = 0

        self._loaded = False

    # ── data loading ─────────────────────────────────────────────────────────

    def fetch(self) -> "FriendsClipDataset":
        """Load fMRI + audio for this clip into memory. Idempotent."""
        if self._loaded:
            return self

        # ── fMRI per subject, concatenated along time axis ────────────────────
        fmri_blocks: list[np.ndarray] = []
        subj_indices: list[int] = []
        n_TRs: Optional[int] = None
        for sub_id in self.manifest.subjects:
            path = build_clip_fmri_path(self.manifest.roi_dir, sub_id, self.clip_name)
            data = np.load(path)
            per_roi = [data[r].astype(np.float32) for r in self.manifest.roi_names]
            stacked = np.concatenate(per_roi, axis=1)  # (n_TRs, V_total)
            if n_TRs is None:
                n_TRs = stacked.shape[0]
            elif stacked.shape[0] != n_TRs:
                raise ValueError(
                    f"Clip {self.clip_name!r} has inconsistent n_TRs across subjects: "
                    f"sub-{sub_id} has {stacked.shape[0]} vs expected {n_TRs}."
                )
            fmri_blocks.append(stacked)
            subj_indices.extend([self.manifest.subject_to_idx[sub_id]] * stacked.shape[0])

        fmri_full = np.concatenate(fmri_blocks, axis=0)  # (n_subj * n_TRs, V_total)
        # Per-ROI z-score across the combined time axis to match the trainer's
        # convention. (Each clip's npz is already z-scored per subject — this
        # is a re-z-score after the cross-subject stacking to keep scale
        # consistent.)
        roi_offsets = self._roi_offsets()
        for start, end in roi_offsets:
            block = fmri_full[:, start:end]
            m = block.mean(axis=0, keepdims=True)
            s = block.std(axis=0, keepdims=True)
            s = np.where(s < 1e-12, 1.0, s)
            fmri_full[:, start:end] = (block - m) / s

        # ── audio: load + align to per-TR windows + HRF-delay stack + trim ──
        wav = load_audio(build_clip_audio_path(self.manifest.audio_dir, self.clip_name))
        aligned = _align_wav_for_friends(wav, n_TRs, tr_len=self.tr_length, sr=self.sampling_rate)
        delays = list(range(0, self.hrf_delay + 1))
        delayed = make_delayed(aligned, delays).float()

        if self.trim_end > 0:
            delayed = delayed[self.trim_start : -self.trim_end]
        else:
            delayed = delayed[self.trim_start :]

        n_TRs_used = delayed.shape[0]
        self.n_TRs_per_subject = n_TRs_used

        # Trim fMRI symmetrically per subject. fmri_full is laid out as
        # [sub0_t0, sub0_t1, ..., sub0_tN-1, sub1_t0, ...]; subject-aware trim.
        keep_mask = np.zeros(n_TRs, dtype=bool)
        end_idx = n_TRs - self.trim_end if self.trim_end > 0 else n_TRs
        keep_mask[self.trim_start : end_idx] = True
        per_subj_keep = np.tile(keep_mask, len(self.manifest.subjects))
        fmri_trimmed = fmri_full[per_subj_keep]
        subj_idx_trimmed = np.asarray(subj_indices)[per_subj_keep]

        self.fmri_data = torch.from_numpy(fmri_trimmed).float()
        self.subj_idx = torch.from_numpy(subj_idx_trimmed).long()
        self.delayed_wav = delayed
        self._loaded = True
        return self

    def release(self) -> None:
        """Free per-clip tensors (call after iterating a clip in training)."""
        self.delayed_wav = torch.empty(0)
        self.fmri_data = torch.empty(0)
        self.subj_idx = torch.empty(0, dtype=torch.long)
        self._loaded = False

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.fmri_data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if not self._loaded:
            raise RuntimeError(
                f"FriendsClipDataset(clip={self.clip_name!r}) must be .fetch()-ed before indexing."
            )
        wav = self.delayed_wav[idx % self.n_TRs_per_subject]
        if self.processor is not None:
            wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
            wav = self.processor(
                wav, return_tensors="pt", sampling_rate=self.sampling_rate,
            ).input_values[0]
        return wav, self.fmri_data[idx], int(self.subj_idx[idx])

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _roi_offsets(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        start = 0
        for r in self.manifest.roi_names:
            end = start + FS6_ROI_DIMS[r]
            out.append((start, end))
            start = end
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Training-time alignment (matches the original Friends fs6 trainer's align_wav_to_trs)
# ─────────────────────────────────────────────────────────────────────────────


def _align_wav_for_friends(
    wav: torch.Tensor, n_trs: int, tr_len: float, sr: int = TARGET_SAMPLE_RATE,
) -> torch.Tensor:
    """Zero-anchored per-TR audio windowing for Friends training.

    TR ``k`` covers the audio interval ``[k*tr_len, (k+1)*tr_len]``. Mirrors
    the original training-time helper exactly so checkpoint forward passes are
    consistent.
    """
    tr_times = np.arange(0, int(n_trs * tr_len), step=tr_len) + tr_len
    expected = int(tr_len * sr)
    aligned: list[torch.Tensor] = []
    for tridx, trtime in enumerate(tr_times):
        sidx = int(sr * tr_times[tridx - 1]) if tridx > 0 else 0
        eidx = int(sr * trtime)
        chunk = wav[sidx:eidx]
        if chunk.shape[0] < expected:
            chunk = torch.cat([torch.zeros(expected - chunk.shape[0], dtype=chunk.dtype), chunk])
        elif chunk.shape[0] > expected:
            chunk = chunk[:expected]
        aligned.append(chunk)
    return torch.vstack(aligned)
