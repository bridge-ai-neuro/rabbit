"""Unit tests for the Friends dataset adapter.

Writes synthetic .npz and .npy files mimicking the on-disk data layout into a
tmp dir, then exercises ``FriendsClipManifest.discover`` +
``FriendsClipDataset.fetch`` + indexing. No real Friends data required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rabbit.data import FriendsClipDataset, FriendsClipManifest, parse_clip_name
from rabbit.model.roi_layout import FS6_ROI_DIMS


def _write_synthetic_clip(
    roi_dir: Path, audio_dir: Path, sub_id: str, season: int, episode: int, part: str,
    n_TRs: int = 30, tr_length: float = 1.49, sample_rate: int = 16_000,
) -> str:
    clip_name = f"s{season:02d}_e{episode:02d}_{part}"
    # fMRI .npz with one key per ROI, each (n_TRs, V_roi).
    npz_path = roi_dir / f"sub-{sub_id}" / f"{clip_name}.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(hash((sub_id, clip_name)) % 2**31)
    payload = {
        roi: np.random.randn(n_TRs, V).astype(np.float32)
        for roi, V in FS6_ROI_DIMS.items()
    }
    np.savez_compressed(npz_path, **payload)

    # Audio .npy (only need to write once per clip; OK to overwrite).
    audio_dir_season = audio_dir / f"s{season}"
    audio_dir_season.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir_season / f"friends_s{season:02d}e{episode:02d}{part}.npy"
    if not audio_path.exists():
        wav = np.random.randn(int(n_TRs * tr_length * sample_rate)).astype(np.float32)
        np.save(audio_path, wav)

    return clip_name


def test_parse_clip_name():
    assert parse_clip_name("s01_e02_a") == (1, 2, "a")
    assert parse_clip_name("s06_e23_b") == (6, 23, "b")
    assert parse_clip_name("not_a_clip") is None


def test_manifest_discovers_only_common_clips(tmp_path: Path):
    roi_dir = tmp_path / "fmri"
    audio_dir = tmp_path / "audio"
    # sub-01 has clips a, b; sub-02 has only clip a.
    _write_synthetic_clip(roi_dir, audio_dir, "01", 1, 2, "a")
    _write_synthetic_clip(roi_dir, audio_dir, "01", 1, 2, "b")
    _write_synthetic_clip(roi_dir, audio_dir, "02", 1, 2, "a")
    # plus an extra subject directory for sub-02 to exist
    (roi_dir / "sub-02").mkdir(exist_ok=True)

    m = FriendsClipManifest.discover(roi_dir, audio_dir, subjects=["01", "02"])
    assert m.clip_names == ["s01_e02_a"], (
        f"Manifest should only return clips present for ALL subjects; got {m.clip_names}."
    )
    assert m.subject_to_idx == {"01": 0, "02": 1}


def test_clip_dataset_fetch_and_index(tmp_path: Path):
    roi_dir = tmp_path / "fmri"
    audio_dir = tmp_path / "audio"
    n_TRs = 50
    for sub_id in ("01", "02"):
        _write_synthetic_clip(roi_dir, audio_dir, sub_id, 1, 2, "a", n_TRs=n_TRs)

    manifest = FriendsClipManifest.discover(roi_dir, audio_dir, subjects=["01", "02"])
    ds = FriendsClipDataset(
        clip_name="s01_e02_a",
        manifest=manifest,
        hrf_delay=4,
        tr_length=1.49,
        trim_start=2,
        trim_end=2,
    )
    ds.fetch()

    # 2 subjects × (n_TRs - trim_start - trim_end) TRs.
    expected_TRs_per_subj = n_TRs - 2 - 2
    expected_total = 2 * expected_TRs_per_subj
    assert len(ds) == expected_total
    assert ds.n_TRs_per_subject == expected_TRs_per_subj

    # Inspect one sample.
    wav, fmri, subj = ds[0]
    # Audio: (hrf_delay+1) * (tr_length * sample_rate) = 5 * 23840 = 119_200
    assert wav.shape == (5 * int(1.49 * 16_000),), wav.shape
    # fMRI: flat over all 30 ROIs.
    assert fmri.shape == (sum(FS6_ROI_DIMS.values()),)
    assert subj == 0  # first samples belong to sub-01 (slot 0)

    # subj wraps to slot 1 in the second half.
    _, _, late_subj = ds[expected_total - 1]
    assert late_subj == 1


def test_clip_dataset_release(tmp_path: Path):
    roi_dir = tmp_path / "fmri"
    audio_dir = tmp_path / "audio"
    _write_synthetic_clip(roi_dir, audio_dir, "01", 1, 2, "a")
    manifest = FriendsClipManifest.discover(roi_dir, audio_dir, subjects=["01"])
    ds = FriendsClipDataset(
        clip_name="s01_e02_a", manifest=manifest,
        hrf_delay=4, trim_start=2, trim_end=2,
    )
    ds.fetch()
    assert len(ds) > 0
    ds.release()
    assert len(ds) == 0
    with pytest.raises(RuntimeError, match="must be .fetch"):
        _ = ds[0]
