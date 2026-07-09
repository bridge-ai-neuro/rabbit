"""Unit tests for the eval metrics + audio alignment."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rabbit.eval import (
    align_wav_to_trs_eval,
    correlation_4fold,
    pearson_r_per_voxel,
    zscore,
)


def test_pearson_r_per_voxel_exact():
    rng = np.random.default_rng(0)
    n_TRs, n_v = 200, 50
    pred = rng.standard_normal((n_TRs, n_v)).astype(np.float64)
    target = pred + 0.5 * rng.standard_normal((n_TRs, n_v))
    r = pearson_r_per_voxel(pred, target)
    assert r.shape == (n_v,)
    # Compute the reference via numpy.corrcoef on each column pair.
    ref = np.array([
        np.corrcoef(pred[:, j], target[:, j])[0, 1] for j in range(n_v)
    ])
    np.testing.assert_allclose(r, ref, rtol=1e-10, atol=1e-12)


def test_pearson_r_zero_variance_vertices():
    pred = np.zeros((100, 3))  # all zero — zero variance
    target = np.random.randn(100, 3)
    r = pearson_r_per_voxel(pred, target)
    np.testing.assert_array_equal(r, np.zeros(3))


def test_correlation_4fold_matches_concatenated_zscore():
    rng = np.random.default_rng(42)
    n_TRs, n_v = 400, 20
    pred = rng.standard_normal((n_TRs, n_v))
    target = pred + 0.3 * rng.standard_normal((n_TRs, n_v))

    r = correlation_4fold(pred, target, n_folds=4)
    # Cross-check: manually z-score per fold, then full Pearson.
    fold = n_TRs // 4
    p_zs, t_zs = np.zeros_like(pred), np.zeros_like(target)
    for i in range(4):
        s = i * fold
        e = (i + 1) * fold if i < 3 else n_TRs
        p_zs[s:e] = zscore(pred[s:e])
        t_zs[s:e] = zscore(target[s:e])
    ref = pearson_r_per_voxel(p_zs, t_zs)
    np.testing.assert_allclose(r, ref, rtol=1e-10, atol=1e-12)


def test_align_wav_friends_first_TR_starts_at_zero():
    # 30 TRs of 1.49 s each at 16 kHz → 30 * 23840 = 715200 samples
    sr = 16_000
    tr_len = 1.49
    n_TRs = 30
    wav = torch.arange(int(n_TRs * tr_len * sr), dtype=torch.float32)
    sound_start = 5.0  # arbitrary
    tr_times = sound_start + (np.arange(n_TRs) + 1) * tr_len
    out = align_wav_to_trs_eval(wav, tr_times, tr_len=tr_len, convention="friends", sample_rate=sr)
    assert out.shape == (n_TRs, int(tr_len * sr))
    # First TR friends convention: samples [0, sr * tr_times[0]] capped at expected.
    # Confirm element 0 starts at 0.0.
    assert float(out[0, 0]) == 0.0


def test_align_wav_moth_first_TR_zero_padded():
    sr = 16_000
    tr_len = 2.0
    n_TRs = 30
    wav = torch.arange(int(n_TRs * tr_len * sr), dtype=torch.float32) + 1.0
    # sound_start=0; first trtime=2.0 → equal to tr_len → branch goes to non-first-TR path.
    # Set sound_start so trtime[0] < tr_len to force the pad branch.
    sound_start = 0.5
    tr_times = sound_start + np.arange(n_TRs) * tr_len  # first = 0.5 < 2.0
    out = align_wav_to_trs_eval(wav, tr_times, tr_len=tr_len, convention="moth", sample_rate=sr)
    # Moth pad branch: first int((tr_len - 1.0) * sr) = int(1*16000) samples zero, then real audio.
    pad_n = sr  # int((2.0 - 1.0) * 16000)
    np.testing.assert_allclose(out[0, :pad_n].numpy(), np.zeros(pad_n))
    # Right after pad, the first audio sample should be 1.0 (i.e. wav[0]).
    assert float(out[0, pad_n]) == 1.0
