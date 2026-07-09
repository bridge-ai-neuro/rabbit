"""Tests for the user-facing inference API (audio_onset, shift, predict_many)
and the viz module."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

import numpy as np
import pytest
import torch
import torch.nn as nn

from rabbit.inference import ROIPredictor
from rabbit.model import RABBiT, build_flat_roi_layout


# ─────────────────────────────────────────────────────────────────────────────
# Tiny model fixture (skip HF download)
# ─────────────────────────────────────────────────────────────────────────────


class _DummyBackbone(nn.Module):
    hidden_size: int = 64
    sampling_rate: int = 16_000
    processor = None

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, self.hidden_size)
        self._stride = 320

    def forward(self, input_values, attention_mask=None):
        B, T = input_values.shape
        T_keep = (T // self._stride) * self._stride
        x = (
            input_values[:, :T_keep]
            .reshape(B, T_keep // self._stride, self._stride)
            .mean(-1, keepdim=True)
        )
        return self.proj(x), None


def _tiny_model() -> RABBiT:
    layout = build_flat_roi_layout(
        OrderedDict(
            [
                ("aud_primary_lh", 30),
                ("aud_primary_rh", 30),
                ("aud_belt_lh", 20),
                ("aud_belt_rh", 20),
                ("ifg_lh", 15),
                ("ifg_rh", 15),
            ]
        )
    )
    backbone = _DummyBackbone()
    shared = {n: torch.randn(8, layout.roi_dims_dict[n]) * 0.1 for n in layout.roi_names}
    dev = {n: [torch.randn(3, layout.roi_dims_dict[n]) * 0.05 for _ in range(2)] for n in layout.roi_names}
    return RABBiT(
        readout_type="shared_deviation",
        backbone=backbone,
        roi_layout=layout,
        hidden_dim=64, nhead=4, num_layers=1, dim_feedforward=128,
        roi_bases=shared, dev_roi_bases=dev,
    )


@pytest.fixture
def predictor():
    return ROIPredictor(_tiny_model(), device="cpu")


# ─────────────────────────────────────────────────────────────────────────────
# audio_onset + shift behaviour
# ─────────────────────────────────────────────────────────────────────────────


def test_predict_default_args(predictor):
    audio = np.random.RandomState(0).randn(60 * 16_000).astype(np.float32)
    res = predictor.predict(
        audio, tr_length=1.49, hrf_delay=4, trim_start=2, trim_end=2,
        apply_processor=False,
    )
    assert res.fmri.shape[1] == predictor.model.output_dim
    assert res.fmri.shape[0] == len(res.tr_times)
    assert np.isfinite(res.fmri).all()


def test_audio_onset_shifts_grid(predictor):
    audio = np.random.RandomState(0).randn(60 * 16_000).astype(np.float32)
    base = predictor.predict(
        audio, tr_length=1.49, hrf_delay=4, trim_start=2, trim_end=2,
        audio_onset=0.0, apply_processor=False,
    )
    shifted = predictor.predict(
        audio, tr_length=1.49, hrf_delay=4, trim_start=2, trim_end=2,
        audio_onset=5.0, apply_processor=False,
    )
    # The grid is offset by audio_onset.
    assert shifted.tr_times[0] >= base.tr_times[0] + 4.0
    # And there should be roughly 5/1.49 ≈ 3 fewer TRs (less usable audio).
    assert base.fmri.shape[0] - shifted.fmri.shape[0] >= 2


def test_shift_advances_tr_times_only(predictor):
    audio = np.random.RandomState(0).randn(60 * 16_000).astype(np.float32)
    r0 = predictor.predict(
        audio, tr_length=1.49, hrf_delay=4, trim_start=2, trim_end=2,
        shift=0, apply_processor=False,
    )
    r1 = predictor.predict(
        audio, tr_length=1.49, hrf_delay=4, trim_start=2, trim_end=2,
        shift=3, apply_processor=False,
    )
    # tr_times shifted by 3 * 1.49
    np.testing.assert_allclose(r1.tr_times, r0.tr_times + 3 * 1.49, rtol=1e-6)
    # fmri values themselves unchanged
    np.testing.assert_allclose(r1.fmri, r0.fmri, atol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# predict_many
# ─────────────────────────────────────────────────────────────────────────────


def test_predict_many_directory(predictor, tmp_path):
    rng = np.random.RandomState(0)
    # Three synthetic .npy waveforms in a directory
    for i in range(3):
        np.save(tmp_path / f"clip_{i}.npy", rng.randn(30 * 16_000).astype(np.float32))
    # A bonus non-audio file that should be ignored.
    (tmp_path / "not_audio.txt").write_text("ignore me")

    results = predictor.predict_many(
        tmp_path,
        tr_length=1.49, hrf_delay=4, trim_start=2, trim_end=2,
        progress=False, apply_processor=False,
    )
    assert set(results.keys()) == {"clip_0", "clip_1", "clip_2"}
    for r in results.values():
        assert r.fmri.shape[1] == predictor.model.output_dim


def test_predict_many_path_list(predictor, tmp_path):
    paths = []
    rng = np.random.RandomState(0)
    for i in range(2):
        p = tmp_path / f"clip_{i}.npy"
        np.save(p, rng.randn(20 * 16_000).astype(np.float32))
        paths.append(p)

    results = predictor.predict_many(
        paths, tr_length=1.49, hrf_delay=4, trim_start=2, trim_end=2,
        progress=False, apply_processor=False,
    )
    assert list(results.keys()) == ["clip_0", "clip_1"]


# ─────────────────────────────────────────────────────────────────────────────
# Viz module — at least it renders without raising
# ─────────────────────────────────────────────────────────────────────────────


def test_audio_envelope_basic():
    from rabbit.viz import audio_envelope
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16_000)).astype(np.float32)
    t, env = audio_envelope(audio, sample_rate=16_000, target_hz=100.0)
    assert t.shape == env.shape
    assert t.shape[0] == 100  # one second at 100 Hz envelope
    assert np.all(env > 0)


def test_plot_audio_and_responses_renders(predictor, tmp_path):
    import matplotlib.pyplot as plt
    from rabbit.viz import plot_audio_and_responses

    audio = np.random.RandomState(0).randn(40 * 16_000).astype(np.float32)
    result = predictor.predict(
        audio, tr_length=1.49, hrf_delay=4, trim_start=2, trim_end=2,
        apply_processor=False,
    )
    fig, axes = plot_audio_and_responses(
        audio=audio, result=result,
        roi_bases=("aud_primary", "aud_belt", "ifg"),
        title="smoke test",
    )
    assert len(axes) == 4  # 1 audio + 3 ROIs
    out_png = tmp_path / "out.png"
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 1000


def test_plot_roi_grid_renders(predictor, tmp_path):
    import matplotlib.pyplot as plt
    from rabbit.viz import plot_roi_grid

    audio = np.random.RandomState(0).randn(40 * 16_000).astype(np.float32)
    result = predictor.predict(
        audio, tr_length=1.49, hrf_delay=4, trim_start=2, trim_end=2,
        apply_processor=False,
    )
    # Only the ROIs the tiny model actually has.
    fig, axes = plot_roi_grid(
        result, roi_bases=("aud_primary", "aud_belt", "ifg"), ncols=3,
        title="grid smoke test",
    )
    out_png = tmp_path / "grid.png"
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 1000


def test_plot_audio_and_responses_time_window(predictor):
    """Cropping with time_window should not raise and should return the right axes."""
    import matplotlib.pyplot as plt
    from rabbit.viz import plot_audio_and_responses

    audio = np.random.RandomState(0).randn(40 * 16_000).astype(np.float32)
    result = predictor.predict(
        audio, tr_length=1.49, hrf_delay=4, trim_start=2, trim_end=2,
        apply_processor=False,
    )
    fig, axes = plot_audio_and_responses(
        audio=audio, result=result,
        roi_bases=("aud_primary",),
        time_window=(5.0, 20.0),
    )
    assert axes[-1].get_xlim() == (5.0, 20.0)
    plt.close(fig)
