"""Forward-shape smoke tests for the RABBiT model.

These tests build a tiny RABBiT (small synthetic PCA bases, 2 subjects, 4
ROIs) and check that the forward pass returns finite tensors of the expected
shapes for each ``readout_type``. They run in <30 s on CPU and require no
downloaded models or checkpoints — the speech backbone is monkey-patched
with a small stand-in module.
"""
from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from rabbit.model import (
    RABBiT,
    ROILayout,
    build_flat_roi_layout,
)


class _DummyBackbone(nn.Module):
    """Minimal stand-in for SpeechBackbone — avoids HF downloads in CI.

    Maps ``(B, T_audio)`` waveform to ``(B, T_frames, hidden_size)``. The
    ``T_frames`` count follows wav2vec2's 320× stride.
    """

    hidden_size: int = 64
    sampling_rate: int = 16_000

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, self.hidden_size)
        self._stride = 320

    def forward(self, input_values: torch.Tensor, attention_mask=None):
        # (B, T_audio) -> (B, T_audio, 1) -> linear -> mean-pool over stride
        B, T = input_values.shape
        T_keep = (T // self._stride) * self._stride
        x = input_values[:, :T_keep].reshape(B, T_keep // self._stride, self._stride).mean(-1, keepdim=True)
        return self.proj(x), None

    def feat_extract_output_lengths(self, n: int) -> int:
        return n // self._stride


def _tiny_layout() -> ROILayout:
    return build_flat_roi_layout(
        OrderedDict([("aud_primary_lh", 50), ("aud_primary_rh", 40), ("ifg_lh", 30), ("ifg_rh", 25)])
    )


def _tiny_bases(layout: ROILayout, K: int = 8) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {name: torch.randn(K, layout.roi_dims_dict[name]) * 0.1 for name in layout.roi_names}


def _tiny_dev_bases(layout: ROILayout, n_subj: int = 2, R: int = 3) -> dict[str, list[torch.Tensor]]:
    torch.manual_seed(1)
    return {
        name: [torch.randn(R, layout.roi_dims_dict[name]) * 0.05 for _ in range(n_subj)]
        for name in layout.roi_names
    }


@pytest.fixture(scope="module")
def tiny_audio() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(2, 16_000)  # 2 batch × 1-second audio


def _model_with_dummy_backbone(readout_type: str, layout: ROILayout, **kwargs) -> RABBiT:
    backbone = _DummyBackbone()
    return RABBiT(
        readout_type=readout_type,
        backbone=backbone,
        roi_layout=layout,
        hidden_dim=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        use_self_attention=True,
        **kwargs,
    )


def test_direct_forward_shapes(tiny_audio):
    layout = _tiny_layout()
    model = _model_with_dummy_backbone("direct", layout)
    out = model(tiny_audio)
    assert out["flat_predictions"].shape == (2, layout.flat_dim)
    assert torch.isfinite(out["flat_predictions"]).all()
    for name in layout.roi_names:
        assert out["predictions_by_roi"][name].shape == (2, layout.roi_dims_dict[name])


def test_parametric_forward_shapes(tiny_audio):
    layout = _tiny_layout()
    bases = _tiny_bases(layout)
    model = _model_with_dummy_backbone("parametric", layout, roi_bases=bases)
    out = model(tiny_audio)
    assert out["flat_predictions"].shape == (2, layout.flat_dim)
    assert torch.isfinite(out["flat_predictions"]).all()


def test_shared_deviation_forward_shapes(tiny_audio):
    layout = _tiny_layout()
    shared = _tiny_bases(layout, K=8)
    dev = _tiny_dev_bases(layout, n_subj=2, R=3)
    model = _model_with_dummy_backbone(
        "shared_deviation", layout, roi_bases=shared, dev_roi_bases=dev,
    )
    subj = torch.tensor([0, 1])
    out = model(tiny_audio, subj)
    assert out["flat_predictions"].shape == (2, layout.flat_dim)
    assert torch.isfinite(out["flat_predictions"]).all()


def test_shared_deviation_requires_subject_indices(tiny_audio):
    layout = _tiny_layout()
    shared = _tiny_bases(layout, K=8)
    dev = _tiny_dev_bases(layout, n_subj=2, R=3)
    model = _model_with_dummy_backbone(
        "shared_deviation", layout, roi_bases=shared, dev_roi_bases=dev,
    )
    with pytest.raises(ValueError, match="subject_indices"):
        model(tiny_audio)


def test_dataset_shared_deviation_forward_shapes(tiny_audio):
    layout = _tiny_layout()
    shared_a = _tiny_bases(layout, K=8)
    shared_b = _tiny_bases(layout, K=8)
    dev = _tiny_dev_bases(layout, n_subj=3, R=3)  # 3 subjects across 2 datasets
    model = _model_with_dummy_backbone(
        "dataset_shared_deviation", layout,
        shared_roi_bases_per_dataset=[shared_a, shared_b],
        dev_roi_bases=dev,
        subject_to_dataset=[0, 0, 1],
    )
    subj = torch.tensor([0, 2])
    out = model(tiny_audio, subj)
    assert out["flat_predictions"].shape == (2, layout.flat_dim)
    assert torch.isfinite(out["flat_predictions"]).all()


def test_avg_dev_changes_predictions(tiny_audio):
    layout = _tiny_layout()
    shared = _tiny_bases(layout, K=8)
    dev = _tiny_dev_bases(layout, n_subj=3, R=3)
    model = _model_with_dummy_backbone(
        "shared_deviation", layout, roi_bases=shared, dev_roi_bases=dev,
    )
    subj0 = torch.tensor([0, 0])

    before = model(tiny_audio, subj0)["flat_predictions"].detach().clone()
    model.use_avg_deviation(slot=0)
    after = model(tiny_audio, subj0)["flat_predictions"].detach()

    # The slot-0 bases were overwritten with the mean across slots 1 and 2,
    # so the prediction for subject_indices=0 must change.
    assert not torch.allclose(before, after)


def test_attention_returned_when_requested(tiny_audio):
    layout = _tiny_layout()
    model = _model_with_dummy_backbone("direct", layout)
    out = model(tiny_audio, return_attn=True)
    attn = out["attention_weights"]
    assert attn is not None
    # (B, n_heads, n_rois, n_tokens). n_tokens = 16000 / 320 = 50.
    assert attn.shape[0] == 2
    assert attn.shape[2] == layout.num_queries
