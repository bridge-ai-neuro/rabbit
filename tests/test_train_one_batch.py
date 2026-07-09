"""End-to-end smoke test: one training step on a tiny synthetic dataset.

Builds a tiny RABBiT (4 ROIs, hidden_dim=64), wraps a synthetic two-clip
dataset, runs one ``Trainer.fit()`` for ``num_epochs=1`` with no backbone
freezing, and asserts:

  * the loss is finite,
  * at least one trainable parameter changed,
  * a ``best.pt`` checkpoint was written.

Runs in <30 s on CPU with no HF downloads — the backbone is monkey-patched.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from rabbit.model import RABBiT, build_flat_roi_layout
from rabbit.train import TrainConfig, Trainer


# ─────────────────────────────────────────────────────────────────────────────
# Tiny synthetic backbone (same shim used in test_forward_shapes)
# ─────────────────────────────────────────────────────────────────────────────


class _DummyBackbone(nn.Module):
    hidden_size: int = 64
    sampling_rate: int = 16_000
    processor = None

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, self.hidden_size)
        self._stride = 320

    def forward(self, input_values: torch.Tensor, attention_mask=None):
        B, T = input_values.shape
        T_keep = (T // self._stride) * self._stride
        x = input_values[:, :T_keep].reshape(
            B, T_keep // self._stride, self._stride
        ).mean(-1, keepdim=True)
        return self.proj(x), None


# ─────────────────────────────────────────────────────────────────────────────
# Tiny clip dataset
# ─────────────────────────────────────────────────────────────────────────────


class _TinyClip(torch.utils.data.Dataset):
    """Random audio + random fMRI for a few TRs across two subjects."""

    def __init__(self, n_TRs: int, audio_len: int, flat_dim: int, n_subj: int = 2):
        torch.manual_seed(0)
        total = n_TRs * n_subj
        self.audio = torch.randn(total, audio_len)
        self.fmri  = torch.randn(total, flat_dim)
        self.subj  = torch.tensor([s for s in range(n_subj) for _ in range(n_TRs)])

    def __len__(self) -> int:
        return self.audio.shape[0]

    def __getitem__(self, idx: int):
        return self.audio[idx], self.fmri[idx], int(self.subj[idx])

    def release(self) -> None:  # mirror FriendsClipDataset API
        pass


def _tiny_model() -> RABBiT:
    layout = build_flat_roi_layout(
        OrderedDict([("aud_primary_lh", 30), ("aud_primary_rh", 30), ("ifg_lh", 20), ("ifg_rh", 20)])
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


def test_one_epoch_end_to_end(tmp_path: Path):
    model = _tiny_model()
    flat_dim = model.output_dim
    audio_len = 16_000  # 1 s @ 16 kHz; gives 50 backbone tokens

    train_clips = ["clip_a", "clip_b"]
    val_clips = ["clip_val"]
    factories = {
        name: _TinyClip(n_TRs=12, audio_len=audio_len, flat_dim=flat_dim)
        for name in train_clips + val_clips
    }

    def factory(name: str):
        return factories[name]

    cfg = TrainConfig(
        num_epochs=1,
        batch_size=8,
        base_lr=1e-3, readout_lr=1e-3, bases_lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        warmup_fraction=0.0,
        freeze_backbone_epochs=0,
        early_stopping_patience=0,
        lambda_corr=1.0, lambda_l2=1.0,
        lambda_anchor=1e-3, mu_ortho=0.0,
        save_dir=tmp_path / "out",
        exp_name="smoke",
    )

    # Snapshot a backbone-side param + a brain-encoder param + a bases param
    # before training so we can assert each group actually receives gradients.
    sample_backbone = next(p.detach().clone() for p in model.backbone.parameters() if p.requires_grad)
    head_name, head_param = next(
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and "decoder" in n
    )
    sample_head = head_param.detach().clone()
    bases_name, bases_param = next(
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and ("bases_shared" in n or "bases_dev" in n)
    )
    sample_bases = bases_param.detach().clone()

    trainer = Trainer(
        model=model,
        train_clips=train_clips, val_clips=val_clips,
        clip_factory=factory, config=cfg, device="cpu",
    )
    trainer.fit()

    # ── assertions ──
    json_log = json_logs(trainer)
    assert len(json_log["epochs"]) == 1
    epoch = json_log["epochs"][0]
    assert np.isfinite(epoch["train_loss"]) and np.isfinite(epoch["eval_loss"])
    assert epoch["n_samples"] > 0

    # Backbone, brain-encoder, and bases groups all updated.
    new_backbone = next(p for p in model.backbone.parameters() if p.requires_grad)
    assert not torch.allclose(sample_backbone, new_backbone.detach()), \
        "Backbone group did not update."

    new_head = dict(model.named_parameters())[head_name]
    assert not torch.allclose(sample_head, new_head.detach()), \
        f"Brain-encoder group did not update ({head_name})."

    new_bases = dict(model.named_parameters())[bases_name]
    assert not torch.allclose(sample_bases, new_bases.detach()), \
        f"Bases group did not update ({bases_name})."

    # Checkpoints written.
    best_ckpt = cfg.save_dir / cfg.exp_name / "best.pt"
    epoch_ckpt = cfg.save_dir / cfg.exp_name / "epoch_0.pt"
    assert best_ckpt.exists(), "best.pt was not saved."
    assert epoch_ckpt.exists(), "epoch_0.pt was not saved."

    # Reload the checkpoint and forward-pass it to confirm round-tripability.
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    assert "model" in ckpt and "rabbit_version" in ckpt
    fresh = _tiny_model()
    fresh.load_state_dict(ckpt["model"])
    with torch.no_grad():
        out = fresh(torch.randn(2, audio_len), torch.zeros(2, dtype=torch.long))
    assert out["flat_predictions"].shape == (2, flat_dim)
    assert torch.isfinite(out["flat_predictions"]).all()


def json_logs(trainer: Trainer) -> dict:
    import json
    return json.loads(trainer.tlog.json_path.read_text())
