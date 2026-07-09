"""Train a RABBiT model from a YAML config.

Usage::

    PYTHONPATH=src python scripts/train.py --config configs/friends_shared_dev.yaml \\
        [--override key.path=value ...]

The config must contain ``readout_type``, ``backbone``, ``lora``,
``encoder``, ``parametric``, ``shared_deviation``, ``data``, ``temporal``,
and ``training`` sections — see ``configs/friends_shared_dev.yaml``.

For Friends + shared_deviation specifically: this script builds the manifest,
splits train/val (val defaults to one held-out clip), instantiates the model
via ``rabbit.inference.checkpoint.build_rabbit_from_config``, and runs the
training loop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rabbit.data import FriendsClipDataset, FriendsClipManifest
from rabbit.inference.checkpoint import build_rabbit_from_config
from rabbit.train import TrainConfig, Trainer
from rabbit.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    config = load_config(str(args.config), overrides=args.override)

    # ── data ──────────────────────────────────────────────────────────────────
    data_cfg = config.get("data", {})
    temp_cfg = config.get("temporal", {})
    train_cfg_raw = config.get("training", {})
    loss_cfg = config.get("loss", {})
    sd_cfg = config.get("shared_deviation", {})
    output_cfg = config.get("output", {})

    manifest = FriendsClipManifest.discover(
        roi_dir=data_cfg["fmri_dir"] if "fmri_dir" in data_cfg else data_cfg["roi_dir"],
        audio_dir=data_cfg["audio_dir"] if "audio_dir" in data_cfg else data_cfg["wav_dir"],
        subjects=data_cfg["subjects"],
        roi_names=data_cfg.get("roi_names"),
    )
    val_clips = data_cfg.get("val_clips", [])
    test_clips = data_cfg.get("test_clips", [])
    if not val_clips and manifest.clip_names:
        val_clips = [manifest.clip_names[-1]]  # default: last clip as val
    train_clips, val_clips, _ = manifest.split(val_clips=val_clips, test_clips=test_clips)
    print(f"[train] manifest: {manifest}")
    print(f"[train] split: {len(train_clips)} train | {len(val_clips)} val")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_rabbit_from_config(config)
    processor = model.backbone.processor

    def clip_factory(clip_name: str) -> FriendsClipDataset:
        ds = FriendsClipDataset(
            clip_name=clip_name,
            manifest=manifest,
            hrf_delay=int(temp_cfg.get("hrf_delay", 6)),
            tr_length=float(temp_cfg.get("tr_length", 1.49)),
            trim_start=int(temp_cfg.get("trim_start", 10)),
            trim_end=int(temp_cfg.get("trim_end", 9)),
            processor=processor,
            sampling_rate=int(model.backbone.sampling_rate),
        )
        return ds.fetch()

    # ── trainer ───────────────────────────────────────────────────────────────
    tc = TrainConfig(
        num_epochs=int(train_cfg_raw.get("num_epochs", 30)),
        batch_size=int(train_cfg_raw.get("batch_size", 256)),
        base_lr=float(train_cfg_raw.get("base_lr", 1e-4)),
        readout_lr=float(train_cfg_raw.get("readout_lr", 1e-3)),
        bases_lr=float(sd_cfg.get("bases_lr", 1e-4)),
        weight_decay=float(train_cfg_raw.get("weight_decay", 1e-2)),
        grad_clip=float(train_cfg_raw.get("grad_clip", 1.0)),
        warmup_fraction=float(train_cfg_raw.get("warmup_fraction", 0.03)),
        freeze_backbone_epochs=int(train_cfg_raw.get("freeze_backbone_epochs", 3)),
        early_stopping_patience=int(train_cfg_raw.get("early_stopping_patience", 5)),
        lambda_corr=float(loss_cfg.get("lambda_corr", 1.0)),
        lambda_l2=float(loss_cfg.get("lambda_l2", 1.0)),
        lambda_anchor=float(sd_cfg.get("lambda_anchor", 1e-3)),
        mu_ortho=float(sd_cfg.get("mu_ortho", 0.0)),
        save_dir=Path(output_cfg.get("save_dir", "./outputs")),
        exp_name=output_cfg.get("exp_name", "rabbit"),
    )
    print(f"[train] config: {tc}")

    trainer = Trainer(
        model=model,
        train_clips=train_clips,
        val_clips=val_clips,
        clip_factory=clip_factory,
        config=tc,
        device=args.device,
    )
    trainer.tlog.set_config(config)
    trainer.fit()


if __name__ == "__main__":
    main()
