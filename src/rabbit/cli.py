"""Command-line entry points.

Today: ``rabbit-predict`` — audio file → predictions NPZ.

Usage::

    rabbit-predict \\
        --checkpoint path/to/best.pt \\
        --config     configs/friends_shared_dev.yaml \\
        --audio      sample.wav \\
        --output     predictions.npz \\
        --tr 1.49 --hrf-delay 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .inference import ROIPredictor


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict fMRI from audio using a trained RABBiT model.")
    p.add_argument("--checkpoint", required=True, type=Path, help="Path to a trained .pt checkpoint.")
    p.add_argument("--config", required=True, type=Path, help="YAML config matching the checkpoint.")
    p.add_argument("--audio", required=True, type=Path, help="Input audio file (wav / flac / npy).")
    p.add_argument("--output", required=True, type=Path, help="Output NPZ destination.")
    p.add_argument("--tr", type=float, default=1.49, help="TR length in seconds.")
    p.add_argument("--hrf-delay", type=int, default=6)
    p.add_argument("--trim-start", type=int, default=10)
    p.add_argument("--trim-end", type=int, default=9)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-avg-dev", action="store_true", help="Disable the avg-dev trick.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-attention", action="store_true")
    return p


def predict_cli() -> None:
    args = _build_parser().parse_args()

    predictor = ROIPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        use_avg_dev=not args.no_avg_dev,
    )

    result = predictor.predict(
        audio_path=args.audio,
        tr_length=args.tr,
        hrf_delay=args.hrf_delay,
        trim_start=args.trim_start,
        trim_end=args.trim_end,
        batch_size=args.batch_size,
        return_attention=args.save_attention,
    )

    out = dict(
        fmri=result.fmri.astype(np.float32),
        tr_times=result.tr_times.astype(np.float64),
        roi_names=np.array(result.roi_names),
    )
    if result.attention is not None:
        out["attention"] = result.attention.astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **out)
    print(f"[rabbit] wrote {args.output} — {result.fmri.shape[0]} TRs × {result.fmri.shape[1]} verts")


if __name__ == "__main__":
    predict_cli()
