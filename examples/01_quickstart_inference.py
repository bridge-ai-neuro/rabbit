"""Quickstart: load a trained RABBiT checkpoint and predict fMRI from audio.

Run this from the repo root after ``pip install -e .``::

    python examples/01_quickstart_inference.py \\
        --checkpoint /path/to/best.pt \\
        --audio       sample.wav

The example assumes ``configs/friends_shared_dev.yaml`` and expects
``RABBIT_BASES_DIR`` to point at the Friends PCA-bases directory (which ships
alongside trained checkpoints).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rabbit.inference import ROIPredictor


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "friends_shared_dev.yaml",
    )
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--tr", type=float, default=1.49)
    ap.add_argument("--hrf-delay", type=int, default=6)
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if None)")
    args = ap.parse_args()

    predictor = ROIPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
        use_avg_dev=True,
    )

    result = predictor.predict(
        audio_path=args.audio,
        tr_length=args.tr,
        hrf_delay=args.hrf_delay,
    )

    print(f"Predicted fMRI shape:  {result.fmri.shape}")
    print(f"TR centre times:       {result.tr_times[:3]}, ..., {result.tr_times[-1]:.2f}")
    print()
    print(f"Per-ROI shapes (first 5 of {len(result.roi_names)}):")
    for name in result.roi_names[:5]:
        arr = result.by_roi[name]
        print(f"  {name:<22}  {arr.shape}   range [{arr.min():.3f}, {arr.max():.3f}]")

    print()
    # Cheap sanity check on the predictions
    finite_frac = np.isfinite(result.fmri).mean()
    print(f"Finite fraction:       {finite_frac:.4f}")


if __name__ == "__main__":
    main()
