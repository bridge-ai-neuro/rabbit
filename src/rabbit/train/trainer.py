"""RABBiT training loop.

One trainer for all readout variants — direct, parametric, shared_deviation,
dataset_shared_deviation. Dataset-agnostic on the inside: the trainer
iterates over a sequence of *clip datasets*, each of which yields
``(audio_window, fmri_TR, subject_idx)`` samples. The Friends dataset adapter
in ``rabbit.data.friends`` is the canonical implementation; other datasets
plug in by exposing the same shape.

End-to-end recipe:

  1. For each epoch, iterate over training clips (one DataLoader per clip).
  2. Per batch, forward through the model, compute loss + anchor + ortho,
     backprop, step three optimiser groups.
  3. (Optional) freeze backbone for the first N epochs.
  4. After each epoch, evaluate on the validation clip(s) using the same
     loss.
  5. Save ``epoch_{N}.pt`` and ``best.pt`` (lowest eval loss).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from ..model import RABBiT
from .loss import UnscaledCorrelationLoss
from .optim import build_param_groups, count_group_params


__all__ = [
    "TrainConfig",
    "Trainer",
    "ClipDatasetFactory",
]


# ─────────────────────────────────────────────────────────────────────────────
# Config + factory protocol
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """Knobs the trainer cares about. Defaults match the paper headline."""

    num_epochs:              int   = 30
    batch_size:              int   = 256
    base_lr:                 float = 1.0e-4
    readout_lr:              float = 1.0e-3
    bases_lr:                float = 1.0e-4
    weight_decay:            float = 1.0e-2
    grad_clip:               float = 1.0
    warmup_fraction:         float = 0.03
    freeze_backbone_epochs:  int   = 3
    early_stopping_patience: int   = 5
    lambda_corr:             float = 1.0
    lambda_l2:               float = 1.0
    lambda_anchor:           float = 1.0e-3
    mu_ortho:                float = 0.0
    seed:                    int   = 122
    num_workers:             int   = 0
    pin_memory:              bool  = False

    save_dir:                Path  = field(default_factory=lambda: Path("./outputs/run"))
    exp_name:                str   = "rabbit"
    log_every_n_steps:       int   = 50


# A ClipDatasetFactory is anything callable that returns a fetched
# torch.utils.data.Dataset for the given clip name. The dataset must yield
# 3-tuples (audio_window, fmri_TR, subject_idx) on each __getitem__.
ClipDatasetFactory = Callable[[str], Dataset]


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────


class _TrainLogger:
    """JSON + .log writer; one row per epoch."""

    def __init__(self, save_dir: Path, exp_name: str) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = save_dir / f"{exp_name}_{ts}.json"
        self.log_path  = save_dir / f"{exp_name}_{ts}.log"
        self._logger = logging.getLogger(f"rabbit.train.{exp_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(self.log_path)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        self._logger.addHandler(fh)
        self._logger.addHandler(ch)
        self._data: dict = {"run_id": ts, "config": {}, "epochs": []}

    def set_config(self, cfg: dict) -> None:
        self._data["config"] = cfg
        self._flush()
        self._logger.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def log_epoch(self, **row) -> None:
        self._data["epochs"].append(row)
        self._flush()

    def _flush(self) -> None:
        with open(self.json_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class Trainer:
    """RABBiT trainer.

    Args:
        model: a constructed ``RABBiT``. Its readout type drives whether
            ``subject_indices`` are passed at forward time and whether the
            anchor + ortho regularisers contribute.
        train_clips: iterable of clip names to iterate per epoch.
        val_clips: clip names used for end-of-epoch evaluation. Loss is
            averaged across them.
        clip_factory: callable ``clip_name -> Dataset``. Each dataset must
            yield ``(audio, fmri, subj_idx)`` 3-tuples.
        config: ``TrainConfig`` knobs (LRs, schedule, regularisers, paths).
        device: torch device. If None, ``cuda`` when available.
    """

    def __init__(
        self,
        model: RABBiT,
        train_clips: Sequence[str],
        val_clips: Sequence[str],
        clip_factory: ClipDatasetFactory,
        config: TrainConfig,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.model = model
        self.train_clips = list(train_clips)
        self.val_clips = list(val_clips)
        self.clip_factory = clip_factory
        self.cfg = config
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.model.to(self.device)

        self.criterion = UnscaledCorrelationLoss(
            roi_slices=self.model.roi_layout.roi_slices,
            lambda_corr=config.lambda_corr,
            lambda_l2=config.lambda_l2,
        )

        self.param_groups = build_param_groups(
            model,
            base_lr=config.base_lr,
            readout_lr=config.readout_lr,
            bases_lr=config.bases_lr,
            weight_decay=config.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(self.param_groups)

        # Linear warmup + linear decay LR schedule; total_steps estimated
        # from train_clips × an average per-clip step count. The estimate
        # only matters for warmup_fraction and decay slope.
        steps_per_clip = max(1, 300 // max(1, config.batch_size))
        total_steps = max(1, steps_per_clip * len(self.train_clips) * config.num_epochs)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(config.warmup_fraction * total_steps),
            num_training_steps=total_steps,
        )

        self.tlog = _TrainLogger(Path(config.save_dir), config.exp_name)
        self.tlog.info(
            f"Trainer ready · model={type(self.model).__name__} · "
            f"readout={self.model.readout_type} · device={self.device}"
        )
        self.tlog.info(
            "Optimiser groups: " + ", ".join(
                f"{k}={v:,}" for k, v in count_group_params(self.param_groups).items()
            )
        )
        if config.freeze_backbone_epochs > 0:
            self._set_backbone_trainable(False)
            self.tlog.info(
                f"Backbone frozen for the first {config.freeze_backbone_epochs} epoch(s)."
            )

        self.best_eval_loss: float = float("inf")
        self.epochs_no_improve: int = 0

    # ── Training entry point ─────────────────────────────────────────────────

    def fit(self) -> None:
        for epoch in range(self.cfg.num_epochs):
            if (
                self.cfg.freeze_backbone_epochs > 0
                and epoch == self.cfg.freeze_backbone_epochs
            ):
                self._set_backbone_trainable(True)
                self.tlog.info(f"Backbone un-frozen at start of epoch {epoch}.")

            train_loss, reg_loss, n_samples = self._train_epoch(epoch)
            eval_loss = self._eval_epoch()
            lr_now = float(self.optimizer.param_groups[0]["lr"])

            self.tlog.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                reg_loss=reg_loss,
                eval_loss=eval_loss,
                lr=lr_now,
                n_samples=n_samples,
            )
            self.tlog.info(
                f"Ep{epoch:>3}  train={train_loss:.5f}  eval={eval_loss:.5f}  "
                f"reg={reg_loss:.6f}  samples={n_samples}  lr={lr_now:.2e}"
            )
            self._save_checkpoint(epoch=epoch, eval_loss=eval_loss)
            if self._should_early_stop(eval_loss):
                self.tlog.info(f"Early stopping at epoch {epoch}.")
                break

    # ── One training epoch ───────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> tuple[float, float, int]:
        self.model.train()
        recon_losses: list[float] = []
        reg_losses:   list[float] = []
        total_samples = 0

        for clip_name in self.train_clips:
            ds = self.clip_factory(clip_name)
            if ds is None or len(ds) == 0:
                continue
            loader = DataLoader(
                ds, batch_size=self.cfg.batch_size, shuffle=False,
                num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
            )
            for batch in loader:
                wav, fmri, subj = _unpack(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)

                out = self.model(
                    wav,
                    subj if self.model.needs_subject_indices else None,
                )
                recon = self.criterion(out["flat_predictions"], fmri)

                reg = self.model.compute_anchor_loss() * self.cfg.lambda_anchor
                if self.cfg.mu_ortho > 0:
                    reg = reg + self.cfg.mu_ortho * self.model.compute_ortho_loss()
                total = recon + reg

                total.backward()
                if self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                recon_losses.append(float(recon.detach()))
                reg_losses.append(float(reg.detach()) if torch.is_tensor(reg) else float(reg))
                total_samples += wav.shape[0]

            # Free per-clip cache between clips.
            if hasattr(ds, "release"):
                ds.release()

        train_mean = float(np.mean(recon_losses)) if recon_losses else float("nan")
        reg_mean = float(np.mean(reg_losses)) if reg_losses else 0.0
        return train_mean, reg_mean, total_samples

    # ── Validation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_epoch(self) -> float:
        if not self.val_clips:
            return float("nan")
        self.model.eval()
        clip_losses: list[float] = []
        for clip_name in self.val_clips:
            ds = self.clip_factory(clip_name)
            if ds is None or len(ds) == 0:
                continue
            loader = DataLoader(
                ds, batch_size=self.cfg.batch_size, shuffle=False,
                num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
            )
            losses: list[float] = []
            for batch in loader:
                wav, fmri, subj = _unpack(batch, self.device)
                out = self.model(
                    wav,
                    subj if self.model.needs_subject_indices else None,
                )
                losses.append(float(self.criterion(out["flat_predictions"], fmri).detach()))
            if losses:
                clip_losses.append(float(np.mean(losses)))
            if hasattr(ds, "release"):
                ds.release()
        return float(np.mean(clip_losses)) if clip_losses else float("nan")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.model.backbone.parameters():
            p.requires_grad = trainable

    def _save_checkpoint(self, *, epoch: int, eval_loss: float) -> None:
        save_dir = Path(self.cfg.save_dir) / self.cfg.exp_name
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model": self.model.state_dict(),
            "epoch": epoch,
            "eval_loss": eval_loss,
            "rabbit_version": "0.1",
            "readout_type": self.model.readout_type,
            "roi_names": list(self.model.roi_layout.roi_names),
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch}.pt")
        if eval_loss < self.best_eval_loss:
            torch.save(ckpt, save_dir / "best.pt")
            self.best_eval_loss = eval_loss
            self.epochs_no_improve = 0
            self.tlog.info(f"  new best eval={eval_loss:.5f} — saved best.pt")
        else:
            self.epochs_no_improve += 1

    def _should_early_stop(self, eval_loss: float) -> bool:
        if self.cfg.early_stopping_patience <= 0:
            return False
        return self.epochs_no_improve >= self.cfg.early_stopping_patience


# ─────────────────────────────────────────────────────────────────────────────
# Internal: support 2- or 3-tuple datasets (subject-aware vs not)
# ─────────────────────────────────────────────────────────────────────────────


def _unpack(batch, device):
    if len(batch) == 3:
        wav, fmri, subj = batch
        subj = subj.to(device).long() if torch.is_tensor(subj) else torch.as_tensor(subj, device=device, dtype=torch.long)
    elif len(batch) == 2:
        wav, fmri = batch
        subj = None
    else:
        raise ValueError(f"Dataset must yield 2- or 3-tuples; got {len(batch)} elements.")
    return wav.to(device), fmri.to(device), subj
