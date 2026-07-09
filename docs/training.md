# Training

The unified RABBiT trainer ([`rabbit.train.Trainer`](../src/rabbit/train/trainer.py))
covers every readout variant (`direct`, `parametric`, `shared_deviation`,
`dataset_shared_deviation`) through a single code path. The dataset adapter
is pluggable — Friends ships today in [`rabbit.data.friends`](../src/rabbit/data/friends.py);
additional datasets implement the same per-clip protocol.

## Hardware

The paper headlines were trained in bf16 on a single H200. Empirically,
`bf16 + larger batch on one H200` outperforms `fp16 + DDP on two H100s` —
fp16 has produced NaN / DDP convergence issues for the shared-deviation
readout. Stick with bf16 if you have it.

## Quickstart — Friends shared-dev wav2vec2-base (paper headline)

```bash
export RABBIT_BASES_DIR=/path/to/Friends_Templates_fs6
export RABBIT_DATA_ROOT=/path/to/friends_fs6_semi
export RABBIT_AUDIO_ROOT=/path/to/friends_dataset/stimuli

PYTHONPATH=src python scripts/train.py --config configs/friends_shared_dev.yaml
```

For multi-GPU bf16 (one process per GPU):

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    scripts/train.py --config configs/friends_shared_dev.yaml
```

CLI overrides land on dotted config keys:

```bash
PYTHONPATH=src python scripts/train.py \
    --config configs/friends_shared_dev.yaml \
    --override training.batch_size=128 training.num_epochs=10
```

Total training time on H200 with the default config: ~24 GPU-hours.

## Loss

```
loss = lambda_corr · ⟨ 1 − pearson(pred_roi, target_roi) ⟩_ROI
     + lambda_l2   · MSE(pred, target)
     + lambda_anchor · ‖ bases - bases_init ‖²
     + mu_ortho      · ‖ bases_shared · bases_sharedᵀ − I ‖²
```

The correlation term is the headline gradient signal — it drives per-vertex
`r` upward. The L2 term keeps predictions on the right scale. The anchor
keeps the learnable shared/dev bases near their PCA-derived init. `mu_ortho`
is off by default.

Implementation: [`rabbit.train.UnscaledCorrelationLoss`](../src/rabbit/train/loss.py),
plus the model's `compute_anchor_loss()` / `compute_ortho_loss()` hooks
([wrapper.py](../src/rabbit/model/wrapper.py#L246-L260)).

## Optimiser groups (three-way split)

| Group | Params | LR (default) | Weight decay |
|---|---|---|---|
| backbone        | wav2vec2 / WavLM (LoRA adapters or unfrozen attention) | `1e-4` | `1e-2` |
| brain_encoder   | input proj, queries, decoder, coefficient heads        | `1e-3` | `1e-2` |
| bases           | `bases_shared`, `bases_dev` (shared-deviation only)    | `1e-4` | `0`    |

The bases run on a slower LR and skip weight decay because the anchor
regulariser already controls their drift. Construction:
[`rabbit.train.build_param_groups`](../src/rabbit/train/optim.py).

## Backbone freeze schedule

Backbone gradients are off for the first `freeze_backbone_epochs` epochs
(default 3). This lets the brain encoder fit the LoRA-free representation
before adapter gradients begin to flow.

## Data layout

The expected on-disk layout:

```
$RABBIT_DATA_ROOT/
└── sub-{01..06}/sXX_eYY_{a,b}.npz       # 30-ROI fMRI per (subject, clip)

$RABBIT_AUDIO_ROOT/
└── s{1..6}/friends_sXXeXX{a,b}.npy       # 16 kHz mono waveform per clip

$RABBIT_BASES_DIR/
├── average/pca_bases.npz                 # shared-pathway init
└── sub-{01..06}/pca_bases.npz            # per-subject deviation init
```

Each fMRI `.npz` is keyed by the 30 ROI names with `(n_TRs, V_roi)` float32
arrays, z-scored per ROI at preprocessing. Each clip is ≈400 TRs at TR=1.49 s.

## Checkpoint format

[`Trainer._save_checkpoint`](../src/rabbit/train/trainer.py#L209) writes one
`epoch_N.pt` per epoch plus a `best.pt` updated whenever the validation loss
improves. The dict format:

```python
{
    "model":           state_dict,                  # straight nn.Module.state_dict()
    "epoch":           int,
    "eval_loss":       float,
    "rabbit_version":  "0.1",
    "readout_type":    "shared_deviation",
    "roi_names":       [...],
}
```

Legacy research checkpoints (`{"model": state_dict}` with old module names)
remain loadable through `rabbit.inference.load_from_checkpoint` — see the
key-remap in [`wrapper.py`](../src/rabbit/model/wrapper.py#L260-L297).

## Subject indexing convention

| Dataset | ID type | Subject directory |
|---|---|---|
| moth    | `int` (1..8)  | `UTS0{N}/` |
| friends | `str` (`'01'..'06'`) | `sub-{NN}/` |

The order of `data.subjects` in the config defines the 0-based slot used to
index `bases_dev[slot]`. Keep this list consistent between training and
inference.

## At-a-glance: minimum viable run

```python
from rabbit.data import FriendsClipManifest, FriendsClipDataset
from rabbit.inference.checkpoint import build_rabbit_from_config
from rabbit.train import Trainer, TrainConfig
from rabbit.utils import load_config

cfg = load_config("configs/friends_shared_dev.yaml")
model = build_rabbit_from_config(cfg)
manifest = FriendsClipManifest.discover(
    roi_dir=cfg["data"]["fmri_dir"],
    audio_dir=cfg["data"]["audio_dir"],
    subjects=cfg["data"]["subjects"],
)

def factory(clip):
    return FriendsClipDataset(
        clip_name=clip, manifest=manifest,
        processor=model.backbone.processor,
    ).fetch()

trainer = Trainer(
    model=model,
    train_clips=manifest.clip_names[:-1],
    val_clips=manifest.clip_names[-1:],
    clip_factory=factory,
    config=TrainConfig(num_epochs=30, batch_size=256),
)
trainer.fit()
```
