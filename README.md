# RABBiT

[Website](https://bridge-ai-neuro.github.io/rabbit/) · [Paper (arXiv)](https://arxiv.org/abs/2607.05171) · [Live demo](https://bridge-ai-neuro.github.io/rabbit/demo/) · [Model weights (Hugging Face)](https://huggingface.co/omermosa/rabbit)

**RABBiT** is a speech-to-fMRI brain encoder. It takes raw audio and predicts BOLD responses across 30 cortical ROIs on `fsaverage6`. Internally: a pretrained speech model (wav2vec2 or WavLM) produces frame-rate hidden states, a small transformer decoder with one learned query per ROI cross-attends to those frames, and a per-ROI readout combines a dataset-wide shared subspace with a low-rank per-subject deviation to produce vertex-level predictions.

This repo contains the model definition, inference API, and training/evaluation code behind the [paper](https://arxiv.org/abs/2607.05171).

## Quickstart

```bash
pip install -e .
```

The pretrained checkpoints and the PCA basis files are on the [Hugging Face model
repo](https://huggingface.co/omermosa/rabbit). Download a checkpoint (e.g.
`rabbit_friends_fs6_shared_dev.pt`) and the matching `Friends_Templates_fs6/`
bases, then point `RABBIT_BASES_DIR` at the bases directory:

```bash
export RABBIT_BASES_DIR=/path/to/Friends_Templates_fs6
```

Predict fMRI from an audio clip using a trained checkpoint:

```python
from rabbit.inference import ROIPredictor

predictor = ROIPredictor.from_checkpoint(
    checkpoint_path="path/to/best.pt",
    config_path="configs/friends_shared_dev.yaml",
    use_avg_dev=True,    # held-out-subject mode
)

result = predictor.predict(
    audio_path="path/to/clip.wav",
    tr_length=1.49,
    hrf_delay=6,
)

# result.fmri        — (n_TRs, 41394) fsaverage6 predictions over 30 ROIs
# result.by_roi      — OrderedDict[roi_name -> (n_TRs, V_roi)]
# result.tr_times    — (n_TRs,) audio center time in seconds per TR
```

CLI:

```bash
rabbit-predict \
  --checkpoint path/to/best.pt \
  --config     configs/friends_shared_dev.yaml \
  --audio      sample.wav \
  --output     predictions.npz \
  --tr 1.49 --hrf-delay 6
```

For an end-to-end tour — loading the model, predicting on a single clip,
visualizing the audio waveform alongside per-ROI responses, batch-inference
over a directory, and held-out evaluation on the narratives 21st-year story —
open [`notebooks/demo.ipynb`](notebooks/demo.ipynb). The held-out-evaluation
section also needs the `fs6` extra (`pip install -e ".[fs6]"`) for the
`mne`-based ROI extraction.

A fully in-browser **live demo** (speak into your mic, watch the predicted brain
update in real time) runs on the [project website](https://bridge-ai-neuro.github.io/rabbit/demo/).

## Model overview

```
audio (16 kHz)
  │
  ▼  wav2vec2 / WavLM (with LoRA adapters)
hidden_states (B, T, 768)
  │  + sinusoidal temporal position encoding
  ▼  Linear(768 → 256)
memory_tokens (B, T, 256)
  │
  ▼  transformer decoder
  │  · 30 learned ROI query embeddings (15 bilateral ROIs)
  │  · self-attention among ROI queries
  │  · cross-attention to speech tokens
  │  · two layers, eight heads
roi_tokens (B, 30, 256)
  │
  ▼  per-ROI shared+deviation readout
  │  pred = coeff_shared(h) @ bases_shared
  │       + coeff_dev(h)    @ bases_dev[subject]
flat_predictions (B, 41394)
```

The 30 ROIs are bilateral HCP-MMP1 groupings covering the canonical auditory and language pathway (`aud_primary`, `aud_belt`, `stg_sts`, `posterior_temp`, `temporal_pole`, `angular_gyrus`, `supramarginal`, `ifg`, `mfg_dlpfc`) plus DMN, motor, insula, and visual ROIs.

See [`docs/architecture.md`](docs/architecture.md) for the full mathematical breakdown including the shared+deviation factorization and the `avg-dev` zero-shot trick.

## Repo layout

```
rabbit/
├── src/rabbit/             package source
│   ├── model/              encoder, readouts, backbone, wrapper, ROI layout
│   ├── inference/          ROIPredictor + audio + checkpoint loader
│   ├── eval/               narratives held-out eval pipeline + metrics + fs6 helpers
│   ├── data/               Friends dataset adapter (clip-level)
│   ├── train/              Trainer + loss + optimiser groups
│   └── utils/              YAML config helpers
├── configs/                training / inference YAMLs (paper headlines)
├── examples/               worked end-to-end scripts
├── notebooks/              demo.ipynb — the end-to-end tour
├── scripts/                train.py — training entrypoint
├── tests/                  unit and smoke tests
└── docs/                   architecture, training, ...
```

## Status

**v0.1** — the model code, inference API, and evaluation pipeline, plus a
unified trainer with a synthetic-data smoke test that trains one batch
end-to-end. A full real-data reproduction run is not included here; see
[`docs/training.md`](docs/training.md) for the training pipeline and configs.

## Citation

If you use RABBiT, please cite:

```bibtex
@article{moussa2026rabbit,
  title   = {RABBiT: Rapidly Adaptive {BOLD} Foundation Model via Brain-Tuning
             for Accurate Zero-Shot and Few-Shot Prediction of
             Speech-Elicited Responses in the Brain},
  author  = {Moussa, Omer and others},
  journal = {arXiv preprint arXiv:2607.05171},
  year    = {2026}
}
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
