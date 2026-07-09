# Architecture

A more detailed walk-through of the RABBiT model. For the package code, see
[`src/rabbit/model/`](../src/rabbit/model/).

## End-to-end pipeline

```
audio (16 kHz, mono)
   │
   ▼  SpeechBackbone — wav2vec2-base/large or WavLM-large (HF transformers).
   │      Feature encoder (CNN) is always frozen. Transformer half is either
   │      LoRA-adapted (rank 16) or fully frozen.
hidden_states (B, T_frames, 768 or 1024)        T_frames = T_audio / 320
   │
   ▼  Linear(hidden_size → 256) + LayerNorm + Dropout
memory_tokens (B, T, 256)
   │  + sinusoidal temporal position encoding
   │
   ▼  TemporalROIDecoder
   │     · 30 learned ROI query embeddings (15 bilateral HCP-MMP1 ROIs)
   │     · pre-norm pattern: self-attn → cross-attn → FFN, residuals everywhere
   │     · num_layers = 2, nhead = 8, dim_ff = 1024
roi_tokens (B, 30, 256)
   │
   ▼  per-ROI readout (see Readout variants below)
flat_predictions (B, 41394)                     fs6 z-scored fMRI
```

## Readout variants

Per-ROI heads in [`src/rabbit/model/readouts.py`](../src/rabbit/model/readouts.py).
Each takes a 256-d ROI token and produces a per-vertex prediction.

### `direct`
```
pred = Linear(256 → V_roi)(roi_token)
```
Plain readout. Most parameters, simplest interface. Strong baseline.

### `parametric`
```
coeffs = Linear(256 → K)(roi_token)
pred   = coeffs @ bases                    # bases (K, V) frozen, from PCA of training fMRI
```
PCA-precomputed spatial bases are stored as **buffers** (move with `.to(device)`,
saved in `state_dict`, no gradient). Bias init `1/K` so the initial prediction
is the average of all bases.

### `shared_deviation` (headline)
```
pred = coeff_shared(roi_token) @ bases_shared
     + coeff_dev(roi_token)    @ bases_dev[subject_index]
```
Both `bases_shared` (K, V) and `bases_dev` (n_subjects, R, V) are *learnable*
`nn.Parameter` tensors. Anchor buffers retain the PCA-derived init so the
trainer's anchor loss can penalise drift. `R ≪ K` (typically R=15, K=100).

For "hard" ROIs where subject identity dominates the response (AG, MFG, IFG, SM),
the deviation bias is initialised to `1/R` — same uniform-contribution
convention as the shared bias.

### `dataset_shared_deviation`
Same as `shared_deviation` but the shared basis is **per dataset**:
`bases_shared` is `(n_datasets, K, V)` and each batch element selects its
shared basis via `subject_to_dataset[subject_index]`. Used when training across
heterogeneous datasets (e.g. moth + friends).

## The avg-dev trick

At training time the model is conditioned on subject identity via
`subject_indices`. To run on a held-out subject not in the training set:

```python
model.use_avg_deviation(slot=0)
# Now feed subject_indices = torch.full((B,), 0) for any batch.
```

This overwrites `bases_dev[0]` (and, in the `indp_sd` variant, the per-subject
coeff_dev tensors) with the mean across the other subject slots. The shared
pathway is unchanged. Empirically this gives the best zero-shot transfer for
RABBiT-shared-dev models.

## Why the 256-dim bottleneck

wav2vec2 outputs 768-dim, WavLM-large 1024-dim, but the decoder operates in
256-dim throughout. The projection is deliberate: it caps decoder capacity to
prevent overfitting on the (relatively small) fMRI dataset while leaving the
backbone free to produce richer per-frame features. Multi-head attention with
`nhead=8` then gives 32-dim per head.

## Shapes summary (fs6 30-ROI, wav2vec2-base)

| Tensor                  | Shape                       |
|-------------------------|-----------------------------|
| input_wav               | (B, T_audio)                |
| hidden_states           | (B, T = T_audio / 320, 768) |
| memory_tokens           | (B, T, 256)                 |
| roi_tokens              | (B, 30, 256)                |
| `flat_predictions`      | (B, 41,394)                 |
| `attention_weights`     | (B, 8 heads, 30 rois, T)    |
| `bases_shared` per ROI  | (100, V_roi)                |
| `bases_dev` per ROI     | (n_subjects, 15, V_roi)     |

V_roi varies 77–5188 across the 30 ROIs; the smallest is right-hemisphere A1,
the largest is right-hemisphere motor cortex.
