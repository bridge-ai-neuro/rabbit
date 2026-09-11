# RABBiT web demo — speech → fMRI, fully client-side

This is the live demo for the RABBiT project page; it lives at `site/demo/` and
links back to the paper page (`../`). Recorded examples replay saved population
predictions. Experimental microphone mode runs RABBiT inference on audio in a
browser worker; it does not measure or personalize to the visitor's brain.
Colourmap matches the website hero video (`site/assets/hero_build/
brain_render.py`): uniform neutral cortex, `|value| < 0.05` stays neutral, the
salient steel→neutral→orange→crimson BOLD map for activation.

## Run locally

Preferred — serve the **whole site** from its root so you can click from the
landing page into the demo and back:

```bash
cd site
node serve.mjs             # http://localhost:8000/demo/   (landing at /)
```

Standalone (demo only):

```bash
cd site/demo
node serve.mjs             # http://localhost:8000   (MIME + HTTP range requests)
# or: python3 -m http.server 8000   (works, but no range streaming for .onnx)
```

Choose an example and **Play**, **Pause/Resume**, or **Stop**. Changing examples
stops old audio; starting either microphone or clip playback stops the other.
Brain frames and clip captions follow the AudioContext clock, including pauses.

For microphone mode, choose **Load microphone model (422 MB)**. Download and
initialization have separate states; Cancel and Retry are available. Start the
microphone only after the model is ready. Expect about ten seconds of context
plus CPU processing before the first prediction. Internet access is needed for
Three.js and ONNX runtime modules; weights are cached where supported. Optional
live captions explicitly load their own model and do not feed text to RABBiT.

## How it works

```
clip.f32 ─(JS windowing+HRF stack, pipeline.mjs)→ (TR, 166880)
         ─(onnxruntime-web, assets/rabbit_*.onnx)→ flat (TR, 41394)
         ─(flat_to_fs6 scatter)→ fs6 (81924) → Three.js pial surface, synced to audio
```

`pipeline.mjs` (preprocessing + scatter + colormap) is verified **bit-exact**
against the Python/PyTorch pipeline — see `node_check/parity.mjs`
(`node node_check/parity.mjs`). The Three.js render can't be smoke-tested
headless (sandbox WebGL), so verify it visually in a real browser.

## Assets (`assets/`)

| file | what |
|---|---|
| `rabbit_fp32.onnx` | full model, ONNX (symlink → `scripts/_onnx_proto_out/`). **422 MB.** |
| `pial_coords.f32` / `coords.f32` | fsaverage6 pial / inflated vertices (81924×3) |
| `faces.i32` | shared triangulation (163840×3) |
| `flat_to_fs6.i32` | model output → fs6 vertex map (41394) |
| `clip{1..5}.f32` + `clips.json` | 24 s 16 kHz demo clips (Friends + 21st Year) |
| `preds_clip{1..5}.f32` | precomputed per-TR predictions for clip mode |
| `transcripts.json` | precomputed captions (synced, per clip) |
| `render_manifest.json` | dims, ROI slices |

Regenerate: `scripts/onnx_export_prototype.py` (ONNX),
`node_check/extract_render_assets.py` (surface), `node_check/make_fixture.py`
(parity fixture).

## Deploy to GitHub Pages

The demo deploys as part of the project page: it ships inside `site/demo/`, so
publishing `site/` (see `../README.md`) puts the demo live at
`https://<user>.github.io/<repo>/demo/`. Everything is static.

Clip mode works out of the box. For **mic mode**, the 422 MB ONNX weight
exceeds the Pages 100 MB file limit — host it on the Hugging Face Hub or a CDN
(still serverless) and configure `model-loader.mjs`. `assets/
rabbit_fp32.onnx` is a symlink to `../../scripts/_onnx_proto_out/` for local
dev; do not commit the resolved blob. Do **not** set COOP/COEP — it breaks
onnxruntime-web's worker (see the note in `serve.mjs`).

## Known follow-ups

- **fp16 (211 MB, lossless)** is the right shipping weight but its ONNX
  *packaging* is currently blocked: onnxconverter-common / ORT / half-trace all
  emit graphs onnxruntime rejects at attention dtype boundaries (the fp16 *math*
  is proven lossless — `scripts/fp16_fidelity_check.py`, Pearson 0.999999). Needs
  a manual dtype-unification pass or onnxsim. int8 is unusable (Pearson 0.82
  full / 0.98 backbone-only — visible degradation).
- Few-shot is not live (needs the viewer's own fMRI); show as a JS α-fit replay
  on stored subjects.

## Website QA (2026-09-09)

The current lifecycle, responsive, and error-state browser checks are under
`../../design-review/check-fixes-6-7.mjs`; they use real recorded clips and
controlled model workers/synthetic microphone streams. The separate
`check-real-demo-model.mjs` exercises the actual ONNX worker with recorded speech
fed through a synthetic microphone stream. Both open no server port and access
no microphone hardware. See `../design-qa.md` for outcomes and limits.

`model-loader.mjs` is the sole model URL/revision/cache configuration. Neither
page downloads weights automatically. Older prefetch/gating tests in
`node_check/` target the previous behavior and should not be used as acceptance
checks for explicit model loading.
