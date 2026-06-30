# RABBiT project page

This directory holds the RABBiT paper project page. It deploys as a static
GitHub Pages site with no build step.

## Files

```
site/
├── index.html              the project page
├── serve.mjs               local dev server — serves landing + demo (MIME + range)
├── assets/
│   ├── style.css           shared stylesheet
│   └── *.png               figures (converted from paper/figs/*.pdf)
├── demo/                    the live, fully client-side web demo (speech → fMRI)
│   ├── index.html          demo UI (Three.js pial brain + onnxruntime-web)
│   ├── app.js, *.mjs       demo logic + mic/whisper workers + shared pipeline
│   └── assets/             demo clips, predictions, fs6 surface, ONNX weights
├── .nojekyll               tell GitHub Pages to skip Jekyll
└── README.md               this file
```

The project page links to the demo from the hero ("Try it live"), the sticky
nav ("Live demo"), and the Resources grid; the demo links back via its top bar
("← Overview"). They share the same palette so the two pages read as one site.

## Deploy to GitHub Pages (production)

Deployment is automated by `.github/workflows/deploy-pages.yml`. One-time setup:
repo **Settings → Pages → Source = "GitHub Actions"**. Then every push to `main`
that touches `site/` publishes the page at `https://<user>.github.io/<repo>/`.

The workflow **prunes** the publish tree to ~45 MB so the 1 GB Pages limit is
never at risk — it excludes the dev/build-only weight (`assets/hero_build/` 128 MB,
`demo/node_check/` incl. its 412 MB `node_modules`, the 422 MB `*.onnx` symlink,
the unreferenced `hero_rabbit.gif`, and the dev servers/READMEs). Nothing else is
needed: the site is static, no Jekyll (`.nojekyll`), no bundling.

> If you ever fall back to "Deploy from a branch /site", the same exclusions must
> be enforced via `.gitignore` (already ignores `node_modules/`, `*.onnx`) — but
> `hero_build/` would then ship, so prefer the Actions workflow.

### Mic mode: host the model on Hugging Face

Clip mode (the default) is precomputed and works on Pages with zero setup. **Mic
mode** runs the 422 MB ONNX, which can't live on Pages (100 MB file cap). Host it
on a **public Hugging Face model repo** (CloudFront-backed, CORS-OK, free):

1. Create a public HF model repo and upload `rabbit_fp32.onnx`
   (regenerate via `scripts/onnx_export_prototype.py`).
2. Set `MODEL_HF_REPO` at the top of `demo/app.js` to that repo
   (the weight URL is built as `…/resolve/main/rabbit_fp32.onnx`).

The weight is cached per-device (Cache Storage, key `rabbit-model-v1`) so each
visitor downloads it at most once. Until the repo exists, mic mode degrades
gracefully ("couldn't load the model — it may still be deploying"); the clips
still work. **Do not** set COOP/COEP headers — they break onnxruntime-web's worker.

### Custom domain / base URL

The absolute URLs in `index.html` (`canonical`, `og:image`, `twitter:image`),
`sitemap.xml`, and `robots.txt` assume `https://bridge-ai-neuro.github.io/rabbit/`. If
you move to a custom domain, update those (and add a `CNAME` file).

## Local preview

```bash
cd site
node serve.mjs        # http://localhost:8000  (landing)  ·  /demo/  (live demo)
```

`serve.mjs` serves the whole `site/` tree so you can click straight from the
landing page into the demo and back. Plain `python3 -m http.server 8000` also
works for the landing page, but the **demo** needs `serve.mjs` — its large
`.onnx` weights require correct MIME types and HTTP range requests.

### Deploying the demo

The demo's clip mode is precomputed and works on plain GitHub Pages with no
extra setup. Its **mic mode** needs the 422 MB ONNX weight, which exceeds the
Pages 100 MB file limit — host that file on the Hugging Face Hub or a CDN (still
serverless) and point `MODEL_URLS` in `demo/app.js` at the URL. `demo/assets/
rabbit_fp32.onnx` is a symlink to `scripts/_onnx_proto_out/` for local dev; do
not commit the resolved 422 MB blob to the Pages branch.

## Pre-launch checklist

The page is launch-safe now (no 404s): every not-yet-public link — Paper PDF,
arXiv, Code/repo, demo notebook — is **gated to the in-page `#release-status`
note**. When the real URLs exist, flip them back:

1. **Code repo** — make `github.com/<user>/rabbit` public, then repoint the
   gated links (grep `href="#release-status"` in `index.html` and
   `href="../#release-status"` in `demo/index.html`).
2. **arXiv + Paper PDF** — set the real hrefs once posted (currently gated).
3. **Demo notebook** — link to the repo blob / nbviewer once public (gated).
4. **Mic-mode model** — upload `rabbit_fp32.onnx` to a public HF repo and set
   `MODEL_HF_REPO` in `demo/app.js` (see "Mic mode" above).
5. **BibTeX** — set the real venue in `@inproceedings{moussa2026rabbit, …}` once
   accepted.
6. **Custom domain (optional)** — update the absolute URLs (see above).

## Updating figures

To re-export figures after a paper revision:

```bash
cd paper/figs
for fig in teaser transformer training-process dots_lang_paper \
           "hierarchy_path_2d 2" cv_w2v_saturation_lh_lang \
           vert_story_twosided_bracket_sensorimotorgrouped \
           appendix_variants_lang_sig \
           brain_aggregate_delta_fs6_focus_with_legend; do
  magick -density 180 "$fig.pdf" -background white -alpha remove -alpha off \
    "../../site/assets/$(echo $fig | tr ' ' '_').png"
done
```

## Design language

- **Light theme** (warm white `#faf9f6`)
- **Serif headings** (Iowan Old Style / Charter / Source Serif Pro)
- **Sans body** (Inter)
- **Paul Tol Muted accents** — same palette as paper figures
- **Section colour coding**: zero-shot = steel blue, few-shot = wine,
  TBT = teal, SID = wine
- No JavaScript dependencies; deploys as static HTML

The walk-through under `walk_thru/` uses the same colour language so the two
sites visually rhyme.
