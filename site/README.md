# RABBiT project page

This directory holds the RABBiT paper project page. It deploys as a static
GitHub Pages site with no build step.

## Current local review

The actual website at `/` is one continuous research document. A model definition,
hook, and four icon-led points cover prediction, adaptation, state-of-the-art performance,
and neuroscience analyses. The use cases lead into Results, beginning with
expandable metric definitions, then Zero-shot results: one TRIBEv2 comparison, the regional breakdown, and the
group-level summary. Few-shot results and language-network findings follow,
then model architecture, resources, and citation. Fourteen numbered scientific
figures remain; the individual zero-shot example was removed during review. Navigation links
jump within this document; `/results/`, `/methods/`, and `/resources/` now redirect
to their corresponding sections. `/design-preview/` remains a historical mockup.

```bash
cd /BRAIN/BBT/work/rabbit/site
PORT=8765 node serve.mjs
```

Open `http://localhost:8765/`. Try scrolling through the sections, using the restored side navigator, and
opening the use-case data requirements or metric definitions. Methods equations are visible by default. The demo remains at `/demo/`. If the server is
already running, refresh the page rather than launching another server on the
same port. For a remote workspace, forward the selected port. No build or push
is required.

Before fix 2, the complete working site was archived in
`../backups/site-before-fix2-20260908T005429Z/`. The earlier working-site and
last-commit archives remain in `../backups/site-before-full-redesign-20260908T001335Z/`.
Each backup has a manifest with checksums. The later `site-before-scroll-*` and
`site-before-continuous-document-*` snapshots preserve the subsequent layout
and navigation revisions. Archives stay local.

The main document loads `assets/design-tokens.css`, `assets/redesign.css`,
`assets/pages.css`, and `assets/continuous.css`. Navigation and themes use
`assets/site.js`; `assets/reading.css` and `assets/reading.js` provide the section
index, progress, and reversible scroll fades. Original figures load lazily near
the viewport. All text and sections remain in the HTML, including without JS.

The demo uses `assets/demo-redesign.css`. The old `style.css` and `themes.css`
remain on disk but are not loaded by the main document. Fonts and Feather icons
are local, with licenses alongside. [Visual QA](design-qa.md) records preservation,
browser evidence, and test limits. The main `index.html` is the source of truth;
old one-time scripts in `design-review/` are historical migrations, not a build step.

## Comparison and design review

The TRIBEv2 map now identifies its target and the 15 recordings used in its
export. “How this comparison is measured” opens details on aggregation, surface
mapping, and temporal alignment. Figure images link to their full-size assets;
redundant caption links were removed. Evaluation defines group-level alignment,
individual alignment, and ISC.

The shared style system uses consistent section/component headings, 17px research
prose, 14px captions, and a maximum 1200px content area alongside the sidebar.
The demo uses the same Inter type and blue controls, with larger panel headings.
Source audit and before/after screenshots: `../design-review/fixes-3-4/`.
Saved timing offsets remain a provenance question; no timing error has been
established. See the later clarification in the audit and visual QA notes.

## Files

```
site/
├── index.html              full scrolling research document with concise opening
├── results/index.html      compatibility redirect to the results section
├── methods/index.html      compatibility redirect to model architecture
├── resources/index.html    compatibility redirect to resources/citation
├── serve.mjs               local server for all pages and demo (MIME + range)
├── assets/
│   ├── design-tokens.css   shared colors, typography, and icon treatment
│   ├── redesign.css        base research styles
│   ├── pages.css           shared research layout
│   ├── continuous.css      icon-led intro and full-document chapter spacing
│   ├── site.js             shared navigation and reading themes
│   ├── reading.css         restored sidebar, mobile section index, scroll effects
│   ├── reading.js          active sections, reading progress, reversible fades
│   ├── legacy-links.js     harmless compatibility file for older cached HTML
│   └── *.png               figures (converted from paper/figs/*.pdf)
├── demo/                    the live, fully client-side web demo (speech → fMRI)
│   ├── index.html          demo UI (Three.js pial brain + onnxruntime-web)
│   ├── app.js, *.mjs       demo logic + mic/whisper workers + shared pipeline
│   └── assets/             demo clips, predictions, fs6 surface, ONNX weights
├── .nojekyll               tell GitHub Pages to skip Jekyll
└── README.md               this file
```

The project page links to the demo from the hero ("Explore demo"), the sticky
nav ("Demo"), and the Resources section; the demo links back via its top bar
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

### Microphone model and loading

Recorded examples use precomputed predictions. They do not load the ONNX model.
Microphone mode requires an explicit **Load microphone model (422 MB)** action,
then **Start microphone** after CPU initialization succeeds. Loading can be
cancelled; failures offer a retry. Cached weights are reused when storage permits;
private browsing, quota limits, and browser eviction can require another download.
Optional live captions use a separate model and start only when selected.

`demo/model-loader.mjs` is the single model configuration: repository, pinned
revision, expected byte count, and cache version. It prefers the local ONNX
symlink on localhost; the published site uses the pinned Hugging Face export.
Update these values together when intentionally replacing the export. No model
prefetch runs on the paper page or on demo arrival. The model can run without
Cache Storage. Do not add COOP/COEP without retesting the ONNX worker.

Public availability checked 2026-09-09: `omermosa/rabbit` contains
`rabbit_fp32.onnx` (422,176,732 bytes), at revision
`c6405bf1788c46e7fa710f5c246f16b443ab024e`. PyTorch research checkpoints and PCA
bases were not listed there. The site describes the browser export and makes no
promise about other weights or release dates. Evidence: `../design-review/fixes-6-7/release-audit.md`.

### Custom domain / base URL

The absolute URLs in each page's `index.html` (`canonical`, `og:image`, `twitter:image`),
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
serverless) and configure `demo/model-loader.mjs`. `demo/assets/
rabbit_fp32.onnx` is a symlink to `scripts/_onnx_proto_out/` for local dev; do
not commit the resolved 422 MB blob to the Pages branch.

## Publication details

The paper, PDF, code, notebook, browser-model repository, and source-dataset
links were checked on 2026-09-09. The site uses the arXiv author list: Omer Moussa
and Mariya Toneva. `assets/rabbit.bib` matches the visible citation; copy and
file download controls are in `#cite`. Both pages have canonical/social metadata;
`sitemap.xml` lists the overview and demo. Update venue information only after
an authoritative publication record changes.

The saved comparison map's timing offsets still need tracing through their time
origins to confirm equivalence with the manuscript protocol. This is an open
provenance question, not an established error. See the qualified note in
`../design-review/fixes-3-4/comparison-audit.md`.

Current local validation and limitations: `design-qa.md`. Nothing in this pass
was committed, pushed, or deployed.

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

- White background, graphite text, restrained blue (`#2c5885`) controls and rules.
- Self-hosted Inter for headings and body, with the original RABBiT and MPI logos.
- Hero and page titles capped at 44px; section headings at 30–38px, method component headings at 25–29px.
- Shared page navigation with a labeled Menu on smaller screens; Demo remains prominent.
- Feather line icons accompany navigation, resource labels, and the two use-case workflows.
- Faint blue use-case band and clear spacing/rules separate the landing sections.
- The overview, results, methods, resources, and citation share one continuous scroll. The transparent right sidebar tracks sections; narrow screens have a section menu labeled with the current section. A 4px, unlabeled progress line stays directly below the sticky top navigation and fills as the document scrolls.
- Content fades at viewport edges and is fully visible in the central reading area. Reduced-motion settings, keyboard focus, no JavaScript, and printing retain full visibility.
- Data requirements and equations use native disclosures; anchors and browser history remain native.
- Optional Soft and Dark reading themes retain the same blue identity.
- Scientific colormaps are preserved. Full-size links expose every numbered figure.
- Static HTML with no build step; existing KaTeX and demo dependencies remain.

## Fixes 6 and 7 local review

Open `/demo/` to inspect Play/Pause, Resume, Stop, and model loading. Clips and
microphone are exclusive; changing examples stops existing audio immediately.
The brain supports keyboard rotation/zoom and a reset button. Microphone mode
is available only on supported desktops over HTTPS or localhost.

The hero video keeps native controls and adds Replay. It does not load
asynchronously until visible and motion/data preferences allow it, or until
explicitly played. Its accessible name identifies the pre-rendered, sped-up
sequence. No caption was restored beneath it.

Backup: `../backups/site-before-fixes-6-7-20260909T203923Z/changed-files.tar.gz`.
Current browser checks use request interception and open no server port:

```bash
# From the repository root; requires the existing local browser/test dependencies.
node --test design-review/check-model-loader.mjs
xvfb-run -a node design-review/check-fixes-6-7.mjs
xvfb-run -a node design-review/check-research-fixes-6-7.mjs
xvfb-run -a node design-review/check-real-demo-model.mjs
```

The older `demo/node_check/run_prefetch.mjs` and `run_gating.mjs` describe the
superseded automatic-download behavior and are not acceptance checks for this
version. The current checks above cover explicit loading and readiness.
