# RABBiT website QA — fixes 6 and 7

Date: 2026-09-09. Implemented locally for the next user review. No commit, push,
deployment, listening server, or microphone-hardware access.

## Changes ready to inspect

- `/demo/`: Play/Pause, Resume and Stop; audio-clock synchronization; immediate
  cancellation when changing examples; exclusive microphone/example modes.
- Explicit 422 MB model loading with byte progress, Cancel, Retry and separate
  “weights loaded / initializing / ready” states. No automatic weight downloads.
  Optional live captions disclose their separate model and fail independently.
- Model/worker failures, denied or late microphone permission, disconnected
  streams and leaving the page release resources. Storage failure does not
  prevent use of already-downloaded weights.
- Demo skip link/main landmark, readable disabled controls, keyboard brain
  rotation/zoom, reset view, mobile fitting, and startup/WebGL/no-JS messages.
- Hero native playback controls plus Replay; no automatic video request with
  reduced motion or data saving. The accessible name identifies the pre-rendered,
  sped-up sequence. The removed caption has not been restored.
- Verified resource destinations; accurate browser-export availability; separate
  dataset links; complete two-author BibTeX with copy/download and copy fallback.
  Demo canonical and social metadata now match its publication path.

## Validation

`design-review/fixes-6-7/` contains screenshots, JSON reports and release evidence.

| Check | Result |
| --- | --- |
| Loading helper tests | 4 passed: response validation, cancellation/stalls, worker readiness/errors/timeouts, deployment/local URL selection |
| Main/demo browser interactions | 41 passed; no uncaught page errors |
| Real precomputed example | Original audio, prediction arrays and cortical mesh played; pause held audio-clock progress and transcript; resume continued; stop reset the view |
| Model lifecycle | Cancel/retry, cached-weight reuse, initialization failure, delayed/denied mic permission, mode switching, caption opt-in/failure/cleanup passed with controlled fixtures |
| Real ONNX browser inference | 422,176,732-byte local export initialized on CPU in 6.763 seconds in this environment; recorded speech through a synthetic mic stream produced 31,235 non-neutral vertices; tracks released on Stop; no page errors |
| Responsive review | Main/resources at 320, 390, 1024, 1440 and 1920 px; demo at 320, 390, 768, 1024 and 1440 px; no horizontal document overflow; inspected desktop/mobile captures |
| Continuous research regression | All 15 numbered figures and both equations; native wheel/End scrolling, reversible fades, restored sidebar, mobile menu, reduced motion, no-JS navigation, seven legacy URLs and project-prefix navigation passed |
| Static references | 208 local references resolve; every image has an alt attribute; IDs unique; both canonical URLs listed in sitemap |
| Scientific preservation | Approved intro unchanged; scientific pipeline and ONNX worker byte-identical to backup; scientific figure assets untouched |
| External destinations | All nine unique external page links returned HTTP 200; pinned ONNX HEAD returned expected size and browser CORS header |
| Syntax/whitespace | JavaScript syntax and `git diff --check` passed |

The shared white/blue tokens retain the prior measured text-contrast minima:
Light 5.25:1, Soft 4.87:1, Dark 6.55:1. New controls use these tokens; video
controls/error copy use explicit colors for their white figure surface. Disabled
demo cards no longer dim their explanatory copy.

## Limits and remaining review

Browser checks used Chromium with desktop/mobile viewport and touch emulation,
not physical phones, Safari, Firefox or a screen-reader session. No real
microphone was accessed. The live-caption lifecycle was tested with controlled
workers; the separate speech-recognition model's transcription quality was not
reevaluated. Real RABBiT inference used the full local export delivered through a
Blob-backed response because a single 422 MB Playwright response exceeds its
serialization limit. The public export was verified with metadata and HEAD,
not a full remote browser download.

A test browser profile could not write a 422 MB cache entry. The real-model test
therefore also exercised successful model use when the cache write failed.
Caching remains best-effort, as the page now explains. The measured initialization
time is test evidence, not a visitor-facing performance promise.

Temporal alignment in the headline comparison remains an open provenance
question: saved offsets may include onset corrections or use different time
origins. No timing error has been established. The qualified source note is
`../design-review/fixes-3-4/comparison-audit.md`. No manuscript/evaluation change
was made as part of these website fixes.

## Preview and rollback

```bash
cd /BRAIN/BBT/work/rabbit/site
PORT=8000 node serve.mjs
```

Open `http://localhost:8000/` and `http://localhost:8000/demo/`. Choose another
unused `PORT` if occupied; stop your foreground preview with Ctrl+C. Remote
workspaces may need that port forwarded in the IDE.

Pre-edit snapshot:
`../backups/site-before-fixes-6-7-20260909T203923Z/changed-files.tar.gz`, with a
verified SHA-256 manifest. Earlier full-site backups remain available. The new
`demo/model-loader.mjs` and `assets/rabbit.bib` were added in this pass; a manual
rollback can restore the archived files and remove those two additions.


## Approved few-shot schematic — 2026-09-10

The approved compact figure now appears beside the few-shot introduction on wide
screens and below its unchanged text on narrow screens. “Idiosyncratic” replaces
“Deviation” in the figure’s head label, tuning annotation and accessible text.
All fifteen numbered scientific figures remain. The new methods schematic is
unnumbered and offers a keyboard-accessible full-size SVG link.

Browser checks covered seven widths (320–1920 px), the reading themes, aspect
ratio, unchanged 17px intro text, no page overflow, label fit and active few-shot
navigation. Desktop/mobile captures were inspected. On narrow screens the full
schematic scales down; the full-size link provides access to its fine labels.
Evidence: `../design-review/fewshot-schematic/integrated/checks.json`.
Backup: `../backups/site-before-fewshot-schematic-20260910T071407Z/`.
The SVG can be removed and the archived index/styles restored to revert this
addition. No server, commit, push or deployment in this pass.


## Figure link simplification — 2026-09-10

Per user feedback, removed the 19 redundant Full-size image/figure/schematic and
Paper figure (PDF)/PDF text links beneath figures. The figure images themselves
remain linked. Earlier QA descriptions of caption links describe the prior UI.
Static comparison confirms all image attributes, linked images, section IDs and
15 numbered figures are preserved. No additional runtime behavior changed.


## Zero-shot copy correction — 2026-09-10

Updated zero-shot descriptions to population-level responses without fMRI at
prediction time. Changed the workflow label to No fitting, and added the native
dual-setting sentence with a speech-to-fMRI scope and “To our knowledge” qualifier.
Compared HTML with the verified pre-edit backup: all tags, attributes, images,
links and 15 numbered figures are unchanged. Checked replacement text and
`git diff --check`; no browser regression run for these text-only edits.
The earlier exact hero-copy QA reflects the wording before this user correction.
Backup: `../backups/site-before-zero-shot-copy-20260910T114014Z/`.


## Results sequence and disclosure defaults — 2026-09-10

The Results introduction now contains three collapsed metric definitions,
followed by Zero-shot results with one TRIBEv2 comparison and the regional and
group-level breakdowns. Removed the individual zero-shot example and repeated
Prediction settings section. Both equations are expanded by default. Fourteen
numbered figures remain; the removed example's source asset stays on disk.

Updated Chromium regression passed: exact zero-shot figure order, single
comparison, navigation/scroll progress, reversible fades, reduced motion,
keyboard opening/closing of every metric, default-open equations, native
no-JavaScript disclosures, eight legacy redirects, and project-prefix navigation.
No horizontal overflow at 320, 390, 1024, 1440 and 1920 px; no browser errors.
Reviewed desktop Results/comparison and mobile metric screenshots. A focused
comparison placement check also verified the retained #comparison anchor and
Zero-shot sidebar state at 320, 390 and 1440 px.

Static checks confirmed balanced HTML, unique IDs, valid fragment links, and
unchanged formula text and metric definitions. `git diff --check` passed.
Evidence: `../design-review/zero-shot-restructure/checks.json`,
`comparison-layout-checks.json`, and adjacent screenshots. Chromium viewport
emulation was used; no physical-device or other-browser run in this pass.

Backups: `../backups/site-before-zero-shot-restructure-20260910T114432Z/` and
`../backups/site-before-metrics-disclosures-20260910T114829Z/`.
All changes remain local. No server ports opened, commit, push or deployment.


## Neuroscience wording — 2026-09-10

Updated the neuroscience section heading, both navigation labels and the two
localizer headings per review. Verified replacements and compared HTML tokens:
all tags and attributes remain identical, preserving 14 numbered figures and
section anchors. No browser rerun for this text-only edit.
Backup: `../backups/site-before-neuroscience-headings-20260910T115745Z/`.


## Evaluation summary removed — 2026-09-10

Removed the Evaluation at a glance block. Static HTML checks confirm balanced
tags, unique IDs, valid in-page links and 14 retained numbered figures. Updated
the legacy Methods evaluation redirect to Results. No browser rerun for this
section deletion; formulas and metric disclosures are unchanged.
Backup: `../backups/site-before-evaluation-removal-20260910T115937Z/`.


## Hero video frame removed — 2026-09-10

Desktop/mobile Chromium checks at 1440 and 390 px confirmed the hero container
has a transparent background, zero border/padding/radius, working native control
attribute and no page overflow. Media assets and playback code are unchanged.
Evidence: `../design-review/hero-frame-removal/checks.json` and screenshots.
Backup: `../backups/site-before-hero-frame-removal-20260910T121230Z/`.
No server port opened. `git diff --check` passed.
