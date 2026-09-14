# RABBiT project page

This directory holds the RABBiT paper project page. It deploys as a static
GitHub Pages site with no build step.


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



## Local preview

```bash
cd site
node serve.mjs        # http://localhost:8000  (landing)  ·  /demo/  (live demo)
```

`serve.mjs` serves the whole `site/` tree so you can click straight from the
landing page into the demo and back. Plain `python3 -m http.server 8000` also
works for the landing page, but the **demo** needs `serve.mjs` — its large
`.onnx` weights require correct MIME types and HTTP range requests.



