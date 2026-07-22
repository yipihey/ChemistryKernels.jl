# ChemistryKernels.jl methods site

This site is the HTML methods overview and scientific figure atlas for
ChemistryKernels.jl. Reaction-rate, cooling, UV-background, shielding, metal,
and recombination curves are generated directly from the Julia package rather
than maintained as hand-copied plotting data.

## Automatic scientific-data updates

`npm run dev`, `npm run build`, and `npm test` regenerate
`app/science-data.ts` from `scripts/generate_science_data.jl` before compiling
the site. The generator loads the repository's Julia test environment and calls
the package's current rate, cooling, recombination, helium, dust, UVB, and metal
implementations.

From this directory:

```bash
npm run generate:science  # regenerate figures only
npm run dev               # regenerate, then start the local site
npm run build             # regenerate, then create the deployment build
npm run build:pages       # build a static GitHub Pages artifact
npm test                  # regenerate, build both targets, and test the HTML
```

The repository CI independently regenerates `app/science-data.ts` and fails if
the committed result differs. This makes a physics change that affects a figure
visible in the same pull request, while the HTML build test guards the complete
methods page.

## GitHub Pages

The `GitHub Pages` workflow publishes `dist/github-pages` after every push to
`main`. The static export keeps the interactive plots and rewrites generated
asset and metadata URLs for the repository path:

<https://yipihey.github.io/ChemistryKernels.jl/>

GitHub Pages must use **GitHub Actions** as its publishing source in the
repository settings. The workflow also supports manual runs from the Actions
tab.

## Requirements

- Julia with the repository `test` environment instantiated
- Node.js 22.13 or newer
- npm dependencies installed with `npm ci`

The generated `dist/` directory is intentionally ignored; hosting builds it
from the source and the current scientific dataset.
