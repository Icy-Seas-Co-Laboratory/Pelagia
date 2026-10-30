# Documentation assets

`roi-workflow.svg` is a code-authored schematic for the Pelagia README and
workflow guides. It illustrates the relationship between an immutable source
frame, candidate ROI artifacts, refined masks, and human-curated evidence.
The shapes are illustrative and carry no biological, metric, or scale claim.

To revise it, edit the SVG directly and validate it with an XML parser. The
Pelagia logo and icon in this directory are existing project brand assets; they
are maintained separately from the workflow schematic.

## README logo wrapper

`readme-logo.svg` is a self-contained presentation wrapper around the original `pelagia_logo.png`, with a white rounded-rectangle backing for dark-theme contrast. It embeds the original artwork without recoloring or redrawing it. Regenerate the wrapper if the source logo changes.

The committed `readme-logo.png` derivative is rendered from that wrapper. To regenerate all derivatives without adding repository dependencies:

Run this from the Pelagia repository root:

```bash
npm install --prefix "${TMPDIR:-/tmp}/pelagia-doc-assets" --no-save --no-audit --no-fund @resvg/resvg-js
NODE_PATH="${TMPDIR:-/tmp}/pelagia-doc-assets/node_modules" node <<'JS'
const { Resvg } = require('@resvg/resvg-js');
const fs = require('fs');
const svg = 'docs/assets/readme-logo.svg';
fs.writeFileSync('docs/assets/readme-logo.png', new Resvg(fs.readFileSync(svg, 'utf8')).render().asPng());
JS
```

The wrapper embeds the original artwork as base64; edit or regenerate that wrapper from the original logo when the source logo changes, then render the PNG derivative.

