#!/bin/bash
# Render the frozen G4-nono snapshots to PDF (layout only, text unchanged).
# Tooling: pandoc (the repository's paper-build tool) with xelatex
# (TeX Live 2026); used here: pandoc 3.12.1, XeTeX 3.141592653-2.6-0.999998.
# render.lua is a verbatim copy of docs/osf-filing/e1-small/render.lua;
# strikeout.lua keeps struck-through text (PI decisions) intact.
set -eu
cd "$(dirname "$0")"
COMMIT=14c3d5f86443
render() {
  pandoc -f markdown+lists_without_preceding_blankline "$1.frozen.md" \
    -o "$1.frozen.pdf" \
    --pdf-engine=xelatex --lua-filter=render.lua --lua-filter=strikeout.lua \
  -V header-includes="\\usepackage[normalem]{ulem}" \
    -V mainfont="Arial Unicode MS" -V mainfontoptions="BoldFont=Arial Bold" \
    -V mainfontoptions="ItalicFont=Arial Italic" \
    -V mainfontoptions="BoldItalicFont=Arial Bold Italic" \
    -V monofont="Menlo" -V geometry:margin=2cm -V fontsize=10pt \
    -V colorlinks=true -V pagestyle=plain \
    --metadata title="$2" \
    --metadata subtitle="docs/$1.md at commit $COMMIT (file SHA-256 $3)"
}
render osf-prereg-g4-nono-cross-domain-2026-10-23 \
  "G4-nono cross-domain RECOMBINE pre-registration — frozen snapshot" \
  "9c514909…57e9b28a"
render osf-prereg-erratum-g4-power-2026-10-23 \
  "Erratum: G4 / G5-bis power sentences — frozen snapshot" \
  "c15c202f…1c1371ed"
