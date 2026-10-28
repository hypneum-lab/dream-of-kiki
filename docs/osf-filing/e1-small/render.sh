#!/bin/bash
# Render the frozen E1-small snapshot to PDF (layout only, text unchanged).
# Tooling: pandoc (the repository's paper-build tool) with xelatex
# (TeX Live 2026); used here: pandoc 3.12.1, XeTeX 3.141592653-2.6-0.999998.
set -eu
cd "$(dirname "$0")"
SRC=osf-prereg-e1-small-2026-10-25.frozen.md
pandoc -f markdown+lists_without_preceding_blankline "$SRC" \
  -o osf-prereg-e1-small-2026-10-25.frozen.pdf \
  --pdf-engine=xelatex --lua-filter=render.lua \
  -V mainfont="Arial Unicode MS" -V mainfontoptions="BoldFont=Arial Bold" \
  -V mainfontoptions="ItalicFont=Arial Italic" \
  -V mainfontoptions="BoldItalicFont=Arial Bold Italic" \
  -V monofont="Menlo" -V geometry:margin=2cm -V fontsize=10pt \
  -V colorlinks=true -V pagestyle=plain \
  --metadata title="E1-small pre-registration — frozen snapshot" \
  --metadata subtitle="docs/osf-prereg-e1-small-2026-10-25.md at commit 804b6d14c7cc (file SHA-256 6d47d8b7…93208463)"
