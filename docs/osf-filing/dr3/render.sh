#!/bin/bash
# Render the frozen DR-3 snapshot to PDF (layout only, text unchanged).
# Tooling: pandoc (the repository's paper-build tool) with xelatex
# (TeX Live 2026); used here: pandoc 3.12.1, XeTeX 3.141592653-2.6-0.999998.
# render.lua is a verbatim copy of docs/osf-filing/e1-small/render.lua;
# strikeout.lua keeps struck-through text (PI decisions) intact.
set -eu
cd "$(dirname "$0")"
SRC=osf-prereg-dr3-relational-probe-2026-10-23
pandoc -f markdown+lists_without_preceding_blankline "$SRC.frozen.md" \
  -o "$SRC.frozen.pdf" \
  --pdf-engine=xelatex --lua-filter=render.lua --lua-filter=strikeout.lua \
  -V header-includes="\\usepackage[normalem]{ulem}" \
  -V mainfont="Arial Unicode MS" -V mainfontoptions="BoldFont=Arial Bold" \
  -V mainfontoptions="ItalicFont=Arial Italic" \
  -V mainfontoptions="BoldItalicFont=Arial Bold Italic" \
  -V monofont="Menlo" -V geometry:margin=2cm -V fontsize=10pt \
  -V colorlinks=true -V pagestyle=plain \
  --metadata title="DR-3 global relational probe pre-registration — frozen snapshot" \
  --metadata subtitle="docs/$SRC.md at commit 14c3d5f86443 (file SHA-256 15dcf778…9fc67084)"
