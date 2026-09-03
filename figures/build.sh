#!/usr/bin/env bash
# Render figures.html -> GSI_IGS_System_Figures.pdf (11x8.5in landscape).
# Uses Chrome's own print-to-PDF: no npm deps (the old render.mjs needed puppeteer-core,
# which is not installed). Page size comes from the @page rule in figures.html.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$DIR/../GSI_IGS_System_Figures.pdf}"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless --disable-gpu --no-sandbox \
  --allow-file-access-from-files --force-color-profile=srgb \
  --no-pdf-header-footer --virtual-time-budget=6000 \
  --print-to-pdf="$OUT" "file://$DIR/figures.html" 2>/dev/null
echo "wrote $OUT"
