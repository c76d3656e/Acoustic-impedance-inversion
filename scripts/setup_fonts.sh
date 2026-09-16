#!/usr/bin/env bash
# Install a CJK font so Chinese figure labels render (optional).
#
# The visualization code falls back gracefully to the default font if no CJK
# font is present, so this is only needed when you want Chinese titles/labels.
# Requires sudo. Alternatively, point AI_INVERSION_CJK_FONT at a .ttf/.otf file.
set -euo pipefail

if fc-list 2>/dev/null | grep -qiE 'wqy|noto sans cjk'; then
    echo "CJK font already installed."
    exit 0
fi

sudo apt-get update
sudo apt-get install -y fonts-wqy-microhei fonts-noto-cjk
echo "CJK font installed. Re-run your visualization script."
