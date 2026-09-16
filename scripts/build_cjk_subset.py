"""Build a tiny CJK font subset for the frontend (web UI + Pyodide matplotlib).

Scans the frontend source + dataset manifest for every character actually used,
then subsets the system Noto Sans CJK SC face to just those glyphs. The result
(``frontend/public/fonts/cjk-subset.otf``) is small enough to ship with the
static site, so Chinese renders self-contained with no external font/CDN.

Requires a Noto Sans CJK font to be installed (see AGENTS.md); it is a
build-time-only dependency -- the committed subset is what the site uses.
"""

from __future__ import annotations

import glob
import os

from fontTools import subset
from fontTools.ttLib import TTFont, TTCollection

TTC_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
OUT = "frontend/public/fonts/cjk-subset.otf"
EXTRA = "×·⁶²³μ°±−–—…“”‘’、，。（）：；！？　％‰"


def find_font():
    for p in TTC_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No CJK font found. Install fonts-noto-cjk (see scripts/setup_fonts.sh)."
    )


def sc_font_number(path: str) -> int:
    if not path.endswith(".ttc"):
        return 0
    coll = TTCollection(path)
    for i, f in enumerate(coll.fonts):
        names = " ".join(
            filter(None, [f["name"].getDebugName(1), f["name"].getDebugName(4)])
        )
        if "SC" in names or "Simplified" in names:
            return i
    return 0


def collect_chars() -> str:
    chars = {chr(c) for c in range(0x20, 0x7F)}  # ASCII printable
    paths = glob.glob("frontend/src/**/*", recursive=True)
    paths += ["frontend/index.html", "frontend/public/data/manifest.json"]
    for p in paths:
        if os.path.isfile(p):
            try:
                txt = open(p, encoding="utf-8").read()
            except Exception:
                continue
            chars.update(ch for ch in txt if ord(ch) > 0x7F)
    chars.update(EXTRA)
    return "".join(sorted(chars))


def main() -> None:
    path = find_font()
    num = sc_font_number(path)
    text = collect_chars()

    options = subset.Options()
    options.set(layout_features="*", name_IDs="*", notdef_outline=True,
                recalc_bounds=True, glyph_names=False)
    font = TTFont(path, fontNumber=num, lazy=True)
    ss = subset.Subsetter(options)
    ss.populate(text=text)
    ss.subset(font)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    font.save(OUT)
    size_kb = os.path.getsize(OUT) / 1024
    print(f"Wrote {OUT} ({size_kb:.0f} KB) covering {len(text)} chars "
          f"from {os.path.basename(path)} [font #{num}]")


if __name__ == "__main__":
    main()
