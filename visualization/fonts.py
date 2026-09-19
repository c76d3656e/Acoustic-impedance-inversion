"""Publication fonts: Times/Liberation Serif for Latin + CJK for Chinese.

Chinese labels need a CJK-capable face; numerals, English and math stay in a
Times-compatible serif (Times New Roman on Windows/macOS, Liberation Serif on
Linux).  Matplotlib 3.6+ font fallback walks ``font.family`` per glyph.

:func:`configure_cjk_font` (kept as the public name) selects both faces and
configures rcParams.  Plotting still succeeds if CJK is missing; Chinese then
falls back to boxes.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_CJK_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
]

_SERIF_CANDIDATES = [
    "Times New Roman",
    "Times",
    "Liberation Serif",
    "Nimbus Roman",
    "Nimbus Roman No9 L",
    "DejaVu Serif",
]

_SERIF_FILES = [
    os.path.join(_ROOT, "frontend", "public", "fonts", "liberation-serif.ttf"),
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/serif.ttf",  # Pyodide virtual FS
]

_CJK_FILES = [
    # Do not register frontend/public/fonts/cjk-subset.otf here: that face is a
    # tiny UI subset and would hide the system CJK font of the same family name.
]

_configured = None


def _register_file(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    try:
        fm.fontManager.addfont(path)
        return fm.FontProperties(fname=path).get_name()
    except Exception:
        return None


def _first_installed(names: list[str]) -> str | None:
    available = {f.name for f in fm.fontManager.ttflist}
    return next((n for n in names if n in available), None)


def configure_cjk_font() -> bool:
    """Set Times-compatible serif + CJK fallback. Returns True if CJK is available."""
    global _configured
    if _configured is not None:
        return _configured

    serif = None
    for path in _SERIF_FILES:
        serif = _register_file(path)
        if serif:
            break
    if serif is None:
        serif = _first_installed(_SERIF_CANDIDATES) or "DejaVu Serif"

    cjk = None
    env = os.environ.get("AI_INVERSION_CJK_FONT")
    if env:
        cjk = _register_file(env)
    if cjk is None:
        for path in _CJK_FILES:
            cjk = _register_file(path)
            if cjk:
                break
    if cjk is None:
        cjk = _first_installed(_CJK_CANDIDATES)

    family = [serif]
    if cjk:
        family.append(cjk)
    family.append("DejaVu Serif")

    plt.rcParams["font.family"] = family
    plt.rcParams["font.serif"] = [serif, "DejaVu Serif"]
    if cjk:
        plt.rcParams["font.sans-serif"] = [cjk, serif, "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    _configured = bool(cjk)
    return _configured
