"""CJK font configuration for Chinese figure labels.

Chinese labels require a CJK-capable font.  :func:`configure_cjk_font` selects
the first available one and configures matplotlib; if none is installed it
returns ``False`` and labels fall back to the default font (Chinese glyphs may
render as boxes, but plotting still succeeds).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
]

_configured = None


def configure_cjk_font() -> bool:
    """Set a CJK sans-serif font if available. Returns True on success."""
    global _configured
    if _configured is not None:
        return _configured

    # Allow an explicit .ttf/.otf via env var (works with zero system deps).
    font_file = os.environ.get("AI_INVERSION_CJK_FONT")
    if font_file and os.path.exists(font_file):
        fm.fontManager.addfont(font_file)
        name = fm.FontProperties(fname=font_file).get_name()
        plt.rcParams["font.sans-serif"] = [name] + plt.rcParams.get("font.sans-serif", [])
        plt.rcParams["axes.unicode_minus"] = False
        _configured = True
        return _configured

    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in _CANDIDATES if c in available), None)
    if chosen is not None:
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams.get(
            "font.sans-serif", []
        )
        plt.rcParams["axes.unicode_minus"] = False
        _configured = True
    else:
        _configured = False
    return _configured
