"""Frontend publication figures must keep the docs canvas aspect, not equal-xy."""

from __future__ import annotations

import base64
import importlib.util
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_PY = ROOT / "frontend" / "public" / "py" / "figures.py"


def _load():
    spec = importlib.util.spec_from_file_location("frontend_figures", FIG_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _png_wh(b64: str) -> tuple[int, int]:
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", raw[16:24])
    return int(w), int(h)


def _origin_upper(nx, ny, fn):
    """Match ``extractSlice`` z-axis: row 0 = max y."""
    out = np.empty(ny * nx, dtype=float)
    for ix in range(nx):
        for iy in range(ny):
            out[(ny - 1 - iy) * nx + ix] = fn(ix, iy)
    return out


def test_slice_png_matches_docs_canvas():
    mod = _load()
    nx, ny = 25, 40
    vals = _origin_upper(nx, ny, lambda ix, iy: 20 + 0.4 * ix + 0.2 * iy)
    b64 = mod.render_slice({
        "values": vals.tolist(), "w": nx, "h": ny,
        "extent": [0.0, 50.0, 0.0, 80.0],
        "horizLabel": "X (m)", "vertLabel": "Y (m)",
        "title": "test slice", "unit": "MPa", "scale": 1.0,
        "reverse": False,
        "cmapStops": [
            {"pos": 0.0, "color": [68, 1, 84]},
            {"pos": 1.0, "color": [253, 231, 37]},
        ],
        "boreholes": [[10.0, 10.0], [40.0, 70.0]],
    })
    w, h = _png_wh(b64)
    assert (w, h) == (1080, 840)


def test_compare_png_matches_docs_canvas_not_equal_aspect():
    mod = _load()
    nx, ny = 25, 40
    truth = _origin_upper(nx, ny, lambda ix, iy: 40 + 8 * np.exp(-((ix - 18) ** 2 + (iy - 16) ** 2) / 40))
    mwd = truth + 6.0
    seis = truth - 3.0
    fused = truth + 1.0
    b64 = mod.render_compare({
        "w": nx, "h": ny,
        "extent": [0.0, 50.0, 0.0, 80.0],
        "horizLabel": "X (m)", "vertLabel": "Y (m)",
        "title": "融合优势对比",
        "vmin": float(truth.min()), "vmax": float(truth.max()),
        "scale": 1.0,
        "boreholes": [[10.0, 10.0], [40.0, 40.0]],
        "panels": [
            {"title": "(a) 强度真值", "residualTitle": "", "values": truth.tolist()},
            {"title": "(b) 仅钻孔插值", "residualTitle": "(e) 仅钻孔 $-$ 真值", "values": mwd.tolist()},
            {"title": "(c) 仅波阻抗标定", "residualTitle": "(f) 仅波阻抗 $-$ 真值", "values": seis.tolist()},
            {"title": "(d) 外漂移克里金融合", "residualTitle": "(g) 融合 $-$ 真值", "values": fused.tolist()},
        ],
    })
    w, h = _png_wh(b64)
    # docs/images/fusion_advantage.png is 2220×990 (14.8×6.6 in @ 150 dpi).
    assert (w, h) == (2220, 990)
    assert abs(w / h - 2.242) < 0.02


def test_profile_png_matches_docs_canvas():
    mod = _load()
    z = np.linspace(0.0, -40.0, 20).tolist()
    well = {
        "title": "钻孔",
        "z": z,
        "ucs_true": [30 + i for i in range(20)],
        "ucs_mwd": [31 + i for i in range(20)],
        "ucs_seis": [28 + i for i in range(20)],
        "ucs_fused": [30.5 + i for i in range(20)],
    }
    b64 = mod.render_profile({"title": "沿孔剖面", "wells": [well, well]})
    w, h = _png_wh(b64)
    assert (w, h) == (1560, 840)
