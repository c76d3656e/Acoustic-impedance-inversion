"""Publication matplotlib recipes for the static frontend (Pyodide) and tests.

Layout matches ``visualization.report_style``: ``aspect='auto'`` so the 20×50 m
working-face window fills the axes the same way as ``docs/images``, and a fixed canvas
(no ``bbox_inches='tight'``) so PNG pixel aspect equals figsize × dpi.

  * slice      — 7.2 × 5.6 in @ 150 dpi → 1080 × 840
  * compare    — (3.4×ncols + 1.2) × 6.6 in @ 150 dpi → 2220 × 990 for 4 columns
  * profile    — 5.2×n × 5.6 in @ 150 dpi → 1560 × 840 for 2 wells
  * trislices  — 12.0 × 7.6 in @ 150 dpi → 1800 × 1140
"""

from __future__ import annotations

import base64
import io
import json

import matplotlib

matplotlib.use("Agg")

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SLICE_FIGSIZE = (7.2, 5.6)
COMPARE_ROW_H = 6.6
COMPARE_COL_W = 3.4
COMPARE_COL_PAD = 1.2
PROFILE_PANEL_W = 5.2
PROFILE_H = 5.6
TRISLICES_FIGSIZE = (12.0, 7.6)
SAVE_DPI = 150


def _configure_fonts():
    """Times/Liberation Serif for Latin; CJK face for Chinese glyphs."""
    serif = None
    for path in (
        "/serif.ttf",
        os.path.join(os.path.dirname(__file__), "..", "fonts", "liberation-serif.ttf"),
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ):
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                serif = fm.FontProperties(fname=path).get_name()
                break
            except Exception:
                continue
    cjk = None
    for path in (
        "/cjk.ttf",
        os.path.join(os.path.dirname(__file__), "..", "fonts", "cjk-subset.otf"),
    ):
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                cjk = fm.FontProperties(fname=path).get_name()
                break
            except Exception:
                continue
    family = [serif or "Liberation Serif", cjk or "DejaVu Sans", "DejaVu Serif"]
    plt.rcParams.update({
        "font.family": family,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })


_configure_fonts()


def _payload(p):
    return json.loads(p) if isinstance(p, str) else p


def _save_fixed(fig, dpi: int = SAVE_DPI) -> str:
    """Save using figsize × dpi, never tight-crop (that distorts panel aspect)."""
    old = plt.rcParams.get("savefig.bbox")
    plt.rcParams["savefig.bbox"] = None
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    finally:
        plt.rcParams["savefig.bbox"] = old


def _style_ax(ax, ext, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("auto")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _holes(ax, holes, s_outer=55, s_inner=8):
    for bx, by in holes:
        ax.scatter([bx], [by], s=s_outer, facecolors="none",
                   edgecolors="k", linewidths=1.1, zorder=5)
        ax.scatter([bx], [by], s=s_inner, c="k", zorder=6)


def _xy(p):
    w, h = int(p["w"]), int(p["h"])
    ext = p["extent"]
    xs = np.linspace(ext[0], ext[1], w)
    ys = np.linspace(ext[2], ext[3], h)
    return w, h, ext, xs, ys


def _field(values, w, h, scale):
    # Payload is origin-upper (row 0 = top = ext[3]); contourf wants south row first.
    return np.asarray(values, dtype=float).reshape(h, w)[::-1] / float(scale)


def _cmap_from_stops(p):
    if not p.get("cmapStops"):
        name = "viridis_r" if p.get("reverse") else "viridis"
        return plt.get_cmap(name)
    stops = sorted(p["cmapStops"], key=lambda s: s["pos"])
    pairs = [(s["pos"], tuple(c / 255.0 for c in s["color"])) for s in stops]
    if p.get("reverse"):
        pairs = sorted([(1 - pos, col) for pos, col in pairs], key=lambda t: t[0])
    lo, hi = pairs[0][0], pairs[-1][0]
    if hi <= lo:
        hi = lo + 1.0
    pairs = [(min(1.0, max(0.0, (pos - lo) / (hi - lo))), col) for pos, col in pairs]
    pairs[0] = (0.0, pairs[0][1])
    pairs[-1] = (1.0, pairs[-1][1])
    return LinearSegmentedColormap.from_list("custom", pairs)


def render_slice(payload) -> str:
    p = _payload(payload)
    w, h, ext, xs, ys = _xy(p)
    Z = _field(p["values"], w, h, p["scale"])
    cmap = _cmap_from_stops(p)
    vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    lev = np.linspace(vmin, vmax, 30)

    fig, ax = plt.subplots(figsize=SLICE_FIGSIZE)
    cf = ax.contourf(xs, ys, Z, levels=lev, cmap=cmap, extend="both")
    ax.contour(xs, ys, Z, levels=lev, colors="k", linewidths=0.4,
               linestyles="--", alpha=0.5)
    _holes(ax, p.get("boreholes") or [], s_outer=90, s_inner=10)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(p["unit"])
    _style_ax(ax, ext, p["title"], p["horizLabel"], p["vertLabel"])
    fig.tight_layout()
    return _save_fixed(fig)


def render_compare(payload) -> str:
    p = _payload(payload)
    w, h, ext, xs, ys = _xy(p)
    scale = float(p["scale"])
    panels = p["panels"]
    truth = _field(panels[0]["values"], w, h, scale)
    fields = [_field(m["values"], w, h, scale) for m in panels[1:]]
    residuals = [fld - truth for fld in fields]
    slice_abs = np.concatenate([np.abs(r).ravel() for r in residuals]) if residuals else np.array([1.0])
    err_abs = float(np.percentile(slice_abs, 98)) if slice_abs.size else 1.0
    if not np.isfinite(err_abs) or err_abs < 1e-6:
        err_abs = 1.0

    vmin = float(p["vmin"]) / scale
    vmax = float(p["vmax"]) / scale
    if vmax <= vmin:
        vmax = vmin + 1.0
    lev = np.linspace(vmin, vmax, 20)
    err_lev = np.linspace(-err_abs, err_abs, 21)
    n_m = len(fields)
    n_cols = n_m + 1
    fig_w = COMPARE_COL_W * n_cols + COMPARE_COL_PAD
    fig, axes = plt.subplots(
        2, n_cols, figsize=(fig_w, COMPARE_ROW_H), constrained_layout=True,
    )
    titles_top = [panels[0]["title"]] + [m["title"] for m in panels[1:]]
    vols_top = [truth] + fields
    field_cmap = _cmap_from_stops(p)
    cf0 = None
    for c, (vol, title) in enumerate(zip(vols_top, titles_top)):
        ax = axes[0, c]
        cf0 = ax.contourf(xs, ys, vol, levels=lev, cmap=field_cmap, extend="both")
        ax.contour(xs, ys, vol, levels=lev, colors="k", linewidths=0.3,
                   linestyles="--", alpha=0.4)
        _holes(ax, p.get("boreholes") or [])
        _style_ax(ax, ext, title, "", p["vertLabel"] if c == 0 else "")
        ax.set_xlabel("")

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5, 0.55,
        "下行：预测 − 真值\n红＝估计偏高\n蓝＝估计偏低\n越浅越好",
        transform=axes[1, 0].transAxes, ha="center", va="center",
        fontsize=11, linespacing=1.6,
    )
    cf1 = None
    for c, (res, method) in enumerate(zip(residuals, panels[1:]), start=1):
        ax = axes[1, c]
        cf1 = ax.contourf(
            xs, ys, res, levels=err_lev, cmap="RdBu_r",
            extend="both", vmin=-err_abs, vmax=err_abs,
        )
        ax.contour(xs, ys, res, levels=[0.0], colors="k", linewidths=0.6, alpha=0.45)
        _holes(ax, p.get("boreholes") or [])
        rmse = float(np.sqrt(np.mean(res ** 2)))
        _style_ax(
            ax, ext, f"{method['residualTitle']}\nRMSE {rmse:.1f} MPa",
            p["horizLabel"], p["vertLabel"] if c == 1 else "",
        )
    if cf0 is not None:
        cbar0 = fig.colorbar(cf0, ax=axes[0, :].tolist(), shrink=0.9, pad=0.02)
        cbar0.set_label("UCS (MPa)")
    if cf1 is not None:
        cbar1 = fig.colorbar(cf1, ax=axes[1, 1:].tolist(), shrink=0.9, pad=0.02)
        cbar1.set_label("预测 − 真值 (MPa)")
    if p.get("title"):
        fig.suptitle(p["title"], fontsize=13)
    return _save_fixed(fig)


def render_profile(payload) -> str:
    p = _payload(payload)
    wells = p["wells"]
    n = max(1, len(wells))
    fig, axes = plt.subplots(
        1, n, figsize=(PROFILE_PANEL_W * n, PROFILE_H),
        constrained_layout=True, sharey=True,
    )
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        w = wells[k]
        z = np.array(w["z"], dtype=float)
        ax.plot(w["ucs_true"], z, "k-", lw=2.0, label="真值 UCS")
        ax.plot(w["ucs_seis"], z, "--", color="C1", lw=1.8, label="波阻抗插值")
        ax.plot(w["ucs_mwd"], z, ":", color="C0", lw=1.8, label="钻孔插值")
        ax.plot(w["ucs_fused"], z, "-", color="C2", lw=2.0, label="融合插值")
        ax.set_xlabel("UCS (MPa)")
        ax.set_title(w["title"])
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        if k == 0:
            ax.set_ylabel("标高 (m)")
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(p["title"], fontsize=13)
    return _save_fixed(fig)


def _cad_3d_axes(ax):
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.72, 0.74, 0.78, 1.0))
        axis.pane.set_alpha(1.0)
        axis.line.set_color((0.45, 0.47, 0.50, 1.0))
    ax.grid(True, color="#d5d8de", linestyle="-", linewidth=0.4)
    ax.view_init(elev=22, azim=-58)


def render_trislices(payload) -> str:
    """3-D orthogonal planes + XY / XZ / YZ, white CAD background."""
    p = _payload(payload)
    scale = float(p.get("scale") or 1.0)
    box = p["box"]  # [x0, x1, y0, y1, z0, z1]
    x0, x1, y0, y1, z0, z1 = [float(v) for v in box]
    cmap = _cmap_from_stops(p)
    xy = _field(p["xy"]["values"], int(p["xy"]["w"]), int(p["xy"]["h"]), scale)
    xz = _field(p["xz"]["values"], int(p["xz"]["w"]), int(p["xz"]["h"]), scale)
    yz = _field(p["yz"]["values"], int(p["yz"]["w"]), int(p["yz"]["h"]), scale)
    xc, yc, zc = float(p["xc"]), float(p["yc"]), float(p["zc"])
    if "vmin" in p:
        vmin = float(p["vmin"]) / scale
    else:
        vmin = float(np.nanmin(xy))
    if "vmax" in p:
        vmax = float(p["vmax"]) / scale
    else:
        vmax = float(np.nanmax(xy))
    if vmax <= vmin:
        vmax = vmin + 1.0
    lev = np.linspace(vmin, vmax, 20)
    gx = np.linspace(x0, x1, xy.shape[1])
    gy = np.linspace(y0, y1, xy.shape[0])
    gz = np.linspace(min(z0, z1), max(z0, z1), xz.shape[0])

    fig = plt.figure(figsize=TRISLICES_FIGSIZE, facecolor="white")
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1.70, 0.20, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        left=0.02, right=0.90, top=0.90, bottom=0.07,
        hspace=0.42, wspace=0.05,
    )
    ax3d = fig.add_subplot(gs[:, 0], projection="3d", facecolor="white")
    ax_xy = fig.add_subplot(gs[0, 2])
    ax_xz = fig.add_subplot(gs[1, 2])
    ax_yz = fig.add_subplot(gs[2, 2])

    norm = Normalize(vmin=vmin, vmax=vmax)
    xx_xy, yy_xy = np.meshgrid(gx, gy)
    xx_xz, zz_xz = np.meshgrid(gx, gz)
    yy_yz, zz_yz = np.meshgrid(gy, gz)
    kw = dict(rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax3d.plot_surface(xx_xy, yy_xy, np.full_like(xx_xy, zc), facecolors=cmap(norm(xy)), **kw)
    ax3d.plot_surface(xx_xz, np.full_like(xx_xz, yc), zz_xz, facecolors=cmap(norm(xz)), **kw)
    ax3d.plot_surface(np.full_like(yy_yz, xc), yy_yz, zz_yz, facecolors=cmap(norm(yz)), **kw)
    holes = p.get("boreholes") or []
    if holes:
        hx, hy = zip(*holes)
        ax3d.scatter(hx, hy, [zc] * len(hx), s=12, c="k", depthshade=False)
    _cad_3d_axes(ax3d)
    ax3d.set_xlim(x0, x1)
    ax3d.set_ylim(y0, y1)
    ax3d.set_zlim(min(z0, z1), max(z0, z1))
    try:
        ax3d.set_box_aspect((1.0, 1.55, 1.25), zoom=1.32)
    except TypeError:
        try:
            ax3d.set_box_aspect((1.0, 1.55, 1.25))
        except Exception:
            pass
        ax3d.dist = 8.0
    ax3d.set_anchor("C")
    ax3d.tick_params(labelsize=8, pad=1)
    ax3d.set_xticks([x0, 0.5 * (x0 + x1), x1])
    ax3d.set_yticks([y0, 0.5 * (y0 + y1), y1])
    ax3d.set_zticks([min(z0, z1), 0.5 * (z0 + z1), max(z0, z1)])
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title(f"三正交切面  X={xc:.0f} m, Y={yc:.0f} m, Z={zc:.0f} m", pad=6)

    ext_xy = p["xy"]["extent"]
    xs_xy = np.linspace(ext_xy[0], ext_xy[1], xy.shape[1])
    ys_xy = np.linspace(ext_xy[2], ext_xy[3], xy.shape[0])
    ax_xy.contourf(xs_xy, ys_xy, xy, levels=lev, cmap=cmap, extend="both")
    ax_xy.contour(xs_xy, ys_xy, xy, levels=lev, colors="k", linewidths=0.3,
                  linestyles="--", alpha=0.4)
    _holes(ax_xy, holes)
    _style_ax(ax_xy, ext_xy, f"XY  标高 {zc:.0f} m", "X (m)", "Y (m)")

    ext_xz = p["xz"]["extent"]
    xs_xz = np.linspace(ext_xz[0], ext_xz[1], xz.shape[1])
    zs_xz = np.linspace(ext_xz[2], ext_xz[3], xz.shape[0])
    ax_xz.contourf(xs_xz, zs_xz, xz, levels=lev, cmap=cmap, extend="both")
    ax_xz.contour(xs_xz, zs_xz, xz, levels=lev, colors="k", linewidths=0.3,
                  linestyles="--", alpha=0.4)
    _style_ax(ax_xz, ext_xz, f"XZ  Y={yc:.0f} m",
              p["xz"].get("horizLabel", "X (m)"), p["xz"].get("vertLabel", "Z (m)"))

    ext_yz = p["yz"]["extent"]
    ys_yz = np.linspace(ext_yz[0], ext_yz[1], yz.shape[1])
    zs_yz = np.linspace(ext_yz[2], ext_yz[3], yz.shape[0])
    ax_yz.contourf(ys_yz, zs_yz, yz, levels=lev, cmap=cmap, extend="both")
    ax_yz.contour(ys_yz, zs_yz, yz, levels=lev, colors="k", linewidths=0.3,
                  linestyles="--", alpha=0.4)
    _style_ax(ax_yz, ext_yz, f"YZ  X={xc:.0f} m",
              p["yz"].get("horizLabel", "Y (m)"), p["yz"].get("vertLabel", "Z (m)"))

    ax_xy.locator_params(nbins=4)
    ax_xz.locator_params(nbins=4)
    ax_yz.locator_params(nbins=4)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_xy, ax_xz, ax_yz], shrink=0.82, pad=0.04)
    cbar.set_label(p.get("unit") or "UCS (MPa)")
    if p.get("title"):
        fig.suptitle(p["title"], fontsize=13)
    return _save_fixed(fig)
