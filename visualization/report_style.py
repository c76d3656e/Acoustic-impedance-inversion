"""Report-style horizontal slice maps.

Reproduces the look of the mine report figures: filled ``viridis`` contours,
dashed contour lines, drill-hole "bullseye" markers, and a scientific colorbar
(values shown in units of ``10**scale``).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .fonts import configure_cjk_font


def _contour_levels(data: np.ndarray, levels: int, vmin, vmax):
    lo = float(np.nanmin(data) if vmin is None else vmin)
    hi = float(np.nanmax(data) if vmax is None else vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, int(levels))


def _draw_holes(ax, holes_xy, s_outer: float = 120, s_inner: float = 14):
    if holes_xy is None or len(holes_xy) == 0:
        return
    hx = np.asarray(holes_xy)[:, 0]
    hy = np.asarray(holes_xy)[:, 1]
    ax.scatter(hx, hy, s=s_outer, facecolors="none", edgecolors="k", linewidths=1.3)
    ax.scatter(hx, hy, s=s_inner, c="k")


def _style_slice_ax(ax, gx, gy, title):
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_xlim(gx[0], gx[-1])
    ax.set_ylim(gy[0], gy[-1])
    ax.set_aspect("auto")


def plot_report_slice(
    field: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    z_index: int,
    outfile: str,
    title: str,
    cbar_label: str = "",
    holes_xy: np.ndarray | None = None,
    cmap: str = "viridis",
    levels: int = 30,
    scale: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Save a report-style horizontal slice.

    Parameters
    ----------
    field:
        Volume ``(nx, ny, nz)``.
    z_index:
        Depth/elevation sample to slice.
    holes_xy:
        Optional ``(n, 2)`` array of drill-hole ``(x, y)`` coordinates drawn as
        bullseye markers.
    scale:
        Divide values by this before plotting (e.g. ``1e6`` to show ``x10^6``).
    vmin, vmax:
        Shared colour limits in the *original* field units.  ``None`` uses the
        slice min/max.  Both are divided by ``scale`` before plotting.
    """
    configure_cjk_font()
    vol = np.asarray(field, dtype=float)
    if vol.ndim != 3:
        raise ValueError("field must be 3-D (nx, ny, nz)")
    data = vol[:, :, z_index] / scale
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    vmin_s = None if vmin is None else float(vmin) / scale
    vmax_s = None if vmax is None else float(vmax) / scale
    lev = _contour_levels(data, levels, vmin_s, vmax_s)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cf = ax.contourf(xx, yy, data, levels=lev, cmap=cmap, extend="both")
    ax.contour(xx, yy, data, levels=lev, colors="k", linewidths=0.4,
               linestyles="--", alpha=0.5)
    _draw_holes(ax, holes_xy)
    cbar = fig.colorbar(cf, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    _style_slice_ax(ax, gx, gy, title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_slice_grid(
    panels,
    gx: np.ndarray,
    gy: np.ndarray,
    z_index: int,
    outfile: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    ncols: int = 4,
    levels: int = 20,
    scale: float = 1.0,
    suptitle: str = "",
):
    """Shared-scale report-style grid of horizontal slices.

    ``panels`` is a list of ``(field, title, holes_xy)``.  Empty trailing axes
    are hidden.  One colourbar is shared so panels are visually comparable.
    """
    configure_cjk_font()
    n = len(panels)
    if n == 0:
        raise ValueError("panels must be non-empty")
    ncols = max(1, min(int(ncols), n))
    nrows = int(np.ceil(n / ncols))
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    lev = np.linspace(float(vmin) / scale, float(vmax) / scale, int(levels))

    fig_w = 3.4 * ncols + 0.8
    fig_h = 2.9 * nrows + (0.55 if suptitle else 0.15)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    cf = None
    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.set_visible(False)
            continue
        field, title, holes_xy = panels[i]
        data = np.asarray(field, dtype=float)[:, :, z_index] / scale
        cf = ax.contourf(xx, yy, data, levels=lev, cmap=cmap, extend="both")
        ax.contour(xx, yy, data, levels=lev, colors="k", linewidths=0.3,
                   linestyles="--", alpha=0.45)
        _draw_holes(ax, holes_xy, s_outer=55, s_inner=8)
        _style_slice_ax(ax, gx, gy, title)
        if i % ncols != 0:
            ax.set_ylabel("")
        if i < n - ncols:
            ax.set_xlabel("")
    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes_flat[:n].tolist(), shrink=0.85, pad=0.02)
        cbar.set_label(cbar_label)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_field_residual_grid(
    true_field: np.ndarray,
    methods: list,
    gx: np.ndarray,
    gy: np.ndarray,
    z_index: int,
    outfile: str,
    vmin: float,
    vmax: float,
    holes_xy: np.ndarray | None = None,
    true_title: str = "(a) 强度真值",
    field_cbar: str = "UCS (MPa)",
    residual_cbar: str = "预测 − 真值 (MPa)",
    suptitle: str = "",
    levels: int = 20,
):
    """Two-row comparison: fields on top, ``pred − truth`` heatmaps below.

    ``methods`` is ``[(field, field_title, residual_title), ...]``.  The residual
    row shares one diverging scale so a paler map means a smaller gap.
    """
    configure_cjk_font()
    n_m = len(methods)
    if n_m < 1:
        raise ValueError("methods must be non-empty")
    n_cols = n_m + 1
    true = np.asarray(true_field, dtype=float)
    fields = [np.asarray(m[0], dtype=float) for m in methods]
    residuals = [fld - true for fld in fields]
    slice_abs = np.concatenate(
        [np.abs(r[:, :, z_index]).ravel() for r in residuals]
    )
    err_abs = float(np.percentile(slice_abs, 98))
    if not np.isfinite(err_abs) or err_abs < 1e-6:
        err_abs = 1.0

    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    lev = np.linspace(float(vmin), float(vmax), int(levels))
    err_lev = np.linspace(-err_abs, err_abs, int(levels) + 1)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(3.4 * n_cols + 1.2, 6.6),
        constrained_layout=True,
    )
    titles_top = [true_title] + [m[1] for m in methods]
    vols_top = [true] + fields
    cf0 = None
    for c, (vol, title) in enumerate(zip(vols_top, titles_top)):
        ax = axes[0, c]
        data = vol[:, :, z_index]
        cf0 = ax.contourf(xx, yy, data, levels=lev, cmap="viridis", extend="both")
        ax.contour(xx, yy, data, levels=lev, colors="k", linewidths=0.3,
                   linestyles="--", alpha=0.4)
        _draw_holes(ax, holes_xy, s_outer=55, s_inner=8)
        _style_slice_ax(ax, gx, gy, title)
        if c != 0:
            ax.set_ylabel("")
        ax.set_xlabel("")

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5, 0.55,
        "下行：预测 − 真值\n红＝估计偏高\n蓝＝估计偏低\n越浅越好",
        transform=axes[1, 0].transAxes, ha="center", va="center",
        fontsize=11, linespacing=1.6,
    )
    cf1 = None
    for c, (res, method) in enumerate(zip(residuals, methods), start=1):
        ax = axes[1, c]
        data = res[:, :, z_index]
        cf1 = ax.contourf(
            xx, yy, data, levels=err_lev, cmap="RdBu_r",
            extend="both", vmin=-err_abs, vmax=err_abs,
        )
        ax.contour(xx, yy, data, levels=[0.0], colors="k", linewidths=0.6, alpha=0.45)
        _draw_holes(ax, holes_xy, s_outer=55, s_inner=8)
        rmse = float(np.sqrt(np.mean(data ** 2)))
        _style_slice_ax(ax, gx, gy, f"{method[2]}\nRMSE {rmse:.1f} MPa")
        if c != 1:
            ax.set_ylabel("")
    if cf0 is not None:
        cbar0 = fig.colorbar(cf0, ax=axes[0, :].tolist(), shrink=0.9, pad=0.02)
        cbar0.set_label(field_cbar)
    if cf1 is not None:
        cbar1 = fig.colorbar(cf1, ax=axes[1, 1:].tolist(), shrink=0.9, pad=0.02)
        cbar1.set_label(residual_cbar)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


TRISLICES_FIGSIZE = (12.0, 7.6)


def _cad_3d_axes(ax):
    """White CAD/SolidWorks-like 3-D panes for screenshot-friendly figures."""
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.72, 0.74, 0.78, 1.0))
        axis.pane.set_alpha(1.0)
        axis.line.set_color((0.45, 0.47, 0.50, 1.0))
    ax.grid(True, color="#d5d8de", linestyle="-", linewidth=0.4)
    ax.tick_params(colors="#333333")
    ax.xaxis.label.set_color("#222222")
    ax.yaxis.label.set_color("#222222")
    ax.zaxis.label.set_color("#222222")
    ax.view_init(elev=22, azim=-58)


def plot_orthogonal_trislices(
    field: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    ix: int,
    iy: int,
    iz: int,
    outfile: str,
    title: str = "",
    cbar_label: str = "UCS (MPa)",
    holes_xy: np.ndarray | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    scale: float = 1.0,
    levels: int = 20,
):
    """Publication figure: 3-D orthogonal planes plus XY / XZ / YZ slices.

    White background (AutoCAD / SolidWorks / matplotlib light style).  The
    three cutting planes share one colour scale with the 2-D panels.
    """
    configure_cjk_font()
    vol = np.asarray(field, dtype=float) / float(scale)
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    gz = np.asarray(gz, dtype=float)
    nx, ny, nz = vol.shape
    ix = int(np.clip(ix, 0, nx - 1))
    iy = int(np.clip(iy, 0, ny - 1))
    iz = int(np.clip(iz, 0, nz - 1))
    lo = float(np.nanmin(vol) if vmin is None else vmin / scale)
    hi = float(np.nanmax(vol) if vmax is None else vmax / scale)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    lev = np.linspace(lo, hi, int(levels))
    xc, yc, zc = float(gx[ix]), float(gy[iy]), float(gz[iz])

    xy = vol[:, :, iz].T  # (ny, nx)
    xz = vol[:, iy, :].T  # (nz, nx)
    yz = vol[ix, :, :].T  # (nz, ny)

    fig = plt.figure(figsize=TRISLICES_FIGSIZE, facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.28)
    ax3d = fig.add_subplot(gs[0, :], projection="3d", facecolor="white")
    ax_xy = fig.add_subplot(gs[1, 0])
    ax_xz = fig.add_subplot(gs[1, 1])
    ax_yz = fig.add_subplot(gs[1, 2])

    from matplotlib import cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=lo, vmax=hi)
    cmap_obj = plt.get_cmap(cmap)
    xx_xy, yy_xy = np.meshgrid(gx, gy)
    zz_xy = np.full_like(xx_xy, zc)
    xx_xz, zz_xz = np.meshgrid(gx, gz)
    yy_xz = np.full_like(xx_xz, yc)
    yy_yz, zz_yz = np.meshgrid(gy, gz)
    xx_yz = np.full_like(yy_yz, xc)
    kw = dict(rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax3d.plot_surface(xx_xy, yy_xy, zz_xy, facecolors=cmap_obj(norm(xy)), **kw)
    ax3d.plot_surface(xx_xz, yy_xz, zz_xz, facecolors=cmap_obj(norm(xz)), **kw)
    ax3d.plot_surface(xx_yz, yy_yz, zz_yz, facecolors=cmap_obj(norm(yz)), **kw)
    cf3 = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cf3.set_array([])
    if holes_xy is not None and len(holes_xy):
        hx = np.asarray(holes_xy)[:, 0]
        hy = np.asarray(holes_xy)[:, 1]
        ax3d.scatter(hx, hy, np.full_like(hx, zc), s=12, c="k", depthshade=False)
    _cad_3d_axes(ax3d)
    ax3d.set_xlim(float(gx[0]), float(gx[-1]))
    ax3d.set_ylim(float(gy[0]), float(gy[-1]))
    z0, z1 = float(gz[0]), float(gz[-1])
    ax3d.set_zlim(min(z0, z1), max(z0, z1))
    try:
        # Mild aspect so a 20×50 m face still fills the 3-D panel.
        ax3d.set_box_aspect((1.0, 1.6, 1.15))
    except Exception:
        pass
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title(
        f"三正交切面  X={xc:.0f} m, Y={yc:.0f} m, Z={zc:.0f} m",
        pad=8,
    )

    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    ax_xy.contourf(xx, yy, vol[:, :, iz], levels=lev, cmap=cmap, extend="both")
    ax_xy.contour(xx, yy, vol[:, :, iz], levels=lev, colors="k",
                  linewidths=0.3, linestyles="--", alpha=0.4)
    _draw_holes(ax_xy, holes_xy, s_outer=55, s_inner=8)
    ax_xy.axhline(yc, color="w", lw=0.8, alpha=0.85)
    ax_xy.axvline(xc, color="w", lw=0.8, alpha=0.85)
    _style_slice_ax(ax_xy, gx, gy, f"XY  标高 {zc:.0f} m")

    ax_xz.contourf(gx, gz, xz, levels=lev, cmap=cmap, extend="both")
    ax_xz.contour(gx, gz, xz, levels=lev, colors="k", linewidths=0.3,
                  linestyles="--", alpha=0.4)
    ax_xz.axhline(zc, color="w", lw=0.8, alpha=0.85)
    ax_xz.axvline(xc, color="w", lw=0.8, alpha=0.85)
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title(f"XZ  Y={yc:.0f} m")
    ax_xz.set_xlim(gx[0], gx[-1])
    ax_xz.set_ylim(min(z0, z1), max(z0, z1))
    ax_xz.set_aspect("auto")

    ax_yz.contourf(gy, gz, yz, levels=lev, cmap=cmap, extend="both")
    ax_yz.contour(gy, gz, yz, levels=lev, colors="k", linewidths=0.3,
                  linestyles="--", alpha=0.4)
    ax_yz.axhline(zc, color="w", lw=0.8, alpha=0.85)
    ax_yz.axvline(yc, color="w", lw=0.8, alpha=0.85)
    ax_yz.set_xlabel("Y (m)")
    ax_yz.set_ylabel("Z (m)")
    ax_yz.set_title(f"YZ  X={xc:.0f} m")
    ax_yz.set_xlim(gy[0], gy[-1])
    ax_yz.set_ylim(min(z0, z1), max(z0, z1))
    ax_yz.set_aspect("auto")

    cbar = fig.colorbar(cf3, ax=[ax3d, ax_xy, ax_xz, ax_yz], shrink=0.55, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.savefig(outfile, dpi=150, facecolor="white")
    plt.close(fig)
    return outfile


def plot_spherical_variogram(
    outfile: str,
    range_m: float = 16.5,
    sill: float = 1.0,
    nugget: float = 0.05,
    title: str = "球状变差函数",
):
    """Textbook spherical variogram used by the ordinary-kriging residual model."""
    configure_cjk_font()
    h = np.linspace(0.0, 2.4 * range_m, 400)
    a = float(range_m)
    c0, c1 = float(nugget), float(sill) - float(nugget)
    gamma = np.empty_like(h)
    inside = h < a
    t = h[inside] / a
    gamma[inside] = c0 + c1 * (1.5 * t - 0.5 * t ** 3)
    gamma[~inside] = c0 + c1
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(h, gamma, color="#1f4e79", lw=2.0, label=r"$\gamma(h)$")
    ax.axhline(sill, color="#888", ls="--", lw=1.0, label=f"sill C0+C={sill:g}")
    ax.axvline(a, color="#c45c26", ls="--", lw=1.0, label=rf"range $a={a:g}\,\mathrm{{m}}$")
    ax.scatter([0], [c0], s=28, c="#333", zorder=5, label=f"nugget C0={c0:g}")
    ax.set_xlabel(r"$h$ (m)")
    ax.set_ylabel(r"$\gamma(h)$")
    ax.set_title(title)
    ax.set_xlim(0, h[-1])
    ax.set_ylim(0, 1.15 * sill)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, facecolor="white")
    plt.close(fig)
    return outfile


def plot_working_face(
    field: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    z_index: int,
    outfile: str,
    x0: float,
    y0: float,
    width: float = 20.0,
    height: float = 50.0,
    title: str = "工作面窗口（20×50 m）",
    holes_xy: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Full-block slice with the 20×50 m working-face window drawn on top."""
    configure_cjk_font()
    from matplotlib.patches import Rectangle

    vol = np.asarray(field, dtype=float)
    data = vol[:, :, z_index]
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    lo = float(np.nanmin(data) if vmin is None else vmin)
    hi = float(np.nanmax(data) if vmax is None else vmax)
    lev = np.linspace(lo, hi, 20)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cf = ax.contourf(xx, yy, data, levels=lev, cmap="viridis", extend="both")
    ax.contour(xx, yy, data, levels=lev, colors="k", linewidths=0.3,
               linestyles="--", alpha=0.4)
    _draw_holes(ax, holes_xy, s_outer=70, s_inner=9)
    rect = Rectangle(
        (x0, y0), width, height, fill=False, edgecolor="#d62728",
        linewidth=2.0, linestyle="-", zorder=7,
    )
    ax.add_patch(rect)
    ax.text(
        x0 + 0.6, y0 + height - 1.6,
        f"{width:.0f}×{height:.0f} m 工作面",
        color="#d62728", fontsize=11, va="top",
    )
    _style_slice_ax(ax, gx, gy, title)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("UCS (MPa)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, facecolor="white")
    plt.close(fig)
    return outfile
