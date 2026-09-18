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
