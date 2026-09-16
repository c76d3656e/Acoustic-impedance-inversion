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
    """
    configure_cjk_font()
    vol = np.asarray(field, dtype=float)
    if vol.ndim != 3:
        raise ValueError("field must be 3-D (nx, ny, nz)")
    data = vol[:, :, z_index] / scale
    xx, yy = np.meshgrid(gx, gy, indexing="ij")

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cf = ax.contourf(xx, yy, data, levels=levels, cmap=cmap)
    ax.contour(xx, yy, data, levels=levels, colors="k", linewidths=0.4,
               linestyles="--", alpha=0.5)

    if holes_xy is not None and len(holes_xy):
        hx = np.asarray(holes_xy)[:, 0]
        hy = np.asarray(holes_xy)[:, 1]
        ax.scatter(hx, hy, s=120, facecolors="none", edgecolors="k", linewidths=1.3)
        ax.scatter(hx, hy, s=14, c="k")

    cbar = fig.colorbar(cf, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_xlim(gx[0], gx[-1])
    ax.set_ylim(gy[0], gy[-1])
    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile
