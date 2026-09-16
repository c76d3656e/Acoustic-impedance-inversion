"""Vertical cross-section comparison: true vs inverted vs error."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_cross_section_comparison(
    true_volume: np.ndarray,
    inv_volume: np.ndarray,
    iy: int,
    outfile: str,
):
    """Save a 3-panel figure of an inline cross-section at crossline ``iy``.

    Panels: true impedance, inverted impedance (shared colour scale) and the
    absolute error.
    """
    t = np.asarray(true_volume, dtype=float)
    p = np.asarray(inv_volume, dtype=float)
    if t.shape != p.shape:
        raise ValueError("volumes must have the same shape")
    if t.ndim != 3:
        raise ValueError("volumes must be 3-D (nx, ny, nz)")

    true_sec = t[:, iy, :].T
    inv_sec = p[:, iy, :].T
    err_sec = np.abs(true_sec - inv_sec)

    vmin = float(min(true_sec.min(), inv_sec.min()))
    vmax = float(max(true_sec.max(), inv_sec.max()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, data, name, cmap, lims in (
        (axes[0], true_sec, "True AI", "viridis", (vmin, vmax)),
        (axes[1], inv_sec, "Inverted AI", "viridis", (vmin, vmax)),
        (axes[2], err_sec, "|Error|", "magma", (None, None)),
    ):
        im = ax.imshow(
            data, aspect="auto", origin="upper", cmap=cmap,
            vmin=lims[0], vmax=lims[1],
        )
        ax.set_title(f"{name} (crossline {iy})")
        ax.set_xlabel("Inline (X)")
        ax.set_ylabel("Depth / time sample")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(r"kg / (m$^2\cdot$s)")

    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    return outfile
