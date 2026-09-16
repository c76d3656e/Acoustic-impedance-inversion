"""Horizontal acoustic-impedance slice maps (the final deliverable)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt
import numpy as np


def plot_impedance_map(
    volume: np.ndarray,
    z_index: int,
    outfile: str,
    title: str | None = None,
    levels: int = 30,
):
    """Save a filled-contour map of a horizontal impedance slice.

    Parameters
    ----------
    volume:
        Impedance volume ``(nx, ny, nz)``.
    z_index:
        Depth/time sample to slice.
    outfile:
        PNG path to write.
    """
    vol = np.asarray(volume, dtype=float)
    if vol.ndim != 3:
        raise ValueError("volume must be 3-D (nx, ny, nz)")
    nx, ny, nz = vol.shape
    if not 0 <= z_index < nz:
        raise ValueError("z_index out of range")

    ai_slice = vol[:, :, z_index]
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(xx, yy, ai_slice, levels=levels, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"Acoustic Impedance  (kg / (m$^2\cdot$s))")
    ax.set_xlabel("Inline (X)")
    ax.set_ylabel("Crossline (Y)")
    ax.set_title(title or f"Acoustic impedance slice @ z-index {z_index}")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    return outfile
