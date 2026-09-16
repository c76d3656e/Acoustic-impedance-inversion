"""Multi-panel horizontal-slice figures for the fusion workflow."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _slice_extent(gx, gy):
    return [float(gx[0]), float(gx[-1]), float(gy[0]), float(gy[-1])]


def plot_field_slice(field, gx, gy, z_index, outfile, title, cbar_label, cmap="viridis"):
    """Save a single horizontal slice of a 3-D field."""
    data = np.asarray(field)[:, :, z_index].T  # (ny, nx) for imshow
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, origin="lower", extent=_slice_extent(gx, gy), cmap=cmap, aspect="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    return outfile


def plot_fusion_panels(panels, gx, gy, z_index, elevation, outfile):
    """5-panel comparison at one elevation.

    ``panels`` is an ordered list of ``(volume, title, cbar_label, cmap)``.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    extent = _slice_extent(gx, gy)
    for ax, (vol, title, cbar_label, cmap) in zip(axes, panels):
        data = np.asarray(vol)[:, :, z_index].T
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(cbar_label)
    fig.suptitle(f"Elevation {elevation:.0f} m", fontsize=14)
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    return outfile
