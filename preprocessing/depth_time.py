"""Depth <-> two-way-time (TWT) conversion for well/seismic tie."""

from __future__ import annotations

import numpy as np


def depth_to_twt(depth: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Two-way time from a velocity profile.

    ``t(z) = 2 * integral_0^z dz / Vp(z)`` (seconds when depth is in metres and
    Vp in m/s).
    """
    depth = np.asarray(depth, dtype=float)
    vp = np.asarray(vp, dtype=float)
    if depth.shape != vp.shape:
        raise ValueError("depth and vp must have the same shape")
    dz = np.gradient(depth)
    owt = np.cumsum(dz / vp)
    return 2.0 * owt


def resample_to_time(
    depth: np.ndarray,
    values: np.ndarray,
    vp: np.ndarray,
    dt: float,
    n_samples: int | None = None,
):
    """Resample a depth-domain log onto a regular TWT grid.

    Returns ``(time_axis, values_in_time)``.
    """
    twt = depth_to_twt(depth, vp)
    if n_samples is None:
        n_samples = int(np.floor(twt[-1] / dt)) + 1
    time_axis = np.arange(n_samples) * dt
    resampled = np.interp(time_axis, twt, values)
    return time_axis, resampled
