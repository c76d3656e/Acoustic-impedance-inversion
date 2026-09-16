"""Blind-well validation.

A well is held out of the low-frequency model and used only to score the
inversion.  This is far more convincing than a pretty impedance map because the
inversion never "saw" the withheld impedance profile.
"""

from __future__ import annotations

import numpy as np

from .metrics import summary


def blind_well_profile(volume: np.ndarray, ix: int, iy: int) -> np.ndarray:
    """Extract the impedance profile (trace) at inline/crossline ``(ix, iy)``."""
    vol = np.asarray(volume, dtype=float)
    if vol.ndim != 3:
        raise ValueError("volume must be 3-D (nx, ny, nz)")
    return vol[ix, iy, :]


def blind_well_report(true_volume, inv_volume, ix, iy) -> dict:
    """Metrics at a single blind-well location."""
    t = blind_well_profile(true_volume, ix, iy)
    p = blind_well_profile(inv_volume, ix, iy)
    return summary(t, p)
