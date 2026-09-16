"""Marmousi2 loader (SEG / community mirrors).

Loads P-wave velocity and density models (and optional seismic) from ``.npy``
files.  Accepts a directory containing ``vp.npy`` and ``rho.npy`` (and optionally
``seismic.npy``); community repackagings such as
https://github.com/kk3383/Marmousi- provide impedance/seismic arrays directly.
"""

from __future__ import annotations

import os

import numpy as np

SOURCE = "https://wiki.seg.org/wiki/AGL_Elastic_Marmousi"


def load_marmousi2(directory: str = "data/marmousi2"):
    """Return ``(vp, rho, seismic_or_None)`` numpy arrays from ``directory``."""
    vp_path = os.path.join(directory, "vp.npy")
    rho_path = os.path.join(directory, "rho.npy")
    if not (os.path.exists(vp_path) and os.path.exists(rho_path)):
        raise FileNotFoundError(
            f"Marmousi2 vp.npy/rho.npy not found in {directory}. "
            f"Download from {SOURCE} and convert to .npy."
        )
    vp = np.load(vp_path)
    rho = np.load(rho_path)
    seis_path = os.path.join(directory, "seismic.npy")
    seismic = np.load(seis_path) if os.path.exists(seis_path) else None
    return vp, rho, seismic
