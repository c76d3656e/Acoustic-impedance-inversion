"""Conversions between acoustic impedance and reflectivity."""

from __future__ import annotations

import numpy as np


def reflectivity_from_impedance(impedance: np.ndarray) -> np.ndarray:
    """Normal-incidence reflection coefficients from an impedance log.

    ``R_i = (Z_{i+1} - Z_i) / (Z_{i+1} + Z_i)``

    The result has one fewer sample than the input, describing the interface
    between consecutive layers.
    """
    z = np.asarray(impedance, dtype=float)
    if z.ndim != 1:
        raise ValueError("impedance must be 1-D")
    if np.any(z <= 0):
        raise ValueError("impedance values must be positive")
    return (z[1:] - z[:-1]) / (z[1:] + z[:-1])


def impedance_from_reflectivity(
    reflectivity: np.ndarray, z0: float
) -> np.ndarray:
    """Recursive impedance from reflectivity given a starting impedance ``z0``.

    ``Z_{i+1} = Z_i * (1 + R_i) / (1 - R_i)``
    """
    r = np.asarray(reflectivity, dtype=float)
    if r.ndim != 1:
        raise ValueError("reflectivity must be 1-D")
    if z0 <= 0:
        raise ValueError("z0 must be positive")
    z = np.empty(r.size + 1, dtype=float)
    z[0] = z0
    z[1:] = z0 * np.cumprod((1.0 + r) / (1.0 - r))
    return z
