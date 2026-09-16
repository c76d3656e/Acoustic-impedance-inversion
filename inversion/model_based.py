"""Model-based (Tikhonov-regularised) acoustic impedance inversion.

Seismic data is band-limited, so inversion is stabilised with

* a smooth low-frequency background model ``m0`` (from wells / horizons), and
* a smoothness (second-difference) regulariser.

For a single trace we solve

    min_m ||G m - d||^2 + lam * ||L (m - m0)||^2 + eps * ||m - m0||^2

which has the closed-form normal-equations solution

    m = m0 + (G^T G + lam L^T L + eps I)^{-1} G^T (d - G m0).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .forward import forward_operator


def _second_difference(n: int) -> np.ndarray:
    """Interior second-difference operator (zero boundary rows)."""
    l = np.zeros((n, n), dtype=float)
    idx = np.arange(1, n - 1)
    l[idx, idx - 1] = 1.0
    l[idx, idx] = -2.0
    l[idx, idx + 1] = 1.0
    return l


def background_model(
    impedance: np.ndarray, sigma=8.0
) -> np.ndarray:
    """Smooth low-frequency background impedance.

    Simulates the low-frequency trend that in practice comes from well logs
    interpolated along interpreted horizons.  ``sigma`` is the Gaussian
    smoothing length (samples); a scalar is broadcast across all axes.
    """
    z = np.asarray(impedance, dtype=float)
    return np.exp(gaussian_filter(np.log(z), sigma=sigma))


def _inverse_operator(wavelet, n, lam, eps):
    g = forward_operator(wavelet, n)
    l = _second_difference(n)
    a = g.T @ g + lam * (l.T @ l) + eps * np.eye(n)
    # B maps a residual seismic trace to a log-impedance perturbation.
    b = np.linalg.solve(a, g.T)
    return g, b


def invert_trace(
    seismic: np.ndarray,
    wavelet: np.ndarray,
    background: np.ndarray,
    lam: float = 5.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """Invert a single seismic trace to acoustic impedance."""
    d = np.asarray(seismic, dtype=float)
    if d.ndim != 1:
        raise ValueError("seismic must be 1-D")
    n = d.size
    m0 = np.log(np.asarray(background, dtype=float))
    if m0.shape != d.shape:
        raise ValueError("background must match seismic length")
    g, b = _inverse_operator(wavelet, n, lam, eps)
    dm = b @ (d - g @ m0)
    return np.exp(m0 + dm)


def invert_volume(
    seismic_volume: np.ndarray,
    wavelet: np.ndarray,
    background_volume: np.ndarray,
    lam: float = 5.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """Trace-by-trace inversion of a 3-D seismic volume.

    The system matrix depends only on the (shared) wavelet and trace length, so
    it is factored once and applied to every trace.
    """
    d = np.asarray(seismic_volume, dtype=float)
    if d.ndim != 3:
        raise ValueError("seismic_volume must be 3-D (nx, ny, nz)")
    if background_volume.shape != d.shape:
        raise ValueError("background_volume must match seismic_volume shape")

    nx, ny, nz = d.shape
    traces = d.reshape(-1, nz)
    m0 = np.log(background_volume.reshape(-1, nz))

    g, b = _inverse_operator(wavelet, nz, lam, eps)
    residual = traces - m0 @ g.T
    dm = residual @ b.T
    m = m0 + dm
    return np.exp(m).reshape(nx, ny, nz)
