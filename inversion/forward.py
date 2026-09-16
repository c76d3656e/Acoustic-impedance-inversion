"""Linear forward modelling operators (log-impedance domain).

The convolutional seismic model is

    seismic = W @ r,      r ~= 0.5 * D @ m,      m = ln(impedance)

where ``W`` is the wavelet convolution matrix and ``D`` a first-difference
operator.  Composing them gives the forward operator ``G = W @ (0.5 * D)`` that
maps log-impedance directly to a seismic trace.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import convolution_matrix

from .reflectivity import reflectivity_from_impedance


def difference_operator(n: int) -> np.ndarray:
    """First-difference operator ``D`` of shape ``(n, n)``.

    ``(D m)_i = m_{i+1} - m_i`` for ``i < n - 1``; the final row is zero so the
    operator is square and the output aligns sample-for-sample with the input.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    d = np.zeros((n, n), dtype=float)
    idx = np.arange(n - 1)
    d[idx, idx] = -1.0
    d[idx, idx + 1] = 1.0
    return d


def wavelet_matrix(wavelet: np.ndarray, n: int) -> np.ndarray:
    """Convolution matrix ``W`` (mode ``'same'``) of shape ``(n, n)``."""
    w = np.asarray(wavelet, dtype=float)
    if w.ndim != 1:
        raise ValueError("wavelet must be 1-D")
    if len(w) > n:
        raise ValueError("wavelet longer than the trace it convolves")
    return convolution_matrix(w, n, mode="same")


def forward_operator(wavelet: np.ndarray, n: int) -> np.ndarray:
    """Forward operator ``G = W @ (0.5 * D)`` mapping log-impedance to seismic."""
    return wavelet_matrix(wavelet, n) @ (0.5 * difference_operator(n))


def synthetic_seismic(impedance: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    """Synthetic post-stack trace from an impedance log.

    Uses the *exact* reflectivity formula convolved with the wavelet, so it is a
    faithful (mildly non-linear) forward model rather than the linearised
    operator used during inversion.
    """
    z = np.asarray(impedance, dtype=float)
    if z.ndim != 1:
        raise ValueError("impedance must be 1-D")
    n = z.size
    r = np.zeros(n, dtype=float)
    r[:-1] = reflectivity_from_impedance(z)
    return wavelet_matrix(wavelet, n) @ r


def synthetic_seismic_volume(
    impedance_volume: np.ndarray, wavelet: np.ndarray
) -> np.ndarray:
    """Vectorised :func:`synthetic_seismic` over a 3-D volume.

    The last axis is treated as depth/time; the first two axes are spatial
    (inline, crossline).
    """
    vol = np.asarray(impedance_volume, dtype=float)
    if vol.ndim != 3:
        raise ValueError("impedance_volume must be 3-D (nx, ny, nz)")
    nx, ny, nz = vol.shape
    traces = vol.reshape(-1, nz)
    r = np.zeros_like(traces)
    r[:, :-1] = (traces[:, 1:] - traces[:, :-1]) / (
        traces[:, 1:] + traces[:, :-1]
    )
    w = wavelet_matrix(wavelet, nz)
    seismic = r @ w.T
    return seismic.reshape(nx, ny, nz)
