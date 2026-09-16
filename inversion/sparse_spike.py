"""Sparse-spike inversion baseline.

Estimates a sparse reflectivity series by L1-regularised deconvolution
(iterative soft-thresholding, ISTA):

    min_r 0.5 * ||W r - d||^2 + mu * ||r||_1

The reflectivity is then integrated to log-impedance and merged with a smooth
background so that the (missing) low-frequency content is restored.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .forward import wavelet_matrix


def _soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def sparse_spike_inversion(
    seismic: np.ndarray,
    wavelet: np.ndarray,
    background: np.ndarray | None = None,
    mu: float = 0.05,
    n_iter: int = 400,
    lowcut_sigma: float = 12.0,
):
    """Sparse-spike inversion of one trace.

    Returns acoustic impedance when ``background`` is supplied, otherwise the
    estimated sparse reflectivity series.
    """
    d = np.asarray(seismic, dtype=float)
    if d.ndim != 1:
        raise ValueError("seismic must be 1-D")
    n = d.size
    w = wavelet_matrix(wavelet, n)

    step = 1.0 / (np.linalg.norm(w, 2) ** 2)
    r = np.zeros(n, dtype=float)
    for _ in range(n_iter):
        grad = w.T @ (w @ r - d)
        r = _soft_threshold(r - step * grad, mu * step)

    if background is None:
        return r

    m0 = np.log(np.asarray(background, dtype=float))
    m_hf = 2.0 * np.cumsum(r)
    detail = m_hf - gaussian_filter(m_hf, sigma=lowcut_sigma)
    return np.exp(m0 + detail)
