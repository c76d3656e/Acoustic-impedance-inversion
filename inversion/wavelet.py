"""Seismic source wavelets."""

from __future__ import annotations

import numpy as np


def ricker(n: int, dt: float, freq: float) -> np.ndarray:
    """Return a zero-phase Ricker (Mexican-hat) wavelet.

    Parameters
    ----------
    n:
        Number of samples.  An odd value keeps the peak exactly centred.
    dt:
        Sample interval in seconds.
    freq:
        Central (dominant) frequency in Hz.

    Returns
    -------
    numpy.ndarray
        Wavelet amplitudes with the peak normalised to ``1.0``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")

    t = (np.arange(n) - (n - 1) / 2.0) * dt
    arg = (np.pi * freq * t) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)
