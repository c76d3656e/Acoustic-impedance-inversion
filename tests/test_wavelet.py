import numpy as np

from inversion import ricker


def test_ricker_peak_centered_and_normalised():
    w = ricker(n=101, dt=0.002, freq=30.0)
    assert w.shape == (101,)
    assert np.argmax(w) == 50
    assert np.isclose(w.max(), 1.0)


def test_ricker_symmetric_and_zero_mean():
    w = ricker(n=101, dt=0.002, freq=25.0)
    assert np.allclose(w, w[::-1], atol=1e-12)
    # A Ricker wavelet integrates to ~0.
    assert abs(w.sum()) < 0.5
