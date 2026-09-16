import numpy as np

from inversion import (
    ricker,
    difference_operator,
    wavelet_matrix,
    forward_operator,
    synthetic_seismic,
    synthetic_seismic_volume,
)


def test_operator_shapes_and_composition():
    n = 50
    w = ricker(21, 0.002, 30.0)
    d = difference_operator(n)
    wm = wavelet_matrix(w, n)
    g = forward_operator(w, n)
    assert d.shape == (n, n)
    assert wm.shape == (n, n)
    assert np.allclose(g, wm @ (0.5 * d))


def test_synthetic_seismic_shapes():
    z = np.linspace(2.0e6, 5.0e6, 60)
    w = ricker(21, 0.002, 30.0)
    s = synthetic_seismic(z, w)
    assert s.shape == (60,)

    vol = np.stack([np.stack([z] * 4, axis=0)] * 3, axis=0)  # (3,4,60)
    sv = synthetic_seismic_volume(vol, w)
    assert sv.shape == (3, 4, 60)
