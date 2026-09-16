import numpy as np

from inversion import reflectivity_from_impedance, impedance_from_reflectivity


def test_reflectivity_roundtrip():
    z = np.array([2.0e6, 3.0e6, 3.0e6, 5.0e6, 4.0e6])
    r = reflectivity_from_impedance(z)
    z_rec = impedance_from_reflectivity(r, z0=z[0])
    assert np.allclose(z_rec, z)


def test_reflectivity_sign():
    z = np.array([2.0e6, 4.0e6, 2.0e6])
    r = reflectivity_from_impedance(z)
    assert r[0] > 0  # impedance increase -> positive reflection
    assert r[1] < 0  # impedance decrease -> negative reflection
