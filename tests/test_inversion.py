import numpy as np

from datasets import layered_property_model, acoustic_impedance
from inversion import (
    ricker,
    synthetic_seismic_volume,
    invert_volume,
    background_model,
    sparse_spike_inversion,
    synthetic_seismic,
)
from validation import r2_score


def test_volume_inversion_recovers_impedance_noise_free():
    vp, rho = layered_property_model(shape=(8, 8, 160), seed=1)
    ai_true = acoustic_impedance(vp, rho)
    w = ricker(81, 0.002, 30.0)
    seismic = synthetic_seismic_volume(ai_true, w)
    background = background_model(ai_true, sigma=(2, 2, 14))
    ai_inv = invert_volume(seismic, w, background, lam=5.0)
    assert r2_score(ai_true, ai_inv) > 0.9


def test_sparse_spike_runs_and_is_finite():
    vp, rho = layered_property_model(shape=(1, 1, 160), seed=2)
    ai = acoustic_impedance(vp, rho)[0, 0]
    w = ricker(81, 0.002, 30.0)
    seismic = synthetic_seismic(ai, w)
    bg = background_model(ai.reshape(1, 1, -1), sigma=(0, 0, 14))[0, 0]
    inv = sparse_spike_inversion(seismic, w, background=bg, n_iter=100)
    assert inv.shape == ai.shape
    assert np.all(np.isfinite(inv))
