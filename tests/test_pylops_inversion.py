import numpy as np
import pytest

from datasets import layered_property_model, acoustic_impedance
from inversion import ricker, synthetic_seismic_volume, background_model, poststack_inversion
from validation import r2_score


@pytest.mark.skipif(poststack_inversion is None, reason="pylops not installed")
def test_pylops_poststack_inversion_recovers_impedance():
    vp, rho = layered_property_model(shape=(6, 6, 120), seed=0)
    ai_true = acoustic_impedance(vp, rho)
    w = ricker(41, 0.002, 30.0)
    seismic = synthetic_seismic_volume(ai_true, w)
    background = background_model(ai_true, sigma=(2, 2, 12))
    ai_inv = poststack_inversion(seismic, w, background, epsR=5.0, epsI=1e-3)
    assert ai_inv.shape == ai_true.shape
    assert r2_score(ai_true, ai_inv) > 0.7
