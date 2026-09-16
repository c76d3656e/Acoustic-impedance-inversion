import numpy as np

from mwd import mwd_features, FEATURE_NAMES, specific_energy, PhysicsGuidedGPR


def test_specific_energy_formula():
    V, N, M, F = 2.0, 80.0, 1000.0, 5000.0
    se = specific_energy(V, N, M, F)
    assert np.isclose(se, F + 2 * np.pi * N * M / V)


def test_features_shape():
    n = 10
    rng = np.random.default_rng(0)
    X = mwd_features(rng.random(n), rng.random(n), rng.random(n), rng.random(n))
    assert X.shape == (n, len(FEATURE_NAMES))


def test_pggpr_predicts_mean_and_std():
    rng = np.random.default_rng(1)
    ucs = rng.uniform(20, 150, size=120)
    ucs_norm = (ucs - 20) / 150
    V = 2.0 * np.exp(-1.2 * ucs_norm) + 0.05 * rng.standard_normal(120)
    N = 80 + 3 * rng.standard_normal(120)
    M = 500 + 1500 * ucs_norm + 100 * rng.standard_normal(120)
    F = 5000 + 20000 * ucs_norm + 1000 * rng.standard_normal(120)
    X = mwd_features(V, N, M, F)

    model = PhysicsGuidedGPR().fit(X[:90], ucs[:90])
    mu, sigma = model.predict(X[90:])
    assert mu.shape == (30,)
    assert sigma.shape == (30,)
    assert np.all(sigma > 0)
    # Should track the true UCS reasonably well.
    ss_res = np.sum((ucs[90:] - mu) ** 2)
    ss_tot = np.sum((ucs[90:] - ucs[90:].mean()) ** 2)
    assert 1 - ss_res / ss_tot > 0.7
