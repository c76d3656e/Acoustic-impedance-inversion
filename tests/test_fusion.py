import numpy as np

from fusion import (
    precision_weighted_fusion,
    simple_weighted_fusion,
    doyen_collocated_update,
    ImpedanceStrengthCalibrator,
)


def test_precision_weighting_equal_variance_is_average():
    mu_M = np.array([10.0, 20.0])
    mu_Z = np.array([20.0, 40.0])
    var = np.array([4.0, 4.0])
    mu_f, var_f = precision_weighted_fusion(mu_M, var, mu_Z, var)
    assert np.allclose(mu_f, [15.0, 30.0])
    # Fused variance halves for two equal independent sources.
    assert np.allclose(var_f, [2.0, 2.0])


def test_fused_variance_not_larger_than_inputs():
    rng = np.random.default_rng(0)
    mu_M, mu_Z = rng.random(50), rng.random(50)
    var_M, var_Z = rng.random(50) + 0.1, rng.random(50) + 0.1
    _, var_f = precision_weighted_fusion(mu_M, var_M, mu_Z, var_Z)
    assert np.all(var_f <= var_M + 1e-9)
    assert np.all(var_f <= var_Z + 1e-9)


def test_precision_weighting_prefers_lower_variance():
    mu_f, _ = precision_weighted_fusion(10.0, 0.01, 100.0, 100.0)
    assert abs(mu_f - 10.0) < 1.0  # dominated by the confident source


def test_simple_weighted_fusion():
    assert np.isclose(simple_weighted_fusion(10.0, 20.0, w=0.25), 17.5)


def test_doyen_zero_variance_honors_primary():
    mu_k = np.array([10.0, 30.0])
    var_k = np.array([0.0, 0.0])
    z = np.array([100.0, -50.0])
    mu, var, w = doyen_collocated_update(
        mu_k, var_k, z, rho=0.8,
        mean_primary=20.0, std_primary=10.0,
        mean_secondary=20.0, std_secondary=10.0,
    )
    assert np.allclose(mu, mu_k, atol=1e-9)
    assert np.all(var < 1e-9)
    assert np.all(w > 0.99)


def test_doyen_far_field_is_linear_regression():
    """When kriging variance = sill, the update is ρ toward the secondary."""
    m_y, s_y = 50.0, 10.0
    mu_k = np.array([m_y, m_y])
    var_k = np.array([s_y ** 2, s_y ** 2])
    z = np.array([70.0, 30.0])
    rho = 0.8
    mu, var, w = doyen_collocated_update(
        mu_k, var_k, z, rho=rho,
        mean_primary=m_y, std_primary=s_y,
        mean_secondary=m_y, std_secondary=s_y,
    )
    expected = m_y + rho * (z - m_y)
    assert np.allclose(mu, expected, atol=1e-6)
    assert np.allclose(var, np.full(2, (1.0 - rho ** 2) * s_y ** 2), atol=1e-6)
    assert np.allclose(w, 1.0 - rho ** 2, atol=1e-6)


def test_borehole_anchor_weight_is_one_at_holes_zero_far_away():
    from fusion import borehole_anchor_weight

    gx = np.linspace(0.0, 50.0, 26)
    gy = np.linspace(0.0, 80.0, 41)
    holes = np.array([[25.0, 40.0], [10.0, 20.0]])
    w = borehole_anchor_weight(gx, gy, holes, radius=8.0)
    i = int(np.argmin(np.abs(gx - 25.0)))
    j = int(np.argmin(np.abs(gy - 40.0)))
    assert w[i, j] > 0.95
    # Far from both collars.
    i0 = int(np.argmin(np.abs(gx - 48.0)))
    j0 = int(np.argmin(np.abs(gy - 78.0)))
    assert w[i0, j0] < 0.05


def test_fusion_does_not_cancel_borehole_hard_data():
    """Near holes, fused strength must stay with the MWD branch, not seismic."""
    from datasets import generate_mine
    from fusion import run_fusion_pipeline

    ds = generate_mine(shape=(16, 20, 32), n_holes=6, seed=3)
    res = run_fusion_pipeline(ds, seed=3)
    sm = res.S_M[ds.hole_ix, ds.hole_iy, ds.hole_iz]
    sf = res.S_F[ds.hole_ix, ds.hole_iy, ds.hole_iz]
    sz = res.S_Z[ds.hole_ix, ds.hole_iy, ds.hole_iz]
    # Hole voxels: fusion ≈ MWD, not pulled halfway to impedance.
    assert np.mean(np.abs(sf - sm)) < 0.2 * np.mean(np.abs(sz - sm) + 1e-6)
    assert res.w_anchor.shape == (ds.gx.size, ds.gy.size)
    assert float(res.w_anchor.max()) > 0.9
    assert abs(res.rho) <= 1.0
    # Volume fusion should improve on well-only kriging when several collars
    # constrain the collocated secondary.
    from validation import summary
    assert summary(ds.ucs_true, res.S_F)["RMSE"] <= summary(ds.ucs_true, res.S_M)["RMSE"] + 1e-6


def test_impedance_calibration_monotonic():
    rng = np.random.default_rng(2)
    ai = np.linspace(4e6, 1.1e7, 60)
    ucs = 20 + 1.0e-5 * (ai - 4e6) + 2.0 * rng.standard_normal(60)
    cal = ImpedanceStrengthCalibrator().fit(ai, ucs)
    mu, sigma = cal.predict(np.array([5e6, 9e6]))
    assert mu[1] > mu[0]  # higher impedance -> higher strength
    assert np.all(sigma > 0)


def test_pipeline_accepts_precomputed_impedance_and_one_hole():
    from datasets import generate_mine, subset_holes
    from fusion import run_fusion_pipeline
    from validation import summary

    ds = generate_mine(shape=(10, 8, 32), n_holes=4, seed=5)
    full = run_fusion_pipeline(ds, seed=5)
    reused = run_fusion_pipeline(ds, seed=5, ai_inv=full.ai_inv)
    assert np.allclose(full.S_F, reused.S_F)
    assert np.allclose(full.ai_inv, reused.ai_inv)

    one = subset_holes(ds, 1)
    res1 = run_fusion_pipeline(one, seed=5, ai_inv=full.ai_inv)
    assert res1.S_F.shape == ds.ucs_true.shape
    assert np.all(np.isfinite(res1.S_F))
    mwd = summary(ds.ucs_true, res1.S_M)
    fused = summary(ds.ucs_true, res1.S_F)
    assert fused["RMSE"] <= mwd["RMSE"] + 1e-6
    hole_sz = summary(one.ucs_at_holes, res1.S_Z[one.hole_ix, one.hole_iy, one.hole_iz])
    hole_sf = summary(one.ucs_at_holes, res1.S_F[one.hole_ix, one.hole_iy, one.hole_iz])
    assert hole_sf["RMSE"] <= hole_sz["RMSE"] + 1e-6
