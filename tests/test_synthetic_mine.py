import numpy as np

from datasets import generate_mine


def test_generate_mine_shapes_and_ranges():
    ds = generate_mine(shape=(12, 10, 24), n_holes=6, seed=0)
    assert ds.ucs_true.shape == (12, 10, 24)
    assert ds.ai_true.shape == (12, 10, 24)
    assert ds.ucs_true.min() >= 20.0 - 1e-6
    assert ds.ucs_true.max() <= 170.0 + 1e-6
    # Hole sample arrays are consistent.
    n = ds.ucs_at_holes.size
    assert ds.V.shape == ds.N.shape == ds.M.shape == ds.F.shape == (n,)
    assert ds.hole_xyz.shape == (n, 3)


def test_ai_and_ucs_are_correlated():
    ds = generate_mine(shape=(20, 16, 32), n_holes=8, seed=3)
    corr = np.corrcoef(ds.ai_true.ravel(), ds.ucs_true.ravel())[0, 1]
    assert corr > 0.5  # shared latent competence field


def test_mwd_monotonic_with_strength():
    ds = generate_mine(shape=(16, 12, 24), n_holes=8, seed=1)
    # Harder rock -> lower penetration rate on average.
    hi = ds.ucs_at_holes > np.median(ds.ucs_at_holes)
    assert ds.V[hi].mean() < ds.V[~hi].mean()
    assert ds.F[hi].mean() > ds.F[~hi].mean()
