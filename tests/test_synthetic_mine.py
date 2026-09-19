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
    assert 0.5 < corr < 0.93  # shared competence, but not a rescaled copy


def test_default_block_is_50_by_80():
    ds = generate_mine(n_holes=4, seed=0)
    assert abs(ds.gx[-1] - 50.0) < 1e-6
    assert abs(ds.gy[-1] - 80.0) < 1e-6
    assert abs(ds.gz[-1] + 40.0) < 1e-6
    assert ds.ucs_true.shape == (25, 40, 40)


def test_mwd_monotonic_with_strength():
    ds = generate_mine(shape=(16, 12, 24), n_holes=8, seed=1)
    # Harder rock -> lower penetration rate on average.
    hi = ds.ucs_at_holes > np.median(ds.ucs_at_holes)
    assert ds.V[hi].mean() < ds.V[~hi].mean()
    assert ds.F[hi].mean() > ds.F[~hi].mean()


def test_subset_holes_is_nested_prefix():
    from datasets import subset_holes, unique_hole_xy_indices

    ds = generate_mine(shape=(12, 10, 16), n_holes=5, seed=4)
    pairs = unique_hole_xy_indices(ds)
    assert len(pairs) == 5
    sub = subset_holes(ds, 2)
    kept = unique_hole_xy_indices(sub)
    assert len(kept) == 2
    assert np.array_equal(kept, pairs[:2])
    assert np.array_equal(sub.ucs_true, ds.ucs_true)
    assert sub.ucs_at_holes.size == ds.ucs_at_holes.size * 2 // 5
    try:
        subset_holes(ds, 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_plum_blossom_is_spread_not_clustered():
    from datasets import plum_blossom_hole_indices, unique_hole_xy_indices, subset_holes

    ds = generate_mine(n_holes=12, seed=42)
    pairs = unique_hole_xy_indices(ds)
    assert len(pairs) == 12
    assert ds.meta.get("hole_layout") == "plum_blossom"
    xs, ys = ds.gx[pairs[:, 0]], ds.gy[pairs[:, 1]]
    assert xs.max() - xs.min() > 0.55 * (ds.gx[-1] - ds.gx[0])
    assert ys.max() - ys.min() > 0.55 * (ds.gy[-1] - ds.gy[0])
    Lx = float(ds.gx[-1] - ds.gx[0])
    Ly = float(ds.gy[-1] - ds.gy[0])
    # Interior: leave the free-face strip empty (not parked on the edges).
    assert xs.min() > 0.10 * Lx and xs.max() < 0.90 * Lx
    assert ys.min() > 0.10 * Ly and ys.max() < 0.90 * Ly
    dmin = np.inf
    for a in range(12):
        for b in range(a + 1, 12):
            dmin = min(dmin, np.hypot(xs[a] - xs[b], ys[a] - ys[b]))
    assert dmin > 8.0  # not piled in one corner
    # Staggered 梅花: odd rows sit at extra Y stations, not a 3×4 rectangle.
    assert len(np.unique(np.round(xs, 5))) >= 3
    assert len(np.unique(np.round(ys, 5))) >= 5

    lattice = plum_blossom_hole_indices(ds.gx, ds.gy, 12)
    assert len(lattice) == 12
    assert len(set(lattice)) == 12

    # Nested prefixes stay uniformly spread (farthest-point order).
    sub = subset_holes(ds, 4)
    sp = unique_hole_xy_indices(sub)
    sxs, sys = ds.gx[sp[:, 0]], ds.gy[sp[:, 1]]
    assert sxs.max() - sxs.min() > 0.40 * (ds.gx[-1] - ds.gx[0])
    assert sys.max() - sys.min() > 0.40 * (ds.gy[-1] - ds.gy[0])
