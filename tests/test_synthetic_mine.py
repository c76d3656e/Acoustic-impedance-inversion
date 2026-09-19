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
    from collections import defaultdict

    from datasets import (
        FACE_HEIGHT, FACE_WIDTH, centered_face,
        plum_blossom_hole_indices, unique_hole_xy_indices, subset_holes,
    )

    ds = generate_mine(n_holes=14, seed=42)
    pairs = unique_hole_xy_indices(ds)
    assert len(pairs) == 14
    assert ds.meta.get("hole_layout") == "plum_blossom"
    assert ds.meta.get("hole_window") == (FACE_WIDTH, FACE_HEIGHT)
    xs, ys = ds.gx[pairs[:, 0]], ds.gy[pairs[:, 1]]
    x0, x1, y0, y1 = centered_face(ds.gx, ds.gy)
    fw, fh = x1 - x0, y1 - y0
    # All 14 wells live inside the default 20×50 m working face.
    assert xs.min() >= x0 - 1e-6 and xs.max() <= x1 + 1e-6
    assert ys.min() >= y0 - 1e-6 and ys.max() <= y1 + 1e-6
    assert xs.max() - xs.min() > 0.55 * fw
    assert ys.max() - ys.min() > 0.55 * fh
    # Interior of the face: leave a strip empty (not parked on the window edge).
    assert xs.min() > x0 + 0.06 * fw and xs.max() < x1 - 0.06 * fw
    assert ys.min() > y0 + 0.06 * fh and ys.max() < y1 - 0.06 * fh
    dmin = np.inf
    for a in range(14):
        for b in range(a + 1, 14):
            dmin = min(dmin, np.hypot(xs[a] - xs[b], ys[a] - ys[b]))
    assert dmin > 5.0  # not piled in one corner of the face
    # Staggered 梅花: odd rows sit at extra Y stations, not a 3×4 rectangle.
    assert len(np.unique(np.round(xs, 5))) >= 3
    assert len(np.unique(np.round(ys, 5))) >= 5
    # Centre X-station is filled (not left with only the two mid-face wells).
    cols = defaultdict(list)
    for x, y in zip(xs, ys):
        cols[round(float(x), 1)].append(float(y))
    mid_x = sorted(cols)[len(cols) // 2]
    assert len(cols[mid_x]) >= 4

    lattice = plum_blossom_hole_indices(ds.gx, ds.gy, 14, bbox=(x0, x1, y0, y1))
    assert len(lattice) == 14
    assert len(set(lattice)) == 14
    assert set(lattice) == set(map(tuple, pairs.tolist()))

    # Nested prefixes stay uniformly spread across the face (farthest-point order).
    sub = subset_holes(ds, 4)
    sp = unique_hole_xy_indices(sub)
    sxs, sys = ds.gx[sp[:, 0]], ds.gy[sp[:, 1]]
    assert sxs.max() - sxs.min() > 0.40 * fw
    assert sys.max() - sys.min() > 0.40 * fh
