import numpy as np

from geostats import ordinary_kriging_3d, regression_kriging_3d, make_grid


def _points():
    pts = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [2, 2, 2], [1, 1, 1]],
        dtype=float,
    )
    vals = np.array([10.0, 12.0, 14.0, 16.0, 22.0, 15.0])
    return pts, vals


def test_ok_shapes_and_variance_nonneg():
    pts, vals = _points()
    gx, gy, gz = make_grid(0, 2, 3, 0, 2, 3, 0, 2, 3)
    mean, var = ordinary_kriging_3d(pts, vals, gx, gy, gz, nlags=3)
    assert mean.shape == (3, 3, 3)
    assert var.shape == (3, 3, 3)
    assert np.all(var > -1e-6)


def test_ok_interpolates_at_data_node():
    pts, vals = _points()
    gx, gy, gz = make_grid(0, 2, 3, 0, 2, 3, 0, 2, 3)
    mean, var = ordinary_kriging_3d(pts, vals, gx, gy, gz, nlags=3)
    # Node (1,1,1) coincides with the last data point (value 15).
    assert abs(mean[1, 1, 1] - 15.0) < 2.0


def test_ok_single_vertical_hole_finite():
    # Collinear samples: auto variogram fit has no lateral lags.
    z = np.linspace(0, 10, 8)
    pts = np.column_stack([np.zeros(8), np.zeros(8), z])
    vals = 10.0 + 0.5 * z
    gx, gy, gz = make_grid(-2, 2, 5, -2, 2, 5, 0, 10, 6)
    mean, var = ordinary_kriging_3d(pts, vals, gx, gy, gz, nlags=4)
    assert mean.shape == (5, 5, 6)
    assert np.all(np.isfinite(mean))
    assert np.all(var > -1e-6)


def test_regression_kriging_shapes():
    pts, vals = _points()
    gx, gy, gz = make_grid(0, 2, 3, 0, 2, 3, 0, 2, 3)
    tp = pts[:, [2]]  # trend on z
    nx, ny, nz = 3, 3, 3
    zz = np.broadcast_to(gz[None, None, :, None], (nx, ny, nz, 1))
    mean, var = regression_kriging_3d(pts, vals, tp, gx, gy, gz, zz, nlags=3)
    assert mean.shape == (3, 3, 3)
    assert var.shape == (3, 3, 3)
