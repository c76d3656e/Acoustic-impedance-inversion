"""3-D kriging wrappers built on PyKrige.

``execute('grid', ...)`` in PyKrige returns arrays indexed ``[z, y, x]``; every
function here transposes back to the project convention ``(nx, ny, nz)``.
"""

from __future__ import annotations

import numpy as np
from pykrige.ok3d import OrdinaryKriging3D
from sklearn.linear_model import LinearRegression


def make_grid(x0, x1, nx, y0, y1, ny, z0, z1, nz):
    """Return regular axis vectors ``(gx, gy, gz)``."""
    gx = np.linspace(x0, x1, nx)
    gy = np.linspace(y0, y1, ny)
    gz = np.linspace(z0, z1, nz)
    return gx, gy, gz


def ordinary_kriging_3d(
    points: np.ndarray,
    values: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    variogram_model: str = "spherical",
    nlags: int = 8,
):
    """Ordinary 3-D kriging.

    Parameters
    ----------
    points:
        ``(n, 3)`` array of ``(x, y, z)`` sample locations.
    values:
        ``(n,)`` sample values.

    Returns
    -------
    (mean, variance):
        Two ``(nx, ny, nz)`` arrays.
    """
    pts = np.asarray(points, dtype=float)
    val = np.asarray(values, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (n, 3)")

    ok = OrdinaryKriging3D(
        pts[:, 0], pts[:, 1], pts[:, 2], val,
        variogram_model=variogram_model, nlags=nlags,
    )
    k, ss = ok.execute("grid", gx, gy, gz)  # shape (nz, ny, nx)
    mean = np.asarray(k).transpose(2, 1, 0)
    var = np.asarray(ss).transpose(2, 1, 0)
    return mean, var


def regression_kriging_3d(
    points: np.ndarray,
    values: np.ndarray,
    trend_at_points: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    trend_on_grid: np.ndarray,
    variogram_model: str = "spherical",
    nlags: int = 8,
):
    """Regression kriging: linear trend on covariates + OK of residuals.

    ``trend_at_points`` is ``(n, p)`` covariates at the sample locations;
    ``trend_on_grid`` is ``(nx, ny, nz, p)`` covariates on the grid.

    Returns ``(mean, variance)`` as ``(nx, ny, nz)``.  The variance is the
    residual kriging variance (trend uncertainty is neglected).
    """
    pts = np.asarray(points, dtype=float)
    val = np.asarray(values, dtype=float).ravel()
    tp = np.asarray(trend_at_points, dtype=float)
    tg = np.asarray(trend_on_grid, dtype=float)
    nx, ny, nz = len(gx), len(gy), len(gz)
    if tg.shape[:3] != (nx, ny, nz):
        raise ValueError("trend_on_grid must be (nx, ny, nz, p)")

    reg = LinearRegression().fit(tp, val)
    residual = val - reg.predict(tp)

    res_mean, res_var = ordinary_kriging_3d(
        pts, residual, gx, gy, gz, variogram_model=variogram_model, nlags=nlags
    )
    trend_grid = reg.predict(tg.reshape(-1, tg.shape[-1])).reshape(nx, ny, nz)
    return trend_grid + res_mean, res_var
