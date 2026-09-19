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
    variogram_parameters: dict | None = None,
    anisotropy_scaling_z: float = 1.0,
):
    """Ordinary 3-D kriging.

    Parameters
    ----------
    points:
        ``(n, 3)`` array of ``(x, y, z)`` sample locations.
    values:
        ``(n,)`` sample values.
    variogram_parameters:
        Optional ``{sill, range, nugget}``.  When omitted, PyKrige auto-fits
        (and falls back to a domain-scale spherical model if that fails).
    anisotropy_scaling_z:
        PyKrige z-axis stretch (``>1`` ⇒ longer vertical correlation).

    Returns
    -------
    (mean, variance):
        Two ``(nx, ny, nz)`` arrays.
    """
    pts = np.asarray(points, dtype=float)
    val = np.asarray(values, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (n, 3)")

    ok = _ordinary_kriging_3d_fit(
        pts, val, variogram_model=variogram_model, nlags=nlags,
        gx=gx, gy=gy, gz=gz,
        variogram_parameters=variogram_parameters,
        anisotropy_scaling_z=anisotropy_scaling_z,
    )
    k, ss = ok.execute("grid", gx, gy, gz)  # shape (nz, ny, nx)
    mean = np.asarray(k).transpose(2, 1, 0)
    var = np.asarray(ss).transpose(2, 1, 0)
    return mean, var


def _ordinary_kriging_3d_fit(
    pts, val, variogram_model, nlags, gx, gy, gz,
    variogram_parameters=None, anisotropy_scaling_z=1.0,
):
    """Fit 3-D OK; fall back to a domain-scale spherical model if auto-fit fails.

    A single vertical hole has no lateral lags, so PyKrige cannot estimate a
    3-D variogram from the samples alone.  In that case we use the sample
    variance as the sill and half the model diagonal as the range — the
    interpolator then reverts to the (depth) mean away from the hole, which is
    the correct sparse-well behaviour.
    """
    kw = {"anisotropy_scaling_z": float(anisotropy_scaling_z)}
    if variogram_parameters is not None:
        return OrdinaryKriging3D(
            pts[:, 0], pts[:, 1], pts[:, 2], val,
            variogram_model=variogram_model,
            variogram_parameters=dict(variogram_parameters),
            **kw,
        )
    try:
        return OrdinaryKriging3D(
            pts[:, 0], pts[:, 1], pts[:, 2], val,
            variogram_model=variogram_model, nlags=nlags, **kw,
        )
    except Exception:
        sill = float(np.var(val)) + 1e-6
        dx = float(np.asarray(gx)[-1] - np.asarray(gx)[0])
        dy = float(np.asarray(gy)[-1] - np.asarray(gy)[0])
        dz = float(np.asarray(gz)[-1] - np.asarray(gz)[0])
        vrange = 0.5 * float(np.sqrt(dx * dx + dy * dy + dz * dz))
        return OrdinaryKriging3D(
            pts[:, 0], pts[:, 1], pts[:, 2], val,
            variogram_model=variogram_model,
            variogram_parameters={
                "sill": sill, "range": vrange, "nugget": 0.1 * sill,
            },
            **kw,
        )


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
    residual_range: float | None = None,
    anisotropy_scaling_z: float = 1.0,
    nugget_frac: float = 0.05,
):
    """Regression kriging: linear trend on covariates + OK of residuals.

    ``trend_at_points`` is ``(n, p)`` covariates at the sample locations;
    ``trend_on_grid`` is ``(nx, ny, nz, p)`` covariates on the grid.

    ``residual_range`` (metres) optionally replaces auto-fit with a spherical
    model whose range is the hole-to-hole scale.  Auto-fit on a compact blast
    block otherwise stretches to the domain diagonal and oversmooths the
    mechanical residual that seismic cannot see.

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

    vparams = None
    if residual_range is not None:
        sill = float(np.var(residual)) + 1e-6
        vparams = {
            "sill": sill,
            "range": float(residual_range),
            "nugget": float(nugget_frac) * sill,
        }
    res_mean, res_var = ordinary_kriging_3d(
        pts, residual, gx, gy, gz,
        variogram_model=variogram_model, nlags=nlags,
        variogram_parameters=vparams,
        anisotropy_scaling_z=anisotropy_scaling_z,
    )
    trend_grid = reg.predict(tg.reshape(-1, tg.shape[-1])).reshape(nx, ny, nz)
    return trend_grid + res_mean, res_var
