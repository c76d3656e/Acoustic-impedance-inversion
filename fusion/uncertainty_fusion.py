"""Uncertainty-aware fusion of two strength fields.

Treating each source as a Gaussian estimate ``N(mu, sigma^2)``, the
precision-weighted (inverse-variance) combination is the maximum-likelihood
fusion for independent estimates:

    mu_f    = (mu_M/sigma_M^2 + mu_Z/sigma_Z^2) / (1/sigma_M^2 + 1/sigma_Z^2)
    sigma_f^2 = 1 / (1/sigma_M^2 + 1/sigma_Z^2)

The locally more reliable source (smaller sigma) automatically dominates.
"""

from __future__ import annotations

import numpy as np


def precision_weighted_fusion(mu_M, var_M, mu_Z, var_Z, eps: float = 1e-12):
    """Inverse-variance fusion. Returns ``(mu_f, var_f)`` (same shape)."""
    mu_M = np.asarray(mu_M, dtype=float)
    var_M = np.asarray(var_M, dtype=float)
    mu_Z = np.asarray(mu_Z, dtype=float)
    var_Z = np.asarray(var_Z, dtype=float)

    tau_M = 1.0 / (var_M + eps)
    tau_Z = 1.0 / (var_Z + eps)
    var_f = 1.0 / (tau_M + tau_Z)
    mu_f = (mu_M * tau_M + mu_Z * tau_Z) * var_f
    return mu_f, var_f


def simple_weighted_fusion(mu_M, mu_Z, w: float = 0.5):
    """Fixed-weight baseline: ``w * mu_M + (1 - w) * mu_Z``."""
    if not 0.0 <= w <= 1.0:
        raise ValueError("w must be in [0, 1]")
    return w * np.asarray(mu_M, dtype=float) + (1.0 - w) * np.asarray(mu_Z, dtype=float)


def borehole_anchor_weight(
    gx,
    gy,
    hole_xy,
    radius: float | None = None,
) -> np.ndarray:
    """Compact XY weight that is 1 at borehole collars and 0 beyond ``radius``.

    Precision fusion is applied to the *interpolated* field.  Multiplying the
    borehole branch back in with this kernel keeps hard data from being
    cancelled by impedance.  ``radius`` defaults to 0.4 × median hole spacing
    (or 18 % of the shorter axis for a single hole).
    """
    gx = np.asarray(gx, dtype=float).ravel()
    gy = np.asarray(gy, dtype=float).ravel()
    xy = np.unique(np.asarray(hole_xy, dtype=float).reshape(-1, 2), axis=0)
    nx, ny = gx.size, gy.size
    if xy.size == 0:
        return np.zeros((nx, ny), dtype=float)

    dx = float(np.median(np.abs(np.diff(gx)))) if nx > 1 else 1.0
    dy = float(np.median(np.abs(np.diff(gy)))) if ny > 1 else 1.0
    if radius is None:
        if len(xy) >= 2:
            nn = []
            for i in range(len(xy)):
                d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
                d[i] = np.inf
                nn.append(float(np.min(d)))
            radius = 0.40 * float(np.median(nn))
        else:
            radius = 0.18 * min(float(gx[-1] - gx[0]), float(gy[-1] - gy[0]))
    radius = max(float(radius), 2.0 * max(dx, dy))

    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    dist = np.sqrt(
        (xx[:, :, None] - xy[None, None, :, 0]) ** 2
        + (yy[:, :, None] - xy[None, None, :, 1]) ** 2
    ).min(axis=2)
    r = dist / radius
    return np.clip(1.0 - r * r, 0.0, 1.0) ** 2


def anchor_borehole_hard_data(mu_M, mu_prec, var_M, var_prec, w_xy):
    """Re-inject the borehole branch near holes.

    ``w_xy`` is ``(nx, ny)`` (1 at collars).  Broadcasts over depth.
    Returns ``(mu, var)``.
    """
    w = np.asarray(w_xy, dtype=float)
    if w.ndim == 2:
        w = w[:, :, np.newaxis]
    mu = w * np.asarray(mu_M, dtype=float) + (1.0 - w) * np.asarray(mu_prec, dtype=float)
    var = w * np.asarray(var_M, dtype=float) + (1.0 - w) * np.asarray(var_prec, dtype=float)
    return mu, var
