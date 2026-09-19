"""Well + seismic strength fusion.

Industry practice (Xu et al. SPE 24742; Doyen et al. SPE 36498) treats sparse
borehole UCS as the *primary* variable and the dense seismic field as a
*secondary* / collocated constraint.  That exact-honors wells when the primary
kriging variance vanishes, and does not assume the two maps are independent.

``doyen_collocated_update`` is the Bayesian form of collocated cokriging:
it updates a primary kriging estimate with a collocated secondary using only
the kriging variance and a correlation coefficient.

``precision_weighted_fusion`` is kept as an independent-source baseline; it
can cancel hard data when ``S_M`` and ``S_Z`` are correlated, which is why
the pipeline no longer uses it as ``S_F``.
"""

from __future__ import annotations

import numpy as np


def doyen_collocated_update(
    mu_k,
    var_k,
    secondary,
    rho: float,
    mean_primary: float | None = None,
    std_primary: float | None = None,
    mean_secondary: float | None = None,
    std_secondary: float | None = None,
    eps: float = 1e-12,
):
    """Bayesian collocated cokriging update (Doyen, SPE 36498).

    Primary kriging ``N(mu_k, var_k)`` is updated with a collocated secondary
    field.  In standardized units (Deutsch / Doyen):

        y_k' = (mu_k - m_y) / σ_y ,   y_s' = (z - m_z) / σ_z
        σ_k'² = clip(var_k / σ_y², 0, 1)
        λ = ρ σ_k'² / (ρ² σ_k'² + 1 - ρ²)
        y_cc' = y_k' + λ (y_s' - ρ y_k')
        σ_cc'² = σ_k'² (1 - ρ²) / (ρ² σ_k'² + 1 - ρ²)

    At hard-data nodes ``var_k → 0`` so ``λ → 0`` and ``y_cc → mu_k``.  Far
    from wells ``σ_k'² → 1`` and the estimate shrinks toward the linear
    regression of the secondary.

    Returns ``(mu_cc, var_cc)`` in the original primary units, plus the
    primary weight ``w = 1 - λ ρ`` (1 at wells).
    """
    mu_k = np.asarray(mu_k, dtype=float)
    var_k = np.maximum(np.asarray(var_k, dtype=float), 0.0)
    z = np.asarray(secondary, dtype=float)
    if mu_k.shape != z.shape or var_k.shape != mu_k.shape:
        raise ValueError("mu_k, var_k and secondary must share the same shape")

    rho = float(np.clip(rho, -0.999, 0.999))
    m_y = float(np.mean(mu_k) if mean_primary is None else mean_primary)
    s_y = float(np.std(mu_k) if std_primary is None else std_primary)
    m_z = float(np.mean(z) if mean_secondary is None else mean_secondary)
    s_z = float(np.std(z) if std_secondary is None else std_secondary)
    s_y = max(s_y, eps)
    s_z = max(s_z, eps)

    yk = (mu_k - m_y) / s_y
    ys = (z - m_z) / s_z
    sk2 = np.clip(var_k / (s_y * s_y), 0.0, 1.0)
    one_minus = 1.0 - rho * rho
    denom = rho * rho * sk2 + one_minus + eps
    lam = rho * sk2 / denom
    ycc = yk + lam * (ys - rho * yk)
    var_s = sk2 * one_minus / denom
    mu_cc = ycc * s_y + m_y
    var_cc = var_s * (s_y * s_y)
    w_primary = 1.0 - lam * rho
    return mu_cc, var_cc, w_primary


def precision_weighted_fusion(mu_M, var_M, mu_Z, var_Z, eps: float = 1e-12):
    """Inverse-variance fusion. Returns ``(mu_f, var_f)`` (same shape).

    Assumes independent sources.  Prefer :func:`doyen_collocated_update` when
    the maps share calibration wells.
    """
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
