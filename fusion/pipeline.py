"""Reusable dual-branch fusion pipeline.

Runs both branches on a :class:`datasets.MineDataset` and returns all fields, so
the benchmark and the visualization tooling share a single implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mwd import mwd_features, PhysicsGuidedGPR
from geostats import regression_kriging_3d
from inversion import ricker, synthetic_seismic_volume, invert_volume, background_model
from .calibration import ImpedanceStrengthCalibrator
from .uncertainty_fusion import (
    simple_weighted_fusion,
    doyen_collocated_update,
    borehole_anchor_weight,
)


@dataclass
class FusionResult:
    ai_inv: np.ndarray       # inverted acoustic impedance
    S_M: np.ndarray          # MWD-derived strength
    var_M: np.ndarray
    S_Z: np.ndarray          # impedance-derived strength
    var_Z: np.ndarray
    S_weighted: np.ndarray   # simple-weighted baseline
    S_F: np.ndarray          # collocated cokriging / KED fused strength
    var_F: np.ndarray
    ucs_pts_std_mean: float
    w_anchor: np.ndarray     # (nx, ny) primary (well) weight, 1 at collars
    rho: float = 0.0         # corr(UCS, AI) used by the Doyen update


def _n_unique_collars(ds) -> int:
    xy = np.stack([np.asarray(ds.hole_ix), np.asarray(ds.hole_iy)], axis=1)
    return int(np.unique(xy, axis=0).shape[0])


def _ked_with_impedance(ds, ucs_pts, ai_inv):
    """Xu/Journel kriging with external drift: primary UCS, drift = [depth, AI]."""
    nx, ny, nz = ds.ucs_true.shape
    ai_at_holes = np.asarray(ai_inv, dtype=float)[ds.hole_ix, ds.hole_iy, ds.hole_iz]
    trend_pts = np.column_stack([ds.hole_xyz[:, 2], ai_at_holes])
    z_grid = np.broadcast_to(ds.gz[None, None, :, None], (nx, ny, nz, 1))
    trend_grid = np.concatenate(
        [z_grid, np.asarray(ai_inv, dtype=float)[..., None]], axis=-1,
    )
    return regression_kriging_3d(
        ds.hole_xyz, ucs_pts, trend_pts, ds.gx, ds.gy, ds.gz, trend_grid,
    )


def run_fusion_pipeline(ds, noise: float = 0.05, lam: float = 5.0,
                        freq: float = 30.0, dt: float = 0.002,
                        seed: int = 42, ai_inv=None) -> FusionResult:
    """Run MWD + seismic + fusion.

    ``ai_inv`` may be a precomputed impedance volume (same shape as the mine).
    When given, the seismic inversion is skipped so a well-count series can
    reuse one impedance field while the MWD branch and the AI→UCS calibrator
    see more holes.

    Fusion is collocated cokriging in Doyen's Bayesian form (SPE 36498) when
    at least two collars are available: primary = MWD kriging ``S_M``,
    secondary = calibrated ``S_Z``, ``ρ = corr(UCS, AI)`` at the holes.
    A single collar cannot constrain a 3-D GP transform, so that case falls
    back to kriging with impedance as external drift (Xu et al., SPE 24742).
    Hole voxels are always written back to the MWD point estimates.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = ds.ucs_true.shape

    # --- MWD branch ---
    X = mwd_features(ds.V, ds.N, ds.M, ds.F)
    pggpr = PhysicsGuidedGPR().fit(X, ds.ucs_at_holes)
    ucs_pts, ucs_pts_std = pggpr.predict(X)

    zt_pts = ds.hole_xyz[:, [2]]
    zt_grid = np.broadcast_to(ds.gz[None, None, :, None], (nx, ny, nz, 1))
    S_M, var_krige = regression_kriging_3d(
        ds.hole_xyz, ucs_pts, zt_pts, ds.gx, ds.gy, ds.gz, zt_grid,
    )
    # Hole voxels are hard data: keep the borehole estimate, do not let
    # kriging / nugget smear it before fusion.
    S_M = np.array(S_M, copy=True, dtype=float)
    S_M[ds.hole_ix, ds.hole_iy, ds.hole_iz] = ucs_pts
    var_M = np.maximum(np.asarray(var_krige, dtype=float), 0.0)

    # --- Seismic branch ---
    if ai_inv is None:
        wavelet = ricker(n=31, dt=dt, freq=freq)
        seismic = synthetic_seismic_volume(ds.ai_true, wavelet)
        seismic = seismic + noise * np.std(seismic) * rng.standard_normal(seismic.shape)
        background = background_model(ds.ai_true, sigma=(2, 2, 6))
        ai_inv = invert_volume(seismic, wavelet, background, lam=lam)
    else:
        ai_inv = np.asarray(ai_inv, dtype=float)
        if ai_inv.shape != ds.ucs_true.shape:
            raise ValueError("ai_inv shape must match the mine UCS volume")

    ai_at_holes = ai_inv[ds.hole_ix, ds.hole_iy, ds.hole_iz]
    calib = ImpedanceStrengthCalibrator().fit(ai_at_holes, ds.ucs_at_holes)
    S_Z, sigma_Z = calib.predict(ai_inv)
    var_Z = sigma_Z**2

    # --- Fusion: collocated cokriging (Doyen) or KED with AI drift ----------
    S_weighted = simple_weighted_fusion(S_M, S_Z, w=0.5)
    n_xy = _n_unique_collars(ds)
    if np.std(ucs_pts) > 1e-12 and np.std(ai_at_holes) > 1e-12:
        rho = float(np.corrcoef(np.ravel(ucs_pts), np.ravel(ai_at_holes))[0, 1])
    else:
        rho = 0.0
    rho = float(np.nan_to_num(rho, nan=0.0))

    sz_at_holes = S_Z[ds.hole_ix, ds.hole_iy, ds.hole_iz]
    if n_xy >= 2 and abs(rho) > 0.05:
        S_F, var_F, w_mwd = doyen_collocated_update(
            S_M, var_M, S_Z, rho,
            mean_primary=float(np.mean(ucs_pts)),
            std_primary=float(np.std(ucs_pts)),
            mean_secondary=float(np.mean(sz_at_holes)),
            std_secondary=float(np.std(sz_at_holes) or 1.0),
        )
    else:
        # One collar: the AI→UCS GP overfits that hole (ρ_SZ → 1).  KED
        # uses inverted impedance itself as the drift so the far field is a
        # linear rock-physics transform plus kriged residuals, not the GP.
        S_F, var_ked = _ked_with_impedance(ds, ucs_pts, ai_inv)
        var_F = np.maximum(np.asarray(var_ked, dtype=float), 0.0)
        sill = float(np.percentile(var_F, 95)) + 1e-12
        w_mwd = 1.0 - np.clip(var_F / sill, 0.0, 1.0)

    S_F = np.array(S_F, copy=True, dtype=float)
    S_F[ds.hole_ix, ds.hole_iy, ds.hole_iz] = ucs_pts
    var_F = np.maximum(np.asarray(var_F, dtype=float), 0.0)

    w_xy = np.max(np.asarray(w_mwd, dtype=float), axis=2)
    # Collar columns must read as hard-data = 1 even if the KED variogram
    # left a nugget on var_F.
    w_holes = borehole_anchor_weight(ds.gx, ds.gy, ds.hole_xyz[:, :2])
    w_anchor = np.maximum(w_xy, w_holes)

    return FusionResult(
        ai_inv=ai_inv, S_M=S_M, var_M=var_M, S_Z=S_Z, var_Z=var_Z,
        S_weighted=S_weighted, S_F=S_F, var_F=var_F,
        ucs_pts_std_mean=float(np.mean(ucs_pts_std)),
        w_anchor=w_anchor,
        rho=rho,
    )
