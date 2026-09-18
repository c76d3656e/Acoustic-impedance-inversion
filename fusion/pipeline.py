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
from .uncertainty_fusion import precision_weighted_fusion, simple_weighted_fusion


@dataclass
class FusionResult:
    ai_inv: np.ndarray       # inverted acoustic impedance
    S_M: np.ndarray          # MWD-derived strength
    var_M: np.ndarray
    S_Z: np.ndarray          # impedance-derived strength
    var_Z: np.ndarray
    S_weighted: np.ndarray   # simple-weighted baseline
    S_F: np.ndarray          # uncertainty-aware fused strength
    var_F: np.ndarray
    ucs_pts_std_mean: float


def run_fusion_pipeline(ds, noise: float = 0.05, lam: float = 5.0,
                        freq: float = 30.0, dt: float = 0.002,
                        seed: int = 42, ai_inv=None) -> FusionResult:
    """Run MWD + seismic + fusion.

    ``ai_inv`` may be a precomputed impedance volume (same shape as the mine).
    When given, the seismic inversion is skipped so a well-count series can
    reuse one impedance field while the MWD branch and the AI→UCS calibrator
    see more holes.
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
    var_M = var_krige + float(np.mean(ucs_pts_std**2))

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

    # --- Fusion ---
    S_weighted = simple_weighted_fusion(S_M, S_Z, w=0.5)
    S_F, var_F = precision_weighted_fusion(S_M, var_M, S_Z, var_Z)

    return FusionResult(
        ai_inv=ai_inv, S_M=S_M, var_M=var_M, S_Z=S_Z, var_Z=var_Z,
        S_weighted=S_weighted, S_F=S_F, var_F=var_F,
        ucs_pts_std_mean=float(np.mean(ucs_pts_std)),
    )
