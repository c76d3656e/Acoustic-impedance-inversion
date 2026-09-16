r"""End-to-end MWD-Seismic physics-constrained strength-fusion benchmark.

    ground truth (UCS, AI, co-located MWD)
        |                              |
   MWD branch                     Seismic branch
   V,N,M,F -> PG-GPR -> UCS pts   AI -> seismic -> inversion -> AI_inv
   -> 3D regression kriging       -> AI->UCS calibration (GP)
   -> S_MWD, var_MWD              -> S_Z, var_Z
        \______________ fusion _______________/
                          |
        precision-weighted (uncertainty-aware) fusion
                          |
                  S_fused, sigma_fused   -> compare to UCS_true

Run: python examples/run_fusion_benchmark.py --outdir results
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from datasets import generate_mine, acoustic_impedance  # noqa: F401
from mwd import mwd_features, PhysicsGuidedGPR
from geostats import regression_kriging_3d
from inversion import ricker, synthetic_seismic_volume, invert_volume, background_model
from fusion import (
    ImpedanceStrengthCalibrator,
    precision_weighted_fusion,
    simple_weighted_fusion,
)
from validation import summary
from visualization import plot_fusion_panels, plot_field_slice


def nearest_index(axis, value):
    return int(np.argmin(np.abs(np.asarray(axis) - value)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--n-holes", type=int, default=12)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elevations", type=float, nargs="*", default=[-24, -60, -96])
    args = parser.parse_args()

    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(">> Generating co-located synthetic mine (ground truth) ...")
    ds = generate_mine(n_holes=args.n_holes, seed=args.seed)
    nx, ny, nz = ds.ucs_true.shape
    print(f"   grid {nx}x{ny}x{nz}, {args.n_holes} holes, "
          f"{ds.ucs_at_holes.size} MWD samples")
    print(f"   UCS_true range: {ds.ucs_true.min():.1f}..{ds.ucs_true.max():.1f} MPa")

    # ---------------- MWD branch ----------------------------------------
    print("\n== MWD branch: V,N,M,F -> PG-GPR -> UCS -> regression kriging ==")
    X = mwd_features(ds.V, ds.N, ds.M, ds.F)

    # Spatial blind test: hold out ~30% of holes (by hole id).
    hole_ids = np.stack([ds.hole_ix, ds.hole_iy], axis=1)
    uniq = np.unique(hole_ids, axis=0)
    n_blind = max(1, int(0.3 * len(uniq)))
    blind = uniq[rng.choice(len(uniq), n_blind, replace=False)]
    is_blind = np.array([any((hi == b).all() for b in blind) for hi in hole_ids])

    pggpr = PhysicsGuidedGPR().fit(X[~is_blind], ds.ucs_at_holes[~is_blind])
    ucs_blind_pred, _ = pggpr.predict(X[is_blind])
    r2_blind = summary(ds.ucs_at_holes[is_blind], ucs_blind_pred)["R2"]
    print(f"   PG-GPR spatial-blind UCS R2: {r2_blind:.3f} "
          f"({n_blind}/{len(uniq)} holes held out)")

    pggpr_full = PhysicsGuidedGPR().fit(X, ds.ucs_at_holes)
    ucs_pts, ucs_pts_std = pggpr_full.predict(X)

    zt_pts = ds.hole_xyz[:, [2]]  # elevation as regression-kriging trend
    zt_grid = np.broadcast_to(ds.gz[None, None, :, None], (nx, ny, nz, 1))
    t0 = time.time()
    S_M, var_krige = regression_kriging_3d(
        ds.hole_xyz, ucs_pts, zt_pts, ds.gx, ds.gy, ds.gz, zt_grid,
    )
    var_M = var_krige + float(np.mean(ucs_pts_std**2))
    print(f"   regression kriging -> S_MWD in {time.time() - t0:.1f}s "
          f"(R2 vs true: {summary(ds.ucs_true, S_M)['R2']:.3f})")

    # ---------------- Seismic branch ------------------------------------
    print("\n== Seismic branch: AI -> seismic -> inversion -> AI->UCS calibration ==")
    wavelet = ricker(n=31, dt=0.002, freq=30.0)
    seismic = synthetic_seismic_volume(ds.ai_true, wavelet)
    seismic += args.noise * np.std(seismic) * rng.standard_normal(seismic.shape)
    background = background_model(ds.ai_true, sigma=(2, 2, 6))
    ai_inv = invert_volume(seismic, wavelet, background, lam=5.0)
    print(f"   impedance inversion R2 (AI): {summary(ds.ai_true, ai_inv)['R2']:.3f}")

    ai_at_holes = ai_inv[ds.hole_ix, ds.hole_iy, ds.hole_iz]
    calib = ImpedanceStrengthCalibrator().fit(ai_at_holes, ds.ucs_at_holes)
    S_Z, sigma_Z = calib.predict(ai_inv)
    var_Z = sigma_Z**2
    print(f"   AI->UCS calibration -> S_Z (R2 vs true: {summary(ds.ucs_true, S_Z)['R2']:.3f})")

    # ---------------- Fusion (4 cases) ----------------------------------
    print("\n== Fusion ==")
    S_weighted = simple_weighted_fusion(S_M, S_Z, w=0.5)
    S_F, var_F = precision_weighted_fusion(S_M, var_M, S_Z, var_Z)
    sigma_F = np.sqrt(var_F)

    cases = {
        "Case1 MWD-only":       S_M,
        "Case2 Seismic-only":   S_Z,
        "Case3 Simple-weighted": S_weighted,
        "Case4 Uncertainty-aware (proposed)": S_F,
    }
    print(f"   {'Case':<38}{'R2':>8}{'RMSE':>10}{'MAE':>10}")
    metrics = {}
    for name, field in cases.items():
        s = summary(ds.ucs_true, field)
        metrics[name] = s
        print(f"   {name:<38}{s['R2']:>8.3f}{s['RMSE']:>10.2f}{s['MAE']:>10.2f}")

    # ---------------- Outputs -------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "S_true.npy"), ds.ucs_true)
    np.save(os.path.join(args.outdir, "S_fused.npy"), S_F)
    np.save(os.path.join(args.outdir, "sigma_fused.npy"), sigma_F)

    print("\n>> Rendering figures ...")
    z0 = nearest_index(ds.gz, args.elevations[len(args.elevations) // 2])
    panels = [
        (S_M, "(a) MWD strength S_MWD", "UCS (MPa)", "viridis"),
        (ai_inv, "(b) Seismic impedance AI", r"kg/(m$^2$s)", "cividis"),
        (S_Z, "(c) Impedance-derived S_Z", "UCS (MPa)", "viridis"),
        (S_F, "(d) Fused strength S_F", "UCS (MPa)", "viridis"),
        (sigma_F, "(e) Fusion uncertainty", "sigma (MPa)", "magma"),
    ]
    fig_panels = plot_fusion_panels(
        panels, ds.gx, ds.gy, z0, ds.gz[z0],
        os.path.join(figdir, "fusion_panels.png"),
    )

    slice_files = []
    for elev in args.elevations:
        zi = nearest_index(ds.gz, elev)
        f = plot_field_slice(
            S_F, ds.gx, ds.gy, zi,
            os.path.join(figdir, f"fused_strength_elev_{int(elev)}.png"),
            title=f"Fused strength @ elevation {ds.gz[zi]:.0f} m",
            cbar_label="UCS (MPa)",
        )
        slice_files.append(f)

    # Ground-truth vs fused at the mid elevation for reference.
    gt_file = plot_field_slice(
        ds.ucs_true, ds.gx, ds.gy, z0,
        os.path.join(figdir, "ground_truth_strength.png"),
        title=f"Ground-truth strength @ elevation {ds.gz[z0]:.0f} m",
        cbar_label="UCS (MPa)",
    )

    print("\n=== Fusion benchmark complete ===")
    print(f"   panels          : {fig_panels}")
    print(f"   ground truth    : {gt_file}")
    for f in slice_files:
        print(f"   slice           : {f}")
    best = metrics["Case4 Uncertainty-aware (proposed)"]
    print(f"   proposed fusion : R2={best['R2']:.3f} RMSE={best['RMSE']:.2f} "
          f"MAE={best['MAE']:.2f}")


if __name__ == "__main__":
    main()
