r"""End-to-end MWD-Seismic physics-constrained strength-fusion benchmark.

    ground truth (UCS, AI, co-located MWD)
        |                              |
   MWD branch                     Seismic branch
   V,N,M,F -> PG-GPR -> UCS pts   AI -> seismic -> inversion -> AI_inv
   -> 3D regression kriging       -> AI->UCS calibration (GP)
   -> S_MWD, var_MWD              -> S_Z, var_Z
        \______________ fusion _______________/
                          |
     collocated cokriging (Doyen Bayesian update)
                          |
                  S_fused, sigma_fused   -> compare to UCS_true

Run: python examples/run_fusion_benchmark.py --outdir results
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from datasets import generate_mine
from mwd import mwd_features, PhysicsGuidedGPR
from fusion import run_fusion_pipeline
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
    parser.add_argument("--elevations", type=float, nargs="*", default=[-8, -20, -32])
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

    # MWD spatial-blind test (hold out ~30% of holes).
    X = mwd_features(ds.V, ds.N, ds.M, ds.F)
    hole_ids = np.stack([ds.hole_ix, ds.hole_iy], axis=1)
    uniq = np.unique(hole_ids, axis=0)
    n_blind = max(1, int(0.3 * len(uniq)))
    blind = uniq[rng.choice(len(uniq), n_blind, replace=False)]
    is_blind = np.array([any((hi == b).all() for b in blind) for hi in hole_ids])
    pggpr = PhysicsGuidedGPR().fit(X[~is_blind], ds.ucs_at_holes[~is_blind])
    ucs_blind_pred, _ = pggpr.predict(X[is_blind])
    r2_blind = summary(ds.ucs_at_holes[is_blind], ucs_blind_pred)["R2"]
    print(f"\n>> PG-GPR spatial-blind UCS R2: {r2_blind:.3f} "
          f"({n_blind}/{len(uniq)} holes held out)")

    print(">> Running dual-branch fusion pipeline ...")
    res = run_fusion_pipeline(ds, noise=args.noise, seed=args.seed)
    print(f"   impedance inversion R2 (AI): {summary(ds.ai_true, res.ai_inv)['R2']:.3f}")

    cases = {
        "Case1 MWD-only":       res.S_M,
        "Case2 Seismic-only":   res.S_Z,
        "Case3 Simple-weighted": res.S_weighted,
        "Case4 Collocated cokriging (proposed)": res.S_F,
    }
    print(f"\n   {'Case':<38}{'R2':>8}{'RMSE':>10}{'MAE':>10}")
    metrics = {}
    for name, field in cases.items():
        s = summary(ds.ucs_true, field)
        metrics[name] = s
        print(f"   {name:<38}{s['R2']:>8.3f}{s['RMSE']:>10.2f}{s['MAE']:>10.2f}")

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "S_true.npy"), ds.ucs_true)
    np.save(os.path.join(args.outdir, "S_fused.npy"), res.S_F)
    np.save(os.path.join(args.outdir, "sigma_fused.npy"), np.sqrt(res.var_F))

    print("\n>> Rendering figures ...")
    z0 = nearest_index(ds.gz, args.elevations[len(args.elevations) // 2])
    panels = [
        (res.S_M, "(a) MWD strength S_MWD", "UCS (MPa)", "viridis"),
        (res.ai_inv, "(b) Seismic impedance AI", r"kg/(m$^2$s)", "cividis"),
        (res.S_Z, "(c) Impedance-derived S_Z", "UCS (MPa)", "viridis"),
        (res.S_F, "(d) Fused strength S_F", "UCS (MPa)", "viridis"),
        (np.sqrt(res.var_F), "(e) Fusion uncertainty", "sigma (MPa)", "magma"),
    ]
    fig_panels = plot_fusion_panels(
        panels, ds.gx, ds.gy, z0, ds.gz[z0],
        os.path.join(figdir, "fusion_panels.png"),
    )
    for elev in args.elevations:
        zi = nearest_index(ds.gz, elev)
        plot_field_slice(
            res.S_F, ds.gx, ds.gy, zi,
            os.path.join(figdir, f"fused_strength_elev_{int(elev)}.png"),
            title=f"Fused strength @ elevation {ds.gz[zi]:.0f} m",
            cbar_label="UCS (MPa)",
        )
    plot_field_slice(
        ds.ucs_true, ds.gx, ds.gy, z0,
        os.path.join(figdir, "ground_truth_strength.png"),
        title=f"Ground-truth strength @ elevation {ds.gz[z0]:.0f} m",
        cbar_label="UCS (MPa)",
    )

    best = metrics["Case4 Collocated cokriging (proposed)"]
    print("\n=== Fusion benchmark complete ===")
    print(f"   panels : {fig_panels}")
    print(f"   proposed fusion : R2={best['R2']:.3f} RMSE={best['RMSE']:.2f} "
          f"MAE={best['MAE']:.2f}")


if __name__ == "__main__":
    main()
