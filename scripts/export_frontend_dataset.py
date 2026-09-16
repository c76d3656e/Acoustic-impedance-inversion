"""Export a fixed dataset for the static frontend.

Writes, under ``frontend/public/data/``:

* ``manifest.json``  -- grid, axes, per-field metadata (units, ranges, colormap)
* ``fields/<key>.bin`` -- Float32 volumes, C-order over ``(nx, ny, nz)``
* ``boreholes.json`` -- multi-well along-hole data (V, N, M, F, UCS, AI)

The dataset is deterministic (fixed seed) so the frontend always renders the
same scene and all computation stays client-side.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from datasets import generate_mine
from fusion import run_fusion_pipeline
from mwd import mwd_features, PhysicsGuidedGPR
from geostats import ordinary_kriging_3d


def _write_field(path: str, vol: np.ndarray) -> None:
    np.asarray(vol, dtype="<f4").tofile(path)  # little-endian float32, C-order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="frontend/public/data")
    parser.add_argument("--n-holes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fields_dir = os.path.join(args.outdir, "fields")
    os.makedirs(fields_dir, exist_ok=True)

    print(">> Generating fixed mine dataset + running pipeline ...")
    ds = generate_mine(n_holes=args.n_holes, seed=args.seed)
    res = run_fusion_pipeline(ds, seed=args.seed)
    nx, ny, nz = ds.ucs_true.shape

    # PG-GPR UCS point predictions -> ordinary-kriged continuous MWD field.
    X = mwd_features(ds.V, ds.N, ds.M, ds.F)
    ucs_pts, _ = PhysicsGuidedGPR().fit(X, ds.ucs_at_holes).predict(X)
    mwd_field, _ = ordinary_kriging_3d(ds.hole_xyz, ucs_pts, ds.gx, ds.gy, ds.gz)
    sigma_F = np.sqrt(res.var_F)

    fields = [
        ("impedance", "波阻抗反演场", "×10⁶ kg/(m²·s)", res.ai_inv, 1.0e6, "viridis"),
        ("mwd_strength", "MWD 岩石强度场", "MPa", mwd_field, 1.0, "viridis"),
        ("fused_strength", "融合强度场", "MPa", res.S_F, 1.0, "viridis"),
        ("uncertainty", "融合不确定性", "MPa", sigma_F, 1.0, "magma"),
        ("ground_truth", "真实强度场", "MPa", ds.ucs_true, 1.0, "viridis"),
    ]

    field_meta = []
    for key, name_zh, unit, vol, scale, cmap in fields:
        _write_field(os.path.join(fields_dir, f"{key}.bin"), vol)
        field_meta.append({
            "key": key, "name_zh": name_zh, "unit": unit,
            "file": f"fields/{key}.bin", "scale": scale, "default_cmap": cmap,
            "min": float(np.min(vol)), "max": float(np.max(vol)),
        })
        print(f"   field {key:14s} range {vol.min():.4g}..{vol.max():.4g}")

    manifest = {
        "title_zh": "露天矿波阻抗与岩石强度三维可视化",
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "axes": {"x": ds.gx.tolist(), "y": ds.gy.tolist(), "z": ds.gz.tolist()},
        "extent": {
            "x": [float(ds.gx[0]), float(ds.gx[-1])],
            "y": [float(ds.gy[0]), float(ds.gy[-1])],
            "z": [float(ds.gz[0]), float(ds.gz[-1])],
        },
        "order": "c-xyz",  # flat index = (ix*ny + iy)*nz + iz
        "fields": field_meta,
        "boreholes_file": "boreholes.json",
        "n_holes": int(args.n_holes),
        "seed": int(args.seed),
    }
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---- Multi-well borehole data (points, along-hole logs, strength) -----
    ai_pts = res.ai_inv[ds.hole_ix, ds.hole_iy, ds.hole_iz]
    wells = {}
    for k in range(ds.ucs_at_holes.size):
        ix, iy, iz = int(ds.hole_ix[k]), int(ds.hole_iy[k]), int(ds.hole_iz[k])
        wid = f"W{ix:02d}-{iy:02d}"
        w = wells.setdefault(wid, {
            "id": wid, "x": float(ds.gx[ix]), "y": float(ds.gy[iy]),
            "ix": ix, "iy": iy, "samples": [],
        })
        w["samples"].append({
            "z": float(ds.gz[iz]),
            "V": float(ds.V[k]), "N": float(ds.N[k]),
            "M": float(ds.M[k]), "F": float(ds.F[k]),
            "ucs_true": float(ds.ucs_at_holes[k]),
            "ucs_pred": float(ucs_pts[k]),
            "ai": float(ai_pts[k]),
        })
    for w in wells.values():
        w["samples"].sort(key=lambda s: -s["z"])  # top (0) -> bottom (-120)

    with open(os.path.join(args.outdir, "boreholes.json"), "w") as f:
        json.dump({"wells": list(wells.values())}, f, ensure_ascii=False, indent=2)

    print(f">> Wrote {len(field_meta)} fields, {len(wells)} wells to {args.outdir}")


if __name__ == "__main__":
    main()
