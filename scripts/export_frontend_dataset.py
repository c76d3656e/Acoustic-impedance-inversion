"""Export a fixed dataset for the static frontend.

Writes, under ``frontend/public/data/``:

* ``manifest.json``  -- grid, axes, per-field metadata (units, ranges, colormap)
* ``fields/<key>.bin`` -- Float32 volumes, C-order over ``(nx, ny, nz)``
* ``boreholes.json`` -- multi-well along-hole data (V, N, M, F, UCS branches, AI)

The dataset is deterministic (fixed seed) so the frontend always renders the
same scene and all computation stays client-side.  Volumes come from
``run_fusion_pipeline`` (KED fusion): ``S_M``, ``S_Z``, ``S_F``, inverted AI.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from datasets import generate_mine, unique_hole_xy_indices
from fusion import run_fusion_pipeline

# Live compare view + Pyodide fusion-advantage figure (matches docs/images).
COMPARE_KEYS = (
    "ground_truth",
    "mwd_strength",
    "seismic_strength",
    "fused_strength",
)
COMPARE_TITLES = (
    "(a) 强度真值",
    "(b) 钻孔插值",
    "(c) 波阻抗插值",
    "(d) 融合插值",
)
COMPARE_RESIDUAL_TITLES = (
    "",
    "(e) 钻孔残差",
    "(f) 波阻抗残差",
    "(g) 融合残差",
)
VIEW_WINDOW = {"width": 20, "height": 50}


def _write_field(path: str, vol: np.ndarray) -> None:
    np.asarray(vol, dtype="<f4").tofile(path)  # little-endian float32, C-order


def _profile_well_titles(ds) -> dict[tuple[int, int], str]:
    """Same two wells as ``examples/run_docs_figures.plot_along_holes``."""
    pairs = unique_hole_xy_indices(ds)
    xy = np.column_stack([ds.gx[pairs[:, 0]], ds.gy[pairs[:, 1]]])
    targets: list[np.ndarray] = []
    labels: list[str] = []
    if ds.meta.get("alter_xy") is not None:
        targets.append(np.asarray(ds.meta["alter_xy"], dtype=float))
        labels.append("蚀变晕附近钻孔（力学残差）")
    if ds.meta.get("hard_xy") is not None:
        targets.append(np.asarray(ds.meta["hard_xy"], dtype=float))
        labels.append("硬矿体附近钻孔")
    if not targets:
        return {(int(pairs[0, 0]), int(pairs[0, 1])): "钻孔 1"}
    used: set[tuple[int, int]] = set()
    out: dict[tuple[int, int], str] = {}
    for tgt, lab in zip(targets, labels):
        d = np.sqrt(((xy - tgt) ** 2).sum(axis=1))
        for idx in np.argsort(d):
            key = (int(pairs[idx, 0]), int(pairs[idx, 1]))
            if key not in used:
                used.add(key)
                out[key] = lab
                break
    return out


def fields_from_result(ds, res) -> list[tuple]:
    """``(key, name_zh, unit, volume, scale, cmap)`` from a pipeline result."""
    sigma_F = np.sqrt(np.maximum(np.asarray(res.var_F, dtype=float), 0.0))
    w3 = np.broadcast_to(
        np.asarray(res.w_anchor, dtype=float)[:, :, None], ds.ucs_true.shape,
    ).copy()
    return [
        ("fused_strength", "外漂移克里金融合强度场", "MPa", res.S_F, 1.0, "viridis"),
        ("mwd_strength", "仅钻孔克里金强度场", "MPa", res.S_M, 1.0, "viridis"),
        ("seismic_strength", "波阻抗标定强度场", "MPa", res.S_Z, 1.0, "viridis"),
        ("ground_truth", "真实强度场", "MPa", ds.ucs_true, 1.0, "viridis"),
        ("impedance", "波阻抗反演场", "×10⁶ kg/(m²·s)", res.ai_inv, 1.0e6, "viridis"),
        ("uncertainty", "融合不确定性", "MPa", sigma_F, 1.0, "magma"),
        ("fusion_weight", "钻孔权重", "—", w3, 1.0, "magma"),
    ]


def wells_from_result(ds, res) -> list[dict]:
    """Along-hole samples: true / MWD / seismic / fused UCS plus MWD channels."""
    profile = _profile_well_titles(ds)
    ai_pts = res.ai_inv[ds.hole_ix, ds.hole_iy, ds.hole_iz]
    wells: dict[str, dict] = {}
    for k in range(ds.ucs_at_holes.size):
        ix, iy, iz = int(ds.hole_ix[k]), int(ds.hole_iy[k]), int(ds.hole_iz[k])
        wid = f"W{ix:02d}-{iy:02d}"
        title = profile.get((ix, iy))
        w = wells.setdefault(wid, {
            "id": wid, "x": float(ds.gx[ix]), "y": float(ds.gy[iy]),
            "ix": ix, "iy": iy, "samples": [],
            "profile": title is not None,
            "title_zh": title or "",
        })
        ucs_mwd = float(res.S_M[ix, iy, iz])
        w["samples"].append({
            "z": float(ds.gz[iz]),
            "V": float(ds.V[k]), "N": float(ds.N[k]),
            "M": float(ds.M[k]), "F": float(ds.F[k]),
            "ucs_true": float(ds.ucs_at_holes[k]),
            "ucs_pred": ucs_mwd,
            "ucs_mwd": ucs_mwd,
            "ucs_seis": float(res.S_Z[ix, iy, iz]),
            "ucs_fused": float(res.S_F[ix, iy, iz]),
            "ai": float(ai_pts[k]),
        })
    for w in wells.values():
        w["samples"].sort(key=lambda s: -s["z"])  # top (0) -> bottom
    return list(wells.values())


def write_frontend_dataset(ds, res, outdir: str, n_holes: int, seed: int) -> dict:
    fields_dir = os.path.join(outdir, "fields")
    os.makedirs(fields_dir, exist_ok=True)
    nx, ny, nz = ds.ucs_true.shape
    fields = fields_from_result(ds, res)

    field_meta = []
    for key, name_zh, unit, vol, scale, cmap in fields:
        _write_field(os.path.join(fields_dir, f"{key}.bin"), vol)
        field_meta.append({
            "key": key, "name_zh": name_zh, "unit": unit,
            "file": f"fields/{key}.bin", "scale": scale, "default_cmap": cmap,
            "min": float(np.min(vol)), "max": float(np.max(vol)),
        })
        print(f"   field {key:18s} range {vol.min():.4g}..{vol.max():.4g}")

    wells = wells_from_result(ds, res)
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
        "n_holes": int(n_holes),
        "seed": int(seed),
        "fusion": "ked",
        "default_field": "fused_strength",
        "compare": {
            "keys": list(COMPARE_KEYS),
            "titles_zh": list(COMPARE_TITLES),
            "residual_titles_zh": list(COMPARE_RESIDUAL_TITLES),
        },
        "view_window": dict(VIEW_WINDOW),
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "boreholes.json"), "w") as f:
        json.dump({"wells": wells}, f, ensure_ascii=False, indent=2)
    print(f">> Wrote {len(field_meta)} fields, {len(wells)} wells to {outdir}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="frontend/public/data")
    parser.add_argument("--n-holes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(">> Generating fixed mine dataset + running pipeline ...")
    ds = generate_mine(n_holes=args.n_holes, seed=args.seed)
    res = run_fusion_pipeline(ds, seed=args.seed)
    write_frontend_dataset(ds, res, args.outdir, args.n_holes, args.seed)


if __name__ == "__main__":
    main()
