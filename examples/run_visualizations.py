r"""Generate the full visualization suite for the fusion project.

Outputs (in ``results/viz``):

* Report-style 2-D horizontal slices (like the mine report figures) for the
  seismic **impedance** inversion and the **MWD** strength inversion (and fused).
* Static 3-D renders (PyVista) of impedance / MWD / fused volumes.
* Interactive 3-D HTML pages (Plotly) -- volume, isosurface and slice stacks.

Run: python examples/run_visualizations.py --outdir results/viz
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from datasets import generate_mine
from fusion import run_fusion_pipeline
from mwd import mwd_features, PhysicsGuidedGPR
from geostats import ordinary_kriging_3d
from visualization import (
    plot_report_slice,
    render_volume,
    render_isosurface,
    export_volume_html,
    export_isosurface_html,
    export_slices_html,
)


def nearest_index(axis, value):
    return int(np.argmin(np.abs(np.asarray(axis) - value)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/viz")
    parser.add_argument("--n-holes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elevation", type=float, default=-60.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    html_dir = os.path.join(args.outdir, "interactive")
    os.makedirs(html_dir, exist_ok=True)

    print(">> Generating mine + running dual-branch pipeline ...")
    ds = generate_mine(n_holes=args.n_holes, seed=args.seed)
    res = run_fusion_pipeline(ds, seed=args.seed)
    holes_xy = np.unique(ds.hole_xyz[:, :2], axis=0)

    # Continuous MWD field for display: ordinary kriging of the PG-GPR UCS point
    # predictions (IDW-style borehole interpolation, like the report figures).
    X = mwd_features(ds.V, ds.N, ds.M, ds.F)
    ucs_pts, _ = PhysicsGuidedGPR().fit(X, ds.ucs_at_holes).predict(X)
    mwd_field, _ = ordinary_kriging_3d(ds.hole_xyz, ucs_pts, ds.gx, ds.gy, ds.gz)
    zi = nearest_index(ds.gz, args.elevation)
    elev = ds.gz[zi]
    print(f"   elevation slice: {elev:.0f} m (index {zi})")

    # ---------------- Report-style 2-D slices ---------------------------
    print(">> Report-style 2-D slices ...")
    f_imp = plot_report_slice(
        res.ai_inv, ds.gx, ds.gy, zi,
        os.path.join(args.outdir, "impedance_slice.png"),
        title=rf"{elev:.0f}m标高波阻抗反演场分布 (×$10^6$ kg/(m$^2\cdot$s))",
        cbar_label=r"波阻抗 (×$10^6$ kg/(m$^2\cdot$s))",
        holes_xy=holes_xy, scale=1e6,
    )
    f_mwd = plot_report_slice(
        mwd_field, ds.gx, ds.gy, zi,
        os.path.join(args.outdir, "mwd_strength_slice.png"),
        title=f"{elev:.0f}m标高MWD反演岩石强度场分布 (MPa)",
        cbar_label="UCS (MPa)", holes_xy=holes_xy,
    )
    f_fused = plot_report_slice(
        res.S_F, ds.gx, ds.gy, zi,
        os.path.join(args.outdir, "fused_strength_slice.png"),
        title=f"{elev:.0f}m标高融合岩石强度场分布 (MPa)",
        cbar_label="UCS (MPa)", holes_xy=holes_xy,
    )

    # ---------------- Static 3-D renders --------------------------------
    print(">> Static 3-D renders (PyVista) ...")
    v_imp = render_volume(res.ai_inv, os.path.join(args.outdir, "impedance_volume_3d.png"))
    v_mwd = render_isosurface(mwd_field, os.path.join(args.outdir, "mwd_strength_iso_3d.png"))
    v_fused = render_volume(res.S_F, os.path.join(args.outdir, "fused_strength_volume_3d.png"))

    # ---------------- Interactive 3-D HTML ------------------------------
    print(">> Interactive 3-D HTML (Plotly) ...")
    elev_idx = [nearest_index(ds.gz, e) for e in (-24, -48, -72, -96)]
    h1 = export_volume_html(
        res.ai_inv, ds.gx, ds.gy, ds.gz,
        os.path.join(html_dir, "impedance_volume.html"),
        title="Acoustic impedance volume", colorbar_title="AI",
    )
    h2 = export_isosurface_html(
        res.S_F, ds.gx, ds.gy, ds.gz,
        os.path.join(html_dir, "fused_strength_isosurface.html"),
        title="Fused rock-strength isosurfaces", colorbar_title="UCS (MPa)",
    )
    h3 = export_volume_html(
        mwd_field, ds.gx, ds.gy, ds.gz,
        os.path.join(html_dir, "mwd_strength_volume.html"),
        title="MWD-derived strength volume", colorbar_title="UCS (MPa)",
    )
    h4 = export_slices_html(
        res.S_F, ds.gx, ds.gy, ds.gz,
        os.path.join(html_dir, "fused_strength_slices.html"),
        elevations_idx=elev_idx,
        title="Fused strength - horizontal slices", colorbar_title="UCS (MPa)",
    )

    print("\n=== Visualization suite complete ===")
    for f in [f_imp, f_mwd, f_fused, v_imp, v_mwd, v_fused, h1, h2, h3, h4]:
        print(f"   {f if f else '(3-D render skipped)'}")


if __name__ == "__main__":
    main()
