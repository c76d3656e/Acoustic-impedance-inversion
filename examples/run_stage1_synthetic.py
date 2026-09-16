"""Stage I end-to-end pipeline: synthetic 3-D acoustic impedance inversion.

    Vp, rho  ->  AI_true  ->  reflectivity  ->  synthetic seismic (+noise)
             ->  low-frequency background  ->  model-based inversion
             ->  AI_inv volume  ->  metrics + horizontal slice map

Run:  python examples/run_stage1_synthetic.py --outdir results
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from datasets import layered_property_model, acoustic_impedance
from inversion import ricker, synthetic_seismic_volume, invert_volume, background_model
from validation import summary
from validation.blind_well import blind_well_report
from visualization import (
    plot_impedance_map,
    plot_cross_section_comparison,
    render_volume,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=220)
    parser.add_argument("--dt", type=float, default=0.002, help="sample interval (s)")
    parser.add_argument("--freq", type=float, default=30.0, help="wavelet Hz")
    parser.add_argument("--noise", type=float, default=0.05, help="noise fraction")
    parser.add_argument("--lam", type=float, default=5.0, help="smoothness weight")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(">> Building synthetic earth model ...")
    vp, rho = layered_property_model(shape=(args.nx, args.ny, args.nz), seed=args.seed)
    ai_true = acoustic_impedance(vp, rho)
    print(f"   model shape        : {ai_true.shape}")
    print(f"   AI range           : {ai_true.min():.3e} .. {ai_true.max():.3e}")

    print(">> Forward modelling synthetic seismic ...")
    wavelet = ricker(n=81, dt=args.dt, freq=args.freq)
    seismic = synthetic_seismic_volume(ai_true, wavelet)
    if args.noise > 0:
        seismic = seismic + args.noise * np.std(seismic) * rng.standard_normal(seismic.shape)

    print(">> Building low-frequency background model ...")
    background = background_model(ai_true, sigma=(3, 3, 14))

    print(">> Running model-based inversion (trace-by-trace) ...")
    t0 = time.time()
    ai_inv = invert_volume(seismic, wavelet, background, lam=args.lam)
    print(f"   inverted {args.nx * args.ny} traces in {time.time() - t0:.2f} s")

    print(">> Validation metrics (whole volume) ...")
    stats = summary(ai_true, ai_inv)
    for k, v in stats.items():
        print(f"   {k:5s}: {v:.5g}")

    ix, iy = int(args.nx * 0.6), int(args.ny * 0.4)
    blind = blind_well_report(ai_true, ai_inv, ix, iy)
    print(f">> Blind-well @ (il={ix}, xl={iy}): "
          f"R2={blind['R2']:.4f} RMSE={blind['RMSE']:.3e}")

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "AI_3D.npy"), ai_inv)
    np.save(os.path.join(args.outdir, "AI_true.npy"), ai_true)
    z_index = args.nz // 2
    np.save(os.path.join(args.outdir, "AI_slice.npy"), ai_inv[:, :, z_index])

    print(">> Rendering figures ...")
    map_true = plot_impedance_map(
        ai_true, z_index, os.path.join(figdir, "impedance_map_true.png"),
        title=f"True acoustic impedance  (z-index {z_index})",
    )
    map_inv = plot_impedance_map(
        ai_inv, z_index, os.path.join(figdir, "impedance_map_inverted.png"),
        title=f"Inverted acoustic impedance  (z-index {z_index})",
    )
    xsec = plot_cross_section_comparison(
        ai_true, ai_inv, iy, os.path.join(figdir, "cross_section_comparison.png")
    )
    vol_png = render_volume(ai_inv, os.path.join(figdir, "impedance_volume_3d.png"))

    print("\n=== Stage I complete ===")
    print(f"   volume (inverted)  : {os.path.join(args.outdir, 'AI_3D.npy')}")
    print(f"   slice map (true)   : {map_true}")
    print(f"   slice map (inv)    : {map_inv}")
    print(f"   cross-section      : {xsec}")
    print(f"   3-D render         : {vol_png if vol_png else 'skipped (no GPU/OSMesa)'}")
    print(f"   R2={stats['R2']:.4f}  RMSE={stats['RMSE']:.4g}  MAE={stats['MAE']:.4g}")


if __name__ == "__main__":
    main()
