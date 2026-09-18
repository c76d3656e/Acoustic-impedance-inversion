r"""Generate the tracked publication figures under ``docs/images/``.

Produces:

* fusion-advantage panels (truth / MWD / seismic / fused, plus pred−truth heatmaps)
* a nested well-count series (1 hole, 2 holes, ... all holes) for MWD-only
  interpolation versus impedance-fused strength
* RMSE / R² curves and a CSV of metrics

Chinese labels require a CJK font (``scripts/setup_fonts.sh`` or
``AI_INVERSION_CJK_FONT``).  Run::

    python3 examples/run_docs_figures.py --outdir docs/images
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from datasets import generate_mine, subset_holes, unique_hole_xy_indices
from fusion import run_fusion_pipeline
from validation import summary, rmse as rmse_score, r2_score
from visualization import (
    configure_cjk_font,
    plot_report_slice,
    plot_slice_grid,
    plot_field_residual_grid,
)
from visualization.report_style import _draw_holes, _style_slice_ax


def nearest_index(axis, value):
    return int(np.argmin(np.abs(np.asarray(axis) - value)))


def holes_xy(ds):
    return np.unique(ds.hole_xyz[:, :2], axis=0)


def far_xy_mask(ds, radius: float) -> np.ndarray:
    """``True`` at (x, y) cells farther than ``radius`` m from every used hole."""
    xx, yy = np.meshgrid(ds.gx, ds.gy, indexing="ij")
    hx, hy = holes_xy(ds).T
    dist = np.sqrt(
        (xx[:, :, None] - hx[None, None, :]) ** 2
        + (yy[:, :, None] - hy[None, None, :]) ** 2
    ).min(axis=2)
    return dist > radius


def masked_metrics(true, pred, mask_xy) -> dict:
    t = np.asarray(true)[mask_xy]
    p = np.asarray(pred)[mask_xy]
    if t.size == 0:
        return {"R2": float("nan"), "RMSE": float("nan")}
    return {"R2": r2_score(t, p), "RMSE": rmse_score(t, p)}


def plot_metrics_curves(rows, outfile, far_radius: float = 12.0):
    configure_cjk_font()
    n = [r["n_holes"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)

    axes[0].plot(n, [r["mwd_rmse"] for r in rows], "o-", label="仅钻孔克里金插值")
    axes[0].plot(n, [r["seis_rmse"] for r in rows], "s--", label="仅波阻抗标定")
    axes[0].plot(n, [r["fused_rmse"] for r in rows], "D-", label="协克里金融合")
    if np.isfinite(rows[0].get("mwd_far_rmse", np.nan)):
        axes[0].plot(n, [r["mwd_far_rmse"] for r in rows], "o:", color="C0",
                     alpha=0.7, label=f"仅钻孔（距孔 > {far_radius:.0f} m）")
        axes[0].plot(n, [r["fused_far_rmse"] for r in rows], "D:", color="C2",
                     alpha=0.7, label=f"融合（距孔 > {far_radius:.0f} m）")
    axes[0].plot(n, [r["hole_seis_rmse"] for r in rows], "s:", color="C1",
                 alpha=0.85, label="仅波阻抗（孔轨迹上）")
    axes[0].plot(n, [r["hole_fused_rmse"] for r in rows], "D:", color="C2",
                 alpha=0.85, label="融合（孔轨迹上）")
    axes[0].set_xlabel("钻孔数量")
    axes[0].set_ylabel("RMSE (MPa)")
    axes[0].set_title("体积均方根误差随钻孔数变化")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(n, [r["mwd_r2"] for r in rows], "o-", label="仅钻孔克里金插值")
    axes[1].plot(n, [r["seis_r2"] for r in rows], "s--", label="仅波阻抗标定")
    axes[1].plot(n, [r["fused_r2"] for r in rows], "D-", label="协克里金融合")
    axes[1].set_xlabel("钻孔数量")
    axes[1].set_ylabel("$R^2$")
    axes[1].set_title("与真值的决定系数随钻孔数变化")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(-0.05, 1.0)

    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_compare_rows(ds_list, mwd_fields, fused_fields, true_field,
                      gx, gy, z_index, vmin, vmax, err_vmax, outfile,
                      elevation: float | None = None):
    """Three rows: MWD-only, fused, |fused − truth|; columns = selected n_holes."""
    configure_cjk_font()
    n_cols = len(ds_list)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    lev = np.linspace(vmin, vmax, 20)
    err_lev = np.linspace(0.0, err_vmax, 20)
    fig, axes = plt.subplots(
        3, n_cols, figsize=(3.5 * n_cols + 1.0, 9.4), constrained_layout=True,
    )
    row_cmaps = [
        (mwd_fields, lev, "viridis", "仅钻孔克里金插值"),
        (fused_fields, lev, "viridis", "钻孔 + 波阻抗融合"),
        ([np.abs(f - true_field) for f in fused_fields], err_lev, "magma",
         "融合绝对误差"),
    ]
    last_cf = [None, None, None]
    for r, (fields, levels, cmap, row_title) in enumerate(row_cmaps):
        for c, (field, ds) in enumerate(zip(fields, ds_list)):
            ax = axes[r, c]
            data = np.asarray(field)[:, :, z_index]
            last_cf[r] = ax.contourf(
                xx, yy, data, levels=levels, cmap=cmap, extend="both",
            )
            ax.contour(xx, yy, data, levels=levels, colors="k",
                       linewidths=0.3, linestyles="--", alpha=0.4)
            _draw_holes(ax, holes_xy(ds), s_outer=55, s_inner=8)
            n_h = int(ds.meta["n_holes"])
            title = f"{n_h} 口钻孔" if r == 0 else ""
            _style_slice_ax(ax, gx, gy, title)
            if c != 0:
                ax.set_ylabel("")
            if r != 2:
                ax.set_xlabel("")
            if c == 0:
                ax.set_ylabel(f"{row_title}\nY (m)")
    for r, label in enumerate(["UCS (MPa)", "UCS (MPa)", "绝对误差 (MPa)"]):
        cbar = fig.colorbar(last_cf[r], ax=axes[r, :].tolist(), shrink=0.9, pad=0.02)
        cbar.set_label(label)
    elev_txt = f"{elevation:.0f} m" if elevation is not None else "水平切片"
    fig.suptitle(f"逐步增加钻孔：稀疏插值 vs. 波阻抗约束融合（标高 {elev_txt}）", fontsize=13)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def write_metrics(rows, outdir):
    csv_path = os.path.join(outdir, "well_count_metrics.csv")
    json_path = os.path.join(outdir, "well_count_metrics.json")
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return csv_path, json_path


def hole_trace_rmse(ds, field) -> float:
    return float(rmse_score(ds.ucs_at_holes, np.asarray(field)[ds.hole_ix, ds.hole_iy, ds.hole_iz]))


def typical_hole_spacing(ds) -> float:
    """Median nearest-neighbour hole spacing in metres."""
    xy = holes_xy(ds)
    if len(xy) < 2:
        return 12.0
    dmin = []
    for i in range(len(xy)):
        d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        d[i] = np.inf
        dmin.append(float(d.min()))
    return float(np.median(dmin))


def plot_hole_layout(ds, z_index, outfile):
    """Plan-view 梅花 lattice on the UCS slice, with short triangular edges."""
    from scipy.spatial import Delaunay

    configure_cjk_font()
    xy = holes_xy(ds)
    xx, yy = np.meshgrid(ds.gx, ds.gy, indexing="ij")
    data = np.asarray(ds.ucs_true)[:, :, z_index]
    vmin, vmax = float(data.min()), float(data.max())
    lev = np.linspace(vmin, vmax, 20)
    fig, ax = plt.subplots(figsize=(6.2, 8.0), constrained_layout=True)
    cf = ax.contourf(xx, yy, data, levels=lev, cmap="viridis", extend="both")
    ax.contour(xx, yy, data, levels=lev, colors="k", linewidths=0.3,
               linestyles="--", alpha=0.35)
    if len(xy) >= 3:
        tri = Delaunay(xy)
        max_edge = 1.35 * typical_hole_spacing(ds)
        drawn = set()
        for simplex in tri.simplices:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                i, j = int(simplex[a]), int(simplex[b])
                key = (min(i, j), max(i, j))
                if key in drawn:
                    continue
                p, q = xy[i], xy[j]
                if np.hypot(p[0] - q[0], p[1] - q[1]) <= max_edge:
                    ax.plot([p[0], q[0]], [p[1], q[1]], color="white",
                            lw=1.2, alpha=0.85, zorder=3)
                    ax.plot([p[0], q[0]], [p[1], q[1]], color="k",
                            lw=0.6, alpha=0.9, zorder=4)
                    drawn.add(key)
    _draw_holes(ax, xy, s_outer=90, s_inner=12)
    _style_slice_ax(ax, ds.gx, ds.gy, "梅花布孔（三角网格近似均匀采样）")
    cbar = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.03)
    cbar.set_label("UCS 真值 (MPa)")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_along_holes(ds, res, outfile, n_show: int = 2):
    """UCS vs elevation along holes nearest the alteration halo and hard body."""
    configure_cjk_font()
    pairs = unique_hole_xy_indices(ds)
    xy = np.column_stack([ds.gx[pairs[:, 0]], ds.gy[pairs[:, 1]]])
    targets = []
    labels = []
    if ds.meta.get("alter_xy") is not None:
        targets.append(np.asarray(ds.meta["alter_xy"], dtype=float))
        labels.append("蚀变晕附近钻孔（力学残差）")
    if ds.meta.get("hard_xy") is not None:
        targets.append(np.asarray(ds.meta["hard_xy"], dtype=float))
        labels.append("硬矿体附近钻孔")
    if not targets:
        targets = [xy[0]]
        labels = ["钻孔 1"]
    used = set()
    chosen = []
    titles = []
    for tgt, lab in zip(targets, labels):
        d = np.sqrt(((xy - tgt) ** 2).sum(axis=1))
        for idx in np.argsort(d):
            key = tuple(pairs[idx])
            if key not in used:
                used.add(key)
                chosen.append(pairs[idx])
                titles.append(lab)
                break
    n_show = min(n_show, len(chosen))
    chosen, titles = chosen[:n_show], titles[:n_show]
    fig, axes = plt.subplots(1, n_show, figsize=(5.2 * n_show, 5.6),
                             constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        i, j = chosen[k]
        mask = (ds.hole_ix == i) & (ds.hole_iy == j)
        z = ds.hole_xyz[mask, 2]
        order = np.argsort(z)
        z = z[order]
        true = ds.ucs_at_holes[mask][order]
        sm = res.S_M[i, j, ds.hole_iz[mask]][order]
        sz = res.S_Z[i, j, ds.hole_iz[mask]][order]
        sf = res.S_F[i, j, ds.hole_iz[mask]][order]
        ax.plot(true, z, "k-", lw=2.0, label="真值 UCS")
        ax.plot(sz, z, "--", color="C1", lw=1.8, label="仅波阻抗标定")
        ax.plot(sm, z, ":", color="C0", lw=1.8, label="MWD（孔点）")
        ax.plot(sf, z, "-", color="C2", lw=2.0, label="协克里金融合")
        ax.set_xlabel("UCS (MPa)")
        ax.set_title(titles[k] if k < len(titles) else f"钻孔 {k+1}")
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        if k == 0:
            ax.set_ylabel("标高 (m)")
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("沿孔剖面：融合在孔上钉回真值；波阻抗只给出趋势", fontsize=13)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_mwd_weight(ds, res, z_index, outfile):
    """Where fusion actually listens to boreholes (Doyen primary weight)."""
    w_xy = getattr(res, "w_anchor", None)
    if w_xy is None:
        tau_m = 1.0 / (res.var_M + 1e-12)
        tau_z = 1.0 / (res.var_Z + 1e-12)
        w_m = tau_m / (tau_m + tau_z)
    else:
        w_m = np.broadcast_to(
            np.asarray(w_xy, dtype=float)[:, :, np.newaxis], res.S_F.shape,
        ).copy()
    elev = float(ds.gz[z_index])
    plot_report_slice(
        w_m, ds.gx, ds.gy, z_index, outfile,
        title=rf"{elev:.0f} m 标高融合权重 $w_{{\mathrm{{MWD}}}}$（越亮越信钻孔）",
        cbar_label=r"$w_{\mathrm{MWD}}$", holes_xy=holes_xy(ds),
        cmap="magma", vmin=0.0, vmax=1.0,
    )
    return outfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="docs/images")
    parser.add_argument("--n-holes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--far-radius", type=float, default=None,
                        help="Far-field mask (m). Default: 0.4 × median 梅花 hole spacing.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    configure_cjk_font()

    print(">> Generating co-located synthetic mine ...")
    ds_all = generate_mine(n_holes=args.n_holes, seed=args.seed)
    spacing = typical_hole_spacing(ds_all)
    if args.far_radius is None:
        args.far_radius = max(5.0, 0.40 * spacing)
    xy_all = holes_xy(ds_all)
    pairs = unique_hole_xy_indices(ds_all)
    print(
        f"   梅花 n={args.n_holes}  median spacing {spacing:.1f} m  "
        f"span X {float(xy_all[:, 0].max() - xy_all[:, 0].min()):.1f} m / "
        f"Y {float(xy_all[:, 1].max() - xy_all[:, 1].min()):.1f} m  "
        f"corr(AI,UCS)={ds_all.meta['corr_ai_ucs']:.3f}  "
        f"far-radius {args.far_radius:.1f} m"
    )
    for k, (ix, iy) in enumerate(pairs, 1):
        print(f"     hole {k:2d}  x={ds_all.gx[ix]:6.2f}  y={ds_all.gy[iy]:6.2f}")
    print(">> Running full dual-branch pipeline (impedance inverted once) ...")
    res_all = run_fusion_pipeline(ds_all, seed=args.seed)
    ai_inv = res_all.ai_inv
    zi = nearest_index(ds_all.gz, args.elevation)
    elev = float(ds_all.gz[zi])
    print(f"   slice elevation {elev:.0f} m (index {zi})")

    vmin = float(ds_all.ucs_true.min())
    vmax = float(ds_all.ucs_true.max())
    holes_all = holes_xy(ds_all)

    # ----- Fusion-advantage panels (all holes) -----------------------------
    print(">> Fusion-advantage figures ...")
    plot_report_slice(
        ds_all.ucs_true, ds_all.gx, ds_all.gy, zi,
        os.path.join(args.outdir, "ground_truth_strength.png"),
        title=f"{elev:.0f} m 标高岩石强度真值 (MPa)",
        cbar_label="UCS (MPa)", holes_xy=holes_all, vmin=vmin, vmax=vmax,
    )
    plot_report_slice(
        res_all.S_M, ds_all.gx, ds_all.gy, zi,
        os.path.join(args.outdir, "mwd_only_strength.png"),
        title=f"{elev:.0f} m 标高仅钻孔克里金插值强度场 (MPa)",
        cbar_label="UCS (MPa)", holes_xy=holes_all, vmin=vmin, vmax=vmax,
    )
    plot_report_slice(
        res_all.S_Z, ds_all.gx, ds_all.gy, zi,
        os.path.join(args.outdir, "impedance_derived_strength.png"),
        title=f"{elev:.0f} m 标高波阻抗标定强度场 (MPa)",
        cbar_label="UCS (MPa)", holes_xy=holes_all, vmin=vmin, vmax=vmax,
    )
    plot_report_slice(
        res_all.S_F, ds_all.gx, ds_all.gy, zi,
        os.path.join(args.outdir, "fused_strength.png"),
        title=f"{elev:.0f} m 标高协克里金融合强度场 (MPa)",
        cbar_label="UCS (MPa)", holes_xy=holes_all, vmin=vmin, vmax=vmax,
    )
    plot_report_slice(
        res_all.ai_inv, ds_all.gx, ds_all.gy, zi,
        os.path.join(args.outdir, "impedance_slice.png"),
        title=rf"{elev:.0f} m 标高波阻抗反演场 (×$10^6$ kg/(m$^2\cdot$s))",
        cbar_label=r"波阻抗 (×$10^6$ kg/(m$^2\cdot$s))",
        holes_xy=holes_all, scale=1e6,
    )
    plot_field_residual_grid(
        ds_all.ucs_true,
        [
            (res_all.S_M, "(b) 仅钻孔插值", "(e) 仅钻孔 $-$ 真值"),
            (res_all.S_Z, "(c) 仅波阻抗标定", "(f) 仅波阻抗 $-$ 真值"),
            (res_all.S_F, "(d) 协克里金融合", "(g) 融合 $-$ 真值"),
        ],
        ds_all.gx, ds_all.gy, zi,
        os.path.join(args.outdir, "fusion_advantage.png"),
        vmin=vmin, vmax=vmax, holes_xy=holes_all,
        true_title="(a) 强度真值",
        suptitle=f"融合优势对比（{elev:.0f} m 标高，{args.n_holes} 口钻孔，50×80 m）",
    )
    plot_along_holes(
        ds_all, res_all, os.path.join(args.outdir, "along_hole_profiles.png"),
    )
    plot_mwd_weight(
        ds_all, res_all, zi, os.path.join(args.outdir, "mwd_fusion_weight.png"),
    )
    plot_hole_layout(
        ds_all, zi, os.path.join(args.outdir, "hole_layout.png"),
    )

    m_all = {
        "mwd": summary(ds_all.ucs_true, res_all.S_M),
        "seis": summary(ds_all.ucs_true, res_all.S_Z),
        "simple": summary(ds_all.ucs_true, res_all.S_weighted),
        "fused": summary(ds_all.ucs_true, res_all.S_F),
        "ai": summary(ds_all.ai_true, res_all.ai_inv),
    }
    print("   full-hole metrics:")
    for k, s in m_all.items():
        print(f"     {k:<8} R2={s['R2']:.3f}  RMSE={s['RMSE']:.2f}")

    # ----- Nested well-count series ----------------------------------------
    print(">> Nested well-count series ...")
    rows = []
    fused_fields = []
    mwd_fields = []
    err_fields = []
    ds_series = []
    var_m_1 = var_f_1 = None
    for n in range(1, args.n_holes + 1):
        ds = subset_holes(ds_all, n)
        res = run_fusion_pipeline(ds, seed=args.seed, ai_inv=ai_inv)
        if n == 1:
            var_m_1, var_f_1 = res.var_M, res.var_F
        mwd = summary(ds.ucs_true, res.S_M)
        seis = summary(ds.ucs_true, res.S_Z)
        fused = summary(ds.ucs_true, res.S_F)
        far = far_xy_mask(ds, args.far_radius)
        mwd_far = masked_metrics(ds.ucs_true, res.S_M, far)
        fused_far = masked_metrics(ds.ucs_true, res.S_F, far)
        row = {
            "n_holes": n,
            "mwd_r2": mwd["R2"], "mwd_rmse": mwd["RMSE"], "mwd_mae": mwd["MAE"],
            "seis_r2": seis["R2"], "seis_rmse": seis["RMSE"], "seis_mae": seis["MAE"],
            "fused_r2": fused["R2"], "fused_rmse": fused["RMSE"], "fused_mae": fused["MAE"],
            "mwd_far_rmse": mwd_far["RMSE"], "fused_far_rmse": fused_far["RMSE"],
            "hole_mwd_rmse": hole_trace_rmse(ds, res.S_M),
            "hole_seis_rmse": hole_trace_rmse(ds, res.S_Z),
            "hole_fused_rmse": hole_trace_rmse(ds, res.S_F),
        }
        rows.append(row)
        fused_fields.append(res.S_F)
        mwd_fields.append(res.S_M)
        err_fields.append(np.abs(res.S_F - ds.ucs_true))
        ds_series.append(ds)
        print(
            f"   n={n:2d}  MWD RMSE={mwd['RMSE']:.2f}  "
            f"seis RMSE={seis['RMSE']:.2f}  fused RMSE={fused['RMSE']:.2f}  "
            f"far MWD/fused={mwd_far['RMSE']:.2f}/{fused_far['RMSE']:.2f}"
        )

    csv_path, json_path = write_metrics(rows, args.outdir)
    plot_metrics_curves(rows, os.path.join(args.outdir, "metrics_vs_nholes.png"),
                        far_radius=args.far_radius)

    fused_panels = [
        (fld, f"n = {n} 口钻孔", holes_xy(ds))
        for n, fld, ds in zip(range(1, args.n_holes + 1), fused_fields, ds_series)
    ]
    mwd_panels = [
        (fld, f"n = {n} 口钻孔", holes_xy(ds))
        for n, fld, ds in zip(range(1, args.n_holes + 1), mwd_fields, ds_series)
    ]
    err_vmax = float(np.percentile(err_fields[0][:, :, zi], 95))
    err_panels = [
        (fld, f"n = {n} 口钻孔", holes_xy(ds))
        for n, fld, ds in zip(range(1, args.n_holes + 1), err_fields, ds_series)
    ]
    plot_slice_grid(
        fused_panels, ds_all.gx, ds_all.gy, zi,
        os.path.join(args.outdir, "well_series_fused.png"),
        cbar_label="UCS (MPa)", vmin=vmin, vmax=vmax, ncols=4,
        suptitle=f"{elev:.0f} m 标高融合强度场：钻孔由 1 口增至 {args.n_holes} 口",
    )
    plot_slice_grid(
        mwd_panels, ds_all.gx, ds_all.gy, zi,
        os.path.join(args.outdir, "well_series_mwd.png"),
        cbar_label="UCS (MPa)", vmin=vmin, vmax=vmax, ncols=4,
        suptitle=f"{elev:.0f} m 标高仅钻孔克里金插值：钻孔由 1 口增至 {args.n_holes} 口",
    )
    plot_slice_grid(
        err_panels, ds_all.gx, ds_all.gy, zi,
        os.path.join(args.outdir, "well_series_fused_error.png"),
        cbar_label="绝对误差 (MPa)", vmin=0.0, vmax=err_vmax, cmap="magma",
        ncols=4,
        suptitle=f"{elev:.0f} m 标高融合绝对误差 |S_F − S_true|",
    )

    pick = [1, 3, 6, args.n_holes]
    pick_idx = [k - 1 for k in pick]
    plot_compare_rows(
        [ds_series[i] for i in pick_idx],
        [mwd_fields[i] for i in pick_idx],
        [fused_fields[i] for i in pick_idx],
        ds_all.ucs_true,
        ds_all.gx, ds_all.gy, zi, vmin, vmax, err_vmax,
        os.path.join(args.outdir, "well_series_compare.png"),
        elevation=elev,
    )

    # 1-hole vs all-holes kriging variance (why interpolation is untrusted far away)
    plot_slice_grid(
        [
            (np.sqrt(var_m_1), "1 口钻孔 · MWD 不确定度", holes_xy(ds_series[0])),
            (np.sqrt(res_all.var_M),
             f"{args.n_holes} 口钻孔 · MWD 不确定度", holes_all),
            (np.sqrt(var_f_1), "1 口钻孔 · 融合不确定度", holes_xy(ds_series[0])),
            (np.sqrt(res_all.var_F),
             f"{args.n_holes} 口钻孔 · 融合不确定度", holes_all),
        ],
        ds_all.gx, ds_all.gy, zi,
        os.path.join(args.outdir, "uncertainty_1_vs_all.png"),
        cbar_label=r"$\sigma$ (MPa)",
        vmin=0.0,
        vmax=float(np.percentile(np.sqrt(res_all.var_M[:, :, zi]), 98)),
        cmap="magma", ncols=4,
        suptitle="钻孔稀疏处克里金方差大，协克里金改信连续波阻抗场",
    )

    with open(os.path.join(args.outdir, "full_hole_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump({k: {mk: float(mv) for mk, mv in s.items()} for k, s in m_all.items()},
                  f, indent=2, ensure_ascii=False)

    print("\n=== docs figures complete ===")
    print(f"   outdir : {args.outdir}")
    print(f"   metrics: {csv_path}")
    print(f"            {json_path}")


if __name__ == "__main__":
    main()
