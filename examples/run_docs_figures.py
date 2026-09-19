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

from datasets import (
    centered_face,
    generate_mine,
    subset_holes,
    unique_hole_xy_indices,
)
from fusion import run_fusion_pipeline
from validation import summary, rmse as rmse_score, r2_score
from visualization import (
    configure_cjk_font,
    plot_report_slice,
    plot_slice_grid,
    plot_field_residual_grid,
    plot_orthogonal_trislices,
    plot_spherical_variogram,
    plot_working_face,
)
from visualization.report_style import _draw_holes, _style_slice_ax


def nearest_index(axis, value):
    return int(np.argmin(np.abs(np.asarray(axis) - value)))


def nearest_range(axis, lo, hi):
    """Inclusive index range snapped to the samples nearest ``lo`` / ``hi``."""
    i0 = nearest_index(axis, lo)
    i1 = nearest_index(axis, hi)
    if i1 < i0:
        i0, i1 = i1, i0
    if i1 <= i0:
        i1 = min(len(np.asarray(axis)) - 1, i0 + 1)
    return i0, i1


def holes_xy(ds):
    return np.unique(ds.hole_xyz[:, :2], axis=0)


def far_xy_mask(ds, radius: float, gx=None, gy=None) -> np.ndarray:
    """``True`` at (x, y) cells farther than ``radius`` m from every used hole."""
    gx = ds.gx if gx is None else np.asarray(gx)
    gy = ds.gy if gy is None else np.asarray(gy)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
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

    axes[0].plot(n, [r["mwd_rmse"] for r in rows], "o-", label="钻孔插值")
    axes[0].plot(n, [r["seis_rmse"] for r in rows], "s--", label="波阻抗插值")
    axes[0].plot(n, [r["fused_rmse"] for r in rows], "D-", label="融合插值")
    if np.isfinite(rows[0].get("mwd_far_rmse", np.nan)):
        axes[0].plot(n, [r["mwd_far_rmse"] for r in rows], "o:", color="C0",
                     alpha=0.7, label=f"钻孔插值（距孔 > {far_radius:.0f} m）")
        axes[0].plot(n, [r["fused_far_rmse"] for r in rows], "D:", color="C2",
                     alpha=0.7, label=f"融合（距孔 > {far_radius:.0f} m）")
    axes[0].plot(n, [r["hole_seis_rmse"] for r in rows], "s:", color="C1",
                 alpha=0.85, label="波阻抗插值（孔轨迹上）")
    axes[0].plot(n, [r["hole_fused_rmse"] for r in rows], "D:", color="C2",
                 alpha=0.85, label="融合（孔轨迹上）")
    axes[0].set_xlabel("钻孔数量")
    axes[0].set_ylabel("RMSE (MPa)")
    axes[0].set_title("体积均方根误差随钻孔数变化")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(n, [r["mwd_r2"] for r in rows], "o-", label="钻孔插值")
    axes[1].plot(n, [r["seis_r2"] for r in rows], "s--", label="波阻抗插值")
    axes[1].plot(n, [r["fused_r2"] for r in rows], "D-", label="融合插值")
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
        (mwd_fields, lev, "viridis", "钻孔插值"),
        (fused_fields, lev, "viridis", "融合插值"),
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


def plot_hole_layout(ds, z_index, outfile, gx=None, gy=None, field=None):
    """Plan-view 梅花 lattice on the UCS slice, with short triangular edges."""
    from scipy.spatial import Delaunay

    configure_cjk_font()
    gx = ds.gx if gx is None else np.asarray(gx)
    gy = ds.gy if gy is None else np.asarray(gy)
    vol = ds.ucs_true if field is None else field
    xy = holes_xy(ds)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    data = np.asarray(vol)[:, :, z_index]
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
    _style_slice_ax(ax, gx, gy, "梅花布孔（20×50 m 工作面，三角网格近似均匀采样）")
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
        ax.plot(sz, z, "--", color="C1", lw=1.8, label="波阻抗插值")
        ax.plot(sm, z, ":", color="C0", lw=1.8, label="钻孔插值")
        ax.plot(sf, z, "-", color="C2", lw=2.0, label="融合插值")
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


def plot_mwd_weight(ds, res, z_index, outfile, gx=None, gy=None, weight=None):
    """Where fusion actually listens to boreholes (Doyen primary weight)."""
    if weight is None:
        w_xy = getattr(res, "w_anchor", None)
        if w_xy is None:
            tau_m = 1.0 / (res.var_M + 1e-12)
            tau_z = 1.0 / (res.var_Z + 1e-12)
            weight = tau_m / (tau_m + tau_z)
        else:
            weight = np.broadcast_to(
                np.asarray(w_xy, dtype=float)[:, :, np.newaxis], res.S_F.shape,
            ).copy()
    gx = ds.gx if gx is None else gx
    gy = ds.gy if gy is None else gy
    elev = float(ds.gz[z_index])
    plot_report_slice(
        weight, gx, gy, z_index, outfile,
        title=f"{elev:.0f} m 标高融合权重 w_MWD（越亮越信钻孔）",
        cbar_label="w_MWD", holes_xy=holes_xy(ds),
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
    bbox = ds_all.meta.get("hole_bbox") or centered_face(ds_all.gx, ds_all.gy)
    win_x0, win_x1, win_y0, win_y1 = (float(v) for v in bbox)
    win_w, win_h = win_x1 - win_x0, win_y1 - win_y0
    ix0, ix1 = nearest_range(ds_all.gx, win_x0, win_x1)
    iy0, iy1 = nearest_range(ds_all.gy, win_y0, win_y1)
    gx_w = ds_all.gx[ix0:ix1 + 1]
    gy_w = ds_all.gy[iy0:iy1 + 1]

    def crop(vol):
        return np.asarray(vol)[ix0:ix1 + 1, iy0:iy1 + 1]

    print(
        f"   梅花 n={args.n_holes}  median spacing {spacing:.1f} m  "
        f"span X {float(xy_all[:, 0].max() - xy_all[:, 0].min()):.1f} m / "
        f"Y {float(xy_all[:, 1].max() - xy_all[:, 1].min()):.1f} m  "
        f"window {win_w:.0f}×{win_h:.0f} m  "
        f"[{gx_w[0]:.1f},{gx_w[-1]:.1f}]×[{gy_w[0]:.1f},{gy_w[-1]:.1f}]  "
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
    holes_win = holes_all[
        (holes_all[:, 0] >= gx_w[0] - 1e-6)
        & (holes_all[:, 0] <= gx_w[-1] + 1e-6)
        & (holes_all[:, 1] >= gy_w[0] - 1e-6)
        & (holes_all[:, 1] <= gy_w[-1] + 1e-6)
    ]
    true_w = crop(ds_all.ucs_true)
    sm_w = crop(res_all.S_M)
    sz_w = crop(res_all.S_Z)
    sf_w = crop(res_all.S_F)
    ai_w = crop(res_all.ai_inv)
    face_txt = f"{win_w:.0f}×{win_h:.0f} m"

    # ----- Fusion-advantage panels (working-face window) -------------------
    print(">> Fusion-advantage figures ...")
    plot_report_slice(
        true_w, gx_w, gy_w, zi,
        os.path.join(args.outdir, "ground_truth_strength.png"),
        title=f"{elev:.0f} m 标高岩石强度真值 (MPa)",
        cbar_label="UCS (MPa)", holes_xy=holes_win, vmin=vmin, vmax=vmax,
    )
    plot_report_slice(
        sm_w, gx_w, gy_w, zi,
        os.path.join(args.outdir, "mwd_only_strength.png"),
        title=f"{elev:.0f} m 标高钻孔插值强度场 (MPa)",
        cbar_label="UCS (MPa)", holes_xy=holes_win, vmin=vmin, vmax=vmax,
    )
    plot_report_slice(
        sz_w, gx_w, gy_w, zi,
        os.path.join(args.outdir, "impedance_derived_strength.png"),
        title=f"{elev:.0f} m 标高波阻抗标定强度场 (MPa)",
        cbar_label="UCS (MPa)", holes_xy=holes_win, vmin=vmin, vmax=vmax,
    )
    plot_report_slice(
        sf_w, gx_w, gy_w, zi,
        os.path.join(args.outdir, "fused_strength.png"),
        title=f"{elev:.0f} m 标高融合插值强度场 (MPa)",
        cbar_label="UCS (MPa)", holes_xy=holes_win, vmin=vmin, vmax=vmax,
    )
    plot_report_slice(
        ai_w, gx_w, gy_w, zi,
        os.path.join(args.outdir, "impedance_slice.png"),
        title=f"{elev:.0f} m 标高波阻抗反演场 (×10⁶ kg/(m²·s))",
        cbar_label="波阻抗 (×10⁶ kg/(m²·s))",
        holes_xy=holes_win, scale=1e6,
    )
    plot_field_residual_grid(
        true_w,
        [
            (sm_w, "(b) 钻孔插值", "(e) 钻孔残差"),
            (sz_w, "(c) 波阻抗插值", "(f) 波阻抗残差"),
            (sf_w, "(d) 融合插值", "(g) 融合残差"),
        ],
        gx_w, gy_w, zi,
        os.path.join(args.outdir, "fusion_advantage.png"),
        vmin=vmin, vmax=vmax, holes_xy=holes_win,
        true_title="(a) 强度真值",
        suptitle=f"融合优势对比（{elev:.0f} m 标高，{args.n_holes} 口钻孔，{face_txt}）",
    )
    plot_along_holes(
        ds_all, res_all, os.path.join(args.outdir, "along_hole_profiles.png"),
    )
    plot_mwd_weight(
        ds_all, res_all, zi, os.path.join(args.outdir, "mwd_fusion_weight.png"),
        gx=gx_w, gy=gy_w, weight=crop(
            np.broadcast_to(
                np.asarray(res_all.w_anchor, dtype=float)[:, :, np.newaxis],
                ds_all.ucs_true.shape,
            ).copy()
        ),
    )
    plot_hole_layout(
        ds_all, zi, os.path.join(args.outdir, "hole_layout.png"),
        gx=gx_w, gy=gy_w, field=true_w,
    )
    plot_working_face(
        ds_all.ucs_true, ds_all.gx, ds_all.gy, zi,
        os.path.join(args.outdir, "working_face_window.png"),
        x0=win_x0, y0=win_y0, width=win_w, height=win_h,
        title=f"{elev:.0f} m 标高工作面窗口（{face_txt} / 全块 50×80 m）",
        holes_xy=holes_all, vmin=vmin, vmax=vmax,
    )
    plot_spherical_variogram(
        os.path.join(args.outdir, "spherical_variogram.png"),
        range_m=max(1.15 * spacing, 8.0),
        title="球状变差函数（残差克里金）",
    )
    plot_orthogonal_trislices(
        sf_w,
        gx_w, gy_w, ds_all.gz,
        ix=(ix0 + ix1) // 2 - ix0,
        iy=(iy0 + iy1) // 2 - iy0,
        iz=zi,
        outfile=os.path.join(args.outdir, "trislices.png"),
        title=f"融合插值三维三正交切面（{face_txt} 工作面，{elev:.0f} m 标高）",
        holes_xy=holes_win, vmin=vmin, vmax=vmax,
    )

    m_all = {
        "mwd": summary(true_w, sm_w),
        "seis": summary(true_w, sz_w),
        "simple": summary(true_w, crop(res_all.S_weighted)),
        "fused": summary(true_w, sf_w),
        "ai": summary(crop(ds_all.ai_true), crop(res_all.ai_inv)),
    }
    print("   working-face metrics:")
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
        mwd = summary(true_w, crop(res.S_M))
        seis = summary(true_w, crop(res.S_Z))
        fused = summary(true_w, crop(res.S_F))
        far = far_xy_mask(ds, args.far_radius, gx=gx_w, gy=gy_w)
        mwd_far = masked_metrics(true_w, crop(res.S_M), far)
        fused_far = masked_metrics(true_w, crop(res.S_F), far)
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
        fused_fields.append(crop(res.S_F))
        mwd_fields.append(crop(res.S_M))
        err_fields.append(np.abs(crop(res.S_F) - true_w))
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
        fused_panels, gx_w, gy_w, zi,
        os.path.join(args.outdir, "well_series_fused.png"),
        cbar_label="UCS (MPa)", vmin=vmin, vmax=vmax, ncols=4,
        suptitle=f"{elev:.0f} m 标高融合强度场（{face_txt}）：钻孔由 1 口增至 {args.n_holes} 口",
    )
    plot_slice_grid(
        mwd_panels, gx_w, gy_w, zi,
        os.path.join(args.outdir, "well_series_mwd.png"),
        cbar_label="UCS (MPa)", vmin=vmin, vmax=vmax, ncols=4,
        suptitle=f"{elev:.0f} m 标高钻孔插值（{face_txt}）：钻孔由 1 口增至 {args.n_holes} 口",
    )
    plot_slice_grid(
        err_panels, gx_w, gy_w, zi,
        os.path.join(args.outdir, "well_series_fused_error.png"),
        cbar_label="绝对误差 (MPa)", vmin=0.0, vmax=err_vmax, cmap="magma",
        ncols=4,
        suptitle=f"{elev:.0f} m 标高融合绝对误差 |S_F − S_true|（{face_txt}）",
    )

    pick = [1, 3, 6, args.n_holes]
    pick_idx = [k - 1 for k in pick]
    plot_compare_rows(
        [ds_series[i] for i in pick_idx],
        [mwd_fields[i] for i in pick_idx],
        [fused_fields[i] for i in pick_idx],
        true_w,
        gx_w, gy_w, zi, vmin, vmax, err_vmax,
        os.path.join(args.outdir, "well_series_compare.png"),
        elevation=elev,
    )

    # 1-hole vs all-holes kriging variance (why interpolation is untrusted far away)
    plot_slice_grid(
        [
            (crop(np.sqrt(var_m_1)), "1 口钻孔 · MWD 不确定度", holes_xy(ds_series[0])),
            (crop(np.sqrt(res_all.var_M)),
             f"{args.n_holes} 口钻孔 · MWD 不确定度", holes_win),
            (crop(np.sqrt(var_f_1)), "1 口钻孔 · 融合不确定度", holes_xy(ds_series[0])),
            (crop(np.sqrt(res_all.var_F)),
             f"{args.n_holes} 口钻孔 · 融合不确定度", holes_win),
        ],
        gx_w, gy_w, zi,
        os.path.join(args.outdir, "uncertainty_1_vs_all.png"),
        cbar_label=r"$\sigma$ (MPa)",
        vmin=0.0,
        vmax=float(np.percentile(np.sqrt(crop(res_all.var_M)[:, :, zi]), 98)),
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
