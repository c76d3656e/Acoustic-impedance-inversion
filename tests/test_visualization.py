import os

import numpy as np

from datasets import generate_mine
from fusion import run_fusion_pipeline, FusionResult
from visualization import plot_report_slice, export_volume_html, export_isosurface_html


def test_fusion_pipeline_shapes():
    ds = generate_mine(shape=(10, 8, 60), n_holes=5, seed=0)
    res = run_fusion_pipeline(ds, seed=0)
    assert isinstance(res, FusionResult)
    for arr in (res.ai_inv, res.S_M, res.var_M, res.S_Z, res.var_Z, res.S_F, res.var_F):
        assert arr.shape == ds.ucs_true.shape
    # Fused variance never exceeds either source.
    assert np.all(res.var_F <= res.var_M + 1e-6)
    assert np.all(res.var_F <= res.var_Z + 1e-6)


def test_report_slice_writes_png(tmp_path):
    ds = generate_mine(shape=(12, 10, 24), n_holes=5, seed=1)
    out = str(tmp_path / "slice.png")
    holes = np.unique(ds.hole_xyz[:, :2], axis=0)
    path = plot_report_slice(
        ds.ucs_true, ds.gx, ds.gy, 12, out,
        title="test slice", cbar_label="UCS (MPa)", holes_xy=holes,
    )
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000


def test_slice_grid_writes_png(tmp_path):
    ds = generate_mine(shape=(12, 10, 24), n_holes=4, seed=1)
    holes = np.unique(ds.hole_xyz[:, :2], axis=0)
    out = str(tmp_path / "grid.png")
    from visualization import plot_slice_grid

    path = plot_slice_grid(
        [
            (ds.ucs_true, "n = 1", holes[:1]),
            (ds.ucs_true, "n = 2", holes[:2]),
        ],
        ds.gx, ds.gy, 12, out,
        cbar_label="UCS (MPa)",
        vmin=float(ds.ucs_true.min()),
        vmax=float(ds.ucs_true.max()),
        ncols=2,
        suptitle="grid test",
    )
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000


def test_interactive_html_written(tmp_path):
    ds = generate_mine(shape=(10, 8, 20), n_holes=4, seed=2)
    v_html = str(tmp_path / "vol.html")
    i_html = str(tmp_path / "iso.html")
    export_volume_html(ds.ai_true, ds.gx, ds.gy, ds.gz, v_html,
                       include_plotlyjs=False)
    export_isosurface_html(ds.ucs_true, ds.gx, ds.gy, ds.gz, i_html,
                           include_plotlyjs=False)
    for p in (v_html, i_html):
        assert os.path.exists(p)
        assert os.path.getsize(p) > 500
