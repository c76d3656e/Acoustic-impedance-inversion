"""Frontend dataset must ship the KED comparison volumes (not MWD-only)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "frontend" / "public" / "data"

REQUIRED_FIELDS = {
    "fused_strength",
    "mwd_strength",
    "seismic_strength",
    "ground_truth",
    "impedance",
    "uncertainty",
    "fusion_weight",
}


def _load_export_mod():
    path = ROOT / "scripts" / "export_frontend_dataset.py"
    spec = importlib.util.spec_from_file_location("export_frontend_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_committed_manifest_has_ked_compare_fields():
    manifest = json.loads((DATA / "manifest.json").read_text())
    keys = {f["key"] for f in manifest["fields"]}
    assert REQUIRED_FIELDS <= keys
    assert manifest.get("default_field") == "fused_strength"
    assert manifest.get("fusion") == "ked"
    assert manifest["compare"]["keys"] == [
        "ground_truth", "mwd_strength", "seismic_strength", "fused_strength",
    ]
    nx, ny, nz = manifest["grid"]["nx"], manifest["grid"]["ny"], manifest["grid"]["nz"]
    nbytes = nx * ny * nz * 4
    for key in REQUIRED_FIELDS:
        path = DATA / "fields" / f"{key}.bin"
        assert path.is_file(), key
        assert path.stat().st_size == nbytes, key


def test_committed_bins_are_distinct_branches():
    mwd = np.fromfile(DATA / "fields" / "mwd_strength.bin", dtype="<f4")
    seis = np.fromfile(DATA / "fields" / "seismic_strength.bin", dtype="<f4")
    fused = np.fromfile(DATA / "fields" / "fused_strength.bin", dtype="<f4")
    truth = np.fromfile(DATA / "fields" / "ground_truth.bin", dtype="<f4")
    assert mwd.shape == seis.shape == fused.shape == truth.shape
    assert not np.allclose(mwd, fused, atol=1e-3)
    assert not np.allclose(seis, fused, atol=1e-3)
    assert not np.allclose(mwd, seis, atol=1e-3)


def test_committed_boreholes_have_branch_ucs():
    wells = json.loads((DATA / "boreholes.json").read_text())["wells"]
    assert len(wells) >= 2
    profiled = [w for w in wells if w.get("profile")]
    assert len(profiled) >= 1
    sample = wells[0]["samples"][0]
    for key in ("ucs_true", "ucs_pred", "ucs_mwd", "ucs_seis", "ucs_fused", "ai"):
        assert key in sample, key


def test_fields_from_result_uses_pipeline_volumes(tmp_path):
    from datasets import generate_mine
    from fusion import run_fusion_pipeline

    ds = generate_mine(shape=(10, 8, 32), n_holes=4, seed=0)
    res = run_fusion_pipeline(ds, seed=0)
    mod = _load_export_mod()
    fields = {row[0]: row[3] for row in mod.fields_from_result(ds, res)}
    assert np.allclose(fields["mwd_strength"], res.S_M)
    assert np.allclose(fields["seismic_strength"], res.S_Z)
    assert np.allclose(fields["fused_strength"], res.S_F)
    wells = mod.wells_from_result(ds, res)
    s0 = wells[0]["samples"][0]
    assert "ucs_seis" in s0 and "ucs_fused" in s0
    out = tmp_path / "data"
    mod.write_frontend_dataset(ds, res, str(out), n_holes=4, seed=0)
    man = json.loads((out / "manifest.json").read_text())
    assert {f["key"] for f in man["fields"]} >= REQUIRED_FIELDS
    assert (out / "fields" / "seismic_strength.bin").is_file()
