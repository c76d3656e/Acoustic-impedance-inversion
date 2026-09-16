# Data warehouse

Five co-located data modules. Actual data files are **not** committed (see
`.gitignore`); download them with `scripts/download_datasets.py` and load them
via the `dataio` package.

| Module | Folder | Loader | Source |
| --- | --- | --- | --- |
| MWD-UCS calibration | `mwd_ucs/` | `dataio.load_mwd_ucs` | Sci. Reports 2025 (on request) — https://www.nature.com/articles/s41598-025-93111-4 |
| MWD spatial | `mwd_spatial/` | `dataio.load_mwd_spatial` | Hansen / Zenodo — https://zenodo.org/records/10358374 |
| MWD raw (rig sensors) | `mwd_raw/` | (CSV, 1 Hz) | USBR S&T 21049 — https://catalog.data.gov/dataset/data-from-st-project-21049-improving-subsurface-characterization-with-monitoring-while-dri |
| Marmousi2 | `marmousi2/` | `dataio.load_marmousi2` | SEG / GitHub — https://wiki.seg.org/wiki/AGL_Elastic_Marmousi |
| Penobscot 3D | `penobscot/` | `dataio.load_penobscot` | Zenodo / SEG — https://zenodo.org/records/1325077 |
| synthetic-mine | `synthetic_mine/` | `dataio.load_synthetic_mine` | generated locally (no download) |

## Why a synthetic co-located benchmark?

No public dataset provides `MWD + UCS + 3D seismic + AI` in one coordinate
system. Rather than stitch unrelated datasets into a fake "real" mine, the
`synthetic-mine` module builds a physically-constrained ground truth (driven by
a shared latent competence field) so the dual-branch fusion can be evaluated
against known `UCS_true(x, y, z)`.
