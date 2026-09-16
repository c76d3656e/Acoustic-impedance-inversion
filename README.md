# MWD–Seismic Physics-Constrained Rock-Strength Fusion

Build a **3-D rock-strength field** `S(x, y, z)` for open-pit blast/rock
characterization by fusing two complementary, independently-inverted sources:

- **MWD** (Measurement-While-Drilling) — *local, high-resolution* mechanical
  response along drill holes → UCS.
- **3-D seismic** — *regional, continuous* structure → acoustic impedance
  `AI = ρ·Vp`.

```
        MWD ─▶ PG-GPR UCS ─▶ 3D regression kriging ─▶ S_MWD, σ_MWD
                                                              ╲
 3D seismic ─▶ impedance inversion ─▶ AI→UCS calibration ─▶ S_Z, σ_Z
                                                              ╱
              uncertainty-aware (precision-weighted) fusion ─▶ S_fused, σ_fused
                                              │
                          3D strength volume ─▶ horizontal slices (e.g. z = −60 m)
```

This is **not** a black-box end-to-end network. It is *dual-branch independent
inversion + physics-constrained late fusion*, because no public dataset provides
co-located `MWD + UCS + 3D seismic + AI`. Each branch is validated on realistic
public-style data, and the fusion is evaluated against a co-located synthetic
ground truth.

## Why uncertainty-aware fusion?

Treating each source as `N(μ, σ²)`, the inverse-variance combination

```
μ_f = (μ_M/σ_M² + μ_Z/σ_Z²) / (1/σ_M² + 1/σ_Z²)      σ_f² = 1 / (1/σ_M² + 1/σ_Z²)
```

lets the locally more reliable source dominate automatically — near drill holes
MWD wins; away from them the continuous seismic field takes over. On the
benchmark this beats both single sources **and** naive fixed-weight averaging.

## Project layout

```
├── datasets/        # synthetic 3-D model + co-located synthetic-mine benchmark
├── dataio/          # 5-module data-warehouse loaders (MWD-UCS / MWD-spatial /
│                    #   Marmousi2 / Penobscot / synthetic-mine)
├── mwd/             # MWD features (Teale specific energy) + PG-GPR UCS model
├── inversion/       # wavelet, reflectivity, forward, model-based, sparse-spike,
│                    #   + PyLops post-stack inversion
├── preprocessing/   # SEG-Y (segyio), LAS (lasio), depth↔time tie
├── geostats/        # 3-D ordinary & regression kriging (mean + variance)
├── fusion/          # AI→UCS calibration (GP) + precision-weighted fusion
├── validation/      # R², RMSE, MAE, blind-well test
├── visualization/   # slice maps, cross-sections, fusion panels, PyVista 3-D
├── examples/        # run_stage1_synthetic.py, run_fusion_benchmark.py
├── scripts/         # download_datasets.py (manual, for real data)
└── tests/           # pytest suite
```

## Setup

```bash
python3 -m pip install --user --break-system-packages -e ".[dev]"
```

(Exactly what the Cloud Agent environment `install` step runs. For local work you
can instead use a virtualenv: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`.)

## Run

End-to-end dual-branch fusion benchmark (fully offline, co-located ground truth):

```bash
python3 examples/run_fusion_benchmark.py --outdir results
```

Produces in `results/`:

- `S_fused.npy`, `sigma_fused.npy`, `S_true.npy`
- `figures/fusion_panels.png` — (a) MWD strength, (b) seismic impedance,
  (c) impedance-derived strength, (d) fused strength, (e) fusion uncertainty
- `figures/fused_strength_elev_{-24,-60,-96}.png` — horizontal slices
- a metrics table for the 4 cases (MWD-only, seismic-only, simple weighted,
  uncertainty-aware) plus a **spatial-blind** MWD hole hold-out score

Seismic-only Stage I validation (`AI → seismic → AI`, with ground truth):

```bash
python3 examples/run_stage1_synthetic.py --outdir results
```

## Visualization suite

```bash
python3 examples/run_visualizations.py --outdir results/viz
```

Produces in `results/viz/`:

- `impedance_slice.png`, `mwd_strength_slice.png`, `fused_strength_slice.png` —
  report-style horizontal slices (viridis fill + dashed contours + drill-hole
  bullseyes + scientific colorbar), matching the mine-report figure style.
- `*_3d.png` — static 3-D renders (PyVista, off-screen).
- `interactive/*.html` — **interactive** 3-D pages (Plotly): rotate/zoom/slice
  volumes, isosurfaces and slice stacks in any browser (self-contained).

Chinese labels need a CJK font. It is optional (figures fall back to the default
font otherwise). Enable it with `scripts/setup_fonts.sh` (needs sudo) or point
`AI_INVERSION_CJK_FONT` at a `.ttf`/`.otf` file.

## Interactive web app (no-backend, static)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fc76d3656e%2FAcoustic-impedance-inversion&root-directory=frontend&project-name=mine-fusion-viz&repository-name=mine-fusion-viz)

`frontend/` is a Vite + React + TypeScript + WebGL2 app that ships a **fixed
dataset** (exported from this pipeline) and renders it entirely client-side:
draggable X/Y/Z slices, a semi-transparent 3-D **volume** with a movable
**section (clipping) plane**, multi-well borehole logs, custom strength
colormaps, and **in-browser publication figures via Pyodide + matplotlib** —
with full Chinese localization. **Deploy is 100% static frontend** (Root
Directory = `frontend`); no backend, no serverless functions.

### Is this "pure frontend"? Yes — for what gets deployed.

The **deployed website** is a pure static frontend: it runs entirely in the
browser (WebGL2 for 3-D, Pyodide/WebAssembly for matplotlib) with no server and
no cloud functions. **Python is never deployed and never runs at request time.**
It lives in the repo only as the *offline* science + data pipeline that
**produced the committed dataset** (`frontend/public/data/`) and the subset font.
You can deploy the site without Python installed at all — Vercel only builds
`frontend/`. If you ever want to regenerate the dataset, that's when you'd run
the Python (`scripts/export_frontend_dataset.py`).

```bash
python3 scripts/export_frontend_dataset.py   # regenerate fixed dataset (deterministic)
npm --prefix frontend install
npm --prefix frontend run dev                # http://localhost:5173
```

## Tests

```bash
python3 -m pytest
```

## Two-stage data roadmap

| Stage | Data | Purpose | Status |
| --- | --- | --- | --- |
| **Algorithm validation** | Marmousi2 / synthetic | seismic→AI, MWD→UCS→kriging | ✅ runnable offline |
| **Co-located benchmark** | synthetic-mine (500×400×120 m, 10 benches) | dual-branch fusion vs ground truth | ✅ implemented |
| **Field data** | Penobscot 3D + Hansen MWD + MWD-UCS | real-data validation | I/O + loaders implemented |

Public datasets are **split by nature** (MWD without seismic, seismic without
MWD). We therefore validate each branch on real-style public data, fuse on a
physically-constrained co-located synthetic benchmark, and keep the loaders ready
for field data. Fetch real data with:

```bash
python scripts/download_datasets.py --dataset marmousi2 --dest data/marmousi2
python scripts/download_datasets.py --dataset penobscot --dest data/penobscot
```

See `data/README.md` for all five modules and their sources.
