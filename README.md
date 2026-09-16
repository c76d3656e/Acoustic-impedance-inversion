# Acoustic Impedance Inversion

Recover a **3-D acoustic-impedance volume** `AI(x, y, z) = ρ·Vp` from post-stack
seismic data and extract a **high-density 2-D impedance map** at any
depth/time slice `AI(x, y, z₀)`.

The project is organised as a complete geophysical workflow:

```
3D seismic ──▶ well constraint ──▶ impedance inversion ──▶ AI(x,y,z) volume ──▶ horizontal slice
```

## Two-stage roadmap

| Stage | Data | Purpose | Status |
| --- | --- | --- | --- |
| **I. Synthetic validation** | Self-contained layered model (Marmousi2-style) | Prove the math chain `AI → R → seismic → AI` with ground truth | ✅ implemented & runnable offline |
| **II. Field data** | Penobscot 3D (SEG-Y + wells B-41 / L-30) | Well-constrained inversion of real 3-D data | Preprocessing I/O implemented (`segyio`/`lasio`); full 3-D field run is future work |

Stage I runs **entirely offline** with a reproducible synthetic earth model, so
the environment (and CI) never depends on multi-gigabyte external downloads.
Real datasets are fetched on demand with `scripts/download_datasets.py`.

## Method

Inversion is performed in the log-impedance domain `m = ln(Z)`, where
reflectivity becomes a linear difference operator:

```
r ≈ 0.5 · D·m           (D = first difference)
seismic = W·r = G·m      (W = wavelet convolution,  G = W·(0.5·D))
```

Because seismic is band-limited, we add a smooth low-frequency background model
`m₀` (from wells / horizons) and solve a Tikhonov-regularised least-squares
problem with a closed form:

```
m = m₀ + (GᵀG + λ·LᵀL + ε·I)⁻¹ Gᵀ (d − G·m₀)
```

A `sparse_spike_inversion` (L1 / ISTA) baseline is also provided.

## Project layout

```
acoustic-impedance-inversion/
├── datasets/            # synthetic earth-model builder (Vp, rho, AI)
├── preprocessing/       # SEG-Y (segyio), LAS (lasio), depth<->time
├── inversion/           # wavelet, reflectivity, forward, model-based, sparse-spike
├── validation/          # R², RMSE, MAE, blind-well test
├── visualization/       # slice maps, cross-sections, 3-D (PyVista)
├── examples/            # run_stage1_synthetic.py  (end-to-end demo)
├── scripts/             # download_datasets.py (manual, for real data)
└── tests/               # pytest suite
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

(This is exactly what the Cloud Agent environment `install` step runs.)

## Run the end-to-end demo

```bash
.venv/bin/python examples/run_stage1_synthetic.py --outdir results
```

Outputs (in `results/`):

- `AI_3D.npy` — inverted impedance volume `(nx, ny, nz)`
- `AI_true.npy` — ground-truth impedance volume
- `AI_slice.npy` — one horizontal impedance map
- `figures/impedance_map_inverted.png` — the 2-D high-density impedance map
- `figures/cross_section_comparison.png` — true vs inverted vs error
- printed metrics: `R²`, `RMSE`, `MAE`, and a blind-well score

## Tests

```bash
.venv/bin/pytest
```

## Stage II: real data

```bash
python scripts/download_datasets.py --dataset marmousi2 --dest data/marmousi2
python scripts/download_datasets.py --dataset penobscot --dest data/penobscot
```

- AGL Elastic Marmousi: https://wiki.seg.org/wiki/AGL_Elastic_Marmousi
- Penobscot 3D: https://wiki.seg.org/wiki/Penobscot_3D — https://zenodo.org/records/1325077

Load SEG-Y with `preprocessing.read_segy`, well logs with
`preprocessing.read_las` / `well_acoustic_impedance`, tie depth↔time with
`preprocessing.depth_to_twt`, then reuse the same `inversion.invert_volume`
engine validated in Stage I.
