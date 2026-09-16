# AGENTS.md

Project: MWD–Seismic physics-constrained rock-strength fusion (Python). See
`README.md` for the architecture, module layout and canonical run/test commands.

## Cursor Cloud specific instructions

- Dependencies are installed by the environment `install` step
  (`python3 -m pip install --user --break-system-packages -e ".[dev]"`) into the
  **user site** (`~/.local`), not a virtualenv. Run everything with the system
  `python3` (e.g. `python3 -m pytest`, `python3 examples/...`). Do not expect a
  `.venv/` to exist.
- The whole pipeline is **offline/synthetic** — no dataset downloads are needed
  to run tests, the fusion benchmark, or the visualization suite. Real datasets
  (`dataio/`, `scripts/download_datasets.py`) are optional and fetched manually.
- Plotting is headless: matplotlib uses the `Agg` backend and PyVista renders
  off-screen. `visualization.render_volume`/`render_isosurface` degrade to
  `None` (no exception) if off-screen 3-D rendering is unavailable, so a missing
  GPU/OSMesa never fails a run.
- Chinese figure labels need a CJK font, which is **not** on the default image.
  It is optional (labels fall back to the default font). To enable Chinese, run
  `scripts/setup_fonts.sh` (needs sudo) or set `AI_INVERSION_CJK_FONT` to a
  `.ttf`/`.otf` path. Keep exponents/units as mathtext (e.g. `$10^6$`,
  `m$^2$`) — CJK fonts lack superscript glyphs.
- The fusion pipeline lives in `fusion/pipeline.run_fusion_pipeline` and is the
  single source of truth shared by `examples/run_fusion_benchmark.py` and
  `examples/run_visualizations.py`. Change branch logic there, not in the
  examples.
- Kriging (PyKrige) cost scales with the number of borehole samples and grid
  size; the synthetic-mine grid (25×20×48) is chosen to keep runs at a few
  seconds. Increasing `--n-holes` or grid resolution increases kriging time.
