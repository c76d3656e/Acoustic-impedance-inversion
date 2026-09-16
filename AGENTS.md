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
- Tracked Chinese publication figures live in `docs/images/` and are
  regenerated with `python3 examples/run_docs_figures.py --outdir docs/images`
  (CJK font required for labels; see `scripts/setup_fonts.sh`). The technical
  note is `docs/技术说明.md`.

### Frontend (`frontend/`) — no-backend static site

- Stack: Vite + React + TypeScript + three.js (WebGL2, via `@react-three/fiber`)
  + Pyodide (in-browser matplotlib). The `install` step runs
  `npm --prefix frontend install`; dev/build use standard scripts
  (`npm --prefix frontend run dev` / `run build`). See `frontend/README.md`.
- It is a **pure static** app: all computation runs client-side. Figure export
  loads Pyodide + matplotlib from the jsDelivr CDN at runtime (the wasm/packages
  are fetched once and run locally). There is no server/cloud function.
- The **fixed dataset** in `frontend/public/data/` (Float32 `.bin` fields +
  `manifest.json` + `boreholes.json`) is committed and shipped with the site.
  Regenerate it with `python3 scripts/export_frontend_dataset.py` (deterministic,
  fixed seed) — do NOT run this in the environment `install`.
- Chinese in the app is powered by a committed subset font
  `frontend/public/fonts/cjk-subset.otf` (used by both the web UI `@font-face`
  and Pyodide matplotlib). Rebuild it only if UI/label text changes, via
  `python3 scripts/build_cjk_subset.py` — this needs a system Noto CJK font
  (`scripts/setup_fonts.sh`) and is a build-time-only step; the committed subset
  is what ships.
- The root `.gitignore` scopes `/data/**` with a leading slash so it does NOT
  ignore `frontend/public/data` (the shipped dataset).
- Vercel deploy: set the project **Root Directory** to `frontend/` (its
  `vercel.json` builds with Vite to `dist/`). No backend/env vars required.
- Frontend performance conventions (keep it smooth): subscribe to Zustand with
  **narrow selectors** (never destructure the whole store — that re-renders every
  component on each slider tick); in the 3-D view (`Volume3D.tsx`) **mutate**
  persistent three.js geometry/`CanvasTexture` objects on change and dispose them
  on unmount rather than recreating per frame; animate only `transform`/`opacity`
  (see `performance-cheatsheet`); the 3-D view is **code-split** (lazy-loaded) so
  three.js isn't in the initial bundle. Motion via `motion` (Framer Motion) uses
  critically-damped springs (`bounce: 0`) by default; libraries follow Emil
  Kowalski's `pick-ui-library` (motion, @number-flow/react, clsx).
