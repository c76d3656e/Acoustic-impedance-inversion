# 露天矿波阻抗与岩石强度三维可视化（前端）

No-backend static web app for exploring the exported impedance / rock-strength
fields and borehole (MWD) data. All computation runs **client-side** — WebGL2
for 3-D/slice rendering and **Pyodide + matplotlib (Agg)** for publication-style
figure export. Deployable to Vercel as a pure static site (no backend, no cloud
functions).

## Features

- **数据场切换**: acoustic impedance inversion, MWD strength, fused strength,
  fusion uncertainty, ground truth.
- **X / Y / Z 切片** with a draggable slice-position slider (live update) and a
  hover value readout.
- **三维视图 (WebGL2)**: three intersecting textured slice planes + strength-
  colored borehole lines; orbit/zoom.
- **钻孔（多井）**: well selector with along-hole logs (V, N, M, F) and strength
  curves (predicted vs. true UCS); drill-hole markers on slices.
- **配色方案**: presets (Viridis/Magma/Plasma/Turbo/Jet) + a **custom strength
  colormap editor** (color stops) that affects both the live view and the
  exported figure.
- **出图（本地 matplotlib）**: generate a Nature-style figure of the current
  slice in-browser, with Chinese labels, dashed contours and drill holes, then
  download the PNG.
- **全中文界面**, powered by a committed subset CJK font.

## Develop

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # -> dist/ (static)
npm run preview
```

The fixed dataset lives in `public/data/` and the subset font in
`public/fonts/`. Regenerate them from the repo root (see the top-level
`AGENTS.md`):

```bash
python3 scripts/export_frontend_dataset.py   # public/data/*
python3 scripts/build_cjk_subset.py          # public/fonts/cjk-subset.otf
```

## Deploy to Vercel

Import the repo, set **Root Directory = `frontend`** (framework auto-detected as
Vite, output `dist/`). One click, no backend, no environment variables. All
figure computation runs on the user's machine (Pyodide/wasm).
