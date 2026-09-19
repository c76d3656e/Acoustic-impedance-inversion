# 露天矿波阻抗与岩石强度三维可视化（前端）

No-backend static web app for exploring the exported impedance / rock-strength
fields and borehole (MWD) data. All computation runs **client-side** — WebGL2
for 3-D/slice rendering and **Pyodide + matplotlib (Agg)** for publication-style
figure export. Deployable to Vercel as a pure static site (no backend, no cloud
functions).

## Features

- **数据场切换**: 外漂移克里金融合、仅钻孔克里金、波阻抗标定 UCS、真值、波阻抗反演、融合不确定性、钻孔权重。
- **融合对比**: 当前切片上并排真值 / MWD / 地震 UCS / 融合（共用左侧配色），下行预测−真值热力图保持红蓝（与文档 `fusion_advantage` 同一套图）。
- **X / Y / Z 切片** with a draggable slice-position slider (live update) and a
  hover value readout.
- **三维视图 (WebGL2)**: 默认体素渲染，可切到三向正交切片；与二维/出图共用同一组切片控制。
- **钻孔（多井）**: well selector with along-hole logs (V, N, M, F) and strength
  curves (true / MWD / seismic / fused UCS); drill-hole markers on slices.
- **配色方案**: presets (Viridis/Magma/Plasma/Turbo/Jet) + a **custom strength
  colormap editor** (color stops) that affects both the live view and the
  exported figure.
- **出图（本地 matplotlib）**: 当前切片、融合对比 2×4、沿孔剖面，均可在浏览器内生成中文出版图并下载 PNG。
- **X / Y / Z 切片** with a draggable slice-position slider (live update) and a
  hover value readout.
- **三维视图 (WebGL2)**: 默认体素渲染，可切到三向正交切片；与二维/出图共用同一组切片控制。
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

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fc76d3656e%2FAcoustic-impedance-inversion&root-directory=frontend&project-name=mine-fusion-viz&repository-name=mine-fusion-viz)

Click the button (it pre-fills **Root Directory = `frontend`**), or import the
repo manually and set Root Directory to `frontend`. Framework is auto-detected as
Vite (output `dist/`). One click, **no backend, no serverless functions, no
environment variables** — a pure static site. All figure computation runs on the
user's machine (Pyodide/WebAssembly).

> The Python in the repo root is **not** used by the deploy. Vercel only builds
> `frontend/` (`npm run build`). Python is an offline tool that produced the
> committed dataset + font; the shipped site is 100% static frontend.
