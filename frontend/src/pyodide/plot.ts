// In-browser publication-quality figure generation.
// Loads Pyodide + matplotlib (Agg) and renders the currently selected slice
// locally (no backend). A subsetted CJK font is registered so Chinese labels
// render. The SAME colormap stops used by the WebGL preview are rebuilt as a
// matplotlib LinearSegmentedColormap so the figure matches the on-screen view.

import type { Colormap } from "../types";

const PYODIDE_VERSION = "0.26.4";
const CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const base = import.meta.env.BASE_URL;

declare global {
  interface Window {
    loadPyodide?: (opts: { indexURL: string }) => Promise<any>;
  }
}

let pyodidePromise: Promise<any> | null = null;

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Pyodide 脚本加载失败"));
    document.head.appendChild(s);
  });
}

const SETUP_PY = `
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_name = 'DejaVu Sans'
try:
    fm.fontManager.addfont('/cjk.ttf')
    _name = fm.FontProperties(fname='/cjk.ttf').get_name()
except Exception as e:
    print('CJK font not registered:', e)

# Nature-style publication defaults.
plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 220, 'font.size': 11,
    'axes.linewidth': 0.8, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'savefig.bbox': 'tight',
    'font.sans-serif': [_name, 'DejaVu Sans'], 'axes.unicode_minus': False,
})
_name
`;

const PLOT_PY = `
import json, base64, io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

p = json.loads(PAYLOAD)
w, h = p['w'], p['h']
Z = np.array(p['values'], dtype=float).reshape(h, w) / p['scale']

stops = sorted(p['cmapStops'], key=lambda s: s['pos'])
pairs = [(s['pos'], tuple(c / 255.0 for c in s['color'])) for s in stops]
if p['reverse']:
    pairs = sorted([(1 - pos, col) for pos, col in pairs], key=lambda t: t[0])
lo, hi = pairs[0][0], pairs[-1][0]
if hi <= lo:
    hi = lo + 1.0
pairs = [(min(1.0, max(0.0, (pos - lo) / (hi - lo))), col) for pos, col in pairs]
# from_list requires the first/last mapping points to be exactly 0 and 1.
pairs[0] = (0.0, pairs[0][1])
pairs[-1] = (1.0, pairs[-1][1])
cmap = LinearSegmentedColormap.from_list('custom', pairs)

ext = p['extent']
fig, ax = plt.subplots(figsize=(6.4, 4.8))
im = ax.imshow(Z, origin='upper', extent=[ext[0], ext[1], ext[2], ext[3]],
               cmap=cmap, aspect='auto', interpolation='bilinear')
try:
    xs = np.linspace(ext[0], ext[1], w)
    ys = np.linspace(ext[3], ext[2], h)
    ax.contour(xs, ys, Z, colors='k', linewidths=0.4, linestyles='--', alpha=0.45)
except Exception as e:
    print('contour skipped:', e)

for bx, by in p['boreholes']:
    ax.scatter([bx], [by], s=90, facecolors='none', edgecolors='k', linewidths=1.2)
    ax.scatter([bx], [by], s=10, c='k')

cb = fig.colorbar(im, ax=ax)
cb.set_label(p['unit'])
ax.set_xlabel(p['horizLabel'])
ax.set_ylabel(p['vertLabel'])
ax.set_title(p['title'])
fig.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=220)
plt.close(fig)
base64.b64encode(buf.getvalue()).decode()
`;

async function ensurePyodide(onStatus?: (s: string) => void): Promise<any> {
  if (pyodidePromise) return pyodidePromise;
  pyodidePromise = (async () => {
    onStatus?.("加载 Pyodide 运行时…");
    await loadScript(`${CDN}pyodide.js`);
    const py = await window.loadPyodide!({ indexURL: CDN });
    onStatus?.("加载 numpy 与 matplotlib…");
    await py.loadPackage(["numpy", "matplotlib"]);
    onStatus?.("注册中文字体…");
    try {
      const resp = await fetch(`${base}fonts/cjk-subset.otf`);
      if (resp.ok) {
        const buf = new Uint8Array(await resp.arrayBuffer());
        py.FS.writeFile("/cjk.ttf", buf);
      }
    } catch {
      /* font optional */
    }
    await py.runPythonAsync(SETUP_PY);
    return py;
  })();
  return pyodidePromise;
}

export interface FigureParams {
  values: Float32Array;
  w: number;
  h: number;
  extent: [number, number, number, number];
  horizLabel: string;
  vertLabel: string;
  title: string;
  unit: string;
  scale: number;
  colormap: Colormap;
  reverse: boolean;
  boreholes: Array<[number, number]>;
}

export interface ComparePanel {
  title: string;
  residualTitle: string;
  values: Float32Array;
}

export interface CompareParams {
  w: number;
  h: number;
  extent: [number, number, number, number];
  horizLabel: string;
  vertLabel: string;
  title: string;
  vmin: number;
  vmax: number;
  scale: number;
  panels: ComparePanel[]; // [truth, mwd, seis, fused]
  boreholes: Array<[number, number]>;
}

export interface ProfileWell {
  title: string;
  z: number[];
  ucs_true: number[];
  ucs_mwd: number[];
  ucs_seis: number[];
  ucs_fused: number[];
}

export interface ProfileParams {
  title: string;
  wells: ProfileWell[];
}

async function runPlot(code: string, payload: unknown, onStatus?: (s: string) => void): Promise<string> {
  const py = await ensurePyodide(onStatus);
  onStatus?.("本地渲染图片…");
  py.globals.set("PAYLOAD", JSON.stringify(payload));
  const b64: string = await py.runPythonAsync(code);
  py.globals.delete("PAYLOAD");
  return `data:image/png;base64,${b64}`;
}

export async function renderFigure(
  params: FigureParams,
  onStatus?: (s: string) => void,
): Promise<string> {
  return runPlot(PLOT_PY, {
    values: Array.from(params.values),
    w: params.w,
    h: params.h,
    extent: params.extent,
    horizLabel: params.horizLabel,
    vertLabel: params.vertLabel,
    title: params.title,
    unit: params.unit,
    scale: params.scale,
    reverse: params.reverse,
    cmapStops: params.colormap.stops,
    boreholes: params.boreholes,
  }, onStatus);
}

const COMPARE_PY = `
import json, base64, io
import numpy as np
import matplotlib.pyplot as plt

p = json.loads(PAYLOAD)
w, h = p['w'], p['h']
scale = float(p['scale'])
panels = p['panels']
truth = np.array(panels[0]['values'], dtype=float).reshape(h, w) / scale
fields = [np.array(m['values'], dtype=float).reshape(h, w) / scale for m in panels[1:]]
residuals = [fld - truth for fld in fields]
slice_abs = np.concatenate([np.abs(r).ravel() for r in residuals])
err_abs = float(np.percentile(slice_abs, 98)) if slice_abs.size else 1.0
if not np.isfinite(err_abs) or err_abs < 1e-6:
    err_abs = 1.0

vmin, vmax = float(p['vmin']) / scale, float(p['vmax']) / scale
lev = np.linspace(vmin, vmax, 20)
err_lev = np.linspace(-err_abs, err_abs, 21)
n_m = len(fields)
n_cols = n_m + 1
ext = p['extent']
xs = np.linspace(ext[0], ext[1], w)
# values are origin-upper (row 0 = top = ext[3]); contourf wants south row first
ys = np.linspace(ext[2], ext[3], h)

def unflip(Z):
    return Z[::-1]

fig, axes = plt.subplots(2, n_cols, figsize=(3.4 * n_cols + 1.2, 6.6),
                         constrained_layout=True)
titles_top = [panels[0]['title']] + [m['title'] for m in panels[1:]]
vols_top = [truth] + fields
cf0 = None
for c, (vol, title) in enumerate(zip(vols_top, titles_top)):
    ax = axes[0, c]
    data = unflip(vol)
    cf0 = ax.contourf(xs, ys, data, levels=lev, cmap='viridis', extend='both')
    ax.contour(xs, ys, data, levels=lev, colors='k', linewidths=0.3,
               linestyles='--', alpha=0.4)
    for bx, by in p['boreholes']:
        ax.scatter([bx], [by], s=55, facecolors='none', edgecolors='k', linewidths=1.1, zorder=5)
        ax.scatter([bx], [by], s=8, c='k', zorder=6)
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    if c == 0:
        ax.set_ylabel(p['vertLabel'])
    else:
        ax.set_ylabel('')
    ax.set_xlabel('')

axes[1, 0].axis('off')
axes[1, 0].text(
    0.5, 0.55,
    '下行：预测 $-$ 真值\\n红＝估计偏高\\n蓝＝估计偏低\\n越浅越好',
    transform=axes[1, 0].transAxes, ha='center', va='center',
    fontsize=11, linespacing=1.6,
)
cf1 = None
for c, (res, method) in enumerate(zip(residuals, panels[1:]), start=1):
    ax = axes[1, c]
    data = unflip(res)
    cf1 = ax.contourf(xs, ys, data, levels=err_lev, cmap='RdBu_r',
                      extend='both', vmin=-err_abs, vmax=err_abs)
    ax.contour(xs, ys, data, levels=[0.0], colors='k', linewidths=0.6, alpha=0.45)
    for bx, by in p['boreholes']:
        ax.scatter([bx], [by], s=55, facecolors='none', edgecolors='k', linewidths=1.1, zorder=5)
        ax.scatter([bx], [by], s=8, c='k', zorder=6)
    rmse = float(np.sqrt(np.mean(data ** 2)))
    ax.set_title(f"{method['residualTitle']}\\nRMSE {rmse:.1f} MPa", fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel(p['horizLabel'])
    if c == 1:
        ax.set_ylabel(p['vertLabel'])
    else:
        ax.set_ylabel('')
if cf0 is not None:
    cbar0 = fig.colorbar(cf0, ax=axes[0, :].tolist(), shrink=0.9, pad=0.02)
    cbar0.set_label('UCS (MPa)')
if cf1 is not None:
    cbar1 = fig.colorbar(cf1, ax=axes[1, 1:].tolist(), shrink=0.9, pad=0.02)
    cbar1.set_label(r'预测 $-$ 真值 (MPa)')
if p['title']:
    fig.suptitle(p['title'], fontsize=13)

buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=150)
plt.close(fig)
base64.b64encode(buf.getvalue()).decode()
`;

const PROFILE_PY = `
import json, base64, io
import numpy as np
import matplotlib.pyplot as plt

p = json.loads(PAYLOAD)
wells = p['wells']
n = max(1, len(wells))
fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6),
                         constrained_layout=True, sharey=True)
axes = np.atleast_1d(axes)
for k, ax in enumerate(axes):
    w = wells[k]
    z = np.array(w['z'], dtype=float)
    ax.plot(w['ucs_true'], z, 'k-', lw=2.0, label='真值 UCS')
    ax.plot(w['ucs_seis'], z, '--', color='C1', lw=1.8, label='仅波阻抗标定')
    ax.plot(w['ucs_mwd'], z, ':', color='C0', lw=1.8, label='MWD（孔点）')
    ax.plot(w['ucs_fused'], z, '-', color='C2', lw=2.0, label='外漂移克里金融合')
    ax.set_xlabel('UCS (MPa)')
    ax.set_title(w['title'])
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    if k == 0:
        ax.set_ylabel('标高 (m)')
        ax.legend(fontsize=8, loc='best')
fig.suptitle(p['title'], fontsize=13)

buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=150)
plt.close(fig)
base64.b64encode(buf.getvalue()).decode()
`;

export async function renderCompareFigure(
  params: CompareParams,
  onStatus?: (s: string) => void,
): Promise<string> {
  return runPlot(COMPARE_PY, {
    w: params.w,
    h: params.h,
    extent: params.extent,
    horizLabel: params.horizLabel,
    vertLabel: params.vertLabel,
    title: params.title,
    vmin: params.vmin,
    vmax: params.vmax,
    scale: params.scale,
    panels: params.panels.map((m) => ({
      title: m.title,
      residualTitle: m.residualTitle,
      values: Array.from(m.values),
    })),
    boreholes: params.boreholes,
  }, onStatus);
}

export async function renderProfileFigure(
  params: ProfileParams,
  onStatus?: (s: string) => void,
): Promise<string> {
  return runPlot(PROFILE_PY, params, onStatus);
}
