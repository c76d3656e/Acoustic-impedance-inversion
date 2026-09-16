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

export async function renderFigure(
  params: FigureParams,
  onStatus?: (s: string) => void,
): Promise<string> {
  const py = await ensurePyodide(onStatus);
  onStatus?.("本地渲染图片…");
  const payload = JSON.stringify({
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
  });
  py.globals.set("PAYLOAD", payload);
  const b64: string = await py.runPythonAsync(PLOT_PY);
  py.globals.delete("PAYLOAD");
  return `data:image/png;base64,${b64}`;
}
