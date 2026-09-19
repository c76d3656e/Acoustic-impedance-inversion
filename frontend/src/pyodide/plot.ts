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
    'figure.dpi': 110, 'savefig.dpi': 150, 'font.size': 11,
    'axes.linewidth': 0.8, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white',
    'font.sans-serif': [_name, 'DejaVu Sans'], 'axes.unicode_minus': False,
})
_name
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

let figuresPromise: Promise<void> | null = null;

async function ensureFigures(py: any, onStatus?: (s: string) => void): Promise<void> {
  if (figuresPromise) return figuresPromise;
  figuresPromise = (async () => {
    onStatus?.("加载出版图模板…");
    const resp = await fetch(`${base}py/figures.py`);
    if (!resp.ok) throw new Error("加载出版图脚本失败");
    await py.runPythonAsync(await resp.text());
  })();
  return figuresPromise;
}

async function runNamed(
  fn: "render_slice" | "render_compare" | "render_profile",
  payload: unknown,
  onStatus?: (s: string) => void,
): Promise<string> {
  const py = await ensurePyodide(onStatus);
  await ensureFigures(py, onStatus);
  onStatus?.("本地渲染图片…");
  py.globals.set("PAYLOAD", JSON.stringify(payload));
  const b64: string = await py.runPythonAsync(`${fn}(PAYLOAD)`);
  py.globals.delete("PAYLOAD");
  return `data:image/png;base64,${b64}`;
}

export async function renderFigure(
  params: FigureParams,
  onStatus?: (s: string) => void,
): Promise<string> {
  return runNamed("render_slice", {
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

export async function renderCompareFigure(
  params: CompareParams,
  onStatus?: (s: string) => void,
): Promise<string> {
  return runNamed("render_compare", {
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
  return runNamed("render_profile", params, onStatus);
}
