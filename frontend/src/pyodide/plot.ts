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

_serif = 'DejaVu Serif'
_cjk = 'DejaVu Sans'
try:
    fm.fontManager.addfont('/serif.ttf')
    _serif = fm.FontProperties(fname='/serif.ttf').get_name()
except Exception as e:
    print('serif font not registered:', e)
try:
    fm.fontManager.addfont('/cjk.ttf')
    _cjk = fm.FontProperties(fname='/cjk.ttf').get_name()
except Exception as e:
    print('CJK font not registered:', e)

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 150, 'font.size': 11,
    'axes.linewidth': 0.8, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': [_serif, _cjk, 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})
_serif
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
    try {
      const resp = await fetch(`${base}fonts/liberation-serif.ttf`);
      if (resp.ok) {
        const buf = new Uint8Array(await resp.arrayBuffer());
        py.FS.writeFile("/serif.ttf", buf);
      }
    } catch {
      /* serif optional */
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
  colormap: Colormap;
  reverse: boolean;
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

export interface SlicePanel {
  values: Float32Array;
  w: number;
  h: number;
  extent: [number, number, number, number];
  horizLabel: string;
  vertLabel: string;
}

export interface TrislicesParams {
  title: string;
  unit: string;
  scale: number;
  vmin: number;
  vmax: number;
  colormap: Colormap;
  reverse: boolean;
  boreholes: Array<[number, number]>;
  xc: number;
  yc: number;
  zc: number;
  box: [number, number, number, number, number, number];
  xy: SlicePanel;
  xz: SlicePanel;
  yz: SlicePanel;
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
  fn: "render_slice" | "render_compare" | "render_profile" | "render_trislices",
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
    reverse: params.reverse,
    cmapStops: params.colormap.stops,
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

function packSlice(s: SlicePanel) {
  return {
    values: Array.from(s.values),
    w: s.w, h: s.h, extent: s.extent,
    horizLabel: s.horizLabel, vertLabel: s.vertLabel,
  };
}

export async function renderTrislicesFigure(
  params: TrislicesParams,
  onStatus?: (s: string) => void,
): Promise<string> {
  return runNamed("render_trislices", {
    title: params.title,
    unit: params.unit,
    scale: params.scale,
    vmin: params.vmin,
    vmax: params.vmax,
    reverse: params.reverse,
    cmapStops: params.colormap.stops,
    boreholes: params.boreholes,
    xc: params.xc, yc: params.yc, zc: params.zc,
    box: params.box,
    xy: packSlice(params.xy),
    xz: packSlice(params.xz),
    yz: packSlice(params.yz),
  }, onStatus);
}
