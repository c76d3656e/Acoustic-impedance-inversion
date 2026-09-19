import type { Manifest } from "../types";

/** Default working-face size inside the larger synthetic block (metres). */
export const VIEW_WINDOW_X = 20;
export const VIEW_WINDOW_Y = 50;

export interface ViewCrop {
  ix0: number;
  ix1: number; // inclusive
  iy0: number;
  iy1: number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

function spanOf(m: Manifest): { lx: number; ly: number } {
  return {
    lx: m.view_window?.width ?? VIEW_WINDOW_X,
    ly: m.view_window?.height ?? VIEW_WINDOW_Y,
  };
}

export function defaultWindowOrigin(m: Manifest): { x0: number; y0: number } {
  const [X0, X1] = m.extent.x;
  const [Y0, Y1] = m.extent.y;
  const { lx, ly } = spanOf(m);
  const x0 = m.view_window?.x0 ?? (X0 + X1 - lx) / 2;
  const y0 = m.view_window?.y0 ?? (Y0 + Y1 - ly) / 2;
  return clampWindowOrigin(m, x0, y0);
}

export function windowMaxOrigin(m: Manifest): { x: number; y: number } {
  const [X0, X1] = m.extent.x;
  const [Y0, Y1] = m.extent.y;
  const { lx, ly } = spanOf(m);
  return {
    x: Math.max(X0, X1 - lx),
    y: Math.max(Y0, Y1 - ly),
  };
}

export function clampWindowOrigin(
  m: Manifest, x0: number, y0: number,
): { x0: number; y0: number } {
  const [X0] = m.extent.x;
  const [Y0] = m.extent.y;
  const max = windowMaxOrigin(m);
  return {
    x0: Math.min(max.x, Math.max(X0, x0)),
    y0: Math.min(max.y, Math.max(Y0, y0)),
  };
}

function nearestIndex(coords: number[], value: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < coords.length; i++) {
    const d = Math.abs(coords[i] - value);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

function axisRange(coords: number[], lo: number, hi: number): [number, number] {
  // Snap to nearest samples so a 20×50 m window stays ~20×50 m on the grid
  // (strict interior bounds shrink the crop by up to one cell on each side).
  let i0 = nearestIndex(coords, lo);
  let i1 = nearestIndex(coords, hi);
  if (i1 < i0) {
    const tmp = i0;
    i0 = i1;
    i1 = tmp;
  }
  if (i1 <= i0) i1 = Math.min(coords.length - 1, i0 + 1);
  return [i0, i1];
}

export function cropWindow(m: Manifest, winX0: number, winY0: number): ViewCrop {
  const { lx, ly } = spanOf(m);
  const { x0, y0 } = clampWindowOrigin(m, winX0, winY0);
  const x1 = x0 + lx;
  const y1 = y0 + ly;
  const [ix0, ix1] = axisRange(m.axes.x, x0, x1);
  const [iy0, iy1] = axisRange(m.axes.y, y0, y1);
  return {
    ix0, ix1, iy0, iy1,
    x0: m.axes.x[ix0], x1: m.axes.x[ix1],
    y0: m.axes.y[iy0], y1: m.axes.y[iy1],
  };
}

export function wellInCrop(w: { x: number; y: number }, crop: ViewCrop): boolean {
  return w.x >= crop.x0 && w.x <= crop.x1 && w.y >= crop.y0 && w.y <= crop.y1;
}

export function clampIndexToCrop(
  axis: "x" | "y" | "z",
  index: number,
  crop: ViewCrop,
  nz: number,
): number {
  if (axis === "x") return Math.min(crop.ix1, Math.max(crop.ix0, index));
  if (axis === "y") return Math.min(crop.iy1, Math.max(crop.iy0, index));
  return Math.min(nz - 1, Math.max(0, index));
}
