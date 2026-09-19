import type { FieldVolume, Manifest, SliceAxis } from "../types";
import { idx } from "../data/loader";
import type { ViewCrop } from "./window";

export interface Slice2D {
  w: number;
  h: number;
  values: Float32Array; // row-major, row 0 = top (max vertical coord)
  horiz: { coords: number[]; label: string };
  vert: { coords: number[]; label: string };
  extent: [number, number, number, number]; // [h0, h1, vLow, vHigh] for imshow origin='upper'
}

export function extractSlice(
  vol: FieldVolume,
  m: Manifest,
  axis: SliceAxis,
  index: number,
  crop?: ViewCrop | null,
): Slice2D {
  const { nx, ny, nz } = m.grid;
  const { x, y, z } = m.axes;
  const d = vol.data;
  const ix0 = crop?.ix0 ?? 0;
  const ix1 = crop?.ix1 ?? nx - 1;
  const iy0 = crop?.iy0 ?? 0;
  const iy1 = crop?.iy1 ?? ny - 1;

  if (axis === "z") {
    const iz = Math.min(nz - 1, Math.max(0, index));
    const w = ix1 - ix0 + 1, h = iy1 - iy0 + 1;
    const values = new Float32Array(w * h);
    for (let ix = ix0; ix <= ix1; ix++)
      for (let iy = iy0; iy <= iy1; iy++)
        values[(iy1 - iy) * w + (ix - ix0)] = d[idx(ix, iy, iz, ny, nz)];
    return {
      w, h, values,
      horiz: { coords: x.slice(ix0, ix1 + 1), label: "X (m)" },
      vert: { coords: y.slice(iy0, iy1 + 1), label: "Y (m)" },
      extent: [x[ix0], x[ix1], y[iy0], y[iy1]],
    };
  }
  if (axis === "x") {
    const ix = Math.min(ix1, Math.max(ix0, index));
    const w = iy1 - iy0 + 1, h = nz;
    const values = new Float32Array(w * h);
    for (let iy = iy0; iy <= iy1; iy++)
      for (let iz = 0; iz < nz; iz++)
        values[iz * w + (iy - iy0)] = d[idx(ix, iy, iz, ny, nz)];
    return {
      w, h, values,
      horiz: { coords: y.slice(iy0, iy1 + 1), label: "Y (m)" },
      vert: { coords: z, label: "标高 (m)" },
      extent: [y[iy0], y[iy1], z[nz - 1], z[0]],
    };
  }
  const iy = Math.min(iy1, Math.max(iy0, index));
  const w = ix1 - ix0 + 1, h = nz;
  const values = new Float32Array(w * h);
  for (let ix = ix0; ix <= ix1; ix++)
    for (let iz = 0; iz < nz; iz++)
      values[iz * w + (ix - ix0)] = d[idx(ix, iy, iz, ny, nz)];
  return {
    w, h, values,
    horiz: { coords: x.slice(ix0, ix1 + 1), label: "X (m)" },
    vert: { coords: z, label: "标高 (m)" },
    extent: [x[ix0], x[ix1], z[nz - 1], z[0]],
  };
}

export function sliceToImageData(
  slice: Slice2D,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
): ImageData {
  const { w, h, values } = slice;
  const img = new ImageData(w, h);
  const span = vmax - vmin || 1;
  for (let i = 0; i < w * h; i++) {
    let t = (values[i] - vmin) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const li = Math.round(t * 255) * 3;
    const o = i * 4;
    img.data[o] = lut[li];
    img.data[o + 1] = lut[li + 1];
    img.data[o + 2] = lut[li + 2];
    img.data[o + 3] = 255;
  }
  return img;
}

/** Draw an ImageData scaled (smoothed) into a display canvas. */
export function drawScaled(canvas: HTMLCanvasElement, img: ImageData): void {
  const off = document.createElement("canvas");
  off.width = img.width;
  off.height = img.height;
  off.getContext("2d")!.putImageData(img, 0, 0);
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, img.width, img.height, 0, 0, canvas.width, canvas.height);
}
