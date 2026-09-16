import type { FieldVolume, Manifest, SliceAxis } from "../types";
import { idx } from "../data/loader";

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
): Slice2D {
  const { nx, ny, nz } = m.grid;
  const { x, y, z } = m.axes;
  const d = vol.data;

  if (axis === "z") {
    const w = nx, h = ny;
    const values = new Float32Array(w * h);
    for (let ix = 0; ix < nx; ix++)
      for (let iy = 0; iy < ny; iy++)
        values[(ny - 1 - iy) * w + ix] = d[idx(ix, iy, index, ny, nz)];
    return {
      w, h, values,
      horiz: { coords: x, label: "X (m)" },
      vert: { coords: y, label: "Y (m)" },
      extent: [x[0], x[nx - 1], y[0], y[ny - 1]],
    };
  }
  if (axis === "x") {
    const w = ny, h = nz;
    const values = new Float32Array(w * h);
    for (let iy = 0; iy < ny; iy++)
      for (let iz = 0; iz < nz; iz++)
        values[iz * w + iy] = d[idx(index, iy, iz, ny, nz)];
    return {
      w, h, values,
      horiz: { coords: y, label: "Y (m)" },
      vert: { coords: z, label: "标高 (m)" },
      extent: [y[0], y[ny - 1], z[nz - 1], z[0]],
    };
  }
  // axis === "y"
  const w = nx, h = nz;
  const values = new Float32Array(w * h);
  for (let ix = 0; ix < nx; ix++)
    for (let iz = 0; iz < nz; iz++)
      values[iz * w + ix] = d[idx(ix, index, iz, ny, nz)];
  return {
    w, h, values,
    horiz: { coords: x, label: "X (m)" },
    vert: { coords: z, label: "标高 (m)" },
    extent: [x[0], x[nx - 1], z[nz - 1], z[0]],
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
