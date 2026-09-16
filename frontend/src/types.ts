export interface FieldMeta {
  key: string;
  name_zh: string;
  unit: string;
  file: string;
  scale: number;
  default_cmap: string;
  min: number;
  max: number;
}

export interface Manifest {
  title_zh: string;
  grid: { nx: number; ny: number; nz: number };
  axes: { x: number[]; y: number[]; z: number[] };
  extent: { x: [number, number]; y: [number, number]; z: [number, number] };
  order: string;
  fields: FieldMeta[];
  boreholes_file: string;
  n_holes: number;
  seed: number;
}

export interface WellSample {
  z: number;
  V: number;
  N: number;
  M: number;
  F: number;
  ucs_true: number;
  ucs_pred: number;
  ai: number;
}

export interface Well {
  id: string;
  x: number;
  y: number;
  ix: number;
  iy: number;
  samples: WellSample[];
}

export interface FieldVolume {
  meta: FieldMeta;
  data: Float32Array; // C-order (ix*ny + iy)*nz + iz
}

export type SliceAxis = "x" | "y" | "z";

/** A colormap defined by anchor stops; used identically in WebGL and matplotlib. */
export interface Colormap {
  key: string;
  name_zh: string;
  stops: Array<{ pos: number; color: [number, number, number] }>; // 0..1, rgb 0..255
}
