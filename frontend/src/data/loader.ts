import type { FieldVolume, Manifest, Well } from "../types";

const base = import.meta.env.BASE_URL;

async function getJSON<T>(rel: string): Promise<T> {
  const res = await fetch(`${base}data/${rel}`);
  if (!res.ok) throw new Error(`加载失败: ${rel} (${res.status})`);
  return (await res.json()) as T;
}

export async function loadManifest(): Promise<Manifest> {
  return getJSON<Manifest>("manifest.json");
}

export async function loadWells(manifest: Manifest): Promise<Well[]> {
  const data = await getJSON<{ wells: Well[] }>(manifest.boreholes_file);
  return data.wells;
}

export async function loadField(
  manifest: Manifest,
  key: string,
): Promise<FieldVolume> {
  const meta = manifest.fields.find((f) => f.key === key);
  if (!meta) throw new Error(`未知数据场: ${key}`);
  const res = await fetch(`${base}data/${meta.file}`);
  if (!res.ok) throw new Error(`加载失败: ${meta.file} (${res.status})`);
  const buf = await res.arrayBuffer();
  return { meta, data: new Float32Array(buf) };
}

/** Flat index for C-order (ix*ny + iy)*nz + iz. */
export function idx(
  ix: number,
  iy: number,
  iz: number,
  ny: number,
  nz: number,
): number {
  return (ix * ny + iy) * nz + iz;
}
