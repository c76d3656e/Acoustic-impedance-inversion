import type { Colormap } from "../types";

// Anchor-stop colormaps. The SAME stops drive both the WebGL/canvas LUT and the
// matplotlib LinearSegmentedColormap, so preview and exported figure match.
const hex = (h: string): [number, number, number] => [
  parseInt(h.slice(1, 3), 16),
  parseInt(h.slice(3, 5), 16),
  parseInt(h.slice(5, 7), 16),
];

const mk = (
  key: string,
  name_zh: string,
  pairs: Array<[number, string]>,
): Colormap => ({
  key,
  name_zh,
  stops: pairs.map(([pos, h]) => ({ pos, color: hex(h) })),
});

export const PRESETS: Colormap[] = [
  mk("viridis", "Viridis（默认）", [
    [0.0, "#440154"], [0.25, "#3b528b"], [0.5, "#21918c"],
    [0.75, "#5ec962"], [1.0, "#fde725"],
  ]),
  mk("magma", "Magma", [
    [0.0, "#000004"], [0.25, "#3b0f70"], [0.5, "#8c2981"],
    [0.75, "#de4968"], [0.9, "#fe9f6d"], [1.0, "#fcfdbf"],
  ]),
  mk("plasma", "Plasma", [
    [0.0, "#0d0887"], [0.25, "#6a00a8"], [0.5, "#b12a90"],
    [0.75, "#e16462"], [0.9, "#fca636"], [1.0, "#f0f921"],
  ]),
  mk("turbo", "Turbo", [
    [0.0, "#30123b"], [0.25, "#28bceb"], [0.5, "#a2fc3c"],
    [0.75, "#fb8022"], [1.0, "#7a0403"],
  ]),
  mk("jet", "Jet", [
    [0.0, "#000080"], [0.125, "#0000ff"], [0.375, "#00ffff"],
    [0.625, "#ffff00"], [0.875, "#ff0000"], [1.0, "#800000"],
  ]),
];

export const DEFAULT_CUSTOM: Colormap = mk("custom", "自定义", [
  [0.0, "#1b2a6b"], [0.5, "#2ca25f"], [1.0, "#f7fcb9"],
]);

/** Build a 256x3 (Uint8) lookup table from a colormap. */
export function buildLUT(cmap: Colormap, reverse = false): Uint8Array {
  const stops = [...cmap.stops].sort((a, b) => a.pos - b.pos);
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    let t = i / 255;
    if (reverse) t = 1 - t;
    let j = 0;
    while (j < stops.length - 1 && t > stops[j + 1].pos) j++;
    const a = stops[Math.max(0, j)];
    const b = stops[Math.min(stops.length - 1, j + 1)];
    const span = b.pos - a.pos || 1;
    const f = Math.min(1, Math.max(0, (t - a.pos) / span));
    for (let c = 0; c < 3; c++) {
      lut[i * 3 + c] = Math.round(a.color[c] + (b.color[c] - a.color[c]) * f);
    }
  }
  return lut;
}

export function rgbToHex([r, g, b]: [number, number, number]): string {
  const h = (n: number) => n.toString(16).padStart(2, "0");
  return `#${h(r)}${h(g)}${h(b)}`;
}

export function hexToRgb(h: string): [number, number, number] {
  return hex(h);
}
