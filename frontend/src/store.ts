import { create } from "zustand";
import type { Colormap, FieldVolume, Manifest, SliceAxis, Well } from "./types";
import { loadField, loadManifest, loadWells } from "./data/loader";
import { PRESETS, DEFAULT_CUSTOM } from "./viz/colormaps";

interface AppState {
  manifest: Manifest | null;
  wells: Well[];
  fields: Record<string, FieldVolume>;
  fieldKey: string;
  axis: SliceAxis;
  sliceIndex: Record<SliceAxis, number>;
  colormap: Colormap;
  reverse: boolean;
  view: "2d" | "3d";
  showBoreholes: boolean;
  showSlices: boolean;
  selectedWell: string | null;
  error: string | null;
  ready: boolean;

  init: () => Promise<void>;
  setFieldKey: (k: string) => Promise<void>;
  setAxis: (a: SliceAxis) => void;
  setSliceIndex: (a: SliceAxis, v: number) => void;
  setColormap: (c: Colormap) => void;
  setReverse: (r: boolean) => void;
  setView: (v: "2d" | "3d") => void;
  setShowBoreholes: (b: boolean) => void;
  setShowSlices: (b: boolean) => void;
  setSelectedWell: (id: string | null) => void;
  currentField: () => FieldVolume | null;
}

export const useStore = create<AppState>((set, get) => ({
  manifest: null,
  wells: [],
  fields: {},
  fieldKey: "impedance",
  axis: "z",
  sliceIndex: { x: 12, y: 10, z: 24 },
  colormap: PRESETS[0],
  reverse: false,
  view: "2d",
  showBoreholes: true,
  showSlices: true,
  selectedWell: null,
  error: null,
  ready: false,

  init: async () => {
    try {
      const manifest = await loadManifest();
      const wells = await loadWells(manifest);
      const first = manifest.fields[0];
      const vol = await loadField(manifest, first.key);
      const { nx, ny, nz } = manifest.grid;
      set({
        manifest,
        wells,
        fields: { [first.key]: vol },
        fieldKey: first.key,
        sliceIndex: {
          x: Math.floor(nx / 2),
          y: Math.floor(ny / 2),
          z: Math.floor(nz / 2),
        },
        selectedWell: wells[0]?.id ?? null,
        ready: true,
      });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  setFieldKey: async (k) => {
    const { manifest, fields } = get();
    if (!manifest) return;
    if (!fields[k]) {
      const vol = await loadField(manifest, k);
      set({ fields: { ...get().fields, [k]: vol } });
    }
    const meta = manifest.fields.find((f) => f.key === k)!;
    const preset = PRESETS.find((p) => p.key === meta.default_cmap) ?? PRESETS[0];
    set({ fieldKey: k, colormap: preset });
  },

  setAxis: (a) => set({ axis: a }),
  setSliceIndex: (a, v) =>
    set({ sliceIndex: { ...get().sliceIndex, [a]: v } }),
  setColormap: (c) => set({ colormap: c }),
  setReverse: (r) => set({ reverse: r }),
  setView: (v) => set({ view: v }),
  setShowBoreholes: (b) => set({ showBoreholes: b }),
  setShowSlices: (b) => set({ showSlices: b }),
  setSelectedWell: (id) => set({ selectedWell: id }),
  currentField: () => get().fields[get().fieldKey] ?? null,
}));

export { PRESETS, DEFAULT_CUSTOM };
