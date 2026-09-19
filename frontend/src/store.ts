import { create } from "zustand";
import type { Colormap, FieldVolume, Manifest, SliceAxis, ThemeMode, ViewMode, VolumeStyle, Well } from "./types";
import { DEFAULT_COMPARE_KEYS } from "./types";
import { loadField, loadManifest, loadWells } from "./data/loader";
import { PRESETS, DEFAULT_CUSTOM } from "./viz/colormaps";
import {
  clampIndexToCrop, clampWindowOrigin, cropWindow, defaultWindowOrigin,
} from "./viz/window";
import type { ViewCrop } from "./viz/window";

interface AppState {
  manifest: Manifest | null;
  wells: Well[];
  fields: Record<string, FieldVolume>;
  fieldKey: string;
  axis: SliceAxis;
  sliceIndex: Record<SliceAxis, number>;
  colormap: Colormap;
  reverse: boolean;
  view: ViewMode;
  volumeStyle: VolumeStyle;
  volumeOpacity: number;
  sectionReverse: boolean;
  showBoreholes: boolean;
  selectedWell: string | null;
  winX0: number;
  winY0: number;
  theme: ThemeMode;
  error: string | null;
  ready: boolean;

  init: () => Promise<void>;
  ensureFields: (keys: string[]) => Promise<void>;
  setFieldKey: (k: string) => Promise<void>;
  setAxis: (a: SliceAxis) => void;
  setSliceIndex: (a: SliceAxis, v: number) => void;
  setColormap: (c: Colormap) => void;
  setReverse: (r: boolean) => void;
  setView: (v: ViewMode) => void;
  setVolumeStyle: (v: VolumeStyle) => void;
  setVolumeOpacity: (v: number) => void;
  setSectionReverse: (b: boolean) => void;
  setShowBoreholes: (b: boolean) => void;
  setSelectedWell: (id: string | null) => void;
  setWindowOrigin: (x0: number, y0: number) => void;
  setTheme: (t: ThemeMode) => void;
  currentField: () => FieldVolume | null;
  crop: () => ViewCrop | null;
}

function compareKeysOf(manifest: Manifest | null): string[] {
  return manifest?.compare?.keys?.length
    ? manifest.compare.keys
    : [...DEFAULT_COMPARE_KEYS];
}

export const useStore = create<AppState>((set, get) => ({
  manifest: null,
  wells: [],
  fields: {},
  fieldKey: "fused_strength",
  axis: "z",
  sliceIndex: { x: 12, y: 10, z: 20 },
  colormap: PRESETS[0],
  reverse: false,
  view: "2d",
  volumeStyle: "voxel",
  volumeOpacity: 0.45,
  sectionReverse: false,
  showBoreholes: true,
  selectedWell: null,
  winX0: 15,
  winY0: 15,
  theme: (typeof localStorage !== "undefined"
    && localStorage.getItem("mine-fusion-theme") === "dark")
    ? "dark" : "light",
  error: null,
  ready: false,

  init: async () => {
    try {
      const manifest = await loadManifest();
      const wells = await loadWells(manifest);
      const preferred = manifest.default_field ?? "fused_strength";
      const first =
        manifest.fields.find((f) => f.key === preferred) ?? manifest.fields[0];
      const vol = await loadField(manifest, first.key);
      const preset = PRESETS.find((p) => p.key === first.default_cmap) ?? PRESETS[0];
      const { nz } = manifest.grid;
      const origin = defaultWindowOrigin(manifest);
      const crop = cropWindow(manifest, origin.x0, origin.y0);
      set({
        manifest,
        wells,
        fields: { [first.key]: vol },
        fieldKey: first.key,
        colormap: preset,
        winX0: origin.x0,
        winY0: origin.y0,
        sliceIndex: {
          x: Math.floor((crop.ix0 + crop.ix1) / 2),
          y: Math.floor((crop.iy0 + crop.iy1) / 2),
          z: Math.floor(nz / 2),
        },
        selectedWell: wells.find((w) => w.profile)?.id ?? wells[0]?.id ?? null,
        ready: true,
      });
      // Preload comparison volumes so field/compare switches stay instant.
      void get().ensureFields(compareKeysOf(manifest));
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  ensureFields: async (keys) => {
    const { manifest } = get();
    if (!manifest) return;
    const missing = keys.filter((k) => k && !get().fields[k]
      && manifest.fields.some((f) => f.key === k));
    if (!missing.length) return;
    const loaded = await Promise.all(missing.map((k) => loadField(manifest, k)));
    const next = { ...get().fields };
    for (const vol of loaded) next[vol.meta.key] = vol;
    set({ fields: next });
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
  setView: (v) => {
    set({ view: v });
    if (v === "compare") {
      void get().ensureFields(compareKeysOf(get().manifest));
    }
  },
  setShowBoreholes: (b) => set({ showBoreholes: b }),
  setVolumeStyle: (v) => set({ volumeStyle: v }),
  setVolumeOpacity: (v) => set({ volumeOpacity: v }),
  setSectionReverse: (b) => set({ sectionReverse: b }),
  setSelectedWell: (id) => set({ selectedWell: id }),
  setWindowOrigin: (x0, y0) => {
    const { manifest, sliceIndex } = get();
    if (!manifest) return;
    const origin = clampWindowOrigin(manifest, x0, y0);
    const crop = cropWindow(manifest, origin.x0, origin.y0);
    const nz = manifest.grid.nz;
    set({
      winX0: origin.x0,
      winY0: origin.y0,
      sliceIndex: {
        x: clampIndexToCrop("x", sliceIndex.x, crop, nz),
        y: clampIndexToCrop("y", sliceIndex.y, crop, nz),
        z: clampIndexToCrop("z", sliceIndex.z, crop, nz),
      },
    });
  },
  setTheme: (theme) => {
    try { localStorage.setItem("mine-fusion-theme", theme); } catch { /* ignore */ }
    set({ theme });
  },
  currentField: () => get().fields[get().fieldKey] ?? null,
  crop: () => {
    const { manifest, winX0, winY0 } = get();
    if (!manifest) return null;
    return cropWindow(manifest, winX0, winY0);
  },
}));

export { PRESETS, DEFAULT_CUSTOM };
