import { create } from "zustand";
import type { Colormap, FieldVolume, Manifest, SliceAxis, ViewMode, VolumeStyle, Well } from "./types";
import { DEFAULT_COMPARE_KEYS } from "./types";
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
  view: ViewMode;
  volumeStyle: VolumeStyle;
  volumeOpacity: number;
  sectionReverse: boolean;
  showBoreholes: boolean;
  selectedWell: string | null;
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
  currentField: () => FieldVolume | null;
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
      const { nx, ny, nz } = manifest.grid;
      set({
        manifest,
        wells,
        fields: { [first.key]: vol },
        fieldKey: first.key,
        colormap: preset,
        sliceIndex: {
          x: Math.floor(nx / 2),
          y: Math.floor(ny / 2),
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
  currentField: () => get().fields[get().fieldKey] ?? null,
}));

export { PRESETS, DEFAULT_CUSTOM };
