import { useEffect, useMemo, useRef } from "react";
import { useShallow } from "zustand/react/shallow";
import clsx from "clsx";
import { useStore } from "../store";
import { RDBU, buildLUT } from "../viz/colormaps";
import { drawScaled, extractSlice, sliceToImageData } from "../viz/slice";
import type { Slice2D } from "../viz/slice";
import { DEFAULT_COMPARE_KEYS } from "../types";
import { t } from "../i18n";

function percentileAbs(values: number[], p: number): number {
  if (!values.length) return 1;
  const s = values.slice().sort((a, b) => a - b);
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

function rmseOf(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s / a.length);
}

function drawHoles(
  cv: HTMLCanvasElement,
  slice: Slice2D,
  wells: Array<{ x: number; y: number }>,
) {
  const ctx = cv.getContext("2d")!;
  const W = cv.width;
  const H = cv.height;
  const [x0, x1, y0, y1] = slice.extent;
  for (const w of wells) {
    const px = ((w.x - x0) / (x1 - x0)) * W;
    const py = (1 - (w.y - y0) / (y1 - y0)) * H;
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1.6;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(px, py, 1.6, 0, Math.PI * 2);
    ctx.fillStyle = "#111";
    ctx.fill();
  }
}

function Panel({
  title,
  slice,
  lut,
  vmin,
  vmax,
  wells,
  showHoles,
  onClick,
}: {
  title: string;
  slice: Slice2D;
  lut: Uint8Array;
  vmin: number;
  vmax: number;
  wells: Array<{ x: number; y: number }>;
  showHoles: boolean;
  onClick?: () => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const [h0, h1, vlo, vhi] = slice.extent;
    const aspect = Math.min(1.35, Math.max(0.35, (vhi - vlo) / (h1 - h0)));
    const W = 280;
    const H = Math.round(W * aspect);
    cv.width = W;
    cv.height = H;
    const img = sliceToImageData(slice, lut, vmin, vmax);
    drawScaled(cv, img);
    if (showHoles) drawHoles(cv, slice, wells);
  }, [slice, lut, vmin, vmax, wells, showHoles]);

  return (
    <div
      className={clsx("compare-cell", onClick && "clickable")}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={onClick ? (e) => { if (e.key === "Enter") onClick(); } : undefined}
    >
      <div className="compare-title">{title}</div>
      <canvas ref={ref} />
    </div>
  );
}

function Colorbar({
  lut, vmin, vmax, unit,
}: {
  lut: Uint8Array; vmin: number; vmax: number; unit: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cb = ref.current;
    if (!cb) return;
    cb.width = 18;
    cb.height = 160;
    const ctx = cb.getContext("2d")!;
    for (let i = 0; i < cb.height; i++) {
      const li = Math.round((1 - i / (cb.height - 1)) * 255) * 3;
      ctx.fillStyle = `rgb(${lut[li]},${lut[li + 1]},${lut[li + 2]})`;
      ctx.fillRect(0, i, cb.width, 1);
    }
  }, [lut]);
  return (
    <div className="colorbar compact">
      <div className="cbar-max">{vmax.toFixed(1)}</div>
      <canvas ref={ref} />
      <div className="cbar-min">{vmin.toFixed(1)}</div>
      <div className="cbar-unit">{unit}</div>
    </div>
  );
}

function residualSlice(pred: Slice2D, truth: Slice2D): Slice2D {
  const values = new Float32Array(pred.values.length);
  for (let i = 0; i < values.length; i++) values[i] = pred.values[i] - truth.values[i];
  return { ...pred, values };
}

export default function CompareView2D() {
  const manifest = useStore((s) => s.manifest);
  const axis = useStore((s) => s.axis);
  const sliceIndex = useStore(useShallow((s) => s.sliceIndex));
  const wells = useStore((s) => s.wells);
  const showBoreholes = useStore((s) => s.showBoreholes);
  const fields = useStore(useShallow((s) => s.fields));
  const setFieldKey = useStore((s) => s.setFieldKey);
  const setView = useStore((s) => s.setView);
  const ensureFields = useStore((s) => s.ensureFields);
  const colormap = useStore((s) => s.colormap);
  const reverse = useStore((s) => s.reverse);

  const keys = manifest?.compare?.keys ?? DEFAULT_COMPARE_KEYS;
  const titles = manifest?.compare?.titles_zh;
  const resTitles = manifest?.compare?.residual_titles_zh;

  useEffect(() => {
    void ensureFields([...keys]);
  }, [ensureFields, keys]);

  const fieldLut = useMemo(() => buildLUT(colormap, reverse), [colormap, reverse]);
  const rdbu = useMemo(() => buildLUT(RDBU), []);

  const slices = useMemo(() => {
    if (!manifest) return null;
    const out: Slice2D[] = [];
    for (const k of keys) {
      const v = fields[k];
      if (!v) return null;
      out.push(extractSlice(v, manifest, axis, sliceIndex[axis]));
    }
    return out;
  }, [manifest, fields, keys, axis, sliceIndex]);

  const ready = Boolean(slices);
  const truthMeta = fields[keys[0]]?.meta;
  const vmin = truthMeta?.min ?? 20;
  const vmax = truthMeta?.max ?? 90;

  const residuals = useMemo(() => {
    if (!slices) return null;
    const truth = slices[0];
    return slices.slice(1).map((s) => residualSlice(s, truth));
  }, [slices]);

  const errAbs = useMemo(() => {
    if (!residuals) return 1;
    const abs: number[] = [];
    for (const r of residuals) {
      for (let i = 0; i < r.values.length; i++) abs.push(Math.abs(r.values[i]));
    }
    const p = percentileAbs(abs, 98);
    return Number.isFinite(p) && p > 1e-6 ? p : 1;
  }, [residuals]);

  if (!manifest) return null;
  if (!ready || !slices || !residuals) {
    return <div className="loading3d">{t.loadingCompare}</div>;
  }

  const showHoles = showBoreholes && axis === "z";
  const holeXY = wells.map((w) => ({ x: w.x, y: w.y }));
  const openField = (key: string) => {
    void setFieldKey(key);
    setView("2d");
  };

  return (
    <div className="compare2d">
      <div className="compare-row">
        {slices.map((s, i) => (
          <Panel
            key={`f-${keys[i]}`}
            title={titles?.[i] ?? fields[keys[i]]?.meta.name_zh ?? keys[i]}
            slice={s}
            lut={fieldLut}
            vmin={vmin}
            vmax={vmax}
            wells={holeXY}
            showHoles={showHoles}
            onClick={() => openField(keys[i])}
          />
        ))}
        <Colorbar lut={fieldLut} vmin={vmin} vmax={vmax} unit="UCS (MPa)" />
      </div>
      <div className="compare-row">
        <div className="compare-cell legend-cell">
          <div className="compare-legend">{t.residualNote}</div>
        </div>
        {residuals.map((s, i) => {
          const rmse = rmseOf(slices[i + 1].values, slices[0].values);
          const label = resTitles?.[i + 1]
            ?? `${fields[keys[i + 1]]?.meta.name_zh ?? keys[i + 1]} − 真值`;
          return (
            <Panel
              key={`r-${keys[i + 1]}`}
              title={`${label.replace("$-$", "−")}\nRMSE ${rmse.toFixed(1)} MPa`}
              slice={s}
              lut={rdbu}
              vmin={-errAbs}
              vmax={errAbs}
              wells={holeXY}
              showHoles={showHoles}
            />
          );
        })}
        <Colorbar lut={rdbu} vmin={-errAbs} vmax={errAbs} unit={t.residualCbar} />
      </div>
      <div className="hover-readout">{t.clickPanel}</div>
    </div>
  );
}
