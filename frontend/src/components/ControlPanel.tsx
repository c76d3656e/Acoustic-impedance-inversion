import { motion } from "motion/react";
import NumberFlow from "@number-flow/react";
import clsx from "clsx";
import { useStore } from "../store";
import type { SliceAxis, ViewMode } from "../types";
import { t } from "../i18n";
import ColormapEditor from "./ColormapEditor";

function Segmented<T extends string>({
  value, options, onChange, layoutId,
}: {
  value: T;
  options: Array<{ value: T; label: string }>;
  onChange: (v: T) => void;
  layoutId: string;
}) {
  return (
    <div className="segmented">
      {options.map((o) => (
        <button
          key={o.value}
          className={clsx("seg-btn", value === o.value && "on")}
          onClick={() => onChange(o.value)}
        >
          {value === o.value && (
            <motion.span
              layoutId={layoutId}
              className="seg-highlight"
              transition={{ type: "spring", bounce: 0, duration: 0.35 }}
            />
          )}
          <span className="seg-label">{o.label}</span>
        </button>
      ))}
    </div>
  );
}

export default function ControlPanel() {
  const manifest = useStore((s) => s.manifest);
  const fieldKey = useStore((s) => s.fieldKey);
  const setFieldKey = useStore((s) => s.setFieldKey);
  const axis = useStore((s) => s.axis);
  const setAxis = useStore((s) => s.setAxis);
  const sliceIndex = useStore((s) => s.sliceIndex[axis]);
  const setSliceIndex = useStore((s) => s.setSliceIndex);
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const showBoreholes = useStore((s) => s.showBoreholes);
  const setShowBoreholes = useStore((s) => s.setShowBoreholes);
  if (!manifest) return null;

  const { nx, ny, nz } = manifest.grid;
  const axisMax: Record<SliceAxis, number> = { x: nx - 1, y: ny - 1, z: nz - 1 };
  const axisCoords: Record<SliceAxis, number[]> = {
    x: manifest.axes.x, y: manifest.axes.y, z: manifest.axes.z,
  };
  const coord = axisCoords[axis][sliceIndex] ?? 0;

  return (
    <div className="panel">
      <section>
        <h3>{t.panelField}</h3>
        <select value={fieldKey} onChange={(e) => setFieldKey(e.target.value)}>
          {manifest.fields.map((f) => (
            <option key={f.key} value={f.key}>
              {f.name_zh}（{f.unit}）
            </option>
          ))}
        </select>
      </section>

      <section>
        <h3>{t.panelView}</h3>
        <Segmented<ViewMode>
          layoutId="view-seg"
          value={view}
          onChange={(v) => setView(v)}
          options={[
            { value: "2d", label: t.view2d },
            { value: "3d", label: t.view3d },
            { value: "compare", label: t.viewCompare },
          ]}
        />
        <div className="flags">
          <label className="row checkbox">
            <input type="checkbox" checked={showBoreholes}
              onChange={(e) => setShowBoreholes(e.target.checked)} />
            <span>{t.showBoreholes}</span>
          </label>
        </div>
      </section>

      <section>
        <h3>{t.panelSlice}</h3>
        <label className="row">
          <span>{t.axis}</span>
          <select value={axis} onChange={(e) => setAxis(e.target.value as SliceAxis)}>
            <option value="z">{t.axisZ}</option>
            <option value="x">{t.axisX}</option>
            <option value="y">{t.axisY}</option>
          </select>
        </label>
        <div className="slice-slider">
          <div className="pos-label">
            <span>{axis === "z" ? t.elevation : `${axis.toUpperCase()}`}</span>
            <span className="pos-value">
              <NumberFlow value={Math.round(coord)} /> {t.meter}
            </span>
          </div>
          <input
            type="range" min={0} max={axisMax[axis]} step={1}
            value={sliceIndex}
            onChange={(e) => setSliceIndex(axis, parseInt(e.target.value))}
          />
        </div>
        {view === "3d" && <p className="hint">{t.sliceWysiwygHint}</p>}
      </section>

      {view !== "compare" && (
      <section>
        <h3>{t.panelColormap}</h3>
        <ColormapEditor />
      </section>
      )}
    </div>
  );
}
