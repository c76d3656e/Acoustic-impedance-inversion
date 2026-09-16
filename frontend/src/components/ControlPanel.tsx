import { useStore } from "../store";
import type { SliceAxis } from "../types";
import { t } from "../i18n";
import ColormapEditor from "./ColormapEditor";

export default function ControlPanel() {
  const {
    manifest, fieldKey, setFieldKey, axis, setAxis, sliceIndex, setSliceIndex,
    view, setView, showBoreholes, setShowBoreholes, showSlices, setShowSlices,
  } = useStore();
  if (!manifest) return null;

  const { nx, ny, nz } = manifest.grid;
  const axisMax: Record<SliceAxis, number> = { x: nx - 1, y: ny - 1, z: nz - 1 };
  const axisCoords: Record<SliceAxis, number[]> = {
    x: manifest.axes.x, y: manifest.axes.y, z: manifest.axes.z,
  };
  const coord = axisCoords[axis][sliceIndex[axis]] ?? 0;
  const posLabel = axis === "z" ? `${t.elevation} ${coord.toFixed(0)} ${t.meter}`
    : `${axis.toUpperCase()} = ${coord.toFixed(0)} ${t.meter}`;

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
        <div className="segmented">
          <button className={view === "2d" ? "on" : ""} onClick={() => setView("2d")}>
            {t.view2d}
          </button>
          <button className={view === "3d" ? "on" : ""} onClick={() => setView("3d")}>
            {t.view3d}
          </button>
        </div>
        {view === "3d" && (
          <div className="flags">
            <label className="row checkbox">
              <input type="checkbox" checked={showSlices}
                onChange={(e) => setShowSlices(e.target.checked)} />
              <span>{t.showSlices}</span>
            </label>
            <label className="row checkbox">
              <input type="checkbox" checked={showBoreholes}
                onChange={(e) => setShowBoreholes(e.target.checked)} />
              <span>{t.showBoreholes}</span>
            </label>
          </div>
        )}
        {view === "2d" && (
          <label className="row checkbox">
            <input type="checkbox" checked={showBoreholes}
              onChange={(e) => setShowBoreholes(e.target.checked)} />
            <span>{t.showBoreholes}</span>
          </label>
        )}
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
          <div className="pos-label">{t.slicePos}：{posLabel}</div>
          <input
            type="range" min={0} max={axisMax[axis]} step={1}
            value={sliceIndex[axis]}
            onChange={(e) => setSliceIndex(axis, parseInt(e.target.value))}
          />
        </div>
      </section>

      <section>
        <h3>{t.panelColormap}</h3>
        <ColormapEditor />
      </section>
    </div>
  );
}
