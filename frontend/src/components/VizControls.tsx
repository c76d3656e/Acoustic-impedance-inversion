import { motion } from "motion/react";
import NumberFlow from "@number-flow/react";
import clsx from "clsx";
import { useStore } from "../store";
import type { SliceAxis, ViewMode, VolumeStyle } from "../types";
import { t } from "../i18n";

export function Segmented<T extends string>({
  value, options, onChange, layoutId, compact,
}: {
  value: T;
  options: Array<{ value: T; label: string }>;
  onChange: (v: T) => void;
  layoutId: string;
  compact?: boolean;
}) {
  return (
    <div className={clsx("segmented", compact && "is-compact")}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
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

/** View mode, 3D style, axis and slice slider — the controls that belong next to the figure. */
export default function VizControls({ compact = false }: { compact?: boolean }) {
  const manifest = useStore((s) => s.manifest);
  const axis = useStore((s) => s.axis);
  const setAxis = useStore((s) => s.setAxis);
  const sliceIndex = useStore((s) => s.sliceIndex[axis]);
  const setSliceIndex = useStore((s) => s.setSliceIndex);
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const volumeStyle = useStore((s) => s.volumeStyle);
  const setVolumeStyle = useStore((s) => s.setVolumeStyle);
  const volumeOpacity = useStore((s) => s.volumeOpacity);
  const setVolumeOpacity = useStore((s) => s.setVolumeOpacity);
  const sectionReverse = useStore((s) => s.sectionReverse);
  const setSectionReverse = useStore((s) => s.setSectionReverse);
  const showBoreholes = useStore((s) => s.showBoreholes);
  const setShowBoreholes = useStore((s) => s.setShowBoreholes);
  if (!manifest) return null;

  const { nx, ny, nz } = manifest.grid;
  const axisMax: Record<SliceAxis, number> = { x: nx - 1, y: ny - 1, z: nz - 1 };
  const axisCoords: Record<SliceAxis, number[]> = {
    x: manifest.axes.x, y: manifest.axes.y, z: manifest.axes.z,
  };
  const coord = axisCoords[axis][sliceIndex] ?? 0;
  const id = compact ? "hud" : "panel";

  const viewOptions = compact
    ? [
        { value: "2d" as const, label: t.hudView2d },
        { value: "3d" as const, label: t.hudView3d },
        { value: "compare" as const, label: t.hudViewCompare },
      ]
    : [
        { value: "2d" as const, label: t.view2d },
        { value: "3d" as const, label: t.view3d },
        { value: "compare" as const, label: t.viewCompare },
      ];

  return (
    <div className={clsx("viz-controls", compact && "is-compact")}>
      <section>
        {!compact && <h3>{t.panelView}</h3>}
        <Segmented<ViewMode>
          layoutId={`${id}-view-seg`}
          value={view}
          onChange={(v) => setView(v)}
          options={viewOptions}
          compact={compact}
        />
        <div className={clsx("flags", compact && "hud-checks")}>
          <label className="row checkbox">
            <input type="checkbox" checked={showBoreholes}
              onChange={(e) => setShowBoreholes(e.target.checked)} />
            <span>{t.showBoreholes}</span>
          </label>
          {compact && view === "3d" && volumeStyle === "voxel" && (
            <label className="row checkbox">
              <input type="checkbox" checked={sectionReverse}
                onChange={(e) => setSectionReverse(e.target.checked)} />
              <span>{t.sectionReverse}</span>
            </label>
          )}
        </div>
        {view === "3d" && (
          <>
            <Segmented<VolumeStyle>
              layoutId={`${id}-vol-style-seg`}
              value={volumeStyle}
              onChange={setVolumeStyle}
              options={[
                { value: "voxel", label: t.volumeStyleVoxel },
                { value: "slices", label: t.volumeStyleSlices },
              ]}
              compact={compact}
            />
            {volumeStyle === "voxel" && (
              <>
                <div className="slice-slider">
                  <div className="pos-label">
                    <span>{t.volumeOpacity}</span>
                    <span className="pos-value">{Math.round(volumeOpacity * 100)}%</span>
                  </div>
                  <input type="range" min={0.05} max={1} step={0.05}
                    value={volumeOpacity}
                    onChange={(e) => setVolumeOpacity(parseFloat(e.target.value))} />
                </div>
                {!compact && (
                  <label className="row checkbox">
                    <input type="checkbox" checked={sectionReverse}
                      onChange={(e) => setSectionReverse(e.target.checked)} />
                    <span>{t.sectionReverse}</span>
                  </label>
                )}
              </>
            )}
          </>
        )}
      </section>

      <section>
        {!compact && <h3>{t.panelSlice}</h3>}
        {compact ? (
          <Segmented<SliceAxis>
            layoutId={`${id}-axis-seg`}
            value={axis}
            onChange={setAxis}
            compact
            options={[
              { value: "z", label: t.hudAxisZ },
              { value: "x", label: t.hudAxisX },
              { value: "y", label: t.hudAxisY },
            ]}
          />
        ) : (
          <label className="row">
            <span>{t.axis}</span>
            <select value={axis} onChange={(e) => setAxis(e.target.value as SliceAxis)}>
              <option value="z">{t.axisZ}</option>
              <option value="x">{t.axisX}</option>
              <option value="y">{t.axisY}</option>
            </select>
          </label>
        )}
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
            onChange={(e) => setSliceIndex(axis, parseInt(e.target.value, 10))}
          />
        </div>
        {!compact && view === "3d" && (
          <p className="hint">
            {volumeStyle === "voxel" ? t.sliceVoxelHint : t.sliceWysiwygHint}
          </p>
        )}
      </section>
    </div>
  );
}
