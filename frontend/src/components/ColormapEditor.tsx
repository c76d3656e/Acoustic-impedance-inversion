import { useState } from "react";
import { useStore, PRESETS, DEFAULT_CUSTOM } from "../store";
import { buildLUT, rgbToHex, hexToRgb } from "../viz/colormaps";
import type { Colormap } from "../types";
import { t } from "../i18n";

function GradientBar({ cmap, reverse }: { cmap: Colormap; reverse: boolean }) {
  const lut = buildLUT(cmap, reverse);
  const stops: string[] = [];
  for (let i = 0; i <= 10; i++) {
    const li = Math.round((i / 10) * 255) * 3;
    stops.push(`rgb(${lut[li]},${lut[li + 1]},${lut[li + 2]}) ${i * 10}%`);
  }
  return (
    <div
      className="gradient-bar"
      style={{ background: `linear-gradient(90deg, ${stops.join(",")})` }}
    />
  );
}

export default function ColormapEditor() {
  const { colormap, reverse, setColormap, setReverse } = useStore();
  const [custom, setCustom] = useState<Colormap>(DEFAULT_CUSTOM);
  const isCustom = colormap.key === "custom";

  const applyPreset = (key: string) => {
    if (key === "custom") setColormap(custom);
    else setColormap(PRESETS.find((p) => p.key === key)!);
  };

  const updateCustom = (next: Colormap) => {
    setCustom(next);
    setColormap(next);
  };

  const setStopColor = (i: number, hex: string) => {
    const stops = custom.stops.map((s, j) =>
      j === i ? { ...s, color: hexToRgb(hex) } : s,
    );
    updateCustom({ ...custom, stops });
  };
  const setStopPos = (i: number, pos: number) => {
    const stops = custom.stops.map((s, j) => (j === i ? { ...s, pos } : s));
    updateCustom({ ...custom, stops });
  };
  const addStop = () =>
    updateCustom({
      ...custom,
      stops: [...custom.stops, { pos: 0.5, color: [200, 200, 200] }],
    });
  const removeStop = (i: number) =>
    updateCustom({ ...custom, stops: custom.stops.filter((_, j) => j !== i) });

  return (
    <div className="colormap-editor">
      <label className="row">
        <span>{t.colormapPreset}</span>
        <select
          value={colormap.key}
          onChange={(e) => applyPreset(e.target.value)}
        >
          {PRESETS.map((p) => (
            <option key={p.key} value={p.key}>
              {p.name_zh}
            </option>
          ))}
          <option value="custom">{t.colormapCustom}</option>
        </select>
      </label>

      <GradientBar cmap={colormap} reverse={reverse} />

      <label className="row checkbox">
        <input
          type="checkbox"
          checked={reverse}
          onChange={(e) => setReverse(e.target.checked)}
        />
        <span>{t.reverse}</span>
      </label>

      {isCustom && (
        <div className="custom-stops">
          {custom.stops.map((s, i) => (
            <div className="stop-row" key={i}>
              <input
                type="color"
                value={rgbToHex(s.color)}
                onChange={(e) => setStopColor(i, e.target.value)}
              />
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={s.pos}
                onChange={(e) => setStopPos(i, parseFloat(e.target.value))}
              />
              <span className="stop-pos">{s.pos.toFixed(2)}</span>
              <button
                className="mini"
                onClick={() => removeStop(i)}
                disabled={custom.stops.length <= 2}
              >
                {t.removeStop}
              </button>
            </div>
          ))}
          <button className="mini add" onClick={addStop}>
            + {t.addStop}
          </button>
        </div>
      )}
    </div>
  );
}
