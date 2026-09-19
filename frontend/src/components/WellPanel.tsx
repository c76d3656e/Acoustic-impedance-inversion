import { useEffect, useRef } from "react";
import { useStore } from "../store";
import type { Well } from "../types";
import { t } from "../i18n";

interface Series { label: string; color: string; get: (s: Well["samples"][0]) => number; }

function drawLog(
  canvas: HTMLCanvasElement, well: Well, series: Series[],
  zmin: number, zmax: number,
) {
  const ctx = canvas.getContext("2d")!;
  const W = canvas.width, H = canvas.height;
  const padL = 8, padR = 8, padT = 10, padB = 10;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#161b22";
  ctx.fillRect(0, 0, W, H);
  const zto = (z: number) => padT + ((zmax - z) / (zmax - zmin)) * (H - padT - padB);

  for (const s of series) {
    const vals = well.samples.map(s.get);
    const vmin = Math.min(...vals), vmax = Math.max(...vals);
    const span = vmax - vmin || 1;
    const xto = (v: number) => padL + ((v - vmin) / span) * (W - padL - padR);
    ctx.beginPath();
    well.samples.forEach((smp, i) => {
      const px = xto(vals[i]);
      const py = zto(smp.z);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.8;
    ctx.stroke();
  }
}

export default function WellPanel() {
  const wells = useStore((s) => s.wells);
  const selectedWell = useStore((s) => s.selectedWell);
  const setSelectedWell = useStore((s) => s.setSelectedWell);
  const manifest = useStore((s) => s.manifest);
  const strengthRef = useRef<HTMLCanvasElement>(null);
  const drillRef = useRef<HTMLCanvasElement>(null);
  const well = wells.find((w) => w.id === selectedWell) ?? wells[0];

  useEffect(() => {
    if (!well || !manifest) return;
    const [zmax, zmin] = [manifest.extent.z[1], manifest.extent.z[0]]; // 0 .. -120
    if (strengthRef.current) {
      strengthRef.current.width = 150;
      strengthRef.current.height = 220;
      drawLog(strengthRef.current, well, [
        { label: t.curveUCStrue, color: "#ffb74d", get: (s) => s.ucs_true },
        { label: t.curveUCSpred, color: "#4fc3f7", get: (s) => s.ucs_mwd ?? s.ucs_pred },
        { label: t.curveUCSseis, color: "#ff8a65", get: (s) => s.ucs_seis ?? s.ucs_pred },
        { label: t.curveUCSfused, color: "#81c784", get: (s) => s.ucs_fused ?? s.ucs_pred },
      ], zmin, zmax);
    }
    if (drillRef.current) {
      drillRef.current.width = 150;
      drillRef.current.height = 220;
      drawLog(drillRef.current, well, [
        { label: t.curveV, color: "#81c784", get: (s) => s.V },
        { label: t.curveN, color: "#e57373", get: (s) => s.N },
        { label: t.curveM, color: "#ba68c8", get: (s) => s.M },
        { label: t.curveF, color: "#fff176", get: (s) => s.F },
      ], zmin, zmax);
    }
  }, [well, manifest]);

  if (!well) return null;

  return (
    <div className="well-panel">
      <label className="row">
        <span>{t.wellSelect}</span>
        <select value={well.id} onChange={(e) => setSelectedWell(e.target.value)}>
          {wells.map((w) => (
            <option key={w.id} value={w.id}>
              {w.id}（X={w.x.toFixed(0)}, Y={w.y.toFixed(0)}）
            </option>
          ))}
        </select>
      </label>
      <div className="well-meta">
        {t.wellCount}：{wells.length} · {t.wellPoints}：{well.samples.length}
      </div>

      <div className="well-logs">
        <div className="log-col">
          <div className="log-title">{t.wellStrength}</div>
          <canvas ref={strengthRef} />
          <div className="legend">
            <span style={{ color: "#ffb74d" }}>■ {t.curveUCStrue}</span>
            <span style={{ color: "#4fc3f7" }}>■ {t.curveUCSpred}</span>
            <span style={{ color: "#ff8a65" }}>■ {t.curveUCSseis}</span>
            <span style={{ color: "#81c784" }}>■ {t.curveUCSfused}</span>
          </div>
        </div>
        <div className="log-col">
          <div className="log-title">{t.wellLogTitle}</div>
          <canvas ref={drillRef} />
          <div className="legend small">
            <span style={{ color: "#81c784" }}>■{t.curveV}</span>
            <span style={{ color: "#e57373" }}>■{t.curveN}</span>
            <span style={{ color: "#ba68c8" }}>■{t.curveM}</span>
            <span style={{ color: "#fff176" }}>■{t.curveF}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
