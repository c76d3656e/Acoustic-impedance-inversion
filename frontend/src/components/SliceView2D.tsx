import { useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { useStore } from "../store";
import { buildLUT } from "../viz/colormaps";
import { drawScaled, extractSlice, sliceToImageData } from "../viz/slice";
import { cropWindow } from "../viz/window";
import { t } from "../i18n";

export default function SliceView2D() {
  const manifest = useStore((s) => s.manifest);
  const axis = useStore((s) => s.axis);
  const sliceIndex = useStore(useShallow((s) => s.sliceIndex));
  const colormap = useStore((s) => s.colormap);
  const reverse = useStore((s) => s.reverse);
  const wells = useStore((s) => s.wells);
  const showBoreholes = useStore((s) => s.showBoreholes);
  const field = useStore((s) => s.currentField());
  const winX0 = useStore((s) => s.winX0);
  const winY0 = useStore((s) => s.winY0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cbarRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<string>("");

  const lut = useMemo(() => buildLUT(colormap, reverse), [colormap, reverse]);
  const crop = useMemo(
    () => (manifest ? cropWindow(manifest, winX0, winY0) : null),
    [manifest, winX0, winY0],
  );

  const slice = useMemo(() => {
    if (!manifest || !field) return null;
    return extractSlice(field, manifest, axis, sliceIndex[axis], crop);
  }, [manifest, field, axis, sliceIndex, crop]);

  const vmin = field?.meta.min ?? 0;
  const vmax = field?.meta.max ?? 1;
  const scale = field?.meta.scale ?? 1;

  useEffect(() => {
    if (!slice || !canvasRef.current) return;
    const [h0, h1, vlo, vhi] = slice.extent;
    const aspect = Math.min(1.4, Math.max(0.3, (vhi - vlo) / (h1 - h0)));
    const W = 640;
    const H = Math.round(W * aspect);
    const cv = canvasRef.current;
    cv.width = W;
    cv.height = H;
    const img = sliceToImageData(slice, lut, vmin, vmax);
    drawScaled(cv, img);

    if (showBoreholes && axis === "z" && manifest) {
      const ctx = cv.getContext("2d")!;
      const [x0, x1, y0, y1] = slice.extent;
      for (const w of wells) {
        if (crop && (w.x < crop.x0 || w.x > crop.x1 || w.y < crop.y0 || w.y > crop.y1)) continue;
        const px = ((w.x - x0) / (x1 - x0)) * W;
        const py = (1 - (w.y - y0) / (y1 - y0)) * H;
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.strokeStyle = "#111";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(px, py, 2.2, 0, Math.PI * 2);
        ctx.fillStyle = "#111";
        ctx.fill();
      }
    }
  }, [slice, lut, vmin, vmax, showBoreholes, axis, wells, manifest, crop]);

  // Colorbar
  useEffect(() => {
    const cb = cbarRef.current;
    if (!cb) return;
    cb.width = 24;
    cb.height = 240;
    const ctx = cb.getContext("2d")!;
    for (let i = 0; i < cb.height; i++) {
      const li = Math.round((1 - i / (cb.height - 1)) * 255) * 3;
      ctx.fillStyle = `rgb(${lut[li]},${lut[li + 1]},${lut[li + 2]})`;
      ctx.fillRect(0, i, cb.width, 1);
    }
  }, [lut]);

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!slice) return;
    const cv = canvasRef.current!;
    const rect = cv.getBoundingClientRect();
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * slice.w);
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * slice.h);
    if (col < 0 || row < 0 || col >= slice.w || row >= slice.h) return;
    const v = slice.values[row * slice.w + col] / scale;
    const hc = slice.horiz.coords[col] ?? 0;
    const vc = slice.vert.coords[slice.vert.coords.length - 1 - row] ?? 0;
    setHover(
      `${slice.horiz.label}=${hc.toFixed(0)}  ${slice.vert.label}=${vc.toFixed(0)}  ${t.value}=${v.toFixed(2)}`,
    );
  };

  if (!field || !slice) return null;

  return (
    <div className="slice2d">
      <div className="slice2d-main">
        <canvas
          ref={canvasRef}
          className="slice-canvas"
          onMouseMove={onMove}
          onMouseLeave={() => setHover("")}
        />
        <div className="colorbar">
          <div className="cbar-max">{(vmax / scale).toFixed(2)}</div>
          <canvas ref={cbarRef} />
          <div className="cbar-min">{(vmin / scale).toFixed(2)}</div>
          <div className="cbar-unit">{field.meta.unit}</div>
        </div>
      </div>
      <div className="hover-readout">{hover || "\u00A0"}</div>
    </div>
  );
}
