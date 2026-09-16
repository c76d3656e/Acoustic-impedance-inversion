import { useMemo, useState } from "react";
import { useStore } from "../store";
import { extractSlice } from "../viz/slice";
import { renderFigure } from "../pyodide/plot";
import { t } from "../i18n";

export default function FigureExport() {
  const { manifest, axis, sliceIndex, colormap, reverse, wells, showBoreholes } =
    useStore();
  const field = useStore((s) => s.currentField());
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [png, setPng] = useState<string | null>(null);

  const title = useMemo(() => {
    if (!manifest || !field) return "";
    const coord = manifest.axes[axis][sliceIndex[axis]] ?? 0;
    const where =
      axis === "z" ? `${coord.toFixed(0)}m标高` : `${axis.toUpperCase()}=${coord.toFixed(0)}m`;
    return `${where} ${field.meta.name_zh}（${field.meta.unit}）`;
  }, [manifest, field, axis, sliceIndex]);

  const onExport = async () => {
    if (!manifest || !field) return;
    setBusy(true);
    setPng(null);
    try {
      const slice = extractSlice(field, manifest, axis, sliceIndex[axis]);
      const boreholes: Array<[number, number]> =
        axis === "z" && showBoreholes ? wells.map((w) => [w.x, w.y]) : [];
      const url = await renderFigure(
        {
          values: slice.values,
          w: slice.w,
          h: slice.h,
          extent: slice.extent,
          horizLabel: slice.horiz.label,
          vertLabel: slice.vert.label,
          title,
          unit: field.meta.unit,
          scale: field.meta.scale,
          colormap,
          reverse,
          boreholes,
        },
        setStatus,
      );
      setPng(url);
      setStatus("");
    } catch (e) {
      setStatus(`出错：${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="figure-export">
      <p className="hint">{t.exportHint}</p>
      <button className="primary" onClick={onExport} disabled={busy}>
        {busy ? t.exporting : t.exportButton}
      </button>
      {status && <div className="status">{status}</div>}
      {png && (
        <div className="figure-result">
          <img src={png} alt="figure" />
          <a className="download" href={png} download="figure.png">
            {t.download}
          </a>
        </div>
      )}
    </div>
  );
}
