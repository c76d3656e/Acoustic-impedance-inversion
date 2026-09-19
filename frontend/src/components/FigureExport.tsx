import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import clsx from "clsx";
import { useShallow } from "zustand/react/shallow";
import { useStore } from "../store";
import { extractSlice } from "../viz/slice";
import { cropWindow, wellInCrop } from "../viz/window";
import { renderCompareFigure, renderFigure, renderProfileFigure, renderTrislicesFigure } from "../pyodide/plot";
import { t } from "../i18n";
import { DEFAULT_COMPARE_KEYS } from "../types";
import type { ExportKind } from "../types";

function wellUcs(s: { ucs_mwd?: number; ucs_pred: number; ucs_seis?: number; ucs_fused?: number }, key: "mwd" | "seis" | "fused") {
  if (key === "mwd") return s.ucs_mwd ?? s.ucs_pred;
  if (key === "seis") return s.ucs_seis ?? s.ucs_pred;
  return s.ucs_fused ?? s.ucs_pred;
}

export default function FigureExport() {
  const manifest = useStore((s) => s.manifest);
  const axis = useStore((s) => s.axis);
  const sliceIndex = useStore(useShallow((s) => s.sliceIndex));
  const colormap = useStore((s) => s.colormap);
  const reverse = useStore((s) => s.reverse);
  const wells = useStore((s) => s.wells);
  const showBoreholes = useStore((s) => s.showBoreholes);
  const view = useStore((s) => s.view);
  const field = useStore((s) => s.currentField());
  const ensureFields = useStore((s) => s.ensureFields);
  const winX0 = useStore((s) => s.winX0);
  const winY0 = useStore((s) => s.winY0);
  const [kind, setKind] = useState<ExportKind>(view === "compare" ? "compare" : "slice");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [png, setPng] = useState<string | null>(null);
  const [filename, setFilename] = useState("figure.png");

  const sliceTitle = useMemo(() => {
    if (!manifest || !field) return "";
    const coord = manifest.axes[axis][sliceIndex[axis]] ?? 0;
    const where =
      axis === "z" ? `${coord.toFixed(0)}m标高` : `${axis.toUpperCase()}=${coord.toFixed(0)}m`;
    return `${where} ${field.meta.name_zh}（${field.meta.unit}）`;
  }, [manifest, field, axis, sliceIndex]);

  const hint = kind === "compare" ? t.exportHintCompare
    : kind === "profile" ? t.exportHintProfile
    : kind === "trislices" ? t.exportHintTrislices
    : t.exportHint;

  const onExport = async () => {
    if (!manifest) return;
    setBusy(true);
    setPng(null);
    try {
      const crop = cropWindow(manifest, winX0, winY0);
      const coord = manifest.axes[axis][sliceIndex[axis]] ?? 0;
      const where = axis === "z"
        ? `${coord.toFixed(0)} m 标高`
        : `${axis.toUpperCase()}=${coord.toFixed(0)} m`;
      const winWells = wells.filter((w) => wellInCrop(w, crop));
      const boreholes: Array<[number, number]> =
        showBoreholes ? winWells.map((w) => [w.x, w.y]) : [];
      const lx = (crop.x1 - crop.x0).toFixed(0);
      const ly = (crop.y1 - crop.y0).toFixed(0);

      if (kind === "compare") {
        const keys = manifest.compare?.keys ?? [...DEFAULT_COMPARE_KEYS];
        await ensureFields(keys);
        const latest = useStore.getState().fields;
        const truth = latest[keys[0]];
        if (!truth) throw new Error("对比场未加载");
        const slice0 = extractSlice(truth, manifest, axis, sliceIndex[axis], crop);
        const titles = manifest.compare?.titles_zh ?? [];
        const resTitles = manifest.compare?.residual_titles_zh ?? [];
        const panels = keys.map((k, i) => {
          const vol = latest[k];
          if (!vol) throw new Error(`缺少数据场 ${k}`);
          const sl = extractSlice(vol, manifest, axis, sliceIndex[axis], crop);
          return {
            title: titles[i] ?? vol.meta.name_zh,
            residualTitle: (resTitles[i] ?? "").replace("−", "$-$") || vol.meta.name_zh,
            values: sl.values,
          };
        });
        const url = await renderCompareFigure({
          w: slice0.w,
          h: slice0.h,
          extent: slice0.extent,
          horizLabel: slice0.horiz.label,
          vertLabel: slice0.vert.label,
          title: `${t.compareSuptitle}（${where}，${manifest.n_holes} 口钻孔，${lx}×${ly} m）`,
          vmin: truth.meta.min,
          vmax: truth.meta.max,
          scale: truth.meta.scale,
          panels,
          boreholes: axis === "z" ? boreholes : [],
          colormap,
          reverse,
        }, setStatus);
        setPng(url);
        setFilename("fusion_advantage.png");
      } else if (kind === "profile") {
        const featured = wells
          .filter((w) => w.profile)
          .sort((a, b) => {
            const ra = (a.title_zh || "").includes("蚀变") ? 0 : 1;
            const rb = (b.title_zh || "").includes("蚀变") ? 0 : 1;
            return ra - rb;
          });
        const chosen = (featured.length ? featured : wells).slice(0, 2);
        if (!chosen.length) throw new Error("没有钻孔数据");
        const url = await renderProfileFigure({
          title: t.profileSuptitle,
          wells: chosen.map((w) => ({
            title: w.title_zh || w.id,
            z: w.samples.map((s) => s.z),
            ucs_true: w.samples.map((s) => s.ucs_true),
            ucs_mwd: w.samples.map((s) => wellUcs(s, "mwd")),
            ucs_seis: w.samples.map((s) => wellUcs(s, "seis")),
            ucs_fused: w.samples.map((s) => wellUcs(s, "fused")),
          })),
        }, setStatus);
        setPng(url);
        setFilename("along_hole_profiles.png");
      } else if (kind === "trislices") {
        if (!field) return;
        const xy = extractSlice(field, manifest, "z", sliceIndex.z, crop);
        const xz = extractSlice(field, manifest, "y", sliceIndex.y, crop);
        const yz = extractSlice(field, manifest, "x", sliceIndex.x, crop);
        const url = await renderTrislicesFigure({
          title: `${field.meta.name_zh} 三正交切面（${lx}×${ly} m 工作面）`,
          unit: field.meta.unit,
          scale: field.meta.scale,
          vmin: field.meta.min,
          vmax: field.meta.max,
          colormap,
          reverse,
          boreholes,
          xc: manifest.axes.x[sliceIndex.x] ?? 0,
          yc: manifest.axes.y[sliceIndex.y] ?? 0,
          zc: manifest.axes.z[sliceIndex.z] ?? 0,
          box: [crop.x0, crop.x1, crop.y0, crop.y1, manifest.extent.z[1], manifest.extent.z[0]],
          xy: { values: xy.values, w: xy.w, h: xy.h, extent: xy.extent,
            horizLabel: xy.horiz.label, vertLabel: xy.vert.label },
          xz: { values: xz.values, w: xz.w, h: xz.h, extent: xz.extent,
            horizLabel: xz.horiz.label, vertLabel: xz.vert.label },
          yz: { values: yz.values, w: yz.w, h: yz.h, extent: yz.extent,
            horizLabel: yz.horiz.label, vertLabel: yz.vert.label },
        }, setStatus);
        setPng(url);
        setFilename(`${field.meta.key}_trislices.png`);
      } else {
        if (!field) return;
        const slice = extractSlice(field, manifest, axis, sliceIndex[axis], crop);
        const url = await renderFigure(
          {
            values: slice.values,
            w: slice.w,
            h: slice.h,
            extent: slice.extent,
            horizLabel: slice.horiz.label,
            vertLabel: slice.vert.label,
            title: sliceTitle,
            unit: field.meta.unit,
            scale: field.meta.scale,
            colormap,
            reverse,
            boreholes: axis === "z" ? boreholes : [],
          },
          setStatus,
        );
        setPng(url);
        setFilename(`${field.meta.key}_slice.png`);
      }
      setStatus("");
    } catch (e) {
      setStatus(`出错：${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="figure-export">
      <div className="segmented export-kinds">
        {([
          ["slice", t.exportKindSlice],
          ["compare", t.exportKindCompare],
          ["profile", t.exportKindProfile],
          ["trislices", t.exportKindTrislices],
        ] as Array<[ExportKind, string]>).map(([k, label]) => (
          <button
            key={k}
            className={clsx("seg-btn", kind === k && "on")}
            onClick={() => { setKind(k); setPng(null); }}
          >
            <span className="seg-label">{label}</span>
          </button>
        ))}
      </div>
      <p className="hint">{hint}</p>
      <button className="primary" onClick={onExport} disabled={busy}>
        {busy ? t.exporting : t.exportButton}
      </button>
      {status && <div className="status">{status}</div>}
      <AnimatePresence>
        {png && (
          <motion.div
            className="figure-result"
            initial={{ opacity: 0, scale: 0.96, filter: "blur(8px)" }}
            animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0, scale: 0.98, filter: "blur(6px)" }}
            transition={{ type: "spring", bounce: 0, duration: 0.45 }}
          >
            <img src={png} alt="figure" />
            <a className="download" href={png} download={filename}>
              {t.download}
            </a>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
