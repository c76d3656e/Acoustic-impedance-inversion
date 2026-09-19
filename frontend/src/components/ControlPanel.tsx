import { useStore } from "../store";
import { t } from "../i18n";
import ColormapEditor from "./ColormapEditor";
import VizControls from "./VizControls";

export default function ControlPanel({ omitViz = false }: { omitViz?: boolean }) {
  const manifest = useStore((s) => s.manifest);
  const fieldKey = useStore((s) => s.fieldKey);
  const setFieldKey = useStore((s) => s.setFieldKey);
  const view = useStore((s) => s.view);
  if (!manifest) return null;

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

      {!omitViz && <VizControls />}

      <section>
        <h3>{t.panelColormap}</h3>
        <ColormapEditor />
        {view === "compare" && <p className="hint">{t.compareColormapHint}</p>}
      </section>
    </div>
  );
}
