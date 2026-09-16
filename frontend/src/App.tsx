import { useEffect } from "react";
import { useStore } from "./store";
import { t } from "./i18n";
import ControlPanel from "./components/ControlPanel";
import SliceView2D from "./components/SliceView2D";
import Volume3D from "./components/Volume3D";
import WellPanel from "./components/WellPanel";
import FigureExport from "./components/FigureExport";

export default function App() {
  const { ready, error, init, view, manifest } = useStore();

  useEffect(() => {
    init();
  }, [init]);

  if (error) return <div className="fullscreen error">加载错误：{error}</div>;
  if (!ready) return <div className="fullscreen">{t.loading}</div>;

  return (
    <div className="app">
      <header className="app-header">
        <h1>{manifest?.title_zh ?? t.appTitle}</h1>
        <span className="subtitle">{t.subtitle}</span>
      </header>

      <div className="app-body">
        <aside className="sidebar left">
          <ControlPanel />
        </aside>

        <main className="stage">
          {view === "2d" ? <SliceView2D /> : <Volume3D />}
        </main>

        <aside className="sidebar right">
          <section>
            <h3>{t.panelWells}</h3>
            <WellPanel />
          </section>
          <section>
            <h3>{t.panelExport}</h3>
            <FigureExport />
          </section>
        </aside>
      </div>
    </div>
  );
}
