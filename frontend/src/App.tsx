import { lazy, Suspense, useEffect } from "react";
import { AnimatePresence, motion } from "motion/react";
import { useStore } from "./store";
import { t } from "./i18n";
import ControlPanel from "./components/ControlPanel";
import SliceView2D from "./components/SliceView2D";
import WellPanel from "./components/WellPanel";
import FigureExport from "./components/FigureExport";

// Code-split the WebGL/three.js view so it (and its ~700KB of deps) only loads
// when the user opens the 3D tab — smaller initial bundle, faster first paint.
const Volume3D = lazy(() => import("./components/Volume3D"));

const easeOut = [0.16, 1, 0.3, 1] as const;

export default function App() {
  const ready = useStore((s) => s.ready);
  const error = useStore((s) => s.error);
  const init = useStore((s) => s.init);
  const view = useStore((s) => s.view);
  const title = useStore((s) => s.manifest?.title_zh);

  useEffect(() => {
    init();
  }, [init]);

  if (error) return <div className="fullscreen error">加载错误：{error}</div>;
  if (!ready)
    return (
      <div className="fullscreen">
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: easeOut }}
        >
          {t.loading}
        </motion.div>
      </div>
    );

  return (
    <div className="app">
      <header className="app-header">
        <div className="traffic" aria-hidden>
          <span className="tl red" />
          <span className="tl yellow" />
          <span className="tl green" />
        </div>
        <h1>{title ?? t.appTitle}</h1>
        <span className="subtitle">{t.subtitle}</span>
      </header>

      <div className="app-body">
        <motion.aside
          className="sidebar left"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.45, ease: easeOut }}
        >
          <ControlPanel />
        </motion.aside>

        <main className="stage">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={view}
              className="stage-inner"
              initial={{ opacity: 0, scale: 0.99 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.99 }}
              transition={{ type: "spring", bounce: 0, duration: 0.35 }}
            >
              {view === "2d" ? (
                <SliceView2D />
              ) : (
                <Suspense fallback={<div className="loading3d">{t.view3d}…</div>}>
                  <Volume3D />
                </Suspense>
              )}
            </motion.div>
          </AnimatePresence>
        </main>

        <motion.aside
          className="sidebar right"
          initial={{ opacity: 0, x: 10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.45, ease: easeOut }}
        >
          <section>
            <h3>{t.panelWells}</h3>
            <WellPanel />
          </section>
          <section>
            <h3>{t.panelExport}</h3>
            <FigureExport />
          </section>
        </motion.aside>
      </div>
    </div>
  );
}
