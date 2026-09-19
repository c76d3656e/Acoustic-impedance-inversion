import { lazy, Suspense, useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import clsx from "clsx";
import { useStore } from "./store";
import { t } from "./i18n";
import ControlPanel from "./components/ControlPanel";
import VizControls from "./components/VizControls";
import SliceView2D from "./components/SliceView2D";
import CompareView2D from "./components/CompareView2D";
import WellPanel from "./components/WellPanel";
import FigureExport from "./components/FigureExport";

// Code-split the WebGL/three.js view so it (and its ~700KB of deps) only loads
// when the user opens the 3D tab — smaller initial bundle, faster first paint.
const Volume3D = lazy(() => import("./components/Volume3D"));

const easeOut = [0.16, 1, 0.3, 1] as const;
const sheetSpring = { type: "spring" as const, bounce: 0, duration: 0.4 };
const DESKTOP_MQ = "(min-width: 1100px)";

type MobilePane = "scene" | "controls" | "data";

function useIsDesktop() {
  const [isDesktop, setIsDesktop] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia(DESKTOP_MQ).matches : true,
  );
  useEffect(() => {
    const mq = window.matchMedia(DESKTOP_MQ);
    const onChange = () => setIsDesktop(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return isDesktop;
}

function SheetHead({ title, onClose }: { title: string; onClose: () => void }) {
  return (
    <div className="sheet-head">
      <span className="sheet-handle" aria-hidden />
      <h2>{title}</h2>
      <button type="button" className="sheet-close" onClick={onClose} aria-label={t.mobileClose}>
        {t.mobileClose}
      </button>
    </div>
  );
}

function DockIconSliders() {
  return (
    <svg className="dock-icon" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M4 8h16M4 16h16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <circle cx="9" cy="8" r="2.3" fill="currentColor" />
      <circle cx="15" cy="16" r="2.3" fill="currentColor" />
    </svg>
  );
}

function DockIconWell() {
  return (
    <svg className="dock-icon" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="6.2" r="2.4" stroke="currentColor" strokeWidth="2" />
      <path d="M12 8.6V20M9 20h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export default function App() {
  const ready = useStore((s) => s.ready);
  const error = useStore((s) => s.error);
  const init = useStore((s) => s.init);
  const view = useStore((s) => s.view);
  const title = useStore((s) => s.manifest?.title_zh);
  const theme = useStore((s) => s.theme);
  const setTheme = useStore((s) => s.setTheme);
  const isDesktop = useIsDesktop();
  const [mobilePane, setMobilePane] = useState<MobilePane>("scene");

  useEffect(() => {
    init();
  }, [init]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "light" ? "#eceef1" : "#0b0e14");
  }, [theme]);

  useEffect(() => {
    if (isDesktop) setMobilePane("scene");
  }, [isDesktop]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && mobilePane !== "scene") setMobilePane("scene");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mobilePane]);

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

  const closeSheet = () => setMobilePane("scene");
  const controlsOpen = isDesktop || mobilePane === "controls";
  const dataOpen = isDesktop || mobilePane === "data";

  return (
    <div className={clsx("app", !isDesktop && "is-mobile")}>
      <header className="app-header">
        <div className="traffic" aria-hidden>
          <span className="tl red" />
          <span className="tl yellow" />
          <span className="tl green" />
        </div>
        <h1>{title ?? t.appTitle}</h1>
        <span className="subtitle">{t.subtitle}</span>
        <span className="header-spacer" />
        <div className="theme-toggle" role="group" aria-label={t.themeToggle}>
          <button
            type="button"
            className={clsx("theme-btn", theme === "light" && "on")}
            onClick={() => setTheme("light")}
          >
            {t.themeLight}
          </button>
          <button
            type="button"
            className={clsx("theme-btn", theme === "dark" && "on")}
            onClick={() => setTheme("dark")}
          >
            {t.themeDark}
          </button>
        </div>
      </header>

      <div className="app-body">
        <motion.aside
          className={clsx("sidebar left", mobilePane === "controls" && "is-open")}
          initial={false}
          animate={
            isDesktop
              ? { opacity: 1, x: 0, y: 0 }
              : { x: 0, y: mobilePane === "controls" ? "0%" : "110%" }
          }
          transition={sheetSpring}
          role={!isDesktop ? "dialog" : undefined}
          aria-modal={!isDesktop && mobilePane === "controls" ? true : undefined}
          aria-hidden={!controlsOpen}
          aria-label={t.mobileControls}
        >
          {!isDesktop && <SheetHead title={t.mobileControls} onClose={closeSheet} />}
          <ControlPanel omitViz={!isDesktop} />
        </motion.aside>

        <main className="stage">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={view}
              className={`stage-inner${view === "compare" ? " is-compare" : ""}`}
              initial={{ opacity: 0, scale: 0.99 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.99 }}
              transition={{ type: "spring", bounce: 0, duration: 0.35 }}
            >
              {view === "2d" ? (
                <SliceView2D />
              ) : view === "compare" ? (
                <CompareView2D />
              ) : (
                <Suspense fallback={<div className="loading3d">{t.view3d}…</div>}>
                  <Volume3D />
                </Suspense>
              )}
            </motion.div>
          </AnimatePresence>
          {!isDesktop && (
            <div className="viz-hud" aria-label={t.mobileHud}>
              <VizControls compact />
            </div>
          )}
        </main>

        <motion.aside
          className={clsx("sidebar right", mobilePane === "data" && "is-open")}
          initial={false}
          animate={
            isDesktop
              ? { opacity: 1, x: 0, y: 0 }
              : { x: 0, y: mobilePane === "data" ? "0%" : "110%" }
          }
          transition={sheetSpring}
          role={!isDesktop ? "dialog" : undefined}
          aria-modal={!isDesktop && mobilePane === "data" ? true : undefined}
          aria-hidden={!dataOpen}
          aria-label={t.mobileData}
        >
          {!isDesktop && <SheetHead title={t.mobileData} onClose={closeSheet} />}
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

      <AnimatePresence>
        {!isDesktop && mobilePane !== "scene" && (
          <motion.button
            key="scrim"
            type="button"
            className="mobile-scrim"
            aria-label={t.mobileClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={closeSheet}
          />
        )}
      </AnimatePresence>

      <nav className="mobile-dock" aria-label={t.mobileDock}>
        <button
          type="button"
          className={clsx("dock-btn", mobilePane === "controls" && "on")}
          aria-pressed={mobilePane === "controls"}
          onClick={() => setMobilePane((p) => (p === "controls" ? "scene" : "controls"))}
        >
          <DockIconSliders />
          <span>{t.mobileControls}</span>
        </button>
        <button
          type="button"
          className={clsx("dock-btn", mobilePane === "data" && "on")}
          aria-pressed={mobilePane === "data"}
          onClick={() => setMobilePane((p) => (p === "data" ? "scene" : "data"))}
        >
          <DockIconWell />
          <span>{t.mobileData}</span>
        </button>
      </nav>
    </div>
  );
}
