import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { FilesPanel } from "../features/files/FilesPanel";
import { useFiles } from "../features/files/useFiles";
import { HistoryPanel } from "../features/history/HistoryPanel";
import { LivePanel } from "../features/live/LivePanel";
import { useLive } from "../features/live/useLive";
import { ModelsPanel } from "../features/models/ModelsPanel";
import { SettingsPanel } from "../features/settings/SettingsPanel";
import { I18nProvider, resolveLocale, useT } from "../i18n";
import { hostText } from "../i18n/host";
import { api, errorText, onBackend, onDrag, onFilesDropped } from "../lib/api";
import { demoBanner, engineChip } from "../lib/engine";
import { isMac, label, shortcutFor } from "../lib/shortcuts";
import type { BackendState, Settings } from "../types/protocol";

const TABS = ["live", "files", "history", "models", "settings"] as const;
type Tab = (typeof TABS)[number];

const DEFAULTS: Settings = { device_id: null, theme: "system", caption_size: 28, save_history: true, experimental_backends: false, language: "auto" };
const MAC = isMac(navigator.platform);

/** Owns the saved settings and the language; everything visible lives in `Main`, inside the translation context. */
export function App() {
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0);

  useEffect(() => void api.settingsGet().then(setSettings).catch((e) => setError(errorText(e))), []);
  // only the newest save may update the screen: replies to earlier saves arrive late and would move a slider back
  const update = useCallback((s: Settings) => {
    setSettings(s);
    const mine = ++seq.current;
    api.settingsSet(s).then((saved) => mine === seq.current && setSettings(saved)).catch((e) => setError(errorText(e)));
  }, []);

  return (
    <I18nProvider locale={resolveLocale(settings.language, navigator.languages)}>
      <Main settings={settings} update={update} error={error} setError={setError} />
    </I18nProvider>
  );
}

interface MainProps {
  settings: Settings;
  update: (s: Settings) => void;
  error: string | null;
  setError: (e: string | null) => void;
}

function Main({ settings, update, error, setError }: MainProps) {
  const t = useT();
  const [tab, setTab] = useState<Tab>("live");
  const [bs, setBs] = useState<BackendState | null>(null);
  const [dragging, setDragging] = useState(false);
  const live = useLive();
  const files = useFiles();

  // The host pushes the backend state on every change (activation, test result, settings, model loading), so the chip and
  // the banner never wait for a tab switch or a second click.
  useEffect(() => {
    let alive = true;
    const offs: (() => void)[] = [];
    const keep = (p: Promise<() => void>) => p.then((u) => (alive ? offs.push(u) : u()));
    keep(onBackend(setBs));
    keep(onDrag(setDragging));
    keep(onFilesDropped(() => setTab("files"))); // useFiles starts the jobs; this only brings them into view
    void api.backendState().then((v) => alive && setBs(v)).catch((e) => setError(errorText(e)));
    return () => {
      alive = false;
      offs.forEach((f) => f());
    };
  }, [setError]);
  useEffect(() => {
    if (settings.theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = settings.theme;
  }, [settings.theme]);

  const selectedDevice = (live.devices ?? []).some((d) => d.id === settings.device_id) ? settings.device_id : null;
  const toggleLive = useCallback(() => {
    setTab("live");
    if (live.status.state === "recording") void live.stop();
    else if (live.status.state === "idle" || live.status.state === "error") void live.start(selectedDevice);
  }, [live, selectedDevice]);

  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "TEXTAREA" || (target.tagName === "INPUT" && /text|number|search/.test((target as HTMLInputElement).type)))) return;
      if (e.key === "Escape") return void setError(null);
      const a = shortcutFor(e, MAC, TABS.length);
      if (!a) return;
      e.preventDefault();
      if (a.type === "tab") setTab(TABS[a.index]!);
      else if (a.type === "toggle-live") toggleLive();
      else if (a.type === "choose-file") {
        setTab("files");
        void files.pick();
      } else update({ ...settings, caption_size: Math.max(12, Math.min(72, settings.caption_size + a.delta)) });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleLive, files, settings, update, setError]);

  // ARIA tabs: arrows, Home and End move and activate
  const onTabKey = (e: KeyboardEvent) => {
    const i = TABS.indexOf(tab);
    const next = e.key === "ArrowRight" ? (i + 1) % TABS.length : e.key === "ArrowLeft" ? (i + TABS.length - 1) % TABS.length : e.key === "Home" ? 0 : e.key === "End" ? TABS.length - 1 : -1;
    if (next < 0) return;
    e.preventDefault();
    setTab(TABS[next]!);
    document.getElementById(`tab-${TABS[next]!}`)?.focus();
  };

  const chip = engineChip(bs, t);
  const banner = demoBanner(bs, t);
  const openModels = () => setTab("models");
  const toggleKey = label(MAC, "Enter");

  return (
    <div className="app">
      {banner.show && (
        <div className="banner" role="note">
          <span>{t("banner.text")} {banner.detail}</span>
          <button onClick={openModels}>{banner.openModelsLabel}</button>
        </div>
      )}
      <div className="top">
        <nav role="tablist" aria-label={t("tab.list")} onKeyDown={onTabKey}>
          {TABS.map((id, i) => (
            <button
              key={id}
              id={`tab-${id}`}
              role="tab"
              aria-selected={id === tab}
              aria-controls={`panel-${id}`}
              aria-keyshortcuts={label(MAC, String(i + 1))}
              tabIndex={id === tab ? 0 : -1}
              className={id === tab ? "active" : ""}
              title={`${t(`tab.${id}`)} (${label(MAC, String(i + 1))})`}
              onClick={() => setTab(id)}
            >
              {t(`tab.${id}`)}
            </button>
          ))}
        </nav>
        <button className={`engine ${chip.tone}`} onClick={openModels} title={chip.title} aria-label={t("engine.aria", { text: chip.text })}>
          <span className="ellipsis">{chip.text}</span>
        </button>
      </div>
      {bs?.load.error && <p className="error" role="alert">{t("app.load_error", { detail: hostText(bs.load.error, t) })}</p>}
      {error && (
        <p className="error row" role="alert">
          <span>{error}</span>
          <button className="link" onClick={() => setError(null)} aria-label={t("app.dismiss")}>×</button>
        </p>
      )}
      <main>
        {/* the live and file panels stay mounted so a recording and running jobs survive switching tabs */}
        <div role="tabpanel" id="panel-live" aria-labelledby="tab-live" hidden={tab !== "live"} className="panel live-panel">
          <LivePanel live={live} settings={settings} onSettings={update} backend={bs} shortcut={toggleKey} onOpenModels={openModels} />
        </div>
        <div role="tabpanel" id="panel-files" aria-labelledby="tab-files" hidden={tab !== "files"} className="panel">
          <FilesPanel files={files} settings={settings} backend={bs} onOpenModels={openModels} shortcut={label(MAC, "O")} />
        </div>
        {tab === "history" && <div role="tabpanel" id="panel-history" aria-labelledby="tab-history" className="panel"><HistoryPanel settings={settings} /></div>}
        {tab === "models" && <div role="tabpanel" id="panel-models" aria-labelledby="tab-models" className="panel"><ModelsPanel backend={bs} settings={settings} onSettings={update} /></div>}
        {tab === "settings" && <div role="tabpanel" id="panel-settings" aria-labelledby="tab-settings" className="panel"><SettingsPanel settings={settings} onSettings={update} onOpenModels={openModels} mac={MAC} /></div>}
      </main>
      {dragging && <div className="drop-overlay" aria-hidden="true"><p>{t("app.drop_overlay")}</p></div>}
    </div>
  );
}
