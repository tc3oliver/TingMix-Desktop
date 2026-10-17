import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { FilesPanel } from "../features/files/FilesPanel";
import { useFiles } from "../features/files/useFiles";
import { HistoryPanel } from "../features/history/HistoryPanel";
import { LivePanel } from "../features/live/LivePanel";
import { useLive } from "../features/live/useLive";
import { ModelsPanel } from "../features/models/ModelsPanel";
import { SettingsPanel } from "../features/settings/SettingsPanel";
import { api, errorText, onBackend, onDrag, onFilesDropped } from "../lib/api";
import { demoBanner, engineChip } from "../lib/engine";
import { isMac, label, shortcutFor } from "../lib/shortcuts";
import type { BackendState, Settings } from "../types/protocol";

const TABS = [
  { id: "live", label: "Live" },
  { id: "files", label: "Files" },
  { id: "history", label: "History" },
  { id: "models", label: "Models" },
  { id: "settings", label: "Settings" },
] as const;
type Tab = (typeof TABS)[number]["id"];

const DEFAULTS: Settings = { device_id: null, theme: "system", caption_size: 28, save_history: true, experimental_backends: false };
const MAC = isMac(navigator.platform);

export function App() {
  const [tab, setTab] = useState<Tab>("live");
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [error, setError] = useState<string | null>(null);
  const [bs, setBs] = useState<BackendState | null>(null);
  const [dragging, setDragging] = useState(false);
  const live = useLive();
  const files = useFiles();
  const seq = useRef(0);

  useEffect(() => void api.settingsGet().then(setSettings).catch((e) => setError(errorText(e))), []);
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
  }, []);
  useEffect(() => {
    if (settings.theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = settings.theme;
  }, [settings.theme]);

  // only the newest save may update the screen: replies to earlier saves arrive late and would move a slider back
  const update = useCallback((s: Settings) => {
    setSettings(s);
    const mine = ++seq.current;
    api.settingsSet(s).then((saved) => mine === seq.current && setSettings(saved)).catch((e) => setError(errorText(e)));
  }, []);

  const selectedDevice = (live.devices ?? []).some((d) => d.id === settings.device_id) ? settings.device_id : null;
  const toggleLive = useCallback(() => {
    setTab("live");
    if (live.status.state === "recording") void live.stop();
    else if (live.status.state === "idle" || live.status.state === "error") void live.start(selectedDevice);
  }, [live, selectedDevice]);

  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && /text|number|search/.test((t as HTMLInputElement).type)))) return;
      if (e.key === "Escape") return void setError(null);
      const a = shortcutFor(e, MAC, TABS.length);
      if (!a) return;
      e.preventDefault();
      if (a.type === "tab") setTab(TABS[a.index]!.id);
      else if (a.type === "toggle-live") toggleLive();
      else if (a.type === "choose-file") {
        setTab("files");
        void files.pick();
      } else update({ ...settings, caption_size: Math.max(12, Math.min(72, settings.caption_size + a.delta)) });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleLive, files, settings, update]);

  // ARIA tabs: arrows, Home and End move and activate
  const onTabKey = (e: KeyboardEvent) => {
    const i = TABS.findIndex((t) => t.id === tab);
    const next = e.key === "ArrowRight" ? (i + 1) % TABS.length : e.key === "ArrowLeft" ? (i + TABS.length - 1) % TABS.length : e.key === "Home" ? 0 : e.key === "End" ? TABS.length - 1 : -1;
    if (next < 0) return;
    e.preventDefault();
    setTab(TABS[next]!.id);
    document.getElementById(`tab-${TABS[next]!.id}`)?.focus();
  };

  const chip = engineChip(bs);
  const banner = demoBanner(bs);
  const openModels = () => setTab("models");
  const toggleKey = label(MAC, "Enter");

  return (
    <div className="app">
      {banner.show && (
        <div className="banner" role="note">
          <span>Demo mode: TingMix isn't recognising speech yet — output is placeholder words (w1 w2 …). {banner.detail}</span>
          <button onClick={openModels}>{banner.openModelsLabel}</button>
        </div>
      )}
      <div className="top">
        <nav role="tablist" aria-label="Sections" onKeyDown={onTabKey}>
          {TABS.map((t, i) => (
            <button
              key={t.id}
              id={`tab-${t.id}`}
              role="tab"
              aria-selected={t.id === tab}
              aria-controls={`panel-${t.id}`}
              aria-keyshortcuts={label(MAC, String(i + 1))}
              tabIndex={t.id === tab ? 0 : -1}
              className={t.id === tab ? "active" : ""}
              title={`${t.label} (${label(MAC, String(i + 1))})`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <button className={`engine ${chip.tone}`} onClick={openModels} title={chip.title} aria-label={`Speech engine: ${chip.text}. Open Models`}>
          <span className="ellipsis">{chip.text}</span>
        </button>
      </div>
      {bs?.load.error && <p className="error" role="alert">The selected model could not be loaded: {bs.load.error}</p>}
      {error && (
        <p className="error row" role="alert">
          <span>{error}</span>
          <button className="link" onClick={() => setError(null)} aria-label="Dismiss error">×</button>
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
      {dragging && <div className="drop-overlay" aria-hidden="true"><p>Drop audio files to transcribe</p></div>}
    </div>
  );
}
