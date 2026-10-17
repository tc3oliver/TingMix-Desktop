import { useEffect, useState } from "react";
import { DiagnosticsPanel } from "../features/diagnostics/DiagnosticsPanel";
import { FilesPanel } from "../features/files/FilesPanel";
import { HistoryPanel } from "../features/history/HistoryPanel";
import { LivePanel } from "../features/live/LivePanel";
import { ModelsPanel } from "../features/models/ModelsPanel";
import { SettingsPanel } from "../features/settings/SettingsPanel";
import { api, errorText, onModelOp } from "../lib/api";
import type { BackendInfo, Settings } from "../types/protocol";

const TABS = ["Live", "Files", "History", "Models", "Settings", "Diagnostics"] as const;
type Tab = (typeof TABS)[number];

const DEFAULTS: Settings = { device_id: null, theme: "system", caption_size: 28, save_history: true, experimental_backends: false };

export function App() {
  const [tab, setTab] = useState<Tab>("Live");
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [error, setError] = useState<string | null>(null);
  const [backend, setBackend] = useState<BackendInfo | null>(null);

  useEffect(() => void api.settingsGet().then(setSettings).catch((e) => setError(errorText(e))), []);
  // the banner follows the selection: it changes with the experimental switch and when a model operation finishes
  useEffect(() => {
    const load = () => api.backendInfo().then(setBackend).catch((e) => setError(errorText(e)));
    void load();
    let alive = true;
    let off: (() => void) | undefined;
    onModelOp((o) => o.state !== "running" && load()).then((u) => (alive ? (off = u) : u()));
    return () => {
      alive = false;
      off?.();
    };
  }, [settings.experimental_backends]);
  useEffect(() => {
    if (settings.theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = settings.theme;
  }, [settings.theme]);

  const update = (s: Settings) => {
    setSettings(s);
    api.settingsSet(s).then(setSettings).catch((e) => setError(errorText(e)));
  };

  return (
    <div className="app">
      {(!backend || backend.mock) && (
        <div className="banner" role="note">Mock — not ASR: this build produces placeholder words (w1 w2 …), not recognized speech.</div>
      )}
      {backend && !backend.mock && (
        <div className="banner" role="note">{backend.experimental ? "EXPERIMENTAL" : "Active"} backend: {backend.label}. {backend.experimental ? "Output has not been validated against the reference." : ""}</div>
      )}
      <nav role="tablist">
        {TABS.map((t) => (
          <button key={t} role="tab" aria-selected={t === tab} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>{t}</button>
        ))}
      </nav>
      {error && <p className="error" role="alert">{error}</p>}
      <main>
        {/* the live and file panels stay mounted so their state survives switching tabs */}
        <div hidden={tab !== "Live"}><LivePanel settings={settings} onSettings={update} /></div>
        <div hidden={tab !== "Files"}><FilesPanel settings={settings} /></div>
        {tab === "History" && <HistoryPanel settings={settings} />}
        {tab === "Models" && <ModelsPanel />}
        {tab === "Settings" && <SettingsPanel settings={settings} onSettings={update} />}
        {tab === "Diagnostics" && <DiagnosticsPanel />}
      </main>
    </div>
  );
}
