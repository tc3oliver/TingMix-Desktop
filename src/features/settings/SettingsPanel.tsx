import { useState } from "react";
import { api, errorText } from "../../lib/api";
import type { Settings, UpdateInfo } from "../../types/protocol";

function Updates() {
  const [info, setInfo] = useState<UpdateInfo | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const run = async (f: () => Promise<unknown>) => {
    setMsg(null);
    try {
      await f();
    } catch (e) {
      setMsg(errorText(e));
    }
  };
  return (
    <div>
      <h3>Application updates</h3>
      <p className="muted small">Updates replace the app only. Installed models are never changed, replaced or re-activated by an app update.</p>
      <div className="row">
        <button onClick={() => run(async () => setInfo(await api.updaterCheck()))}>Check for updates</button>
        {info?.available && <button className="primary" onClick={() => run(async () => { await api.updaterInstall(); setMsg("Update installed. Restart TingMix to use it."); })}>Install {info.version}</button>}
      </div>
      {info && !info.enabled && <p className="muted">{info.notes}</p>}
      {info?.enabled && !info.available && <p className="muted">You are up to date.</p>}
      {info?.available && info.notes && <p className="muted small">{info.notes}</p>}
      {msg && <p className="error" role="alert">{msg}</p>}
    </div>
  );
}

export function SettingsPanel({ settings, onSettings }: { settings: Settings; onSettings: (s: Settings) => void }) {
  return (
    <section className="form">
      <label>
        Theme{" "}
        <select value={settings.theme} onChange={(e) => onSettings({ ...settings, theme: e.target.value as Settings["theme"] })}>
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
      <label>
        Caption size: {settings.caption_size}px{" "}
        <input type="range" min={12} max={72} value={settings.caption_size} onChange={(e) => onSettings({ ...settings, caption_size: Number(e.target.value) })} />
      </label>
      <label>
        <input type="checkbox" checked={settings.save_history} onChange={(e) => onSettings({ ...settings, save_history: e.target.checked })} /> Save finished transcripts to history
      </label>
      <label>
        <input type="checkbox" checked={settings.experimental_backends} onChange={(e) => onSettings({ ...settings, experimental_backends: e.target.checked })} /> Allow EXPERIMENTAL inference backends
        <div className="muted small">Off by default. These backends have not been validated; an active model on one of them is used only while this is on, and the banner says so.</div>
      </label>
      <Updates />
      <p className="muted small">Settings are stored locally and applied immediately.</p>
    </section>
  );
}
