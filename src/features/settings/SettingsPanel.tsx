import { useEffect, useRef, useState } from "react";
import { api, errorText } from "../../lib/api";
import { label } from "../../lib/shortcuts";
import type { Settings, UpdateInfo } from "../../types/protocol";
import { DiagnosticsPanel } from "../diagnostics/DiagnosticsPanel";

function Updates() {
  const [info, setInfo] = useState<UpdateInfo | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const run = async (f: () => Promise<void>) => {
    setMsg(null);
    setErr(null);
    try {
      await f();
    } catch (e) {
      setErr(errorText(e));
    }
  };
  return (
    <fieldset>
      <legend><h2>Updates</h2></legend>
      <p className="muted small">Updates replace the app only. Installed models are never changed, replaced or re-activated by an app update.</p>
      <div className="row">
        <button onClick={() => run(async () => setInfo(await api.updaterCheck()))}>Check for updates</button>
        {info?.available && (
          <button className="primary" onClick={() => run(async () => { await api.updaterInstall(); setMsg("Update installed. Restart TingMix to use it."); })}>
            Install {info.version}
          </button>
        )}
      </div>
      {info && !info.enabled && <p className="muted" role="status">{info.notes}</p>}
      {info?.enabled && !info.available && <p className="muted" role="status">You are up to date.</p>}
      {info?.available && info.notes && <p className="muted small">{info.notes}</p>}
      {msg && <p className="muted" role="status">{msg}</p>}
      {err && <p className="error" role="alert">{err}</p>}
    </fieldset>
  );
}

interface Props {
  settings: Settings;
  onSettings: (s: Settings) => void;
  onOpenModels: () => void;
  mac: boolean;
}

export function SettingsPanel({ settings, onSettings, onOpenModels, mac }: Props) {
  // the slider keeps its own value while dragging and saves once it settles, so replies from earlier saves cannot move it
  const [size, setSize] = useState(settings.caption_size);
  const timer = useRef<number>(0);
  useEffect(() => setSize(settings.caption_size), [settings.caption_size]);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const commit = (v: number) => {
    window.clearTimeout(timer.current);
    if (v !== settings.caption_size) onSettings({ ...settings, caption_size: v });
  };
  const [diag, setDiag] = useState(false);

  return (
    <section className="settings">
      <fieldset>
        <legend><h2>Appearance</h2></legend>
        <label>
          Theme{" "}
          <select value={settings.theme} onChange={(e) => onSettings({ ...settings, theme: e.target.value as Settings["theme"] })}>
            <option value="system">Follow the system</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </label>
        {/* Language goes here (task 25) */}
        <label>
          Caption size: {size} px{" "}
          <input
            type="range"
            min={12}
            max={72}
            value={size}
            onChange={(e) => {
              const v = Number(e.target.value);
              setSize(v);
              window.clearTimeout(timer.current);
              timer.current = window.setTimeout(() => commit(v), 250);
            }}
            onPointerUp={() => commit(size)}
            onKeyUp={() => commit(size)}
          />
        </label>
        <p className="sample" style={{ fontSize: size }}>字幕預覽 Caption preview</p>
      </fieldset>

      <fieldset>
        <legend><h2>History</h2></legend>
        <label>
          <input type="checkbox" checked={settings.save_history} onChange={(e) => onSettings({ ...settings, save_history: e.target.checked })} /> Save finished transcripts to History
        </label>
      </fieldset>

      <fieldset>
        <legend><h2>Speech engine</h2></legend>
        <label>
          <input
            type="checkbox"
            aria-describedby="exp-hint"
            checked={settings.experimental_backends}
            onChange={(e) => onSettings({ ...settings, experimental_backends: e.target.checked })}
          />{" "}
          Allow experimental engines
        </label>
        <p id="exp-hint" className="muted small">
          Off by default. Experimental engines have not been validated; a model that needs one is used only while this is on, and the engine chip says so.
        </p>
        <button className="link" onClick={onOpenModels}>Open Models</button>
      </fieldset>

      <Updates />

      <fieldset>
        <legend><h2>About &amp; diagnostics</h2></legend>
        <p className="muted small">
          Shortcuts: {label(mac, "1")}–{label(mac, "5")} switch tabs · {label(mac, "Enter")} start/stop · {label(mac, "O")} choose a file ·{" "}
          {label(mac, "=")} / {label(mac, "-")} caption size · Esc closes a dialog or dismisses an error.
        </p>
        <details onToggle={(e) => setDiag((e.currentTarget as HTMLDetailsElement).open)}>
          <summary>Diagnostics</summary>
          {diag && <DiagnosticsPanel />}
        </details>
      </fieldset>
    </section>
  );
}
