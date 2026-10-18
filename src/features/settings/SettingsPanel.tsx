import { useEffect, useRef, useState } from "react";
import { LOCALES, useT, type Translate } from "../../i18n";
import { hostText } from "../../i18n/host";
import { api, errorText } from "../../lib/api";
import { label } from "../../lib/shortcuts";
import type { Settings, UpdateInfo } from "../../types/protocol";
import { DiagnosticsPanel } from "../diagnostics/DiagnosticsPanel";

function Updates({ t }: { t: Translate }) {
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
      <legend><h2>{t("settings.updates")}</h2></legend>
      <p className="muted small">{t("settings.updates_note")}</p>
      <div className="row">
        <button onClick={() => run(async () => setInfo(await api.updaterCheck()))}>{t("settings.check_updates")}</button>
        {info?.available && (
          <button className="primary" onClick={() => run(async () => { await api.updaterInstall(); setMsg(t("settings.update_installed")); })}>
            {t("settings.install_update", { version: info.version ?? "" })}
          </button>
        )}
      </div>
      {info && !info.enabled && <p className="muted" role="status">{hostText(info.notes, t)}</p>}
      {info?.enabled && !info.available && <p className="muted" role="status">{t("settings.up_to_date")}</p>}
      {info?.available && info.notes && <p className="muted small">{info.notes}</p>}
      {msg && <p className="muted" role="status">{msg}</p>}
      {err && <p className="error" role="alert">{err}</p>}
    </fieldset>
  );
}

interface Props {
  settings: Settings;
  onSettings: (s: Settings) => void;
  mac: boolean;
}

export function SettingsPanel({ settings, onSettings, mac }: Props) {
  const t = useT();
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
        <legend><h2>{t("settings.appearance")}</h2></legend>
        <label>
          {t("settings.theme")}{" "}
          <select value={settings.theme} onChange={(e) => onSettings({ ...settings, theme: e.target.value as Settings["theme"] })}>
            <option value="system">{t("settings.theme.system")}</option>
            <option value="light">{t("settings.theme.light")}</option>
            <option value="dark">{t("settings.theme.dark")}</option>
          </select>
        </label>
        <label>
          {t("settings.language")}{" "}
          <select value={settings.language} onChange={(e) => onSettings({ ...settings, language: e.target.value })}>
            <option value="auto">{t("settings.language.auto")}</option>
            {Object.entries(LOCALES).map(([id, l]) => (
              <option key={id} value={id}>{l.name}</option>
            ))}
          </select>
        </label>
        <label>
          {t("settings.caption_size", { size })}{" "}
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
        <p className="sample" style={{ fontSize: size }}>{t("settings.sample")}</p>
      </fieldset>

      <fieldset>
        <legend><h2>{t("settings.history")}</h2></legend>
        <label>
          <input type="checkbox" checked={settings.save_history} onChange={(e) => onSettings({ ...settings, save_history: e.target.checked })} /> {t("settings.save_history")}
        </label>
      </fieldset>

      <Updates t={t} />

      <fieldset>
        <legend><h2>{t("settings.about")}</h2></legend>
        <p className="muted small">
          {t("settings.shortcuts", {
            tabs: `${label(mac, "1")}–${label(mac, "5")}`,
            toggle: label(mac, "Enter"),
            choose: label(mac, "O"),
            caption: `${label(mac, "=")} / ${label(mac, "-")}`,
          })}
        </p>
        <details onToggle={(e) => setDiag((e.currentTarget as HTMLDetailsElement).open)}>
          <summary>{t("settings.diagnostics")}</summary>
          {diag && <DiagnosticsPanel />}
        </details>
      </fieldset>
    </section>
  );
}
