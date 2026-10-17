import { useEffect, useState } from "react";
import { useT } from "../../i18n";
import { hostText } from "../../i18n/host";
import { api, errorText } from "../../lib/api";
import type { Diagnostics } from "../../types/protocol";

export function DiagnosticsPanel() {
  const t = useT();
  const [d, setD] = useState<Diagnostics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const load = () => api.diagnostics().then(setD).catch((e) => setError(errorText(e)));
    void load();
    const timer = setInterval(load, 2000);
    return () => clearInterval(timer);
  }, []);
  if (error) return <p className="error" role="alert">{error}</p>;
  if (!d) return <p className="muted">{t("diag.loading")}</p>;
  const rows: [string, string][] = [
    [t("diag.app"), `${d.app_version} · ${d.os}/${d.arch}`],
    [t("diag.backend"), d.backend],
    [t("diag.why"), hostText(d.backend_reason, t)],
    [t("diag.ort"), d.ort_library ?? t("diag.ort_missing")],
    [t("diag.updater"), d.updater_enabled ? t("diag.updater_on") : t("diag.updater_off")],
    [t("diag.live_state"), `${d.live.state}${d.live.message ? ` (${hostText(d.live.message, t)})` : ""}`],
    [t("diag.samples"), String(d.live.samples_in)],
    [t("diag.dropped"), String(d.live.overflow_samples)],
    [t("diag.coalesced"), String(d.live.coalesced_partials)],
    [t("diag.jobs"), String(d.running_jobs)],
    [t("diag.config"), d.config_dir],
    [t("diag.data"), d.data_dir],
  ];
  return (
    <section>
      <table className="kv"><tbody>{rows.map(([k, v]) => <tr key={k}><th>{k}</th><td>{v}</td></tr>)}</tbody></table>
      <h3>{t("diag.devices")}</h3>
      {d.device_error && <p className="error">{d.device_error}</p>}
      {d.input_devices.length === 0 ? <p className="muted">{t("diag.none")}</p> : (
        <ul>
          {d.input_devices.map((x) => (
            <li key={x.id}>{t("diag.device_line", { name: x.name, rate: x.sample_rate, channels: x.channels })}{x.is_default ? ` · ${t("diag.device_default")}` : ""}</li>
          ))}
        </ul>
      )}
      <h3>{t("diag.errors")}</h3>
      {d.recent_errors.length === 0 ? <p className="muted">{t("diag.no_errors")}</p> : <ul>{d.recent_errors.map((e, i) => <li key={i}>{e}</li>)}</ul>}
    </section>
  );
}
