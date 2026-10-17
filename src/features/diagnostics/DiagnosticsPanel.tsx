import { useEffect, useState } from "react";
import { api, errorText } from "../../lib/api";
import type { Diagnostics } from "../../types/protocol";

export function DiagnosticsPanel() {
  const [d, setD] = useState<Diagnostics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const load = () => api.diagnostics().then(setD).catch((e) => setError(errorText(e)));
    void load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, []);
  if (error) return <p className="error" role="alert">{error}</p>;
  if (!d) return <p className="muted">Loading…</p>;
  const rows: [string, string][] = [
    ["App", `${d.app_version} · ${d.os}/${d.arch}`],
    ["Backend", d.backend],
    ["Why", d.backend_reason],
    ["ONNX Runtime library", d.ort_library ?? "not found (set by the bundle or TINGMIX_ORT_DYLIB)"],
    ["App updater", d.updater_enabled ? "enabled" : "disabled (no update signing key in this build)"],
    ["Live state", `${d.live.state}${d.live.message ? ` (${d.live.message})` : ""}`],
    ["Samples captured", String(d.live.samples_in)],
    ["Samples dropped (ring overflow)", String(d.live.overflow_samples)],
    ["Coalesced partial decodes", String(d.live.coalesced_partials)],
    ["Running file jobs", String(d.running_jobs)],
    ["Config", d.config_dir],
    ["Data", d.data_dir],
  ];
  return (
    <section>
      <table className="kv"><tbody>{rows.map(([k, v]) => <tr key={k}><th>{k}</th><td>{v}</td></tr>)}</tbody></table>
      <h3>Input devices</h3>
      {d.device_error && <p className="error">{d.device_error}</p>}
      {d.input_devices.length === 0 ? <p className="muted">None detected.</p> : (
        <ul>{d.input_devices.map((x) => <li key={x.id}>{x.name} · {x.sample_rate} Hz · {x.channels} ch{x.is_default ? " · default" : ""}</li>)}</ul>
      )}
      <h3>Recent errors</h3>
      {d.recent_errors.length === 0 ? <p className="muted">None.</p> : <ul>{d.recent_errors.map((e, i) => <li key={i}>{e}</li>)}</ul>}
    </section>
  );
}
