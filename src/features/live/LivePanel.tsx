import { ExportButtons } from "../../components/ExportButtons";
import { TranscriptView } from "../../components/TranscriptView";
import type { Settings } from "../../types/protocol";
import { finalsOf, openOf } from "./reducer";
import { useLive } from "./useLive";

const clock = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export function LivePanel({ settings, onSettings }: { settings: Settings; onSettings: (s: Settings) => void }) {
  const { status, transcript, devices, error, mic, start, stop, refreshDevices, requestMic, openMicSettings } = useLive();
  const busy = status.state === "recording" || status.state === "starting" || status.state === "stopping";
  const finals = finalsOf(transcript);
  const meter = Math.max(0, Math.min(100, ((status.level_db + 80) / 80) * 100));
  const selected = devices.some((d) => d.id === settings.device_id) ? settings.device_id : "";

  return (
    <section>
      <div className="row toolbar">
        <label>
          Microphone{" "}
          <select
            value={selected ?? ""}
            disabled={busy}
            onFocus={refreshDevices}
            onChange={(e) => onSettings({ ...settings, device_id: e.target.value || null })}
          >
            <option value="">System default</option>
            {devices.map((d) => (
              <option key={d.id} value={d.id}>{d.name}{d.is_default ? " (default)" : ""}</option>
            ))}
          </select>
        </label>
        {status.state === "recording" || status.state === "stopping" ? (
          <button className="danger" onClick={stop} disabled={status.state === "stopping"}>Stop</button>
        ) : (
          <button className="primary" onClick={() => start(selected || null)} disabled={busy}>Start</button>
        )}
        <span className={`pill ${status.state}`} role="status">{status.state}</span>
        <span className="mono" title="elapsed">{clock(status.elapsed_s)}</span>
        <span className="meter" title={`${status.level_db.toFixed(0)} dBFS`}><i style={{ width: `${meter}%` }} /></span>
      </div>
      {mic === "not_determined" && (
        <p className="row">
          <span className="muted">TingMix has not asked for microphone access yet.</span>
          <button onClick={requestMic}>Allow microphone access</button>
        </p>
      )}
      {(mic === "denied" || mic === "restricted") && (
        <p className="row error" role="alert">
          <span>
            {mic === "denied"
              ? "Microphone access is denied for TingMix, so nothing can be recorded."
              : "Microphone access is restricted on this Mac (profile or parental controls)."}
          </span>
          {mic === "denied" && <button onClick={openMicSettings}>Open System Settings</button>}
        </p>
      )}
      {devices.length === 0 && <p className="muted">No microphone found. Connect one, then press Start to try the system default.</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {status.message && <p className="muted small">{status.message}</p>}
      <TranscriptView finals={finals} open={openOf(transcript)} size={settings.caption_size} empty="Press Start and speak. Captions appear here." />
      <div className="row footer">
        <ExportButtons name="live-transcript" segments={finals} />
        <span className="muted small">
          {status.device ? `${status.device} · ` : ""}backend: {status.backend || "Mock — not ASR"}
          {status.overflow_samples > 0 ? ` · ${status.overflow_samples} samples dropped` : ""}
        </span>
      </div>
    </section>
  );
}
