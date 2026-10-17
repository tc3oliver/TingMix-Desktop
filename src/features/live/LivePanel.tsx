import { useEffect, useRef, useState } from "react";
import { ExportButtons } from "../../components/ExportButtons";
import { Provenance } from "../../components/Provenance";
import { TranscriptView } from "../../components/TranscriptView";
import { clock } from "../../lib/format";
import { initialSignal, observe, warningFor } from "../../lib/signal";
import type { BackendState, Settings } from "../../types/protocol";
import { finalsOf, openOf } from "./reducer";
import type { useLive } from "./useLive";

type Live = ReturnType<typeof useLive>;

const STATE_LABEL = { idle: "Ready", starting: "Starting…", recording: "Recording", stopping: "Stopping…", error: "Error" } as const;

interface Props {
  live: Live;
  settings: Settings;
  onSettings: (s: Settings) => void;
  backend: BackendState | null;
  shortcut: string;
  onOpenModels: () => void;
}

export function LivePanel({ live, settings, onSettings, backend, shortcut, onOpenModels }: Props) {
  const { status, transcript, devices, error, mic, start, stop, refreshDevices, requestMic, openMicSettings } = live;
  const list = devices ?? [];
  const busy = status.state === "recording" || status.state === "starting" || status.state === "stopping";
  const finals = finalsOf(transcript);
  const selected = list.some((d) => d.id === settings.device_id) ? settings.device_id : null;
  const select = useRef<HTMLSelectElement>(null);

  const [sig, setSig] = useState(initialSignal());
  useEffect(() => setSig((p) => observe(p, status)), [status.session_id, status.elapsed_s, status.level_db, status.samples_in]); // eslint-disable-line react-hooks/exhaustive-deps
  const warning = warningFor(sig, status);
  const deviceName = status.device || list.find((d) => d.id === selected)?.name || "";

  const stopAndChoose = async () => {
    await stop();
    select.current?.focus();
  };
  const meterValue = Math.round(Math.max(-80, Math.min(0, status.level_db)));
  const meterPct = status.state === "recording" ? ((meterValue + 80) / 80) * 100 : 0;
  const mock = !backend || backend.info.mock;

  return (
    <section className="live">
      <div className="toolbar">
        <div className="group">
          {status.state === "recording" || status.state === "stopping" ? (
            <button className="danger fixed" onClick={stop} disabled={status.state === "stopping"} title={`Stop (${shortcut})`} aria-keyshortcuts={shortcut}>
              Stop
            </button>
          ) : (
            <button className="primary fixed" onClick={() => start(selected)} disabled={busy} title={`Start (${shortcut})`} aria-keyshortcuts={shortcut}>
              Start
            </button>
          )}
          <span className={`state ${status.state}`} role="status">
            <i className="dot" aria-hidden="true" />
            {STATE_LABEL[status.state]}
          </span>
          <span className="clock mono" aria-label="Elapsed time">{clock(status.elapsed_s)}</span>
        </div>
        <div className="group right">
          <label className="mic">
            <span className="visually-hidden">Microphone</span>
            <select ref={select} value={selected ?? ""} disabled={busy} onFocus={refreshDevices} onChange={(e) => onSettings({ ...settings, device_id: e.target.value || null })}>
              <option value="">System default</option>
              {list.map((d) => (
                <option key={d.id} value={d.id}>{d.is_default ? `${d.name} (default)` : d.name}</option>
              ))}
            </select>
          </label>
          <span
            className={`meter ${warning ? "alert" : ""}`}
            role="meter"
            aria-label="Input level"
            aria-valuemin={-80}
            aria-valuemax={0}
            aria-valuenow={meterValue}
            aria-valuetext={status.state === "recording" ? `${meterValue} dB` : "Not recording"}
          >
            <i style={{ width: `${meterPct}%` }} />
          </span>
        </div>
      </div>

      {mic === "not_determined" && (
        <p className="notice">
          <span>TingMix has not asked for microphone access yet.</span>
          <button onClick={requestMic}>Allow microphone access</button>
        </p>
      )}
      {(mic === "denied" || mic === "restricted") && (
        <p className="notice danger" role="alert">
          <span>
            {mic === "denied"
              ? "Microphone access is denied for TingMix, so nothing can be recorded."
              : "Microphone access is restricted on this Mac (profile or parental controls)."}
          </span>
          {mic === "denied" && <button onClick={openMicSettings}>Open System Settings</button>}
        </p>
      )}
      {devices !== null && list.length === 0 && <p className="muted">No microphone found. Connect one, then press Start to try the system default.</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {status.message && <p className="muted small">{status.message}</p>}
      {warning && (
        <div className="notice warn" role="alert">
          <div>
            <strong>{warning === "silence" ? (deviceName ? `No sound from “${deviceName}”` : "No sound from the system default microphone") : deviceName ? `No audio is arriving from “${deviceName}”.` : "No audio is arriving from the microphone."}</strong>
            {warning === "silence" && (
              <p>TingMix is receiving silence from this input. Check that it isn't muted, or choose another microphone. Remote-desktop and virtual audio devices often send silence.</p>
            )}
          </div>
          <button onClick={stopAndChoose}>Stop and choose another microphone</button>
        </div>
      )}

      <Provenance label={status.backend} />
      <TranscriptView
        label="Live captions"
        finals={finals}
        open={openOf(transcript)}
        size={settings.caption_size}
        follow
        live
        resetKey={status.session_id}
        empty={
          mock ? (
            <>
              <p>Demo mode — captions will be placeholder words. Set up a speech model to transcribe real speech.</p>
              <button onClick={onOpenModels}>Set up a speech model</button>
            </>
          ) : (
            <p>Press Start ({shortcut}) and speak. Captions appear here.</p>
          )
        }
      />
      <div className="row footer">
        <ExportButtons name="live-transcript" segments={finals} backend={status.backend} />
        <span className="muted small ellipsis" title={`${status.device} ${status.backend}`}>
          {status.device ? `${status.device} · ` : ""}
          {status.backend}
        </span>
        {status.overflow_samples > 0 && <span className="chip warn small">Some audio was dropped — the computer couldn't keep up</span>}
      </div>
    </section>
  );
}
