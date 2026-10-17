import { useEffect, useRef, useState } from "react";
import { ExportButtons } from "../../components/ExportButtons";
import { Provenance } from "../../components/Provenance";
import { TranscriptView } from "../../components/TranscriptView";
import { useT } from "../../i18n";
import { hostText } from "../../i18n/host";
import { clock } from "../../lib/format";
import { labelText } from "../../lib/provenance";
import { initialSignal, observe, warningFor } from "../../lib/signal";
import type { BackendState, Settings } from "../../types/protocol";
import { finalsOf, openOf } from "./reducer";
import type { useLive } from "./useLive";

type Live = ReturnType<typeof useLive>;

interface Props {
  live: Live;
  settings: Settings;
  onSettings: (s: Settings) => void;
  backend: BackendState | null;
  shortcut: string;
  onOpenModels: () => void;
}

export function LivePanel({ live, settings, onSettings, backend, shortcut, onOpenModels }: Props) {
  const t = useT();
  const { status, transcript, devices, error, mic, start, stop, refreshDevices, requestMic, openMicSettings } = live;
  const list = devices ?? [];
  const busy = status.state === "recording" || status.state === "starting" || status.state === "stopping";
  const finals = finalsOf(transcript);
  const selected = list.some((d) => d.id === settings.device_id) ? settings.device_id : null;
  const select = useRef<HTMLSelectElement>(null);

  const [sig, setSig] = useState(initialSignal());
  useEffect(() => setSig((p) => observe(p, status)), [status.session_id, status.elapsed_s, status.level_db, status.samples_in]); // eslint-disable-line react-hooks/exhaustive-deps
  const warning = warningFor(sig, status);
  const device = status.device || list.find((d) => d.id === selected)?.name || "";

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
            <button className="danger fixed" onClick={stop} disabled={status.state === "stopping"} title={t("live.stop_title", { shortcut })} aria-keyshortcuts={shortcut}>
              {t("live.stop")}
            </button>
          ) : (
            <button className="primary fixed" onClick={() => start(selected)} disabled={busy} title={t("live.start_title", { shortcut })} aria-keyshortcuts={shortcut}>
              {t("live.start")}
            </button>
          )}
          <span className={`state ${status.state}`} role="status">
            <i className="dot" aria-hidden="true" />
            {t(`live.state.${status.state}`)}
          </span>
          <span className="clock mono" aria-label={t("live.elapsed")}>{clock(status.elapsed_s)}</span>
        </div>
        <div className="group right">
          <label className="mic">
            <span className="visually-hidden">{t("live.mic")}</span>
            <select ref={select} value={selected ?? ""} disabled={busy} onFocus={refreshDevices} onChange={(e) => onSettings({ ...settings, device_id: e.target.value || null })}>
              <option value="">{t("live.mic_default")}</option>
              {list.map((d) => (
                <option key={d.id} value={d.id}>{d.is_default ? t("live.mic_default_device", { name: d.name }) : d.name}</option>
              ))}
            </select>
          </label>
          <span
            className={`meter ${warning ? "alert" : ""}`}
            role="meter"
            aria-label={t("live.meter")}
            aria-valuemin={-80}
            aria-valuemax={0}
            aria-valuenow={meterValue}
            aria-valuetext={status.state === "recording" ? t("live.meter_value", { db: meterValue }) : t("live.meter_idle")}
          >
            <i style={{ width: `${meterPct}%` }} />
          </span>
        </div>
      </div>

      {mic === "not_determined" && (
        <p className="notice">
          <span>{t("mic.not_asked")}</span>
          <button onClick={requestMic}>{t("mic.allow")}</button>
        </p>
      )}
      {(mic === "denied" || mic === "restricted") && (
        <p className="notice danger" role="alert">
          <span>{mic === "denied" ? t("mic.denied") : t("mic.restricted")}</span>
          {mic === "denied" && <button onClick={openMicSettings}>{t("mic.open_settings")}</button>}
        </p>
      )}
      {devices !== null && list.length === 0 && <p className="muted">{t("live.no_mic")}</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {status.message && <p className="muted small">{hostText(status.message, t)}</p>}
      {warning && (
        <div className="notice warn" role="alert">
          <div>
            <strong>
              {warning === "silence"
                ? device ? t("signal.silence_title", { device }) : t("signal.silence_title_default")
                : device ? t("signal.no_data", { device }) : t("signal.no_data_default")}
            </strong>
            {warning === "silence" && <p>{t("signal.silence_body")}</p>}
          </div>
          <button onClick={stopAndChoose}>{t("signal.stop_choose")}</button>
        </div>
      )}

      <Provenance label={status.backend} />
      <TranscriptView
        label={t("live.captions")}
        finals={finals}
        open={openOf(transcript)}
        size={settings.caption_size}
        follow
        live
        resetKey={status.session_id}
        empty={
          mock ? (
            <>
              <p>{t("live.empty_demo")}</p>
              <button onClick={onOpenModels}>{t("banner.setup")}</button>
            </>
          ) : (
            <p>{t("live.empty_real", { shortcut })}</p>
          )
        }
      />
      <div className="row footer">
        <ExportButtons name="live-transcript" segments={finals} backend={status.backend} />
        <span className="muted small ellipsis" title={`${status.device} ${labelText(status.backend, t)}`}>
          {status.device ? `${status.device} · ` : ""}
          {labelText(status.backend, t)}
        </span>
        {status.overflow_samples > 0 && <span className="chip warn small">{t("live.dropped")}</span>}
      </div>
    </section>
  );
}
