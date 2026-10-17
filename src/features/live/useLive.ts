import { useCallback, useEffect, useState } from "react";
import { api, errorText, onLive } from "../../lib/api";
import type { Device, LiveStatus, MicStatus } from "../../types/protocol";
import { applySessionEvent, applySnapshot, emptyTranscript, type TranscriptState } from "./reducer";

const IDLE: LiveStatus = {
  state: "idle", session_id: null, elapsed_s: 0, samples_in: 0, overflow_samples: 0,
  level_db: -120, device: "", backend: "", coalesced_partials: 0, message: null,
};

export function useLive() {
  const [status, setStatus] = useState<LiveStatus>(IDLE);
  const [transcript, setTranscript] = useState<TranscriptState>(emptyTranscript());
  const [devices, setDevices] = useState<Device[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [mic, setMic] = useState<MicStatus | null>(null);

  useEffect(() => {
    let off: (() => void) | undefined;
    let alive = true;
    // subscribe first, then ask for the snapshot: the reducer is idempotent, so overlap is harmless
    onLive((m) => {
      if (m.kind === "event") setTranscript((t) => applySessionEvent(t, m.session_id, m.event));
      else {
        const { kind: _k, ...st } = m;
        setStatus(st);
      }
    }).then((u) => (alive ? (off = u) : u()));
    api.liveSnapshot().then((s) => {
      if (!alive) return;
      setStatus(s.status);
      setTranscript((t) => applySnapshot(t.sessionId === s.status.session_id ? t : emptyTranscript(s.status.session_id), s.snapshot));
    }).catch((e) => setError(errorText(e)));
    api.listInputDevices().then((d) => alive && setDevices(d)).catch((e) => setError(errorText(e)));
    // the user may change the permission in System Settings and come back
    const refreshMic = () => api.micStatus().then((m) => alive && setMic(m)).catch(() => undefined);
    void refreshMic();
    window.addEventListener("focus", refreshMic);
    return () => {
      alive = false;
      off?.();
      window.removeEventListener("focus", refreshMic);
    };
  }, []);

  const refreshDevices = useCallback(() => api.listInputDevices().then(setDevices).catch((e) => setError(errorText(e))), []);

  const start = useCallback(async (deviceId: string | null) => {
    setError(null);
    try {
      setTranscript(emptyTranscript());
      setStatus(await api.liveStart(deviceId));
    } catch (e) {
      setError(errorText(e));
    } finally {
      // the first Start shows the system prompt, so the status has usually changed by now
      api.micStatus().then(setMic).catch(() => undefined);
    }
  }, []);

  const requestMic = useCallback(async () => {
    try {
      setMic(await api.micRequest());
    } catch (e) {
      setError(errorText(e));
    }
  }, []);

  const openMicSettings = useCallback(() => api.micOpenSettings().catch((e) => setError(errorText(e))), []);

  const stop = useCallback(async () => {
    try {
      setStatus(await api.liveStop());
    } catch (e) {
      setError(errorText(e));
    }
  }, []);

  return { status, transcript, devices, error, mic, start, stop, refreshDevices, requestMic, openMicSettings };
}
