import type { LiveStatus } from "../types/protocol";

/** Level above which a read counts as sound. The host reports the RMS of the last read chunk in dBFS (digital silence = -120). */
export const SOUND_DB = -90;
export const SILENCE_AFTER_S = 3;
export const NO_DATA_AFTER_S = 2;

export interface SignalState {
  session: string | null;
  lastSoundAt: number;
  lastSamplesAt: number;
  lastSamples: number;
}

export type SignalWarning = "silence" | "no_data" | null;

export const initialSignal = (session: string | null = null): SignalState => ({ session, lastSoundAt: 0, lastSamplesAt: 0, lastSamples: 0 });

/** Fold one status update (every 250 ms from the host) into the watcher state; resets on a new session. */
export function observe(prev: SignalState, s: Pick<LiveStatus, "session_id" | "elapsed_s" | "level_db" | "samples_in">): SignalState {
  const base = prev.session === s.session_id ? prev : initialSignal(s.session_id);
  return {
    session: s.session_id,
    lastSoundAt: s.level_db > SOUND_DB ? s.elapsed_s : base.lastSoundAt,
    lastSamplesAt: s.samples_in > base.lastSamples ? s.elapsed_s : base.lastSamplesAt,
    lastSamples: Math.max(base.lastSamples, s.samples_in),
  };
}

/** Only while recording: no new samples for 2 s means no audio is arriving; samples that are all silent for 3 s means a muted or virtual input. */
export function warningFor(st: SignalState, s: Pick<LiveStatus, "state" | "elapsed_s">): SignalWarning {
  if (s.state !== "recording") return null;
  if (s.elapsed_s - st.lastSamplesAt >= NO_DATA_AFTER_S) return "no_data";
  if (s.elapsed_s - st.lastSoundAt >= SILENCE_AFTER_S) return "silence";
  return null;
}
