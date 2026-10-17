import type { FinalEvent, PartialEvent, Snapshot, TranscriptEvent } from "../../types/protocol";

/** Same rules as `tingmix_core::Reducer`: the newest partial wins per open segment, finals are immutable. */
export interface TranscriptState {
  sessionId: string | null;
  finals: Record<number, FinalEvent>;
  open: Record<number, PartialEvent>;
}

export const emptyTranscript = (sessionId: string | null = null): TranscriptState => ({ sessionId, finals: {}, open: {} });

/** Returns the same object when the event is rejected (closed segment or revision not newer). */
export function applyEvent(s: TranscriptState, ev: TranscriptEvent): TranscriptState {
  if (ev.segment_id in s.finals) return s;
  const cur = s.open[ev.segment_id];
  if (cur && ev.revision <= cur.revision) return s;
  if (ev.type === "final") {
    const { [ev.segment_id]: _closed, ...open } = s.open;
    return { ...s, finals: { ...s.finals, [ev.segment_id]: ev }, open };
  }
  return { ...s, open: { ...s.open, [ev.segment_id]: ev } };
}

/** Merge a host snapshot into the state. Idempotent, so it is safe against events delivered around it. */
export function applySnapshot(s: TranscriptState, snap: Snapshot): TranscriptState {
  let out = s;
  for (const f of snap.finals) out = applyEvent(out, { type: "final", ...f });
  for (const p of snap.open) out = applyEvent(out, { type: "partial", ...p });
  return out;
}

/** Events carry their session; a different id means a new recording, so the old transcript is dropped from view. */
export function applySessionEvent(s: TranscriptState, sessionId: string, ev: TranscriptEvent): TranscriptState {
  const base = s.sessionId === sessionId ? s : emptyTranscript(sessionId);
  return applyEvent(base, ev);
}

export const finalsOf = (s: TranscriptState): FinalEvent[] =>
  Object.values(s.finals).sort((a, b) => a.segment_id - b.segment_id);

export const openOf = (s: TranscriptState): PartialEvent[] =>
  Object.values(s.open).sort((a, b) => a.segment_id - b.segment_id);
