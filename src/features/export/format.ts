export interface Segment {
  start: number;
  end: number;
  text: string;
}

/** `HH:MM:SS,mmm`; rounds to the millisecond first so 59.9996 s becomes 00:01:00,000 and never `00:00:60,000`. */
export function srtTime(seconds: number): string {
  const total = Math.max(0, Math.round((Number.isFinite(seconds) ? seconds : 0) * 1000));
  const ms = total % 1000;
  const s = Math.floor(total / 1000) % 60;
  const m = Math.floor(total / 60_000) % 60;
  const h = Math.floor(total / 3_600_000);
  const p = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${p(h)}:${p(m)}:${p(s)},${p(ms, 3)}`;
}

/** Cues are numbered from 1, empty segments are dropped, and a cue never ends before it starts. */
export function toSrt(segments: Segment[]): string {
  const cues = segments
    .map((s) => ({ ...s, text: s.text.trim() }))
    .filter((s) => s.text !== "")
    .map((s, i) => {
      const end = Math.max(s.end, s.start + 0.001);
      return `${i + 1}\n${srtTime(s.start)} --> ${srtTime(end)}\n${s.text}\n`;
    });
  return cues.join("\n");
}

/** One segment per line. */
export function toTxt(segments: Segment[]): string {
  const lines = segments.map((s) => s.text.trim()).filter((t) => t !== "");
  return lines.length ? lines.join("\n") + "\n" : "";
}
