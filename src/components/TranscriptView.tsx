import { useEffect, useRef } from "react";
import type { FinalEvent, PartialEvent } from "../types/protocol";

interface Props {
  finals: Pick<FinalEvent, "segment_id" | "text" | "start">[];
  open?: PartialEvent[];
  size: number;
  empty: string;
}

const stamp = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export function TranscriptView({ finals, open = [], size, empty }: Props) {
  const end = useRef<HTMLDivElement>(null);
  const count = finals.length + open.length;
  useEffect(() => end.current?.scrollIntoView({ block: "end" }), [count, open[0]?.revision]);
  if (count === 0) return <p className="muted">{empty}</p>;
  return (
    <div className="transcript" style={{ fontSize: size }} aria-live="polite">
      {finals.map((f) => (
        <p key={f.segment_id} className="final">
          <time>{stamp(f.start)}</time> {f.text}
        </p>
      ))}
      {open.map((p) => {
        const chars = [...p.text];
        return (
          <p key={p.segment_id} className="partial" data-revision={p.revision}>
            <time>{stamp(p.start)}</time> {chars.slice(0, p.stable_prefix_len).join("")}
            <span className="unstable">{chars.slice(p.stable_prefix_len).join("")}</span>
          </p>
        );
      })}
      <div ref={end} />
    </div>
  );
}
