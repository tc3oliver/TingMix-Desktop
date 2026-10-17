import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { clock } from "../lib/format";
import type { FinalEvent, PartialEvent } from "../types/protocol";

interface Props {
  finals: Pick<FinalEvent, "segment_id" | "text" | "start">[];
  open?: PartialEvent[];
  size: number;
  /** Shown inside the (always present) container while there is nothing to show. */
  empty: ReactNode;
  /** Accessible name of the scrollable region. */
  label: string;
  /** Keep the newest line in view while the user has not scrolled up (live captions, running jobs). */
  follow?: boolean;
  /** Announce finals to screen readers, once each; partials are never announced. */
  live?: boolean;
  /** A new value (a new recording) resumes following. */
  resetKey?: string | null;
}

const NEAR_BOTTOM_PX = 48;

export function TranscriptView({ finals, open = [], size, empty, label, follow = false, live = false, resetKey = null }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const atBottom = useRef(true);
  const [away, setAway] = useState(false);
  const count = finals.length + open.length;
  const signature = `${finals.length}:${open.map((p) => `${p.segment_id}.${p.revision}`).join(",")}`;

  useEffect(() => {
    atBottom.current = true;
    setAway(false);
  }, [resetKey]);

  // scrolls the container only (never scrollIntoView, which moves every ancestor), and only while the user is at the bottom
  useLayoutEffect(() => {
    const el = box.current;
    if (!el || !follow || count === 0) return;
    if (atBottom.current) el.scrollTop = el.scrollHeight;
    else setAway(true);
  }, [signature, follow, count]);

  const onScroll = () => {
    const el = box.current;
    if (!el) return;
    atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
    if (atBottom.current) setAway(false);
  };
  const jump = () => {
    const el = box.current;
    if (el) el.scrollTop = el.scrollHeight;
    atBottom.current = true;
    setAway(false);
  };

  return (
    <div ref={box} className="transcript" role="region" aria-label={label} tabIndex={0} onScroll={onScroll} style={{ fontSize: size }}>
      {count === 0 && <div className="placeholder muted">{empty}</div>}
      <div className="finals" role={live ? "log" : undefined} aria-relevant={live ? "additions" : undefined}>
        {finals.map((f) => (
          <p key={f.segment_id} className="line final">
            <time>{clock(f.start)}</time>
            <span>{f.text}</span>
          </p>
        ))}
      </div>
      <div className="partials">
        {open.map((p) => {
          const chars = [...p.text];
          return (
            <p key={p.segment_id} className="line partial" data-revision={p.revision}>
              <time>{clock(p.start)}</time>
              <span>
                {chars.slice(0, p.stable_prefix_len).join("")}
                <span className="unstable">{chars.slice(p.stable_prefix_len).join("")}</span>
              </span>
            </p>
          );
        })}
      </div>
      {away && (
        <button className="jump" onClick={jump}>
          Jump to latest
        </button>
      )}
    </div>
  );
}
