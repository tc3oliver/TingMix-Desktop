import { describe, expect, it } from "vitest";
import type { FinalEvent, PartialEvent } from "../../types/protocol";
import { applyEvent, applySessionEvent, applySnapshot, emptyTranscript, finalsOf, openOf } from "./reducer";

const partial = (segment_id: number, revision: number, text: string): PartialEvent => ({
  type: "partial", segment_id, revision, text, start: 0, end: 1, stable_prefix_len: 0,
});
const final = (segment_id: number, revision: number, text: string): FinalEvent => ({
  type: "final", segment_id, revision, text, start: 0, end: 1,
});

describe("transcript reducer", () => {
  it("replaces a partial in place by revision", () => {
    let s = emptyTranscript();
    s = applyEvent(s, partial(0, 0, "a"));
    s = applyEvent(s, partial(0, 1, "ab"));
    expect(openOf(s).map((p) => p.text)).toEqual(["ab"]);
  });

  it("ignores stale and duplicate revisions without changing the state object", () => {
    let s = applyEvent(emptyTranscript(), partial(0, 2, "new"));
    expect(applyEvent(s, partial(0, 1, "old"))).toBe(s);
    expect(applyEvent(s, partial(0, 2, "dup"))).toBe(s);
  });

  it("finalizing removes the open partial and finals are immutable", () => {
    let s = applyEvent(emptyTranscript(), partial(0, 0, "a"));
    s = applyEvent(s, final(0, 1, "a b"));
    expect(openOf(s)).toEqual([]);
    expect(applyEvent(s, partial(0, 5, "late"))).toBe(s);
    expect(applyEvent(s, final(0, 9, "rewrite"))).toBe(s);
    expect(finalsOf(s).map((f) => f.text)).toEqual(["a b"]);
  });

  it("accepts a final that arrives without any partial", () => {
    expect(finalsOf(applyEvent(emptyTranscript(), final(3, 0, "x")))).toHaveLength(1);
  });

  it("orders finals by segment id regardless of arrival order", () => {
    let s = applyEvent(emptyTranscript(), final(2, 0, "c"));
    s = applyEvent(s, final(0, 0, "a"));
    s = applyEvent(s, final(1, 0, "b"));
    expect(finalsOf(s).map((f) => f.text)).toEqual(["a", "b", "c"]);
  });

  it("recovers from a host snapshot after a reload and tolerates replayed events", () => {
    const snap = {
      finals: [{ segment_id: 0, revision: 3, text: "hello", start: 0, end: 2 }],
      open: [{ segment_id: 1, revision: 1, text: "wor", start: 2, end: 3, stable_prefix_len: 1 }],
    };
    let s = applySnapshot(emptyTranscript("live-1"), snap);
    expect(finalsOf(s)).toHaveLength(1);
    expect(openOf(s)[0]?.text).toBe("wor");
    s = applyEvent(s, partial(1, 1, "stale replay"));
    s = applySnapshot(s, snap);
    expect(openOf(s)[0]?.text).toBe("wor");
  });

  it("starts a fresh transcript when an event belongs to another session", () => {
    let s = applySessionEvent(emptyTranscript(), "live-1", final(0, 0, "old"));
    s = applySessionEvent(s, "live-2", partial(0, 0, "new"));
    expect(finalsOf(s)).toEqual([]);
    expect(s.sessionId).toBe("live-2");
    expect(openOf(s)[0]?.text).toBe("new");
  });

  it("an empty final retracts the caption: the partial disappears, no blank row appears, late partials stay out", () => {
    let s = applyEvent(emptyTranscript(), partial(0, 0, "background noise"));
    s = applyEvent(s, final(0, 1, ""));
    expect(openOf(s)).toEqual([]);
    expect(finalsOf(s)).toEqual([]);
    expect(applyEvent(s, partial(0, 5, "late"))).toBe(s);
    s = applyEvent(s, final(1, 0, "real speech"));
    expect(finalsOf(s).map((f) => f.text)).toEqual(["real speech"]);
    // the host's snapshot contains the empty final too
    const snap = { finals: [{ segment_id: 0, revision: 1, text: "  ", start: 0, end: 1 }], open: [] };
    expect(finalsOf(applySnapshot(emptyTranscript(), snap))).toEqual([]);
  });
});
