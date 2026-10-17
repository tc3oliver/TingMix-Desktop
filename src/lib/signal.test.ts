import { describe, expect, it } from "vitest";
import { initialSignal, observe, warningFor } from "./signal";

const st = (session_id: string | null, elapsed_s: number, level_db: number, samples_in: number) => ({ session_id, elapsed_s, level_db, samples_in });

function run(frames: [number, number, number][], session = "s1") {
  let s = initialSignal(session);
  let w = null as ReturnType<typeof warningFor>;
  for (const [t, db, n] of frames) {
    s = observe(s, st(session, t, db, n));
    w = warningFor(s, { state: "recording", elapsed_s: t });
  }
  return { s, w };
}

describe("signal watcher", () => {
  it("is quiet while sound arrives", () => {
    const frames: [number, number, number][] = Array.from({ length: 40 }, (_, i) => [i * 0.25, -50, (i + 1) * 4000]);
    expect(run(frames).w).toBeNull();
  });
  it("warns about silence after 3 s of digital silence, naming no counter", () => {
    const frames: [number, number, number][] = Array.from({ length: 20 }, (_, i) => [i * 0.25, -120, (i + 1) * 4000]);
    expect(run(frames).w).toBe("silence");
    expect(run(frames.slice(0, 10)).w).toBeNull(); // 2.25 s
  });
  it("warns about missing data when the sample count stops growing", () => {
    const frames: [number, number, number][] = [[0, -50, 4000], [0.25, -50, 8000], [1, -50, 8000], [2.5, -50, 8000]];
    expect(run(frames).w).toBe("no_data");
  });
  it("clears after one reading above the threshold", () => {
    const silent: [number, number, number][] = Array.from({ length: 20 }, (_, i) => [i * 0.25, -120, (i + 1) * 4000]);
    expect(run([...silent, [5, -40, 100_000]]).w).toBeNull();
  });
  it("starts over on a new session", () => {
    const a = run(Array.from({ length: 20 }, (_, i) => [i * 0.25, -120, (i + 1) * 4000] as [number, number, number]));
    const b = observe(a.s, st("s2", 0.25, -120, 4000));
    expect(b.lastSoundAt).toBe(0);
    expect(warningFor(b, { state: "recording", elapsed_s: 0.25 })).toBeNull();
  });
  it("never warns when not recording", () => {
    expect(warningFor(initialSignal(), { state: "idle", elapsed_s: 99 })).toBeNull();
  });
});
