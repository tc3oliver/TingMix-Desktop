import { describe, expect, it } from "vitest";
import { makeI18n, translator } from "../../i18n";
import type { BackendState, ModelOp } from "../../types/protocol";
import { cardStatus, etaSeconds, etaText, onboardStage, pushSample } from "./onboard";

const t = translator("en");
const tz = translator("zh-TW");
const coded = (code: string, text: string, params: Record<string, unknown> = {}) => JSON.stringify({ code, params, text });
const op = (p: Partial<ModelOp>): ModelOp => ({ kind: "import", name: "m1", state: "running", message: null, bytes_done: 0, bytes_total: 0, ...p });
const backend = (p: { mock: boolean; label: string; reason?: string }): BackendState => ({
  info: { label: p.label, mock: p.mock, experimental: !p.mock, reason: p.reason ?? "", rejected: [], ort_library: null, demo_allowed: true, usable: true },
  load: { loading: false, label: null, error: null },
});
const real = backend({ mock: false, label: "m1 (mlx) — EXPERIMENTAL" });
const untested = coded("backend.untested", "This model hasn't been tested on this Mac yet.");
const cant = (why: string) => coded("backend.cant_use", "m1 can't be used yet.", { model: "m1", why });
const mock = backend({ mock: true, label: "Mock — not ASR", reason: cant(untested) });

describe("onboarding card", () => {
  it("narrates the steps in order", () => {
    expect(onboardStage(op({}), mock, t)?.text).toBe("Finishing setup, please wait…");
    expect(onboardStage(op({ kind: "download", bytes_done: 5 << 20, bytes_total: 10 << 20 }), mock, t)?.text).toBe("Downloading… 5 MB of 10 MB");
    expect(onboardStage(op({ state: "done" }), mock, t)).toEqual({ phase: "working", text: "Finishing setup, please wait…" });
    expect(onboardStage(op({ kind: "probe" }), mock, t)?.text).toBe("Finishing setup, please wait…");
  });
  it("ends in Ready only when the backend in use is this model", () => {
    expect(onboardStage(op({ kind: "probe", state: "done" }), real, t)).toEqual({ phase: "ready", text: "Ready: M1" });
    expect(onboardStage(op({ kind: "probe", state: "done", name: "other" }), real, t)?.phase).toBe("problem");
  });
  it("explains a finished test that did not lead to a usable backend, in the active language", () => {
    const s = onboardStage(op({ kind: "probe", state: "done" }), mock, t);
    expect(s?.phase).toBe("problem");
    expect(s?.text).toContain("hasn't checked the speech model");
    expect(onboardStage(op({ kind: "probe", state: "done" }), mock, tz)?.text).toContain("還沒有");
  });
  it("shows failures and cancellations as problems", () => {
    const failed = coded("backend.missing_engine", "x", { engine: "MLX", mlx: true });
    expect(onboardStage(op({ state: "failed", message: failed }), mock, t)?.text).toContain("doesn't include the MLX engine");
    expect(onboardStage(op({ state: "cancelled" }), mock, t)?.phase).toBe("problem");
    expect(onboardStage(null, mock, t)).toBeNull();
  });
});

describe("model card status line", () => {
  const m = { id: "m1", active: true };
  const withReason = (why: string) => backend({ mock: true, label: "Mock — not ASR", reason: cant(why) });
  it("says Ready only for the model in use", () => {
    expect(cardStatus(m, real, null, t)).toBe("Ready");
    expect(cardStatus({ id: "m1", active: false }, real, null, t)).toBe("Not selected");
  });
  it("shows a running test and maps the host's reason codes to short lines", () => {
    expect(cardStatus(m, mock, op({ kind: "probe" }), t)).toBe("Checking…");
    expect(cardStatus(m, withReason(coded("backend.missing_engine", "x", { engine: "MLX", mlx: true })), null, t)).toBe("Not supported in this build.");
    expect(cardStatus(m, mock, null, t)).toBe("Not tested yet.");
    expect(cardStatus(m, withReason(coded("backend.test_failed", "x")), null, t)).toMatch(/^Test failed\. The speech model failed/);
    expect(cardStatus(m, mock, null, tz)).toBe("尚未測試。");
  });
});

describe("download rate and time left", () => {
  const gb = 1e9;
  it("stays unknown until 5 s of data exist, then estimates from the last 10 s", () => {
    let s = pushSample([], { at: 0, bytes: 0 });
    expect(etaSeconds(s, gb)).toBeNull();
    s = pushSample(s, { at: 4000, bytes: 40e6 });
    expect(etaSeconds(s, gb)).toBeNull();
    s = pushSample(s, { at: 10_000, bytes: 100e6 });
    expect(etaSeconds(s, gb)).toBeCloseTo(90, 0); // 10 MB/s, 900 MB left
    s = pushSample(s, { at: 30_000, bytes: 300e6 });
    expect(s.length).toBe(1); // older samples fell out of the window
  });
  it("restarts when the byte count goes backwards and is unknown when nothing arrives", () => {
    expect(pushSample([{ at: 0, bytes: 500 }], { at: 100, bytes: 0 })).toEqual([{ at: 100, bytes: 0 }]);
    expect(etaSeconds([{ at: 0, bytes: 5 }, { at: 6000, bytes: 5 }], 100)).toBeNull();
  });
  it("rounds to whole minutes in the UI language", () => {
    const n = (l: "en" | "zh-TW") => makeI18n(l).n;
    expect(etaText(30, n("en"), t)).toBe("less than a minute");
    expect(etaText(190, n("en"), t)).toBe("3 minutes");
    expect(etaText(190, n("zh-TW"), tz)).toBe("3 分鐘");
  });
});
