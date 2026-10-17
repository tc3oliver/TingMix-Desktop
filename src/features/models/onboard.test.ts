import { describe, expect, it } from "vitest";
import type { BackendState, ModelOp } from "../../types/protocol";
import { modelStatus, onboardStage } from "./onboard";

const op = (p: Partial<ModelOp>): ModelOp => ({ kind: "import", name: "m1", state: "running", message: null, bytes_done: 0, bytes_total: 0, ...p });
const backend = (p: { mock: boolean; label: string; reason?: string }): BackendState => ({
  info: { label: p.label, mock: p.mock, experimental: !p.mock, reason: p.reason ?? "", rejected: [], ort_library: null, experimental_allowed: true },
  load: { loading: false, label: null, error: null },
});
const real = backend({ mock: false, label: "m1 (mlx) — EXPERIMENTAL" });
const mock = backend({ mock: true, label: "Mock — not ASR", reason: "m1 can't be used yet. This model hasn't been tested on this Mac yet." });

describe("onboarding card", () => {
  it("narrates the steps in order", () => {
    expect(onboardStage(op({}), mock)?.text).toBe("Verifying and copying files…");
    expect(onboardStage(op({ kind: "download", bytes_done: 5 << 20, bytes_total: 10 << 20 }), mock)?.text).toBe("Downloading… 5 MB of 10 MB");
    expect(onboardStage(op({ state: "done" }), mock)).toEqual({ phase: "working", text: "Files verified. Testing on this Mac…" });
    expect(onboardStage(op({ kind: "probe" }), mock)?.text).toBe("Testing on this Mac…");
  });
  it("ends in Ready only when the backend in use is this model", () => {
    expect(onboardStage(op({ kind: "probe", state: "done" }), real)).toEqual({ phase: "ready", text: "Ready: m1 (mlx) — EXPERIMENTAL" });
    expect(onboardStage(op({ kind: "probe", state: "done", name: "other" }), real)?.phase).toBe("problem");
  });
  it("explains a finished test that did not lead to a usable backend, in the host's plain words", () => {
    const s = onboardStage(op({ kind: "probe", state: "done" }), mock);
    expect(s?.phase).toBe("problem");
    expect(s?.text).toContain("hasn't been tested");
  });
  it("shows failures and cancellations as problems", () => {
    expect(onboardStage(op({ state: "failed", message: "This build of TingMix doesn't include the MLX engine." }), mock)).toEqual({
      phase: "problem",
      text: "This build of TingMix doesn't include the MLX engine.",
    });
    expect(onboardStage(op({ state: "cancelled" }), mock)?.phase).toBe("problem");
    expect(onboardStage(null, mock)).toBeNull();
  });
  it("labels installed models", () => {
    expect(modelStatus({ id: "m1", active: false }, real)).toBe("Installed");
    expect(modelStatus({ id: "m1", active: true }, real)).toBe("In use");
    expect(modelStatus({ id: "m1", active: true }, mock)).toContain("Selected, but not in use.");
  });
});
