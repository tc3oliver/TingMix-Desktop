import { describe, expect, it } from "vitest";
import type { BackendState } from "../types/protocol";
import { demoBanner, engineChip } from "./engine";

const bs = (p: { mock: boolean; experimental?: boolean; label?: string; reason?: string; loading?: boolean; error?: string }): BackendState => ({
  info: { label: p.label ?? "Mock — not ASR", mock: p.mock, experimental: p.experimental ?? false, reason: p.reason ?? "", rejected: [], ort_library: null, experimental_allowed: true },
  load: { loading: p.loading ?? false, label: null, error: p.error ?? null },
});

describe("engine chip", () => {
  it("covers every state of the brief", () => {
    expect(engineChip(null)).toMatchObject({ text: "Checking engine…", tone: "neutral" });
    expect(engineChip(bs({ mock: true }))).toMatchObject({ text: "Demo mode", tone: "warn" });
    expect(engineChip(bs({ mock: false, label: "tingmix-large-mlx-fp32-r75071a1 (mlx) — EXPERIMENTAL", experimental: true, loading: true }))).toMatchObject({ text: "Loading TingMix Large MLX fp32…", tone: "neutral" });
    expect(engineChip(bs({ mock: false, error: "boom" }))).toMatchObject({ text: "Engine failed to load", tone: "danger", title: "boom" });
    expect(engineChip(bs({ mock: false, label: "tingmix-large-mlx-fp32-r75071a1 (mlx) — EXPERIMENTAL", experimental: true }))).toMatchObject({ text: "TingMix Large MLX fp32 · Experimental", tone: "warn" });
    expect(engineChip(bs({ mock: false, label: "m1 (mlx) — VALIDATED" }))).toMatchObject({ text: "M1", tone: "neutral" });
  });
});

describe("demo banner", () => {
  it("shows only in mock mode, with the generic call to action when no model is chosen", () => {
    expect(demoBanner(null).show).toBe(false);
    expect(demoBanner(bs({ mock: false })).show).toBe(false);
    expect(demoBanner(bs({ mock: true, reason: "No model is active; using the mock backend." }))).toEqual({ show: true, detail: null, openModelsLabel: "Set up a speech model" });
  });
  it("explains why a chosen model is not in use", () => {
    const b = demoBanner(bs({ mock: true, reason: "m1 can't be used yet. This model hasn't been tested on this Mac yet." }));
    expect(b.detail).toContain("hasn't been tested");
    expect(b.openModelsLabel).toBe("Open Models");
  });
});
