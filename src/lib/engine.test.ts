import { describe, expect, it } from "vitest";
import { translator } from "../i18n";
import type { BackendState } from "../types/protocol";
import { demoBanner, engineChip, isFirstRun, needsSetup } from "./engine";

const t = translator("en");
const coded = (code: string, text: string, params: Record<string, unknown> = {}) => JSON.stringify({ code, params, text });
const bs = (p: { mock: boolean; experimental?: boolean; label?: string; reason?: string; loading?: boolean; error?: string; release?: boolean }): BackendState => ({
  info: { label: p.label ?? "Mock — not ASR", mock: p.mock, experimental: p.experimental ?? false, reason: p.reason ?? "", rejected: [], ort_library: null, demo_allowed: !p.release, usable: !(p.mock && p.release) },
  load: { loading: p.loading ?? false, label: null, error: p.error ?? null },
});

describe("engine chip", () => {
  it("covers every state of the brief", () => {
    expect(engineChip(null, t)).toBeNull();
    expect(engineChip(bs({ mock: true }), t)).toMatchObject({ text: "Demo mode", tone: "warn" });
    expect(engineChip(bs({ mock: false, label: "tingmix-large-mlx-fp32-r75071a1 (mlx) — EXPERIMENTAL", experimental: true, loading: true }), t)).toMatchObject({ text: "Getting ready…", tone: "neutral" });
    expect(engineChip(bs({ mock: false, error: "boom" }), t)).toMatchObject({ text: "Speech recognition unavailable", tone: "danger", title: "boom" });
    // a ready app shows no engine, model or status word at all
    expect(engineChip(bs({ mock: false, label: "tingmix-large-mlx-fp32-r75071a1 (mlx) — EXPERIMENTAL", experimental: true }), t)).toBeNull();
    expect(engineChip(bs({ mock: false, label: "m1 (mlx) — VALIDATED" }), t)).toBeNull();
  });
  it("speaks Traditional Chinese when asked to", () => {
    expect(engineChip(bs({ mock: true }), translator("zh-TW"))?.text).toBe("示範模式");
  });
});

describe("release builds", () => {
  const release = bs({ mock: true, release: true, reason: coded("backend.no_model", "No model is active") });
  it("send the user to setup instead of offering the placeholder", () => {
    expect(needsSetup(release)).toBe(true);
    expect(needsSetup(bs({ mock: true }))).toBe(false); // debug build: the placeholder is usable
    expect(needsSetup(bs({ mock: false, release: true, label: "m1 (mlx) — EXPERIMENTAL" }))).toBe(false);
    expect(needsSetup(null)).toBe(false); // still loading: no flash of the setup screen
  });
  it("show no demo banner and an unavailable chip", () => {
    expect(demoBanner(release, t).show).toBe(false);
    expect(engineChip(release, t)).toMatchObject({ text: "Speech recognition unavailable", tone: "danger" });
  });
});

describe("demo banner", () => {
  it("shows only in mock mode, with the generic call to action when no model is chosen", () => {
    expect(demoBanner(null, t).show).toBe(false);
    expect(demoBanner(bs({ mock: false }), t).show).toBe(false);
    expect(demoBanner(bs({ mock: true, reason: coded("backend.no_model", "No model is active") }), t)).toEqual({ show: true, detail: null, openModelsLabel: "Set up a speech model" });
  });
  it("explains why a chosen model is not in use", () => {
    const reason = coded("backend.cant_use", "m1 can't be used yet.", { model: "m1", why: coded("backend.untested", "x") });
    const b = demoBanner(bs({ mock: true, reason }), t);
    expect(b.detail).toContain("hasn't checked the speech model");
    expect(b.openModelsLabel).toBe("Open Settings");
  });
});

describe("first run", () => {
  const release = bs({ mock: true, release: true });
  it("takes the whole window until a model is usable, but a broken install keeps the tabs", () => {
    expect(isFirstRun(release, null)).toBe(true);
    expect(isFirstRun(release, { state: "offline", reason: "x" })).toBe(true);
    expect(isFirstRun(release, { state: "problem", reason: "x" })).toBe(false);
    expect(isFirstRun(bs({ mock: false, label: "m1 (mlx) — VALIDATED" }), null)).toBe(false);
    expect(isFirstRun(null, null)).toBe(false);
  });
});
