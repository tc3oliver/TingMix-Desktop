import { describe, expect, it } from "vitest";
import { exportNote, isMock, MOCK_NOTE } from "./provenance";

describe("provenance", () => {
  it("treats the host's mock label (and an unknown label) as mock", () => {
    expect(isMock("Mock — not ASR")).toBe(true);
    expect(isMock("")).toBe(true);
    expect(isMock(null)).toBe(true);
  });
  it("does not mark a real backend as mock", () => {
    expect(isMock("tingmix-large-mlx-fp32-r75071a1 (mlx) — EXPERIMENTAL")).toBe(false);
  });
  it("adds the note to exports only for mock output", () => {
    expect(exportNote("Mock — not ASR")).toBe(MOCK_NOTE);
    expect(exportNote("model (mlx) — EXPERIMENTAL")).toBeUndefined();
  });
});
