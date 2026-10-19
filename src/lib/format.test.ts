import { describe, expect, it } from "vitest";
import { basename, clock, friendlyName } from "./format";

describe("format", () => {
  it("formats durations as m:ss and h:mm:ss", () => {
    expect(clock(0)).toBe("00:00");
    expect(clock(75.9)).toBe("01:15");
    expect(clock(3600)).toBe("1:00:00");
    expect(clock(-3)).toBe("00:00");
    expect(clock(Number.NaN)).toBe("00:00");
  });
  it("takes the last path component", () => {
    expect(basename("/Users/me/notes.txt")).toBe("notes.txt");
    expect(basename("C:\\Users\\me\\notes.srt")).toBe("notes.srt");
  });
  it("turns model ids into readable names", () => {
    expect(friendlyName("tingmix-large-mlx-fp32-r75071a1")).toBe("TingMix Large");
    expect(friendlyName("tingmix-small-onnx-cpu-int8")).toBe("TingMix Small");
    expect(friendlyName("sensevoice-small-onnx-int8-r2365bae")).toBe("SenseVoice Small");
  });
});
