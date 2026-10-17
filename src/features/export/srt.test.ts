import { describe, expect, it } from "vitest";
import { srtTime, toSrt, toTxt } from "./format";

describe("srtTime", () => {
  it("formats hours, minutes, seconds and milliseconds", () => {
    expect(srtTime(0)).toBe("00:00:00,000");
    expect(srtTime(3661.5)).toBe("01:01:01,500");
    expect(srtTime(0.0004)).toBe("00:00:00,000");
  });

  it("carries rounding into the next unit instead of printing 60", () => {
    expect(srtTime(59.9996)).toBe("00:01:00,000");
    expect(srtTime(3599.9999)).toBe("01:00:00,000");
  });

  it("clamps negative and non-finite input", () => {
    expect(srtTime(-5)).toBe("00:00:00,000");
    expect(srtTime(Number.NaN)).toBe("00:00:00,000");
  });
});

describe("toSrt", () => {
  it("numbers cues from 1, separates them with a blank line and keeps CJK text intact", () => {
    const srt = toSrt([
      { start: 0, end: 1.25, text: "今天天氣很好" },
      { start: 2, end: 3.5, text: "hello world" },
    ]);
    expect(srt).toBe("1\n00:00:00,000 --> 00:00:01,250\n今天天氣很好\n\n2\n00:00:02,000 --> 00:00:03,500\nhello world\n");
  });

  it("skips empty segments without leaving gaps in the numbering", () => {
    const srt = toSrt([
      { start: 0, end: 1, text: "  " },
      { start: 1, end: 2, text: "a" },
    ]);
    expect(srt.startsWith("1\n00:00:01,000")).toBe(true);
  });

  it("never lets a cue end before it starts", () => {
    expect(toSrt([{ start: 5, end: 4, text: "x" }])).toContain("00:00:05,000 --> 00:00:05,001");
  });

  it("is empty for no segments", () => {
    expect(toSrt([])).toBe("");
  });
});

describe("toTxt", () => {
  it("writes one trimmed segment per line", () => {
    expect(toTxt([{ start: 0, end: 1, text: " a " }, { start: 1, end: 2, text: "" }, { start: 2, end: 3, text: "b" }])).toBe("a\nb\n");
    expect(toTxt([])).toBe("");
  });
});
