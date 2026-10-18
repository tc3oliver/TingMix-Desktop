import { describe, expect, it } from "vitest";
import { translator } from "../../i18n";
import { entryName } from "./HistoryPanel";

const date = (s: number) => `D${s}`;

describe("history titles", () => {
  it("replace the host's untranslated live name with a dated, translated one", () => {
    const live = { source: "live", name: "Live session", started_at: 7 };
    expect(entryName(live, translator("en"), date)).toBe("Live captions D7");
    expect(entryName(live, translator("zh-TW"), date)).toBe("即時字幕 D7");
  });
  it("keep a file's own name", () => {
    expect(entryName({ source: "file", name: "lecture.m4a", started_at: 1 }, translator("en"), date)).toBe("lecture.m4a");
  });
});
