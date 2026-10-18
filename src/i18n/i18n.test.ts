import { describe, expect, it } from "vitest";
import { en } from "./en";
import { hostCode, hostText, reasonText } from "./host";
import { detectLocale, format, LOCALES, makeI18n, resolveLocale, translator } from "./index";
import { zhTW } from "./zh-TW";

const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]!).sort();

describe("dictionaries", () => {
  it("have the same keys in every language (also enforced by the types)", () => {
    for (const [id, l] of Object.entries(LOCALES)) {
      expect(Object.keys(l.dict).sort(), id).toEqual(Object.keys(en).sort());
    }
  });
  it("use the same parameters in every language and never leave a string empty", () => {
    for (const k of Object.keys(en) as (keyof typeof en)[]) {
      expect(zhTW[k].trim(), k).not.toBe("");
      expect(placeholders(zhTW[k]), k).toEqual(placeholders(en[k]));
    }
  });
  it("keep plural pairs complete", () => {
    for (const k of Object.keys(en).filter((k) => k.endsWith("_one"))) expect(k.replace(/_one$/, "_other") in en, k).toBe(true);
  });
});

describe("locale choice", () => {
  it("follows the first-ranked system language: Traditional Chinese variants, otherwise English", () => {
    for (const l of ["zh-TW", "zh_TW", "zh-Hant", "zh-Hant-TW", "zh-HK", "zh-MO"]) expect(detectLocale([l, "en-US"]), l).toBe("zh-TW");
    expect(detectLocale(["en-US", "zh-TW"])).toBe("en");
    expect(detectLocale(["zh-CN"])).toBe("en");
    expect(detectLocale([])).toBe("en");
  });
  it("lets an explicit setting win over the system language", () => {
    expect(resolveLocale("en", ["zh-TW"])).toBe("en");
    expect(resolveLocale("zh-TW", ["en-US"])).toBe("zh-TW");
    expect(resolveLocale("auto", ["zh-TW"])).toBe("zh-TW");
    expect(resolveLocale("nonsense", ["en"])).toBe("en");
  });
});

describe("formatting", () => {
  it("fills named parameters and leaves a missing one visible", () => {
    expect(format("Saved {file}", { file: "a.txt" })).toBe("Saved a.txt");
    expect(format("Saved {file}")).toBe("Saved {file}");
    expect(translator("zh-TW")("live.start_title", { shortcut: "⌘↵" })).toBe("開始（⌘↵）");
  });
  it("picks plural forms with Intl.PluralRules per locale and formats dates per locale", () => {
    const e = makeI18n("en");
    expect([e.n("history.segments", 1), e.n("history.segments", 2)]).toEqual(["1 segment", "2 segments"]);
    expect(makeI18n("zh-TW").n("history.segments", 3)).toBe("3 段");
    expect(e.date(0)).not.toBe(makeI18n("zh-TW").date(0));
  });
});

describe("platform words", () => {
  it("say this Mac only on macOS, and use the platform's own settings path", () => {
    const mac = translator("en", () => true);
    const win = translator("en", () => false);
    expect(mac("mic.restricted")).toContain("on this Mac");
    expect(win("mic.restricted")).toContain("on this computer");
    expect(mac("mic.denied")).toContain("System Settings › Privacy & Security › Microphone");
    expect(win("mic.denied")).toContain("Settings › Privacy & security › Microphone");
    expect(translator("zh-TW", () => false)("mic.restricted")).toBe("這台電腦限制了麥克風存取（描述檔或家長監護）。");
    expect(win("mic.open_settings")).toBe("Open Settings");
  });
  it("never leave a host detail in user-facing text", () => {
    for (const k of ["host.live.disconnected", "host.setup.offline", "files.failed", "app.load_error", "host.backend.unreadable"] as const) expect(en[k], k).not.toContain("{detail}");
  });
});

describe("host messages", () => {
  const coded = (code: string, text: string, params: Record<string, unknown> = {}) => JSON.stringify({ code, params, text });
  it("translate by code and keep plain text as it is", () => {
    const m = coded("mic.denied", "English fallback");
    expect(hostText(m, translator("en"))).toContain("can't use the microphone");
    expect(hostText(m, translator("zh-TW"))).toContain("麥克風");
    expect(hostText("io error at /x", translator("zh-TW"))).toBe("io error at /x");
    expect(hostText(null)).toBe("");
  });
  it("show a generic sentence in the UI language for a code this UI does not know, never the host's English", () => {
    expect(hostText(coded("future.thing", "Brand new message"), translator("zh-TW"))).toContain("發生問題");
    expect(hostText(coded("future.thing", "Brand new message"), translator("en"))).toContain("Something went wrong");
  });
  it("translate nested reasons and parameters, choosing the MLX variant of the missing-engine text", () => {
    const why = coded("backend.missing_engine", "x", { engine: "MLX", mlx: true });
    const m = coded("backend.cant_use", "x", { model: "m1", why });
    expect(hostText(m, translator("en"))).toContain("m1 can't be used yet. This build of TingMix doesn't include the MLX engine");
    expect(hostText(m, translator("en"))).toContain("Use the MLX build");
    expect(reasonText(m, translator("en"))).toBe("This build of TingMix doesn't include the MLX engine this model needs. Use the MLX build of TingMix.");
    expect(hostCode(m)).toBe("backend.missing_engine");
  });
});

// every code in the host's table has a sentence in both languages: an unknown code would otherwise show the generic text
const hostTable = Object.values(import.meta.glob("../../docs/architecture/host-codes.md", { query: "?raw", import: "default", eager: true }) as Record<string, string>)[0] ?? "";
describe("host codes", () => {
  const codes = [...hostTable.matchAll(/^\| `([a-z_]+\.[a-z_]+)` \|/gm)].map((m) => m[1]!);
  it("finds the documented codes", () => expect(codes.length).toBeGreaterThan(50));
  it("each has an entry in every language", () => {
    for (const c of codes) for (const [id, l] of Object.entries(LOCALES)) expect(`host.${c}` in l.dict, `${id}: ${c}`).toBe(true);
  });
});
