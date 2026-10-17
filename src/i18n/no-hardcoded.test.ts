import { describe, expect, it } from "vitest";

// every component source, as raw text (Vite's glob import, so no Node typings are needed)
const sources = import.meta.glob("../**/*.tsx", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

const strip = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");

// Heuristic scan of the components: JSX text with letters and literal title/aria-label/placeholder/alt attributes must go
// through t(). Text that is not language (symbols such as "×") has no letters; the app name is listed here.
const ALLOWED = new Set(["TingMix"]);
const LETTER = "A-Za-z\\u4e00-\\u9fff";

export function scan(source: string): string[] {
  const code = strip(source);
  const found: string[] = [];
  const text = new RegExp(`(?<!=)>([^<>{}\\n=;()]*[${LETTER}]{2,}[^<>{}\\n=;()]*)<`, "g");
  for (const m of code.matchAll(text)) {
    const s = m[1]!.trim();
    if (s && !ALLOWED.has(s)) found.push(`text “${s}”`);
  }
  const attr = new RegExp(`\\b(title|aria-label|placeholder|alt)="([^"{}]*[${LETTER}]{2,}[^"{}]*)"`, "g");
  for (const m of code.matchAll(attr)) found.push(`${m[1]}="${m[2]}"`);
  return found;
}

describe("the scanner itself", () => {
  it("finds JSX text and literal accessibility attributes, and ignores code and translated text", () => {
    expect(scan("<p>Hello world</p>")).toEqual(["text “Hello world”"]);
    expect(scan('<button aria-label="Close it">{t("x")}</button>')).toEqual(['aria-label="Close it"']);
    expect(scan("<p>設定語音模型</p>")).toEqual(["text “設定語音模型”"]);
    expect(scan('<button>{t("common.cancel")}</button> <i>×</i>')).toEqual([]);
    expect(scan("const f = (): Promise<void> => g<Foo>(x); const h = () => <div />;")).toEqual([]);
  });
});

describe("components contain no hard-coded user-visible text", () => {
  it("finds the component sources", () => expect(Object.keys(sources).length).toBeGreaterThan(8));
  for (const [file, source] of Object.entries(sources)) {
    it(file.replace("../", ""), () => expect(scan(source)).toEqual([]));
  }
});
