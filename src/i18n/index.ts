import { createContext, createElement, useContext, useEffect, useMemo, type ReactNode } from "react";
import { en, type Key } from "./en";
import { zhTW } from "./zh-TW";

// To add a language: create `<id>.ts` typed as `Record<Key, string>`, add one entry here and one in
// `src-tauri/src/locale.rs` (native dialogs and the tray). Nothing else changes.
export const LOCALES = {
  en: { name: "English", dict: en as Record<Key, string>, bcp47: "en", htmlLang: "en" },
  "zh-TW": { name: "繁體中文", dict: zhTW, bcp47: "zh-TW", htmlLang: "zh-Hant" },
} as const;
export type LocaleId = keyof typeof LOCALES;
export type Preference = "auto" | LocaleId;
export const isLocaleId = (s: string): s is LocaleId => s in LOCALES;

/** Traditional Chinese for zh-TW, zh-HK, zh-MO and zh-Hant*; the user's first-ranked language decides, anything else is English. */
export function detectLocale(languages: readonly string[]): LocaleId {
  const first = (languages[0] ?? "en").toLowerCase().replace("_", "-");
  return first === "zh-tw" || first === "zh-hk" || first === "zh-mo" || first.startsWith("zh-hant") ? "zh-TW" : "en";
}

export const resolveLocale = (pref: string, languages: readonly string[]): LocaleId => (isLocaleId(pref) ? pref : detectLocale(languages));

export type Params = Record<string, string | number>;
export type Translate = (key: Key, params?: Params) => string;

/** `{name}` placeholders; a missing parameter is left visible rather than silently dropped. */
export function format(template: string, params: Params = {}): string {
  return template.replace(/\{(\w+)\}/g, (m, k: string) => (k in params ? String(params[k]) : m));
}

/** `{computer}`, `{system_settings}` and `{privacy_path}` in any string follow the OS: "this Mac" only on macOS. */
const onMac = (): boolean => typeof navigator !== "undefined" && /mac|iphone|ipad/i.test(navigator.platform ?? "");

export function translator(locale: LocaleId, mac: () => boolean = onMac): Translate {
  const dict = LOCALES[locale].dict;
  const word = (name: "computer" | "settings" | "privacy") => dict[`os.${name}_${mac() ? "mac" : "other"}` as Key];
  return (key, params) => format(dict[key] ?? en[key], { computer: word("computer"), system_settings: word("settings"), privacy_path: word("privacy"), ...params });
}

/** Plural form chosen with `Intl.PluralRules` for the locale: `<base>_one` / `<base>_other`, with `{count}` filled in. */
export function plural(locale: LocaleId, base: string, count: number, t: Translate): string {
  const form = new Intl.PluralRules(LOCALES[locale].bcp47).select(count) === "one" ? "one" : "other";
  return t(`${base}_${form}` as Key, { count });
}

export interface I18n {
  locale: LocaleId;
  t: Translate;
  /** `history.segments`-style plural helper bound to the locale. */
  n: (base: string, count: number) => string;
  date: (unixSeconds: number) => string;
  number: (n: number, opts?: Intl.NumberFormatOptions) => string;
}

export function makeI18n(locale: LocaleId): I18n {
  const t = translator(locale);
  const bcp = LOCALES[locale].bcp47;
  return {
    locale,
    t,
    n: (base, count) => plural(locale, base, count, t),
    date: (s) => new Intl.DateTimeFormat(bcp, { dateStyle: "medium", timeStyle: "short" }).format(new Date(s * 1000)),
    number: (n, opts) => new Intl.NumberFormat(bcp, opts).format(n),
  };
}

const Ctx = createContext<I18n>(makeI18n("en"));
export const useI18n = (): I18n => useContext(Ctx);
export const useT = (): Translate => useContext(Ctx).t;

/** Current translator for code outside React (host error text); updated by the provider. */
let current: I18n = makeI18n("en");
export const currentI18n = (): I18n => current;

export function I18nProvider({ locale, children }: { locale: LocaleId; children: ReactNode }) {
  const value = useMemo(() => makeI18n(locale), [locale]);
  current = value;
  useEffect(() => {
    document.documentElement.lang = LOCALES[locale].htmlLang; // CJK glyph selection and the screen-reader voice follow it
  }, [locale]);
  return createElement(Ctx.Provider, { value }, children);
}
