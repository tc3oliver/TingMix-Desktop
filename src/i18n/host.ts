import type { Key } from "./en";
import { en } from "./en";
import { currentI18n, type Translate } from "./index";

interface Payload {
  code: string;
  params?: Record<string, unknown>;
  text?: string;
}

function parse(raw: string): Payload | null {
  if (!raw.startsWith("{")) return null;
  try {
    const v = JSON.parse(raw) as Payload;
    return typeof v.code === "string" ? v : null;
  } catch {
    return null;
  }
}

/** The code of a host message ("backend.cant_use" is looked through to the reason it wraps), or null for plain text. */
export function hostCode(raw: string | null | undefined): string | null {
  const p = raw ? parse(raw) : null;
  if (!p) return null;
  return p.code === "backend.cant_use" && typeof p.params?.why === "string" ? (hostCode(p.params.why) ?? p.code) : p.code;
}

/**
 * Translate a message from the host. Coded messages (`{"code","params","text"}`) are looked up as `host.<code>`; parameters that
 * are themselves coded messages are translated first; an unknown code or plain text falls back to the English text the host sent.
 */
export function hostText(raw: string | null | undefined, t: Translate = currentI18n().t): string {
  if (!raw) return "";
  const p = parse(raw);
  if (!p) return raw;
  const params: Record<string, string | number> = {};
  for (const [k, v] of Object.entries(p.params ?? {})) {
    if (typeof v === "string") params[k] = hostText(v, t);
    else if (typeof v === "number") params[k] = v;
  }
  const base = `host.${p.code}`;
  const key = (p.params?.mlx === true && `${base}_mlx` in en ? `${base}_mlx` : base) as Key;
  return key in en ? t(key, params) : (p.text ?? raw);
}

/** Like `hostText`, but for "<model> can't be used yet. <why> …" it returns only the why: cards and banners already name the model. */
export function reasonText(raw: string | null | undefined, t: Translate = currentI18n().t): string {
  const p = raw ? parse(raw) : null;
  return p?.code === "backend.cant_use" && typeof p.params?.why === "string" ? hostText(p.params.why, t) : hostText(raw, t);
}
