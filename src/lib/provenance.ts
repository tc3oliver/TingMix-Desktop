import { friendlyName } from "./format";
import type { Translate } from "../i18n";

/** The host labels placeholder output "Mock — not ASR"; anything else names the model and its status. */
export const isMock = (label: string | null | undefined): boolean => !label || label.startsWith("Mock");

/** Written into exported TXT/SRT files. Export content is deliberately not localised. */
export const MOCK_NOTE = "Mock — not real ASR: these words are placeholders, not recognized speech.";

/** One line saying which backend produced a transcript, written into exports so the file carries it too. */
export function exportNote(label: string | null | undefined): string | undefined {
  return isMock(label) ? MOCK_NOTE : undefined;
}

/** Display name of a backend label: "Demo mode", or the model name for "<id> (<engine>) — STATUS" (the status word is for Diagnostics). */
export function labelText(label: string | null | undefined, t: Translate): string {
  if (isMock(label)) return t("engine.demo");
  const m = /^(.+?) \((.+?)\) — (\w+)$/.exec(label ?? "");
  return m ? friendlyName(m[1]!) : (label ?? "");
}
