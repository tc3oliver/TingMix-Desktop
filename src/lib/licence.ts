import { en, type Key } from "../i18n/en";
import type { Translate } from "../i18n";

/** Display name of a licence: our own wording for the ids we ship, the manifest's name for any other. */
export function licenceName(l: { id: string; name: string }, t: Translate): string {
  const key = `licence.${l.id}` as Key;
  return key in en ? t(key) : l.name;
}
