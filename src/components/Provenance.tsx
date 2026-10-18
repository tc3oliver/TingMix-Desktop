import { useT } from "../i18n";
import { isMock } from "../lib/provenance";

/** Warns when the words are placeholders (debug builds only). Real recognition shows nothing: a ready app does not talk about its engine. */
export function Provenance({ label }: { label: string | null | undefined }) {
  const t = useT();
  if (isMock(label)) return <p className="mock-note" role="note">{t("prov.mock")}</p>;
  return null;
}
