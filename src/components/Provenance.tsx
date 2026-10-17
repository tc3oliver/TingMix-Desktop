import { useT } from "../i18n";
import { isMock, labelText } from "../lib/provenance";

/** Says which backend produced the transcript next to it, so a saved or screenshotted result carries it. */
export function Provenance({ label }: { label: string | null | undefined }) {
  const t = useT();
  if (isMock(label)) return <p className="mock-note" role="note">{t("prov.mock")}</p>;
  return <p className="muted small">{t("prov.produced_by", { name: labelText(label, t) })}</p>;
}
