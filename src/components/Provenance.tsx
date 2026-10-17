import { isMock, MOCK_NOTE } from "../lib/provenance";

/** Says which backend produced the transcript next to it, so a saved or screenshotted result carries it. */
export function Provenance({ label }: { label: string | null | undefined }) {
  if (isMock(label)) return <p className="mock-note" role="note">{MOCK_NOTE}</p>;
  return <p className="muted small">Produced by {label}</p>;
}
