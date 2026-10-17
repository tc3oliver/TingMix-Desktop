import { useEffect, useRef } from "react";
import { useT } from "../i18n";

/** Inline "are you sure" for deletes: focus lands on Cancel, Esc cancels, nothing is deleted by a single click. */
export function ConfirmDelete({ question, action, onConfirm, onCancel }: { question: string; action?: string; onConfirm: () => void; onCancel: () => void }) {
  const t = useT();
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => cancel.current?.focus(), []);
  return (
    <span className="row confirm" role="alertdialog" aria-label={question} onKeyDown={(e) => e.key === "Escape" && (e.stopPropagation(), onCancel())}>
      <span>{question}</span>
      <button className="danger" onClick={onConfirm}>{action ?? t("common.delete")}</button>
      <button ref={cancel} onClick={onCancel}>{t("common.cancel")}</button>
    </span>
  );
}
