import { useEffect, useRef } from "react";

/** Inline "are you sure" for deletes: focus lands on Cancel, Esc cancels, nothing is deleted by a single click. */
export function ConfirmDelete({ question, action = "Delete", onConfirm, onCancel }: { question: string; action?: string; onConfirm: () => void; onCancel: () => void }) {
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => cancel.current?.focus(), []);
  return (
    <span className="row confirm" role="alertdialog" aria-label={question} onKeyDown={(e) => e.key === "Escape" && (e.stopPropagation(), onCancel())}>
      <span>{question}</span>
      <button className="danger" onClick={onConfirm}>{action}</button>
      <button ref={cancel} onClick={onCancel}>Cancel</button>
    </span>
  );
}
