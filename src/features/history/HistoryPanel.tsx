import { useCallback, useEffect, useState } from "react";
import { ConfirmDelete } from "../../components/ConfirmDelete";
import { ExportButtons } from "../../components/ExportButtons";
import { Provenance } from "../../components/Provenance";
import { TranscriptView } from "../../components/TranscriptView";
import { api, errorText } from "../../lib/api";
import { clock } from "../../lib/format";
import { isMock } from "../../lib/provenance";
import type { HistoryEntry, HistorySummary, Settings } from "../../types/protocol";

const SOURCE: Record<HistorySummary["source"], string> = { live: "Live session", file: "Audio file" };
const plural = new Intl.PluralRules("en");
const segments = (n: number) => `${n} ${plural.select(n) === "one" ? "segment" : "segments"}`;
const when = (t: number) => new Date(t * 1000).toLocaleString();

export function HistoryPanel({ settings }: { settings: Settings }) {
  const [items, setItems] = useState<HistorySummary[]>([]);
  const [open, setOpen] = useState<HistoryEntry | null>(null);
  const [confirm, setConfirm] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reload = useCallback(() => api.historyList().then(setItems).catch((e) => setError(errorText(e))), []);
  useEffect(() => void reload(), [reload]);

  const show = (id: string) => api.historyLoad(id).then(setOpen).catch((e) => setError(errorText(e)));
  const remove = async (id: string) => {
    try {
      await api.historyDelete(id);
      if (open?.id === id) setOpen(null);
      setConfirm(null);
      await reload();
    } catch (e) {
      setError(errorText(e));
    }
  };

  return (
    <section>
      {!settings.save_history && <p className="muted">History saving is off (Settings). Existing entries are still listed.</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {items.length === 0 && <p className="muted">No saved transcripts yet. Finished live sessions and files appear here.</p>}
      <ul className="list">
        {items.map((h) => (
          <li key={h.id} className="row between">
            <button className="link ellipsis" aria-current={open?.id === h.id} onClick={() => show(h.id)} title={h.name}>{h.name}</button>
            <span className="muted small">
              {SOURCE[h.source]} · {when(h.started_at)} · {clock(h.duration_s)} · {segments(h.segments)} · {isMock(h.backend) ? "Mock — not real ASR" : h.backend}
            </span>
            <span className="spacer" />
            {confirm === h.id ? (
              <ConfirmDelete question={`Delete “${h.name}”? This can't be undone.`} onConfirm={() => remove(h.id)} onCancel={() => setConfirm(null)} />
            ) : (
              <button onClick={() => setConfirm(h.id)} aria-label={`Delete ${h.name}`}>Delete</button>
            )}
          </li>
        ))}
      </ul>
      {open && (
        <article className="card">
          <header className="row between">
            <strong className="ellipsis" title={open.name}>{open.name}</strong>
            <span className="muted small">{when(open.started_at)}</span>
          </header>
          <Provenance label={open.backend} />
          <TranscriptView label={`Transcript of ${open.name}`} finals={open.finals} size={Math.min(settings.caption_size, 20)} empty="Empty transcript." resetKey={open.id} />
          <ExportButtons name={open.name} segments={open.finals} backend={open.backend} />
        </article>
      )}
    </section>
  );
}
