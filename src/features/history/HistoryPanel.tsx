import { useCallback, useEffect, useState } from "react";
import { ExportButtons } from "../../components/ExportButtons";
import { Provenance } from "../../components/Provenance";
import { TranscriptView } from "../../components/TranscriptView";
import { isMock } from "../../lib/provenance";
import { api, errorText } from "../../lib/api";
import type { HistoryEntry, HistorySummary, Settings } from "../../types/protocol";

export function HistoryPanel({ settings }: { settings: Settings }) {
  const [items, setItems] = useState<HistorySummary[]>([]);
  const [open, setOpen] = useState<HistoryEntry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reload = useCallback(() => api.historyList().then(setItems).catch((e) => setError(errorText(e))), []);
  useEffect(() => void reload(), [reload]);

  const show = (id: string) => api.historyLoad(id).then(setOpen).catch((e) => setError(errorText(e)));
  const remove = async (id: string) => {
    try {
      await api.historyDelete(id);
      if (open?.id === id) setOpen(null);
      await reload();
    } catch (e) {
      setError(errorText(e));
    }
  };

  return (
    <section>
      {!settings.save_history && <p className="muted">History saving is off (Settings). Existing entries are still listed.</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {items.length === 0 && <p className="muted">No saved transcripts yet.</p>}
      <ul className="list">
        {items.map((h) => (
          <li key={h.id} className="row">
            <button onClick={() => show(h.id)}>{h.name}</button>
            <span className="muted small">
              {h.source} · {new Date(h.started_at * 1000).toLocaleString()} · {h.duration_s.toFixed(0)} s · {h.segments} segments · {isMock(h.backend) ? "MOCK — not real ASR" : h.backend}
            </span>
            <button onClick={() => remove(h.id)} aria-label={`Delete ${h.name}`}>Delete</button>
          </li>
        ))}
      </ul>
      {open && (
        <article className="card">
          <Provenance label={open.backend} />
          <TranscriptView finals={open.finals} size={Math.min(settings.caption_size, 20)} empty="Empty transcript." />
          <ExportButtons name={open.name} segments={open.finals} backend={open.backend} />
        </article>
      )}
    </section>
  );
}
