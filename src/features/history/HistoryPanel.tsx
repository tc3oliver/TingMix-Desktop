import { useCallback, useEffect, useState } from "react";
import { ConfirmDelete } from "../../components/ConfirmDelete";
import { ExportButtons } from "../../components/ExportButtons";
import { Provenance } from "../../components/Provenance";
import { TranscriptView } from "../../components/TranscriptView";
import { useI18n, type Translate } from "../../i18n";
import { api, errorText } from "../../lib/api";
import { clock } from "../../lib/format";
import { isMock } from "../../lib/provenance";
import type { HistoryEntry, HistorySummary, Settings } from "../../types/protocol";

/** The host names every live session "Live session"; show a dated, translated title instead. */
export const entryName = (e: { source: string; name: string; started_at: number }, t: Translate, date: (s: number) => string): string =>
  e.source === "live" ? t("history.live_name", { date: date(e.started_at) }) : e.name;

export function HistoryPanel({ settings }: { settings: Settings }) {
  const { t, date } = useI18n();
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
      {!settings.save_history && <p className="muted">{t("history.save_off")}</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {items.length === 0 && <p className="muted">{t("history.none")}</p>}
      <ul className="list">
        {items.map((h) => (
          <li key={h.id} className="row between">
            <button className="link ellipsis" aria-current={open?.id === h.id} onClick={() => show(h.id)} title={entryName(h, t, date)}>{entryName(h, t, date)}</button>
            <span className="muted small">
              {[...(h.source === "live" ? [] : [date(h.started_at)]), clock(h.duration_s), ...(isMock(h.backend) ? [t("prov.mock_short")] : [])].join(" · ")}
            </span>
            <span className="spacer" />
            {confirm === h.id ? (
              <ConfirmDelete question={t("history.confirm", { name: entryName(h, t, date) })} onConfirm={() => remove(h.id)} onCancel={() => setConfirm(null)} />
            ) : (
              <button onClick={() => setConfirm(h.id)} aria-label={t("history.delete_aria", { name: entryName(h, t, date) })}>{t("common.delete")}</button>
            )}
          </li>
        ))}
      </ul>
      {open && (
        <article className="card">
          <header className="row between">
            <strong className="ellipsis" title={entryName(open, t, date)}>{entryName(open, t, date)}</strong>
            <span className="muted small">{date(open.started_at)}</span>
          </header>
          <Provenance label={open.backend} />
          <TranscriptView label={t("history.transcript", { name: entryName(open, t, date) })} finals={open.finals} size={Math.min(settings.caption_size, 20)} empty={t("history.empty")} resetKey={open.id} />
          <ExportButtons name={open.source === "live" ? "live-transcript" : open.name} segments={open.finals} backend={open.backend} />
        </article>
      )}
    </section>
  );
}
