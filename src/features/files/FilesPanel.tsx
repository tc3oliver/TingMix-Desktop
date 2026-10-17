import { ExportButtons } from "../../components/ExportButtons";
import { Provenance } from "../../components/Provenance";
import { TranscriptView } from "../../components/TranscriptView";
import { useT, type Translate } from "../../i18n";
import { hostText } from "../../i18n/host";
import { clock } from "../../lib/format";
import type { BackendState, Settings } from "../../types/protocol";
import { finalsOf, openOf } from "../live/reducer";
import type { Job, useFiles } from "./useFiles";

type Files = ReturnType<typeof useFiles>;

function emptyText(j: Job, t: Translate): string | null {
  switch (j.state) {
    case "running": return t("files.empty_running");
    case "done": return t("files.empty_done");
    case "cancelled": return t("files.empty_cancelled");
    case "failed": return null; // the error line says it
  }
}

interface Props {
  files: Files;
  settings: Settings;
  backend: BackendState | null;
  onOpenModels: () => void;
  shortcut: string;
}

export function FilesPanel({ files, settings, backend, onOpenModels, shortcut }: Props) {
  const t = useT();
  const { jobs, error, pick, cancel, dismiss } = files;
  const mock = !backend || backend.info.mock;
  return (
    <section>
      <div className={`drop ${jobs.length === 0 ? "big" : ""}`}>
        <span>{t("files.drop")} </span>
        <button className="primary" onClick={pick} title={t("files.choose_title", { shortcut })} aria-keyshortcuts={shortcut}>{t("files.choose")}</button>
        <div className="muted small">{t("files.formats")}</div>
      </div>
      {mock && (
        <p className="muted small">
          {t("files.demo_hint")} <button className="link" onClick={onOpenModels}>{t("banner.setup")}</button>
        </p>
      )}
      {error && <p className="error" role="alert">{error}</p>}
      {jobs.length === 0 && <p className="muted">{t("files.empty")}</p>}
      {jobs.map((j) => {
        const finals = finalsOf(j.transcript);
        return (
          <article key={j.id} className="card">
            <header className="row between">
              <strong className="ellipsis" title={j.name}>{j.name}</strong>
              <span className={`pill ${j.state}`}>{t(`files.state.${j.state}`)}</span>
              <span className="muted small">{t("files.audio_done", { time: clock(j.audioS) })}</span>
              <span className="spacer" />
              {j.state === "running" ? (
                <button onClick={() => cancel(j.id)}>{t("common.cancel")}</button>
              ) : (
                <button onClick={() => dismiss(j.id)}>{t("files.remove")}</button>
              )}
            </header>
            {j.state === "running" && <progress aria-label={t("files.progress", { name: j.name })} />}
            {j.state === "failed" && <p className="error" role="alert">{t("files.failed", { detail: hostText(j.message, t) })}</p>}
            <Provenance label={j.backend} />
            <TranscriptView label={t("files.transcript", { name: j.name })} finals={finals} open={openOf(j.transcript)} size={Math.min(settings.caption_size, 20)} follow={j.state === "running"} empty={emptyText(j, t)} resetKey={j.id} />
            <div className="row">
              <ExportButtons name={j.name.replace(/\.[^.]+$/, "")} segments={finals} backend={j.backend} />
              {j.state === "done" && settings.save_history && finals.length > 0 && <span className="muted small">{t("files.saved_history")}</span>}
            </div>
          </article>
        );
      })}
    </section>
  );
}
