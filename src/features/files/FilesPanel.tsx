import { ExportButtons } from "../../components/ExportButtons";
import { Provenance } from "../../components/Provenance";
import { TranscriptView } from "../../components/TranscriptView";
import { clock } from "../../lib/format";
import type { BackendState, Settings } from "../../types/protocol";
import { finalsOf, openOf } from "../live/reducer";
import type { Job, useFiles } from "./useFiles";

type Files = ReturnType<typeof useFiles>;

const STATE_LABEL: Record<Job["state"], string> = { running: "Transcribing…", done: "Done", cancelled: "Cancelled", failed: "Failed" };

function emptyText(j: Job): string | null {
  switch (j.state) {
    case "running": return "Transcribing — text appears as it's recognised.";
    case "done": return "No speech found in this file.";
    case "cancelled": return "Cancelled before any speech was found.";
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
  const { jobs, error, pick, cancel, dismiss } = files;
  const mock = !backend || backend.info.mock;
  return (
    <section>
      <div className={`drop ${jobs.length === 0 ? "big" : ""}`}>
        <span>Drop audio files here or </span>
        <button className="primary" onClick={pick} title={`Choose file (${shortcut})`} aria-keyshortcuts={shortcut}>Choose file…</button>
        <div className="muted small">WAV, MP3, FLAC, OGG, M4A, AAC · processed on this computer, never uploaded</div>
      </div>
      {mock && (
        <p className="muted small">
          Demo mode: results will be placeholder words. <button className="link" onClick={onOpenModels}>Set up a speech model</button>
        </p>
      )}
      {error && <p className="error" role="alert">{error}</p>}
      {jobs.length === 0 && <p className="muted">Nothing transcribed yet.</p>}
      {jobs.map((j) => {
        const finals = finalsOf(j.transcript);
        const empty = emptyText(j);
        return (
          <article key={j.id} className="card">
            <header className="row between">
              <strong className="ellipsis" title={j.name}>{j.name}</strong>
              <span className={`pill ${j.state}`}>{STATE_LABEL[j.state]}</span>
              <span className="muted small">{clock(j.audioS)} of audio transcribed</span>
              <span className="spacer" />
              {j.state === "running" ? (
                <button onClick={() => cancel(j.id)}>Cancel</button>
              ) : (
                <button onClick={() => dismiss(j.id)}>Remove from list</button>
              )}
            </header>
            {j.state === "running" && <progress aria-label={`Transcribing ${j.name}`} />}
            {j.state === "failed" && <p className="error" role="alert">Could not transcribe this file: {j.message}</p>}
            <Provenance label={j.backend} />
            <TranscriptView label={`Transcript of ${j.name}`} finals={finals} open={openOf(j.transcript)} size={Math.min(settings.caption_size, 20)} follow={j.state === "running"} empty={empty} resetKey={j.id} />
            <div className="row">
              <ExportButtons name={j.name.replace(/\.[^.]+$/, "")} segments={finals} backend={j.backend} />
              {j.state === "done" && settings.save_history && finals.length > 0 && <span className="muted small">Saved in History</span>}
            </div>
          </article>
        );
      })}
    </section>
  );
}
