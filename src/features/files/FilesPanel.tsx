import { useEffect, useState } from "react";
import { ExportButtons } from "../../components/ExportButtons";
import { Provenance } from "../../components/Provenance";
import { TranscriptView } from "../../components/TranscriptView";
import { onDrag } from "../../lib/api";
import type { Settings } from "../../types/protocol";
import { finalsOf, openOf } from "../live/reducer";
import { useFiles } from "./useFiles";

export function FilesPanel({ settings }: { settings: Settings }) {
  const { jobs, error, pick, cancel, dismiss } = useFiles();
  const [dragging, setDragging] = useState(false);
  useEffect(() => {
    let off: (() => void) | undefined;
    let alive = true;
    onDrag(setDragging).then((u) => (alive ? (off = u) : u()));
    return () => {
      alive = false;
      off?.();
    };
  }, []);

  return (
    <section>
      <div className={`drop ${dragging ? "over" : ""}`}>
        <button className="primary" onClick={pick}>Choose audio file…</button>
        <span className="muted"> or drop files here (wav, mp3, flac, ogg, m4a, aac)</span>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {jobs.length === 0 && <p className="muted">Nothing transcribed yet. Files are decoded locally and never uploaded.</p>}
      {jobs.map((j) => {
        const finals = finalsOf(j.transcript);
        return (
          <article key={j.id} className="card">
            <header className="row">
              <strong>{j.name}</strong>
              <span className={`pill ${j.state}`}>{j.state}</span>
              <span className="muted small">{j.audioS.toFixed(1)} s of audio processed</span>
              {j.state === "running" ? (
                <button onClick={() => cancel(j.id)}>Cancel</button>
              ) : (
                <button onClick={() => dismiss(j.id)}>Dismiss</button>
              )}
            </header>
            {j.state === "running" && <progress aria-label={`Transcribing ${j.name}`} />}
            {j.state === "failed" && <p className="error" role="alert">Could not transcribe this file: {j.message}</p>}
            <Provenance label={j.backend} />
            <TranscriptView finals={finals} open={openOf(j.transcript)} size={Math.min(settings.caption_size, 20)} empty={j.state === "running" ? "Working…" : "No speech found."} />
            <ExportButtons name={j.name.replace(/\.[^.]+$/, "")} segments={finals} backend={j.backend} />
          </article>
        );
      })}
    </section>
  );
}
