import { useCallback, useEffect, useState } from "react";
import { api, errorText, onFile, onFilesDropped } from "../../lib/api";
import type { FileRef, JobState } from "../../types/protocol";
import { applyEvent, applySnapshot, emptyTranscript, type TranscriptState } from "../live/reducer";

export interface Job {
  id: string;
  name: string;
  /** Which backend produces this transcript ("Mock — not ASR" for placeholder output). */
  backend: string;
  state: JobState;
  audioS: number;
  message: string | null;
  transcript: TranscriptState;
}

export function useFiles() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);

  const begin = useCallback(async (ref: FileRef) => {
    try {
      const { job_id: id, backend } = await api.fileStart(ref.id);
      // an event can beat this setState; keep what the listener already created and only fix its name
      setJobs((js) => (js.some((j) => j.id === id) ? js.map((j) => (j.id === id ? { ...j, name: ref.name, backend } : j)) : [{ id, name: ref.name, backend, state: "running", audioS: 0, message: null, transcript: emptyTranscript() }, ...js]));
    } catch (e) {
      setError(errorText(e));
    }
  }, []);

  useEffect(() => {
    let alive = true;
    const offs: (() => void)[] = [];
    const keep = (p: Promise<() => void>) => p.then((u) => (alive ? offs.push(u) : u()));
    keep(onFile((m) => {
      setJobs((js) => {
        const known = js.some((j) => j.id === m.job_id);
        const base = known ? js : [{ id: m.job_id, name: "audio file", backend: "", state: "running" as const, audioS: 0, message: null, transcript: emptyTranscript() }, ...js];
        return base.map((j) => {
          if (j.id !== m.job_id) return j;
          switch (m.kind) {
            case "progress": return { ...j, audioS: m.audio_s };
            case "event": return { ...j, transcript: applyEvent(j.transcript, m.event) };
            case "done": return { ...j, state: "done", audioS: m.audio_s };
            case "cancelled": return { ...j, state: "cancelled" };
            case "failed": return { ...j, state: "failed", message: m.message };
          }
        });
      });
    }));
    keep(onFilesDropped((refs) => refs.forEach(begin)));
    api.fileJobs().then((views) => {
      if (!alive) return;
      setJobs((cur) => {
        const byId = new Map(cur.map((j) => [j.id, j]));
        return views.map((v) => ({
          id: v.job_id, name: v.name, backend: v.backend, state: v.state, audioS: v.audio_s, message: v.message,
          transcript: applySnapshot(byId.get(v.job_id)?.transcript ?? emptyTranscript(), v.snapshot),
        })).reverse();
      });
    }).catch((e) => setError(errorText(e)));
    return () => {
      alive = false;
      offs.forEach((o) => o());
    };
  }, [begin]);

  const pick = useCallback(async () => {
    setError(null);
    try {
      const ref = await api.filePick();
      if (ref) await begin(ref);
    } catch (e) {
      setError(errorText(e));
    }
  }, [begin]);

  const cancel = useCallback((id: string) => api.fileCancel(id).catch((e) => setError(errorText(e))), []);
  const dismiss = useCallback((id: string) => setJobs((js) => js.filter((j) => j.id !== id)), []);
  return { jobs, error, pick, cancel, dismiss };
}
