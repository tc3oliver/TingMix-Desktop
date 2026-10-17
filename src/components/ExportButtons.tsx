import { useState } from "react";
import { toSrt, toTxt, type Segment } from "../features/export/format";
import { api, errorText } from "../lib/api";
import { basename } from "../lib/format";
import { exportNote, isMock } from "../lib/provenance";

export function ExportButtons({ name, segments, backend }: { name: string; segments: Segment[]; backend: string | null | undefined }) {
  const [msg, setMsg] = useState<{ text: string; title?: string; error: boolean } | null>(null);
  const run = async (ext: "txt" | "srt") => {
    try {
      const note = exportNote(backend);
      const saved = await api.exportText(isMock(backend) ? `${name} (MOCK)` : name, ext, ext === "srt" ? toSrt(segments, note) : toTxt(segments, note));
      setMsg(saved ? { text: `Saved ${basename(saved)}`, title: saved, error: false } : null);
    } catch (e) {
      setMsg({ text: `Export failed: ${errorText(e)}`, error: true });
    }
  };
  const none = segments.length === 0;
  return (
    <span className="row">
      <button disabled={none} onClick={() => run("txt")}>Export TXT</button>
      <button disabled={none} onClick={() => run("srt")}>Export SRT</button>
      {msg && (
        <span className={msg.error ? "error small" : "muted small"} role={msg.error ? "alert" : "status"} title={msg.title}>
          {msg.text}
        </span>
      )}
    </span>
  );
}
