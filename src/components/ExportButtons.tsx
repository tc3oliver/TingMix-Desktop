import { useState } from "react";
import { toSrt, toTxt, type Segment } from "../features/export/format";
import { api, errorText } from "../lib/api";
import { exportNote, isMock } from "../lib/provenance";

export function ExportButtons({ name, segments, backend }: { name: string; segments: Segment[]; backend: string | null | undefined }) {
  const [msg, setMsg] = useState<string | null>(null);
  const run = async (ext: "txt" | "srt") => {
    try {
      const note = exportNote(backend);
      const saved = await api.exportText(isMock(backend) ? `${name} (MOCK)` : name, ext, ext === "srt" ? toSrt(segments, note) : toTxt(segments, note));
      setMsg(saved ? `Saved ${saved}` : null);
    } catch (e) {
      setMsg(`Export failed: ${errorText(e)}`);
    }
  };
  const none = segments.length === 0;
  return (
    <span className="row">
      <button disabled={none} onClick={() => run("txt")}>Export TXT</button>
      <button disabled={none} onClick={() => run("srt")}>Export SRT</button>
      {msg && <span className="muted small">{msg}</span>}
    </span>
  );
}
