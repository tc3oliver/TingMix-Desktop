import { useState } from "react";
import { toSrt, toTxt, type Segment } from "../features/export/format";
import { api, errorText } from "../lib/api";

export function ExportButtons({ name, segments }: { name: string; segments: Segment[] }) {
  const [msg, setMsg] = useState<string | null>(null);
  const run = async (ext: "txt" | "srt") => {
    try {
      const saved = await api.exportText(name, ext, ext === "srt" ? toSrt(segments) : toTxt(segments));
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
