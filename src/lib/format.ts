/** `m:ss`, or `h:mm:ss` from one hour. */
export function clock(seconds: number): string {
  const total = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const p = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${p(m)}:${p(s)}` : `${p(m)}:${p(s)}`;
}

/** Last path component, for messages that must not show a full path. */
export const basename = (path: string): string => path.split(/[\\/]/).filter(Boolean).pop() ?? path;

/** "tingmix-large-mlx-fp32-r75071a1" -> "TingMix Large MLX fp32" (revision suffix dropped, known words cased). */
export function friendlyName(id: string): string {
  const words: Record<string, string> = { tingmix: "TingMix", mlx: "MLX", onnx: "ONNX", cpu: "CPU", gpu: "GPU", source: "source" };
  return id
    .replace(/-r[0-9a-f]{7,}$/, "")
    .split("-")
    .map((w) => words[w] ?? (/^(fp|bf|int)\d+$/.test(w) ? w : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(" ");
}
