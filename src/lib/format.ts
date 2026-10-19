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

/** Model names whose capitalisation is not just a capital first letter. */
const BRANDS: Record<string, string> = { tingmix: "TingMix", sensevoice: "SenseVoice", qwen3: "Qwen3", asr: "ASR" };

/** "tingmix-large-mlx-fp32-r75071a1" -> "TingMix Large": revision, engine and precision tokens are technical and belong in the details. */
export function friendlyName(id: string): string {
  const technical = /^(mlx|onnx|cpu|gpu|source|safetensors|(fp|bf|int)\d+)$/;
  const words = id
    .replace(/-r[0-9a-f]{7,}$/, "")
    .split("-")
    .filter((w) => !technical.test(w));
  return (words.length ? words : id.split("-"))
    .map((w) => BRANDS[w] ?? w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
