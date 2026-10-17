import type { BackendState, ModelOp } from "../../types/protocol";

export type Phase = "working" | "ready" | "problem";

export interface Stage {
  phase: Phase;
  text: string;
}

const mb = (n: number) => `${(n / (1 << 20)).toFixed(0)} MB`;

/**
 * What the single onboarding card says, from the host's latest model operation and backend state.
 * The host does the steps (verify and copy, activate when none is active, test on this Mac); this only describes them.
 */
export function onboardStage(op: ModelOp | null, backend: BackendState | null): Stage | null {
  if (!op) return null;
  if (op.state === "running") {
    if (op.kind === "probe") return { phase: "working", text: "Testing on this Mac…" };
    if (op.kind === "download")
      return { phase: "working", text: op.bytes_total ? `Downloading… ${mb(op.bytes_done)} of ${mb(op.bytes_total)}` : "Downloading…" };
    return { phase: "working", text: "Verifying and copying files…" };
  }
  if (op.state === "cancelled") return { phase: "problem", text: op.message ?? "Cancelled." };
  if (op.state === "failed") return { phase: "problem", text: op.message ?? "Something went wrong." };
  const info = backend?.info;
  const inUse = !!info && !info.mock && info.label.startsWith(op.name);
  if (op.kind === "probe") {
    return inUse ? { phase: "ready", text: `Ready: ${info.label}` } : { phase: "problem", text: info?.reason ?? "The model could not be used." };
  }
  // import/download finished: the host activates and tests it next when nothing was active
  if (inUse) return { phase: "ready", text: `Ready: ${info.label}` };
  return { phase: "working", text: "Files verified. Testing on this Mac…" };
}

/** One line per installed model: is it the one in use, and if it is selected but not usable, why. */
export function modelStatus(m: { id: string; active: boolean }, backend: BackendState | null): string {
  if (!m.active) return "Installed";
  const info = backend?.info;
  if (info && !info.mock && info.label.startsWith(m.id)) return "In use";
  return `Selected, but not in use. ${info?.reason ?? ""}`.trim();
}
