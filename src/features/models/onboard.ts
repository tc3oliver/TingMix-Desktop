import type { Translate } from "../../i18n";
import { hostCode, hostText, reasonText } from "../../i18n/host";
import type { BackendState, ModelOp } from "../../types/protocol";
import { friendlyName } from "../../lib/format";

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
export function onboardStage(op: ModelOp | null, backend: BackendState | null, t: Translate): Stage | null {
  if (!op) return null;
  if (op.state === "running") {
    if (op.kind === "probe") return { phase: "working", text: t("stage.testing") };
    if (op.kind === "download")
      return { phase: "working", text: op.bytes_total ? t("stage.downloading_of", { done: mb(op.bytes_done), total: mb(op.bytes_total) }) : t("stage.downloading") };
    return { phase: "working", text: t("stage.verifying") };
  }
  if (op.state === "cancelled") return { phase: "problem", text: op.message ? hostText(op.message, t) : t("stage.cancelled") };
  if (op.state === "failed") return { phase: "problem", text: op.message ? hostText(op.message, t) : t("stage.failed") };
  const info = backend?.info;
  const inUse = !!info && !info.mock && info.label.startsWith(op.name);
  if (op.kind === "probe") {
    return inUse ? { phase: "ready", text: t("stage.ready", { name: friendlyName(op.name) }) } : { phase: "problem", text: info ? reasonText(info.reason, t) : t("stage.unusable") };
  }
  // import/download finished: the host activates and tests it next when nothing was active
  if (inUse) return { phase: "ready", text: t("stage.ready", { name: friendlyName(op.name) }) };
  return { phase: "working", text: t("stage.files_verified") };
}

/** The one status line on a model card (brief P0-5). Jargon stays out of it; the details are under "Technical details". */
export function cardStatus(m: { id: string; active: boolean }, backend: BackendState | null, op: ModelOp | null, t: Translate): string {
  if (op && op.state === "running" && op.kind === "probe" && op.name === m.id) return t("models.status.testing");
  if (!m.active) return t("models.status.not_selected");
  const info = backend?.info;
  if (!info) return t("models.status.checking");
  if (!info.mock && info.label.startsWith(m.id)) return t("models.status.ready");
  switch (hostCode(info.reason)) {
    case "backend.missing_engine": return t("models.status.unsupported");
    case "backend.untested": return t("models.status.untested");
    case "backend.test_failed":
    case "backend.test_failed_detail": return t("models.status.failed", { detail: reasonText(info.reason, t) });
    default: return reasonText(info.reason, t);
  }
}

export interface RateSample {
  at: number;
  bytes: number;
}

const WINDOW_MS = 10_000;
const MIN_SPAN_MS = 5_000;

/** Keeps the last 10 s of download progress; a new download (bytes going backwards) starts over. */
export function pushSample(samples: RateSample[], s: RateSample): RateSample[] {
  const last = samples[samples.length - 1];
  const kept = last && s.bytes < last.bytes ? [] : samples;
  return [...kept, s].filter((x) => s.at - x.at <= WINDOW_MS);
}

/** Seconds left at the rate of the last 10 s, or null until 5 s of data exist or while nothing is arriving. */
export function etaSeconds(samples: RateSample[], total: number): number | null {
  const first = samples[0];
  const last = samples[samples.length - 1];
  if (!first || !last || last.at - first.at < MIN_SPAN_MS) return null;
  const rate = (last.bytes - first.bytes) / ((last.at - first.at) / 1000);
  return rate > 0 && total > last.bytes ? (total - last.bytes) / rate : null;
}

/** "about 3 minutes" is coarse on purpose: a second-by-second countdown would be noise. */
export function etaText(seconds: number, n: (base: string, count: number) => string, t: Translate): string {
  return seconds < 60 ? t("setup.under_minute") : n("setup.minutes", Math.round(seconds / 60));
}
