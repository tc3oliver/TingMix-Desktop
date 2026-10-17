import type { BackendState } from "../types/protocol";
import { friendlyName } from "./format";

export type Tone = "neutral" | "warn" | "danger";

export interface Chip {
  text: string;
  tone: Tone;
  /** Only when it is a model name: shown in full in the title when the chip truncates. */
  title: string;
}

/** The single engine status next to the tabs (state table of the UI brief, P0-2). */
export function engineChip(bs: BackendState | null): Chip {
  if (!bs) return { text: "Checking engine…", tone: "neutral", title: "Checking which speech engine is in use" };
  if (bs.load.error) return { text: "Engine failed to load", tone: "danger", title: bs.load.error };
  const { info, load } = bs;
  if (info.mock) return { text: "Demo mode", tone: "warn", title: info.reason };
  const name = friendlyName(info.label.split(" (")[0] ?? info.label);
  if (load.loading) return { text: `Loading ${name}…`, tone: "neutral", title: info.label };
  if (info.experimental) return { text: `${name} · Experimental`, tone: "warn", title: info.label };
  return { text: name, tone: "neutral", title: info.label };
}

/** Whether the full-width demo banner shows, and the sentence after it. A chosen-but-unusable model explains itself. */
export function demoBanner(bs: BackendState | null): { show: boolean; detail: string | null; openModelsLabel: string } {
  if (!bs || !bs.info.mock) return { show: false, detail: null, openModelsLabel: "" };
  const generic = bs.info.reason.startsWith("No model is active") || bs.info.reason === "";
  return { show: true, detail: generic ? null : bs.info.reason, openModelsLabel: generic ? "Set up a speech model" : "Open Models" };
}
