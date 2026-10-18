import type { Translate } from "../i18n";
import { hostCode, hostText, reasonText } from "../i18n/host";
import type { BackendState, SetupView } from "../types/protocol";

export type Tone = "neutral" | "warn" | "danger";

export interface Chip {
  text: string;
  tone: Tone;
  /** Full text for the tooltip when the chip truncates. */
  title: string;
}

/** The one status next to the tabs, or null when everything works: a ready app shows no engine name at all. */
export function engineChip(bs: BackendState | null, t: Translate): Chip | null {
  if (!bs) return null;
  if (bs.load.error) return { text: t("engine.failed"), tone: "danger", title: hostText(bs.load.error, t) };
  const { info, load } = bs;
  if (info.mock) return info.usable ? { text: t("engine.demo"), tone: "warn", title: reasonText(info.reason, t) } : { text: t("engine.failed"), tone: "danger", title: reasonText(info.reason, t) };
  if (load.loading) return { text: t("engine.loading"), tone: "neutral", title: info.label };
  return null;
}

/** Whether the full-width demo banner shows, and the sentence after it. A chosen-but-unusable model explains itself. */
export function demoBanner(bs: BackendState | null, t: Translate): { show: boolean; detail: string | null; openModelsLabel: string } {
  // the placeholder engine exists in debug builds only; a release build shows the setup screen instead of a banner
  if (!bs || !bs.info.mock || !bs.info.demo_allowed) return { show: false, detail: null, openModelsLabel: "" };
  const generic = bs.info.reason === "" || hostCode(bs.info.reason) === "backend.no_model";
  return { show: true, detail: generic ? null : reasonText(bs.info.reason, t), openModelsLabel: generic ? t("banner.setup") : t("banner.open") };
}

/** Release builds never run on the placeholder engine: until a model is usable the first-run screen replaces Live and Files. */
export const needsSetup = (bs: BackendState | null): boolean => bs !== null && !bs.info.usable;

/**
 * First run uses the whole window with no tab bar: nothing usable yet and no installed model that merely failed.
 * A broken installed model ("problem") keeps the tabs so History stays reachable.
 */
export const isFirstRun = (bs: BackendState | null, setup: SetupView | null): boolean => needsSetup(bs) && setup?.state !== "problem";
