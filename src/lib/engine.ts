import type { Translate } from "../i18n";
import { hostCode, hostText, reasonText } from "../i18n/host";
import type { BackendState } from "../types/protocol";
import { friendlyName } from "./format";

export type Tone = "neutral" | "warn" | "danger";

export interface Chip {
  text: string;
  tone: Tone;
  /** Full text for the tooltip when the chip truncates. */
  title: string;
}

/** The single engine status next to the tabs (state table of the UI brief, P0-2). */
export function engineChip(bs: BackendState | null, t: Translate): Chip {
  if (!bs) return { text: t("engine.checking"), tone: "neutral", title: t("engine.checking_title") };
  if (bs.load.error) return { text: t("engine.failed"), tone: "danger", title: hostText(bs.load.error, t) };
  const { info, load } = bs;
  if (info.mock) return { text: info.usable ? t("engine.demo") : t("engine.setup"), tone: "warn", title: reasonText(info.reason, t) };
  const name = friendlyName(info.label.split(" (")[0] ?? info.label);
  if (load.loading) return { text: t("engine.loading", { name }), tone: "neutral", title: info.label };
  return { text: name, tone: "neutral", title: info.label };
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
