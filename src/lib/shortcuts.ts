export type Action =
  | { type: "tab"; index: number }
  | { type: "toggle-live" }
  | { type: "choose-file" }
  | { type: "caption"; delta: 2 | -2 };

export interface Key {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
}

export const isMac = (platform: string): boolean => /mac|iphone|ipad/i.test(platform);

/** ⌘ on macOS, Ctrl elsewhere. Not bound on purpose: ⌘R (reload), ⌘W/Q/M/H (system), Space. */
export function shortcutFor(e: Key, mac: boolean, tabCount: number): Action | null {
  const mod = mac ? e.metaKey && !e.ctrlKey : e.ctrlKey && !e.metaKey;
  if (!mod || e.altKey) return null;
  if (/^[1-9]$/.test(e.key) && Number(e.key) <= tabCount && !e.shiftKey) return { type: "tab", index: Number(e.key) - 1 };
  if (e.key === "Enter") return { type: "toggle-live" };
  if (e.key.toLowerCase() === "o" && !e.shiftKey) return { type: "choose-file" };
  if (e.key === "=" || e.key === "+") return { type: "caption", delta: 2 };
  if (e.key === "-" || e.key === "_") return { type: "caption", delta: -2 };
  return null;
}

/** Label for a title attribute: "⌘↵" or "Ctrl+Enter". */
export const label = (mac: boolean, key: string): string => (mac ? `⌘${key === "Enter" ? "↵" : key}` : `Ctrl+${key}`);
