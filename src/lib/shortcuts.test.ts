import { describe, expect, it } from "vitest";
import { label, shortcutFor, type Key } from "./shortcuts";

const k = (key: string, p: Partial<Key> = {}): Key => ({ key, metaKey: false, ctrlKey: false, altKey: false, shiftKey: false, ...p });

describe("shortcuts", () => {
  it("uses ⌘ on Mac and Ctrl elsewhere, never both", () => {
    expect(shortcutFor(k("1", { metaKey: true }), true, 5)).toEqual({ type: "tab", index: 0 });
    expect(shortcutFor(k("1", { ctrlKey: true }), true, 5)).toBeNull();
    expect(shortcutFor(k("2", { ctrlKey: true }), false, 5)).toEqual({ type: "tab", index: 1 });
    expect(shortcutFor(k("2", { metaKey: true }), false, 5)).toBeNull();
  });
  it("maps the documented actions", () => {
    expect(shortcutFor(k("Enter", { metaKey: true }), true, 5)).toEqual({ type: "toggle-live" });
    expect(shortcutFor(k("o", { metaKey: true }), true, 5)).toEqual({ type: "choose-file" });
    expect(shortcutFor(k("=", { metaKey: true }), true, 5)).toEqual({ type: "caption", delta: 2 });
    expect(shortcutFor(k("-", { metaKey: true }), true, 5)).toEqual({ type: "caption", delta: -2 });
  });
  it("ignores tab numbers beyond the tab count, plain keys, and system combinations", () => {
    expect(shortcutFor(k("6", { metaKey: true }), true, 5)).toBeNull();
    expect(shortcutFor(k("Enter"), true, 5)).toBeNull();
    for (const key of ["r", "w", "q", "m", "h", " "]) expect(shortcutFor(k(key, { metaKey: true }), true, 5)).toBeNull();
    expect(shortcutFor(k("1", { metaKey: true, altKey: true }), true, 5)).toBeNull();
  });
  it("labels shortcuts per platform", () => {
    expect(label(true, "Enter")).toBe("⌘↵");
    expect(label(false, "O")).toBe("Ctrl+O");
  });
});
