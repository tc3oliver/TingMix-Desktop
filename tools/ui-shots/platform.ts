// Must be the first import: the app reads `navigator.platform` once, when its modules load.
const state = new URLSearchParams(location.search).get("state") ?? "";
if (state.endsWith("-windows")) Object.defineProperty(navigator, "platform", { value: "Win32", configurable: true });
