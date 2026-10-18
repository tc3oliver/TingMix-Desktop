// Renders every named state of the real app (mocked host) in both languages, themes and two window sizes.
// Usage: pnpm shots [stateSubstring]   Output: tools/ui-shots/out/<state>/<lang>-<theme>-<w>x<h>.png
import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const PORT = 1431;
const STATES = ["setup-checking", "setup-offer", "setup-downloading", "setup-paused", "setup-failed", "setup-finishing", "setup-offline", "setup-unavailable", "setup-unavailable-windows", "ready-empty", "live-captions", "live-retracted", "live-mic-denied", "live-mic-denied-windows", "live-no-signal", "live-disconnected", "files-idle", "files-running", "files-done", "history-empty", "history-items", "settings", "settings-model-advanced-open", "settings-troubleshooting-open"];
const SIZES = [[1100, 720], [720, 560]];
const only = process.argv[2];

const vite = spawn("pnpm", ["exec", "vite", "--config", join(here, "vite.config.ts")], { stdio: "ignore" });
const stop = () => vite.kill();
process.on("exit", stop);
for (let i = 0; ; i++) {
  try { if ((await fetch(`http://localhost:${PORT}/tools/ui-shots/harness.html`)).ok) break; } catch {}
  if (i > 100) throw new Error("vite did not start");
  await new Promise((r) => setTimeout(r, 200));
}

const browser = await chromium.launch();
let n = 0;
const problems = [];
try {
  for (const state of STATES.filter((s) => !only || s.includes(only))) {
    mkdirSync(join(here, "out", state), { recursive: true });
    for (const lang of ["zh-TW", "en"]) for (const theme of ["light", "dark"]) for (const [width, height] of SIZES) {
      const ctx = await browser.newContext({ viewport: { width, height }, colorScheme: theme, locale: lang, deviceScaleFactor: 1 });
      const page = await ctx.newPage();
      page.on("pageerror", (e) => problems.push(`${state} ${lang}: ${e.message}`));
      await page.goto(`http://localhost:${PORT}/tools/ui-shots/harness.html?state=${state}&lang=${lang}&theme=${theme}`);
      await page.waitForSelector("body[data-ready='1']", { timeout: 15000 });
      await page.screenshot({ path: join(here, "out", state, `${lang}-${theme}-${width}x${height}.png`) });
      await ctx.close();
      n++;
    }
  }
} finally {
  await browser.close();
  stop();
}
console.log(`${n} screenshots in tools/ui-shots/out/`);
if (problems.length) console.log(`page errors:\n${problems.join("\n")}`);
