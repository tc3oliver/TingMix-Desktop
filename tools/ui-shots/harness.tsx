import "./platform";
import { emit } from "@tauri-apps/api/event";
import { mockIPC } from "@tauri-apps/api/mocks";
import { createRoot } from "react-dom/client";
import { StrictMode } from "react";
import type { BackendState, Diagnostics, HistorySummary, JobView, LiveStatus, MicStatus, ModelOp, ModelsOverview, SetupView, Settings } from "../../src/types/protocol";

// Mocked host for screenshots. `?state=<name>&lang=en|zh-TW&theme=light|dark` picks one named state; no real model, audio or file is involved.

const q = new URLSearchParams(location.search);
const stateName = q.get("state") ?? "ready-empty";
const base = stateName.replace(/-windows$/, "");
const lang = q.get("lang") ?? "en";
const theme = q.get("theme") ?? "light";

const MODEL_ID = "tingmix-small-onnx-fp16-r1a2b3c4";
const info = (over: Partial<BackendState["info"]> = {}): BackendState["info"] => ({
  label: `${MODEL_ID} (onnx) — EXPERIMENTAL`, mock: false, experimental: false, reason: "", rejected: [], ort_library: null,
  demo_allowed: false, usable: true, ...over,
});
const READY: BackendState = { info: info(), load: { loading: false, label: null, error: null } };
const NOT_READY: BackendState = { info: info({ label: "Mock — not ASR", mock: true, usable: false, reason: JSON.stringify({ code: "backend.no_model", text: "No model installed" }) }), load: { loading: false, label: null, error: null } };

const lic = (id: string, name: string, spdx: string | null) => ({ id, name, url: `https://example.org/licences/${id}`, spdx, acceptance_required: true, accepted: false, notes: "" });
const download: SetupView = {
  state: "download",
  pending: { token: "t", id: "tingmix-small-onnx-fp16-r1a2b3c4", family: "tingmix-small", backend: "onnx", precision: "fp16", download_bytes: 1_320_000_000, compat_status: "supported", licenses: [lic("tingmix-large-model", "TingMix-Large model terms", null), lic("google-embeddinggemma-2", "Google EmbeddingGemma 2 licence and prohibited use policy", null)] },
};
const op = (state: ModelOp["state"], done: number, message: string | null = null): ModelOp => ({ kind: "download", name: "tingmix-small-onnx-fp16-r1a2b3c4", state, message, bytes_done: done, bytes_total: 1_320_000_000 });

const idle: LiveStatus = { state: "idle", session_id: null, elapsed_s: 0, samples_in: 0, overflow_samples: 0, level_db: -120, device: "", backend: "", coalesced_partials: 0, message: null, demo: false };
const rec = (over: Partial<LiveStatus> = {}): LiveStatus => ({ ...idle, backend: `${MODEL_ID} (onnx) — EXPERIMENTAL`, state: "recording", session_id: "s1", elapsed_s: 42, samples_in: 672_000, level_db: -32, device: "MacBook Pro Microphone", ...over });
const seg = (id: number, text: string, start: number, end: number) => ({ segment_id: id, revision: 3, text, start, end });
const part = (id: number, text: string, start: number, end: number) => ({ ...seg(id, text, start, end), stable_prefix_len: Math.floor(text.length / 2) });

const CAPTIONS = lang === "en"
  ? ["Good morning everyone, thanks for joining the weekly sync.", "Let's start with the release status: the beta build went out on Tuesday.", "Two issues came back from testers, both on the file import screen."]
  : ["各位早安，感謝大家參加每週同步會議。", "我們先從發佈狀態開始：測試版在星期二已經送出。", "測試人員回報了兩個問題，都出在檔案匯入畫面。"];
const OPEN = lang === "en" ? "The first one is that long files" : "第一個問題是比較長的檔案";

interface State {
  backend?: BackendState;
  setup?: SetupView | "never";
  modelOp?: ModelOp | null;
  live?: { status: LiveStatus; finals?: ReturnType<typeof seg>[]; open?: ReturnType<typeof part>[] };
  mic?: MicStatus;
  jobs?: JobView[];
  history?: HistorySummary[];
  tab?: string;
  /** Events pushed after the app has mounted, e.g. a caption that is retracted. */
  after?: () => Promise<void>;
  liveEvents?: boolean;
}

/** Opens a disclosure and scrolls it into view (the screenshot is the window, and the panel scrolls inside it). */
const open = async (selector: string) => {
  const summary = document.querySelector<HTMLElement>(selector);
  summary?.click();
  await new Promise((r) => setTimeout(r, 300));
  summary?.scrollIntoView({ block: "start" });
};

const finals = CAPTIONS.map((c, i) => seg(i, c, i * 6, i * 6 + 5));
const job = (state: JobView["state"], name: string, audio_s: number, n: number): JobView => ({
  job_id: `j-${name}`, name, backend: `${MODEL_ID} (onnx) — EXPERIMENTAL`, state, audio_s, message: null,
  snapshot: { finals: finals.slice(0, n), open: [] },
});
const lines = lang === "en" ? ["Good morning everyone, thanks for joining the weekly sync.", "So the first thing I wanted to ask you about was how you got started…", "Today we are going to look at how speech recognition models are trained…"] : ["各位早安，感謝大家參加每週同步會議。", "我想先請你聊聊，當初是怎麼開始做這件事的……", "今天我們要來看語音辨識模型是怎麼訓練出來的……"];
const hist = (i: number, source: "live" | "file", name: string, ago: number, dur: number): HistorySummary => ({
  id: `h${i}`, source, name, started_at: Math.floor(Date.now() / 1000) - ago, duration_s: dur, backend: `${MODEL_ID} (onnx) — EXPERIMENTAL`, segments: 3 + i * 4, first_line: lines[i - 1]!,
});

const STATES: Record<string, State> = {
  "setup-offer": { backend: NOT_READY, setup: download, modelOp: null },
  "setup-downloading": { backend: NOT_READY, setup: download, modelOp: op("running", 488_400_000) },
  "setup-failed": { backend: NOT_READY, setup: download, modelOp: op("failed", 488_400_000, "The connection was interrupted.") },
  "setup-offline": { backend: NOT_READY, setup: { state: "offline", reason: JSON.stringify({ code: "setup.offline", text: "No internet connection (os error 51)" }) }, modelOp: null },
  "setup-checking": { backend: NOT_READY, setup: "never", modelOp: null },
  "setup-paused": { backend: NOT_READY, setup: download, modelOp: op("cancelled", 488_400_000) },
  "setup-finishing": { backend: NOT_READY, setup: download, modelOp: { ...op("running", 1_320_000_000), kind: "probe" } },
  "setup-unavailable-windows": { backend: NOT_READY, setup: { state: "unavailable", why: "platform", reason: JSON.stringify({ code: "setup.no_platform", text: "x" }) }, modelOp: null },
  "setup-unavailable": { backend: NOT_READY, setup: { state: "unavailable", why: "platform", reason: "TingMix is not available for this computer yet." }, modelOp: null },
  "ready-empty": { live: { status: idle } },
  "live-captions": { live: { status: rec(), finals: finals.slice(0, 2), open: [part(2, OPEN, 12, 15)] } },
  "live-retracted": {
    live: { status: rec(), finals: finals.slice(0, 1), open: [part(1, CAPTIONS[1]!.slice(0, 20), 6, 9)] },
    after: async () => {
      await emit("tingmix://live", { kind: "event", session_id: "s1", event: { type: "final", segment_id: 1, revision: 4, text: "", start: 6, end: 9 } });
    },
  },
  "live-mic-denied": { live: { status: idle }, mic: "denied" },
  "live-mic-denied-windows": { live: { status: idle }, mic: "denied" },
  "live-disconnected": { live: { status: { ...rec(), state: "error", message: JSON.stringify({ code: "live.disconnected", text: "The microphone was disconnected: device lost (0x8889000a)" }) }, finals: finals.slice(0, 2) } },
  "live-no-signal": { live: { status: rec({ level_db: -120, elapsed_s: 8 }) } },
  "files-idle": { tab: "files", jobs: [] },
  "files-running": { tab: "files", jobs: [job("running", "weekly-sync.m4a", 95, 2)] },
  "files-done": { tab: "files", jobs: [job("done", "weekly-sync.m4a", 1260, 3), job("failed", "broken.wav", 0, 0)] },
  "history-empty": { tab: "history", history: [] },
  "history-items": { tab: "history", history: [hist(1, "live", "", 3600, 754), hist(2, "file", "interview-final.mp3", 90000, 2710), hist(3, "file", "lecture-03.m4a", 400000, 5400)] },
  settings: { tab: "settings" },
  "settings-model-advanced-open": { tab: "settings", after: async () => open("details.advanced > summary") },
  "settings-troubleshooting-open": { tab: "settings", after: async () => open("fieldset:last-of-type details > summary") },
};
if (!STATES[stateName]) throw new Error(`unknown state ${stateName}`);
const S = { ...STATES[stateName] };

const settings: Settings = { device_id: null, theme: theme as Settings["theme"], caption_size: 28, save_history: true, language: lang };
const overview = (): ModelsOverview => ({
  registry_available: true, note: "", root: "~/Library/Application Support/TingMix/models",
  capabilities: { os: "macos", arch: "aarch64", ram_bytes: 36 << 30, apple_chip: "Apple M4 Max" },
  installed: S.setup ? [] : [{ id: MODEL_ID, family: "tingmix-small", revision: "1a2b3c4", backend: "onnx", precision: "fp16", size_bytes: 1_320_000_000, serial: 3, compat_status: "EXPERIMENTAL", compat_notes: "", licenses: ["apache-2.0"], active: true, previous: false }], unreadable: [], selection: null, op: S.modelOp ?? null,
});
const diag: Diagnostics = {
  app_version: "0.1.0", os: "macos", arch: "aarch64", backend: "TingMix-Small", backend_reason: "", ort_library: null, updater_enabled: false,
  input_devices: [], device_error: null, live: idle, running_jobs: 0, config_dir: "~/Library/Application Support/TingMix", data_dir: "~/Library/Application Support/TingMix", recent_errors: [],
};

mockIPC((cmd, args) => {
  switch (cmd) {
    case "settings_get": return settings;
    case "settings_set": return (args as { settings: Settings }).settings;
    case "backend_state": return S.backend ?? READY;
    case "setup_check": return S.setup === "never" ? new Promise(() => undefined) : (S.setup ?? { state: "ready" });
    case "models_overview": return overview();
    case "models_catalog": return [{ id: "sensevoice-small-onnx-int8-r9f8e7d6", family: "sensevoice", backend: "onnx-cpu", precision: "int8", download_bytes: 240_000_000, compat_status: "EXPERIMENTAL", installed: false, engine_ready: false }];
    case "list_input_devices": return [{ id: "mic1", name: "MacBook Pro Microphone", is_default: true, sample_rate: 48000, channels: 1 }, { id: "mic2", name: "USB Headset", is_default: false, sample_rate: 48000, channels: 1 }];
    case "live_snapshot": return { status: S.live?.status ?? idle, snapshot: { finals: S.live?.finals ?? [], open: S.live?.open ?? [] } };
    case "mic_permission_status": return S.mic ?? "authorized";
    case "file_jobs": return S.jobs ?? [];
    case "history_list": return S.history ?? [];
    case "diagnostics": return diag;
    case "updater_check": return { enabled: false, available: false, version: null, notes: null };
    default: return null;
  }
}, { shouldMockEvents: true });

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);

import { App } from "../../src/app/App";
import "../../src/app/styles.css";

// after mount: switch tab, run scripted events, then tell the screenshot script the page is settled
setTimeout(async () => {
  if (S.tab) document.getElementById(`tab-${S.tab}`)?.click();
  await new Promise((r) => setTimeout(r, 300));
  await S.after?.();
  setTimeout(() => ((document.body.dataset.ready = "1")), 300);
}, 400);
