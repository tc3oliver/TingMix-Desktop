// Mirrors crates/tingmix-core (events) and src-tauri/src/dto.rs (commands). Field names are the Rust ones.

export interface PartialEvent {
  type: "partial";
  segment_id: number;
  revision: number;
  text: string;
  start: number;
  end: number;
  stable_prefix_len: number;
}

export interface FinalEvent {
  type: "final";
  segment_id: number;
  revision: number;
  text: string;
  start: number;
  end: number;
}

export type TranscriptEvent = PartialEvent | FinalEvent;

/** What the host's reducer returns (core `Snapshot`): finals and open partials carry no `type` tag. */
export interface Snapshot {
  finals: Omit<FinalEvent, "type">[];
  open: Omit<PartialEvent, "type">[];
}

export type LiveState = "idle" | "starting" | "recording" | "stopping" | "error";

export interface LiveStatus {
  state: LiveState;
  session_id: string | null;
  elapsed_s: number;
  samples_in: number;
  overflow_samples: number;
  level_db: number;
  device: string;
  backend: string;
  coalesced_partials: number;
  message: string | null;
  /** True only when the latest session ran on the placeholder engine (debug builds). */
  demo: boolean;
}

export type LiveMsg =
  | { kind: "event"; session_id: string; event: TranscriptEvent }
  | ({ kind: "status" } & LiveStatus);

export interface LiveSnapshot {
  status: LiveStatus;
  snapshot: Snapshot;
}

export interface Device {
  id: string;
  name: string;
  is_default: boolean;
  sample_rate: number;
  channels: number;
}

export interface FileRef {
  id: string;
  name: string;
}

export type JobState = "running" | "done" | "cancelled" | "failed";

export type FileMsg = { job_id: string } & (
  | { kind: "progress"; audio_s: number }
  | { kind: "event"; event: TranscriptEvent }
  | { kind: "done"; audio_s: number }
  | { kind: "cancelled" }
  | { kind: "failed"; message: string }
);

export interface JobView {
  job_id: string;
  name: string;
  backend: string;
  state: JobState;
  audio_s: number;
  snapshot: Snapshot;
  message: string | null;
}

export interface Settings {
  device_id: string | null;
  theme: "system" | "light" | "dark";
  caption_size: number;
  save_history: boolean;
  /** `auto` or a language id (`en`, `zh-TW`). */
  language: string;
}

export interface HistorySummary {
  id: string;
  source: "live" | "file";
  name: string;
  started_at: number;
  duration_s: number;
  backend: string;
  segments: number;
  /** First text of the entry (≤ 80 characters), empty when there is none. */
  first_line: string;
}

export interface HistoryEntry extends Omit<HistorySummary, "segments"> {
  finals: Omit<FinalEvent, "type">[];
}

export interface Diagnostics {
  app_version: string;
  os: string;
  arch: string;
  backend: string;
  backend_reason: string;
  ort_library: string | null;
  updater_enabled: boolean;
  input_devices: Device[];
  device_error: string | null;
  live: LiveStatus;
  running_jobs: number;
  config_dir: string;
  data_dir: string;
  recent_errors: string[];
}

export interface InstalledModel {
  id: string;
  family: string;
  revision: string;
  backend: string;
  precision: string;
  size_bytes: number;
  serial: number;
  compat_status: string;
  compat_notes: string;
  licenses: string[];
  active: boolean;
  previous: boolean;
}

export interface ModelOp {
  kind: "download" | "import" | "probe";
  name: string;
  state: "running" | "done" | "failed" | "cancelled";
  message: string | null;
  bytes_done: number;
  bytes_total: number;
}

export interface ModelsOverview {
  registry_available: boolean;
  note: string;
  root: string;
  capabilities: { os: string; arch: string; ram_bytes: number; apple_chip: string | null } | null;
  installed: InstalledModel[];
  unreadable: [string, string][];
  selection: { chosen: string | null; reason: string; rejected: [string, string][] } | null;
  op: ModelOp | null;
}

export interface LicenseInfo {
  id: string;
  name: string;
  url: string;
  spdx: string | null;
  acceptance_required: boolean;
  accepted: boolean;
  notes: string;
}

export interface PendingInstall {
  token: string;
  id: string;
  family: string;
  backend: string;
  precision: string;
  download_bytes: number;
  compat_status: string;
  licenses: LicenseInfo[];
}

export interface BackendInfo {
  label: string;
  mock: boolean;
  experimental: boolean;
  reason: string;
  rejected: [string, string][];
  ort_library: string | null;
  /** Debug builds may run on the placeholder engine; release builds never do. */
  demo_allowed: boolean;
  /** A session can start: a real engine is chosen, or the placeholder is allowed. */
  usable: boolean;
}

export interface UpdateInfo {
  enabled: boolean;
  available: boolean;
  version: string | null;
  notes: string | null;
}

export type MicStatus = "not_determined" | "authorized" | "denied" | "restricted" | "unsupported";

export interface LoadView {
  loading: boolean;
  label: string | null;
  error: string | null;
}

export interface JobStarted {
  job_id: string;
  backend: string;
}

/** Pushed by the host on `tingmix://backend` whenever the choice or the loading progress changes. */
export interface BackendState {
  info: BackendInfo;
  load: LoadView;
}

export type SetupView =
  | { state: "ready" }
  | { state: "download"; pending: PendingInstall }
  | { state: "problem"; reason: string }
  | { state: "unavailable"; why: "platform" | "build"; reason: string }
  | { state: "offline"; reason: string };
