import { invoke } from "@tauri-apps/api/core";
import { hostText } from "../i18n/host";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  BackendState,
  Device,
  Diagnostics,
  FileMsg,
  FileRef,
  HistoryEntry,
  HistorySummary,
  JobView,
  LiveMsg,
  JobStarted,
  LiveSnapshot,
  LiveStatus,
  MicStatus,
  ModelOp,
  ModelsOverview,
  PendingInstall,
  Settings,
  SetupView,
  UpdateInfo,
} from "../types/protocol";

// One wrapper per host command (src-tauri/src/commands.rs); argument names are camelCase here, snake_case in Rust.
export const api = {
  listInputDevices: () => invoke<Device[]>("list_input_devices"),
  liveStart: (deviceId: string | null) => invoke<LiveStatus>("live_start", { deviceId }),
  liveStop: () => invoke<LiveStatus>("live_stop"),
  liveSnapshot: () => invoke<LiveSnapshot>("live_snapshot"),
  filePick: () => invoke<FileRef | null>("file_pick"),
  fileStart: (fileId: string) => invoke<JobStarted>("file_start", { fileId }),
  fileCancel: (jobId: string) => invoke<void>("file_cancel", { jobId }),
  fileJobs: () => invoke<JobView[]>("file_jobs"),
  exportText: (suggestedName: string, extension: "txt" | "srt", content: string) =>
    invoke<string | null>("export_text", { suggestedName, extension, content }),
  settingsGet: () => invoke<Settings>("settings_get"),
  settingsSet: (settings: Settings) => invoke<Settings>("settings_set", { settings }),
  historyList: () => invoke<HistorySummary[]>("history_list"),
  historyLoad: (id: string) => invoke<HistoryEntry>("history_load", { id }),
  historyDelete: (id: string) => invoke<void>("history_delete", { id }),
  diagnostics: () => invoke<Diagnostics>("diagnostics"),
  modelsOverview: () => invoke<ModelsOverview>("models_overview"),
  modelsBeginImport: (kind: "folder" | "zip") => invoke<PendingInstall | null>("models_import", { kind }),
  modelsBeginInstall: () => invoke<PendingInstall | null>("models_begin_install"),
  modelsInstall: (token: string, accepted: string[]) => invoke<void>("models_install", { token, accepted }),
  modelsCancel: () => invoke<void>("models_cancel"),
  modelsActivate: (id: string) => invoke<void>("models_activate", { id }),
  modelsRollback: () => invoke<string>("models_rollback"),
  modelsRemove: (id: string) => invoke<void>("models_remove", { id }),
  modelsProbe: (id: string) => invoke<void>("models_probe", { id }),
  setupCheck: () => invoke<SetupView>("setup_check"),
  backendState: () => invoke<BackendState>("backend_state"),
  updaterCheck: () => invoke<UpdateInfo>("updater_check"),
  micStatus: () => invoke<MicStatus>("mic_permission_status"),
  micRequest: () => invoke<MicStatus>("mic_permission_request"),
  micOpenSettings: () => invoke<void>("mic_open_settings"),
  updaterInstall: () => invoke<void>("updater_install"),
  modelsVerify: (id: string) => invoke<void>("models_verify", { id }),
};

export const onLive = (f: (m: LiveMsg) => void): Promise<UnlistenFn> => listen<LiveMsg>("tingmix://live", (e) => f(e.payload));
export const onModelOp = (f: (o: ModelOp) => void): Promise<UnlistenFn> => listen<ModelOp>("tingmix://models", (e) => f(e.payload));
export const onFile = (f: (m: FileMsg) => void): Promise<UnlistenFn> => listen<FileMsg>("tingmix://file", (e) => f(e.payload));
export const onFilesDropped = (f: (r: FileRef[]) => void): Promise<UnlistenFn> =>
  listen<FileRef[]>("tingmix://files-dropped", (e) => f(e.payload));
export const onDrag = (f: (over: boolean) => void): Promise<UnlistenFn> => listen<boolean>("tingmix://drag", (e) => f(e.payload));

/** Error from a command as text in the current language: coded host messages are translated, anything else is shown as is. */
export const errorText = (e: unknown): string => hostText(typeof e === "string" ? e : e instanceof Error ? e.message : JSON.stringify(e));
export const onBackend = (f: (v: BackendState) => void): Promise<UnlistenFn> => listen<BackendState>("tingmix://backend", (e) => f(e.payload));
