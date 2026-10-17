//! Payloads of commands and events. Field names are snake_case (the protocol's), command arguments are
//! camelCase on the JS side (Tauri's convention).

use serde::{Deserialize, Serialize};
use tingmix_core::{Event, FinalEvent, Snapshot};

/// The only backend wired today. The UI shows this string verbatim wherever text is produced.
pub const BACKEND_LABEL: &str = "Mock — not ASR";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LiveState {
    #[default]
    Idle,
    Starting,
    Recording,
    Stopping,
    Error,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct LiveStatus {
    pub state: LiveState,
    pub session_id: Option<String>,
    pub elapsed_s: f64,
    pub samples_in: u64,
    pub overflow_samples: u64,
    /// RMS level of the latest audio block in dBFS (-120 for digital silence).
    pub level_db: f64,
    pub device: String,
    pub backend: String,
    pub coalesced_partials: u64,
    pub message: Option<String>,
}

/// Everything emitted on `tingmix://live`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LiveMsg {
    Event { session_id: String, event: Event },
    Status(LiveStatus),
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveSnapshot {
    pub status: LiveStatus,
    pub snapshot: Snapshot,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceDto {
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub sample_rate: u32,
    pub channels: u16,
}

/// A file the user picked or dropped. The frontend only ever holds this opaque id, never a path it could forge.
#[derive(Debug, Clone, Serialize)]
pub struct FileRef {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Running,
    Done,
    Cancelled,
    Failed,
}

/// Everything emitted on `tingmix://file`.
#[derive(Debug, Clone, Serialize)]
pub struct FileMsg {
    pub job_id: String,
    #[serde(flatten)]
    pub body: FileBody,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FileBody {
    Progress { audio_s: f64 },
    Event { event: Event },
    Done { audio_s: f64 },
    Cancelled,
    Failed { message: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct JobView {
    pub job_id: String,
    pub name: String,
    pub state: JobState,
    pub audio_s: f64,
    pub snapshot: Snapshot,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Microphone id as listed by `list_input_devices`; `None` = system default.
    pub device_id: Option<String>,
    /// `system`, `light` or `dark`.
    pub theme: String,
    /// Caption font size in px.
    pub caption_size: u32,
    pub save_history: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            device_id: None,
            theme: "system".into(),
            caption_size: 28,
            save_history: true,
        }
    }
}

impl Settings {
    /// Clamp values a hand-edited file or a buggy caller could make unusable.
    pub fn sanitized(mut self) -> Self {
        if !matches!(self.theme.as_str(), "system" | "light" | "dark") {
            self.theme = "system".into();
        }
        self.caption_size = self.caption_size.clamp(12, 96);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    /// `live` or `file`.
    pub source: String,
    pub name: String,
    pub started_at: u64,
    pub duration_s: f64,
    pub backend: String,
    pub finals: Vec<FinalEvent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistorySummary {
    pub id: String,
    pub source: String,
    pub name: String,
    pub started_at: u64,
    pub duration_s: f64,
    pub backend: String,
    pub segments: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Diagnostics {
    pub app_version: String,
    pub os: String,
    pub arch: String,
    pub backend: String,
    pub input_devices: Vec<DeviceDto>,
    pub device_error: Option<String>,
    pub live: LiveStatus,
    pub running_jobs: usize,
    pub config_dir: String,
    pub data_dir: String,
    pub recent_errors: Vec<String>,
}

/// The model panel's view. `registry_available` is false until `tingmix-models` is wired into the host.
#[derive(Debug, Clone, Serialize)]
pub struct ModelsOverview {
    pub registry_available: bool,
    pub note: String,
}
