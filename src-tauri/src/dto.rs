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

/// Returned by `file_start`: the job and the backend that will transcribe it.
#[derive(Debug, Clone, Serialize)]
pub struct JobStarted {
    pub job_id: String,
    pub backend: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobView {
    pub job_id: String,
    pub name: String,
    pub backend: String,
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
    /// `auto` (follow the OS), or a language id the UI offers: `en`, `zh-TW`.
    #[serde(default = "default_language")]
    pub language: String,
}

fn default_language() -> String {
    "auto".into()
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            device_id: None,
            theme: "system".into(),
            caption_size: 28,
            save_history: true,
            language: default_language(),
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
        let ok = self.language.len() <= 12
            && self
                .language
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-');
        if !ok || self.language.is_empty() {
            self.language = default_language();
        }
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
    pub backend_reason: String,
    pub ort_library: Option<String>,
    pub updater_enabled: bool,
    pub input_devices: Vec<DeviceDto>,
    pub device_error: Option<String>,
    pub live: LiveStatus,
    pub running_jobs: usize,
    pub config_dir: String,
    pub data_dir: String,
    pub recent_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateInfo {
    pub enabled: bool,
    pub available: bool,
    pub version: Option<String>,
    pub notes: Option<String>,
}
