use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;

use crate::dto::*;
use crate::files;
use crate::live::{Live, Persist, Sink};
use crate::state::{AppState, Job};

pub const LIVE_EVENT: &str = "tingmix://live";
pub const FILE_EVENT: &str = "tingmix://file";

fn now_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[tauri::command]
pub async fn list_input_devices() -> Result<Vec<DeviceDto>, String> {
    tingmix_audio::list_input_devices()
        .map_err(|e| e.to_string())
        .map(|d| {
            d.into_iter()
                .map(|d| DeviceDto {
                    id: d.id,
                    name: d.name,
                    is_default: d.is_default,
                    sample_rate: d.sample_rate,
                    channels: d.channels,
                })
                .collect()
        })
}

#[tauri::command]
pub async fn live_start(
    app: AppHandle,
    state: State<'_, AppState>,
    device_id: Option<String>,
) -> Result<LiveStatus, String> {
    let mut slot = state.live.lock().unwrap();
    if slot.as_ref().is_some_and(Live::is_running) {
        return Err("a live session is already running".into());
    }
    let id = state.next_id("live");
    let sink: Sink = {
        let app = app.clone();
        Arc::new(move |m| {
            let _ = app.emit(LIVE_EVENT, m);
        })
    };
    let persist: Persist = {
        let app = app.clone();
        let (id, started) = (id.clone(), now_s());
        Arc::new(move |finals, dur| {
            let st = app.state::<AppState>();
            if !st.store.settings().save_history || finals.is_empty() {
                return;
            }
            let entry = HistoryEntry {
                id: id.clone(),
                source: "live".into(),
                name: "Live session".into(),
                started_at: started,
                duration_s: dur,
                backend: BACKEND_LABEL.into(),
                finals: finals.to_vec(),
            };
            if let Err(e) = st.store.save_history(&entry) {
                st.note_error(format!("history save failed: {e}"));
            }
        })
    };
    match Live::start_mic(id, device_id, sink, persist) {
        Ok(live) => {
            let status = live.shared.status();
            *state.last.lock().unwrap() = Some(live.shared.clone());
            *slot = Some(live);
            Ok(status)
        }
        Err(e) => {
            state.note_error(format!("live start failed: {e}"));
            Err(e)
        }
    }
}

#[tauri::command]
pub async fn live_stop(state: State<'_, AppState>) -> Result<LiveStatus, String> {
    let live = state.live.lock().unwrap().take();
    if let Some(mut live) = live {
        live.stop();
        return Ok(live.shared.status());
    }
    Ok(state
        .last
        .lock()
        .unwrap()
        .as_ref()
        .map(|s| s.status())
        .unwrap_or_default())
}

#[tauri::command]
pub async fn live_snapshot(state: State<'_, AppState>) -> Result<LiveSnapshot, String> {
    let last = state.last.lock().unwrap().clone();
    Ok(match last {
        Some(s) => LiveSnapshot {
            status: s.status(),
            snapshot: s.snapshot(),
        },
        None => LiveSnapshot {
            status: LiveStatus {
                backend: BACKEND_LABEL.into(),
                level_db: -120.0,
                ..Default::default()
            },
            snapshot: Default::default(),
        },
    })
}

#[tauri::command]
pub async fn file_pick(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Option<FileRef>, String> {
    let picked = app
        .dialog()
        .file()
        .set_title("Choose an audio file")
        .add_filter(
            "Audio",
            &[
                "wav", "mp3", "flac", "ogg", "oga", "m4a", "aac", "mp4", "caf", "aiff", "aif",
            ],
        )
        .blocking_pick_file();
    let Some(p) = picked else { return Ok(None) };
    Ok(Some(
        state.register_file(p.into_path().map_err(|e| e.to_string())?),
    ))
}

#[tauri::command]
pub async fn file_start(
    app: AppHandle,
    state: State<'_, AppState>,
    file_id: String,
) -> Result<String, String> {
    let path = state
        .files
        .lock()
        .unwrap()
        .get(&file_id)
        .cloned()
        .ok_or("unknown file id")?;
    let name = path
        .file_name()
        .map_or_else(|| "audio".into(), |n| n.to_string_lossy().into_owned());
    let job_id = state.next_id("job");
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    state.jobs.lock().unwrap().push((
        job_id.clone(),
        Job {
            name: name.clone(),
            cancel: cancel.clone(),
            reducer: Default::default(),
            state: JobState::Running,
            audio_s: 0.0,
            message: None,
        },
    ));
    let jid = job_id.clone();
    std::thread::Builder::new()
        .name("tingmix-file".into())
        .spawn(move || run_job(app, jid, name, path, cancel))
        .map_err(|e| e.to_string())?;
    Ok(job_id)
}

fn run_job(
    app: AppHandle,
    job_id: String,
    name: String,
    path: PathBuf,
    cancel: Arc<std::sync::atomic::AtomicBool>,
) {
    let st = app.state::<AppState>();
    let started = now_s();
    let send = |body: FileBody| {
        let _ = app.emit(
            FILE_EVENT,
            FileMsg {
                job_id: job_id.clone(),
                body,
            },
        );
    };
    // the decoder returns errors, but a panic inside a codec must still not take the app down
    let mut reducer = tingmix_core::Reducer::new();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        files::run(&path, &cancel, &mut reducer, &mut |b| {
            if let FileBody::Event { event } = &b {
                if let Some((_, j)) = st
                    .jobs
                    .lock()
                    .unwrap()
                    .iter_mut()
                    .find(|(i, _)| *i == job_id)
                {
                    let _ = j.reducer.apply(event.clone());
                }
            }
            send(b);
        })
    }))
    .unwrap_or_else(|_| files::Outcome {
        state: JobState::Failed,
        audio_s: 0.0,
        message: Some("decoder crashed on this file".into()),
    });
    if let Some((_, j)) = st
        .jobs
        .lock()
        .unwrap()
        .iter_mut()
        .find(|(i, _)| *i == job_id)
    {
        j.state = outcome.state.clone();
        j.audio_s = outcome.audio_s;
        j.message = outcome.message.clone();
    }
    match outcome.state {
        JobState::Done => {
            send(FileBody::Done {
                audio_s: outcome.audio_s,
            });
            let finals: Vec<_> = reducer.finals().cloned().collect();
            if st.store.settings().save_history && !finals.is_empty() {
                let entry = HistoryEntry {
                    id: job_id.clone(),
                    source: "file".into(),
                    name,
                    started_at: started,
                    duration_s: outcome.audio_s,
                    backend: BACKEND_LABEL.into(),
                    finals,
                };
                if let Err(e) = st.store.save_history(&entry) {
                    st.note_error(format!("history save failed: {e}"));
                }
            }
        }
        JobState::Cancelled => send(FileBody::Cancelled),
        _ => {
            let m = outcome.message.unwrap_or_default();
            st.note_error(format!("file transcription failed: {m}"));
            send(FileBody::Failed { message: m });
        }
    }
}

#[tauri::command]
pub async fn file_cancel(state: State<'_, AppState>, job_id: String) -> Result<(), String> {
    let jobs = state.jobs.lock().unwrap();
    let (_, j) = jobs
        .iter()
        .find(|(i, _)| *i == job_id)
        .ok_or("unknown job")?;
    j.cancel.store(true, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
pub async fn file_jobs(state: State<'_, AppState>) -> Result<Vec<JobView>, String> {
    Ok(state
        .jobs
        .lock()
        .unwrap()
        .iter()
        .map(|(id, j)| j.view(id))
        .collect())
}

/// Save dialog + write. The frontend supplies the content and a suggested name, never a path.
#[tauri::command]
pub async fn export_text(
    app: AppHandle,
    suggested_name: String,
    extension: String,
    content: String,
) -> Result<Option<String>, String> {
    if !matches!(extension.as_str(), "txt" | "srt") {
        return Err(format!("unsupported export format {extension:?}"));
    }
    if content.len() > 64 << 20 {
        return Err("export too large".into());
    }
    let stem: String = suggested_name
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | ' '))
        .take(80)
        .collect();
    let picked = app
        .dialog()
        .file()
        .set_title("Export transcript")
        .set_file_name(format!(
            "{}.{extension}",
            if stem.is_empty() { "transcript" } else { &stem }
        ))
        .add_filter(extension.to_uppercase(), &[extension.as_str()])
        .blocking_save_file();
    let Some(p) = picked else { return Ok(None) };
    let path = p.into_path().map_err(|e| e.to_string())?;
    std::fs::write(&path, content).map_err(|e| e.to_string())?;
    Ok(Some(path.display().to_string()))
}

#[tauri::command]
pub async fn settings_get(state: State<'_, AppState>) -> Result<Settings, String> {
    Ok(state.store.settings())
}

#[tauri::command]
pub async fn settings_set(
    state: State<'_, AppState>,
    settings: Settings,
) -> Result<Settings, String> {
    state.store.save_settings(settings)
}

#[tauri::command]
pub async fn history_list(state: State<'_, AppState>) -> Result<Vec<HistorySummary>, String> {
    Ok(state.store.list_history())
}

#[tauri::command]
pub async fn history_load(state: State<'_, AppState>, id: String) -> Result<HistoryEntry, String> {
    state.store.load_history(&id)
}

#[tauri::command]
pub async fn history_delete(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state.store.delete_history(&id)
}

#[tauri::command]
pub async fn diagnostics(state: State<'_, AppState>) -> Result<Diagnostics, String> {
    let (devices, device_error) = match list_input_devices().await {
        Ok(d) => (d, None),
        Err(e) => (Vec::new(), Some(e)),
    };
    let live = state
        .last
        .lock()
        .unwrap()
        .as_ref()
        .map(|s| s.status())
        .unwrap_or_default();
    Ok(Diagnostics {
        app_version: env!("CARGO_PKG_VERSION").into(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        backend: BACKEND_LABEL.into(),
        input_devices: devices,
        device_error,
        live,
        running_jobs: state
            .jobs
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, j)| j.state == JobState::Running)
            .count(),
        config_dir: state.store.config_dir().display().to_string(),
        data_dir: state.store.data_dir().display().to_string(),
        recent_errors: state.errors.lock().unwrap().iter().cloned().collect(),
    })
}

#[tauri::command]
pub async fn models_overview() -> Result<ModelsOverview, String> {
    Ok(ModelsOverview {
        registry_available: false,
        note: "The model registry is not connected to this build yet. Transcription uses the mock backend, which is not ASR.".into(),
    })
}
