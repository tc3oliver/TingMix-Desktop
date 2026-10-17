use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;

use crate::backends;
use crate::coded::coded;
use crate::dto::*;
use crate::files;
use crate::live::{Live, Persist, Sink};
use crate::locale::{tr, Lang, Text};
use crate::models::{ModelsOverview, PendingInstall};
use crate::state::{AppState, Job};
use serde_json::json;

pub const LIVE_EVENT: &str = "tingmix://live";
pub const FILE_EVENT: &str = "tingmix://file";

fn lang(state: &AppState) -> Lang {
    Lang::current(&state.store.settings())
}

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
    // never start capture silently without access: ask first, refuse when denied
    crate::mic::ensure_access().inspect_err(|e| state.note_error(e.clone()))?;
    let mut slot = state.live.lock().unwrap();
    if slot.as_ref().is_some_and(Live::is_running) {
        return Err(coded(
            "live.busy",
            json!({}),
            "a live session is already running",
        ));
    }
    let id = state.next_id("live");
    let resolved = backends::take(&app).map_err(|e| {
        state.note_error(format!("backend: {e}"));
        e
    })?;
    let label = resolved.label.clone();
    let sink: Sink = {
        let app = app.clone();
        Arc::new(move |m| {
            let _ = app.emit(LIVE_EVENT, m);
        })
    };
    let persist: Persist = {
        let app = app.clone();
        let (id, started, label) = (id.clone(), now_s(), label.clone());
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
                backend: label.clone(),
                finals: finals.to_vec(),
            };
            if let Err(e) = st.store.save_history(&entry) {
                st.note_error(format!("history save failed: {e}"));
            }
        })
    };
    match Live::start_mic(
        id,
        resolved.label,
        resolved.backend,
        device_id,
        sink,
        persist,
    ) {
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
pub async fn live_stop(app: AppHandle, state: State<'_, AppState>) -> Result<LiveStatus, String> {
    let live = state.live.lock().unwrap().take();
    if let Some(mut live) = live {
        live.stop();
        let status = live.shared.status();
        drop(live); // releases the backend before the next one is preloaded
        backends::preload(&app);
        return Ok(status);
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
        .set_title(tr(lang(&state), Text::PickAudio))
        .add_filter(
            tr(lang(&state), Text::FilterAudio),
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
) -> Result<JobStarted, String> {
    let path = state
        .files
        .lock()
        .unwrap()
        .get(&file_id)
        .cloned()
        .ok_or_else(|| coded("files.unknown", json!({}), "unknown file id"))?;
    let name = path
        .file_name()
        .map_or_else(|| "audio".into(), |n| n.to_string_lossy().into_owned());
    let resolved =
        backends::take(&app).inspect_err(|e| state.note_error(format!("backend: {e}")))?;
    let job_id = state.next_id("job");
    let backend_label = resolved.label.clone();
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    state.jobs.lock().unwrap().push((
        job_id.clone(),
        Job {
            name: name.clone(),
            backend: backend_label.clone(),
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
        .spawn(move || run_job(app, jid, name, path, cancel, resolved))
        .map_err(|e| e.to_string())?;
    Ok(JobStarted {
        job_id,
        backend: backend_label,
    })
}

fn run_job(
    app: AppHandle,
    job_id: String,
    name: String,
    path: PathBuf,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    resolved: backends::Resolved,
) {
    let backends::Resolved { backend, label } = resolved;
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
        files::run(&path, backend, &cancel, &mut reducer, &mut |b| {
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
    backends::preload(&app); // the backend moved into the job and is released; load the next one now
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
                    backend: label,
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
        .ok_or_else(|| coded("files.unknown", json!({}), "unknown job"))?;
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
    state: State<'_, AppState>,
    suggested_name: String,
    extension: String,
    content: String,
) -> Result<Option<String>, String> {
    if !matches!(extension.as_str(), "txt" | "srt") {
        return Err(coded(
            "export.format",
            json!({ "ext": extension }),
            format!("unsupported export format {extension:?}"),
        ));
    }
    if content.len() > 64 << 20 {
        return Err(coded("export.too_large", json!({}), "export too large"));
    }
    let stem: String = suggested_name
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | ' '))
        .take(80)
        .collect();
    let picked = app
        .dialog()
        .file()
        .set_title(tr(lang(&state), Text::ExportTitle))
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
    app: AppHandle,
    state: State<'_, AppState>,
    settings: Settings,
) -> Result<Settings, String> {
    let saved = state.store.save_settings(settings)?;
    crate::update_tray_language(&app);
    backends::preload(&app);
    Ok(saved)
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
    let bi = backends::info(
        &state.models,
        state.store.settings().experimental_backends,
        &state.runtimes,
    );
    Ok(Diagnostics {
        app_version: env!("CARGO_PKG_VERSION").into(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        backend: bi.label,
        backend_reason: bi.reason,
        ort_library: bi.ort_library,
        updater_enabled: state.updater,
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

pub const MODELS_EVENT: &str = "tingmix://models";

fn mark_experimental(state: &AppState, mut p: PendingInstall) -> PendingInstall {
    p.experimental =
        backends::is_experimental(&p.backend) && !state.store.settings().experimental_backends;
    p
}

fn models_notify(app: &AppHandle) -> crate::models::Notify {
    let app = app.clone();
    Arc::new(move |v| {
        let finished = v.state != "running";
        let onboard = v.state == "done" && matches!(v.kind.as_str(), "import" | "download");
        let id = v.name.clone();
        let _ = app.emit(MODELS_EVENT, v);
        if onboard {
            // one step for the user: a freshly installed model becomes the active one (when none is) and is tested
            backends::onboard(&app, &id, models_notify(&app));
        }
        if finished {
            // a new, imported or probed model can change which backend the next session uses
            backends::preload(&app);
        }
    })
}

#[tauri::command]
pub async fn models_overview(state: State<'_, AppState>) -> Result<ModelsOverview, String> {
    let mut o = state.models.overview();
    if o.registry_available {
        let d = backends::decide(
            &state.models,
            state.store.settings().experimental_backends,
            &state.runtimes,
        );
        o.selection = Some(crate::models::SelectionDto {
            chosen: d.chosen.map(|c| c.0),
            reason: d.reason,
            rejected: d.rejected,
        });
    }
    Ok(o)
}

/// `kind` is `folder` or `zip`; the dialog is opened here, so the frontend never names a path. The signed manifest in the
/// chosen package is verified and its licences returned; `models_install` with the returned token imports it.
#[tauri::command]
pub async fn models_import(
    app: AppHandle,
    state: State<'_, AppState>,
    kind: String,
) -> Result<Option<PendingInstall>, String> {
    let dialog = app
        .dialog()
        .file()
        .set_title(tr(lang(&state), Text::PickModelPackage));
    let picked = match kind.as_str() {
        "folder" => dialog.blocking_pick_folder(),
        "zip" => dialog
            .add_filter(tr(lang(&state), Text::FilterModelArchive), &["zip"])
            .blocking_pick_file(),
        other => return Err(format!("unknown import kind {other:?}")),
    };
    let Some(p) = picked else { return Ok(None) };
    let path = p.into_path().map_err(|e| e.to_string())?;
    state
        .models
        .begin_import(path, kind == "zip", state.next_id("install"))
        .map(|p| Some(mark_experimental(&state, p)))
        .inspect_err(|e| state.note_error(format!("model import: {e}")))
}

/// Pick a signed manifest, verify its signature and return the licences to accept. Nothing is downloaded yet.
#[tauri::command]
pub async fn models_begin_install(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Option<PendingInstall>, String> {
    let picked = app
        .dialog()
        .file()
        .set_title(tr(lang(&state), Text::PickManifest))
        .add_filter(tr(lang(&state), Text::FilterManifest), &["json"])
        .blocking_pick_file();
    let Some(p) = picked else { return Ok(None) };
    let bytes = crate::models::read_manifest_file(&p.into_path().map_err(|e| e.to_string())?)?;
    state
        .models
        .begin_install(&bytes, state.next_id("install"))
        .map(|p| Some(mark_experimental(&state, p)))
}

#[tauri::command]
pub async fn models_install(
    app: AppHandle,
    state: State<'_, AppState>,
    token: String,
    accepted: Vec<String>,
    enable_experimental: bool,
) -> Result<(), String> {
    // consent given in the confirm card replaces a separate trip to Settings
    if enable_experimental {
        let mut s = state.store.settings();
        s.experimental_backends = true;
        state.store.save_settings(s)?;
        backends::publish(&app);
    }
    state.models.install(&token, &accepted, models_notify(&app))
}

#[tauri::command]
pub async fn models_cancel(state: State<'_, AppState>) -> Result<(), String> {
    state.models.cancel();
    Ok(())
}

#[tauri::command]
pub async fn models_activate(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    state.models.activate(&id)?;
    // "Use this model" is one step: test it right away when no fresh test result exists
    let exp = state.store.settings().experimental_backends;
    let d = backends::decide(&state.models, exp, &state.runtimes);
    if d.chosen.is_none() && d.reason.contains("hasn't been tested") {
        if let Err(e) = backends::start_probe(
            &state.models,
            state.runtimes.clone(),
            id,
            models_notify(&app),
        ) {
            state.note_error(format!("test run: {e}"));
        }
    }
    backends::preload(&app);
    Ok(())
}

#[tauri::command]
pub async fn models_rollback(app: AppHandle, state: State<'_, AppState>) -> Result<String, String> {
    let id = state.models.rollback()?;
    backends::preload(&app);
    Ok(id)
}

#[tauri::command]
pub async fn models_remove(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    state.models.remove(&id)?;
    backends::publish(&app);
    Ok(())
}

#[tauri::command]
pub async fn models_verify(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state.models.verify(&id)
}

#[tauri::command]
pub async fn backend_state(state: State<'_, AppState>) -> Result<backends::BackendState, String> {
    Ok(backends::state_of(&state))
}

/// Compatibility probe of an installed version (loads it, decodes the shipped fixture, records RTF).
#[tauri::command]
pub async fn models_probe(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    backends::start_probe(
        &state.models,
        state.runtimes.clone(),
        id,
        models_notify(&app),
    )
}

fn no_updater() -> String {
    coded(
        "update.disabled",
        json!({}),
        "Updates are disabled: this build has no update signing key configured (see docs/release/signing-checklist.md).",
    )
}

#[tauri::command]
pub async fn updater_check(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<UpdateInfo, String> {
    if !state.updater {
        return Ok(UpdateInfo {
            enabled: false,
            available: false,
            version: None,
            notes: Some(no_updater()),
        });
    }
    use tauri_plugin_updater::UpdaterExt;
    let update = app
        .updater()
        .map_err(|e| e.to_string())?
        .check()
        .await
        .map_err(|e| e.to_string())?;
    Ok(match update {
        Some(u) => UpdateInfo {
            enabled: true,
            available: true,
            version: Some(u.version),
            notes: u.body,
        },
        None => UpdateInfo {
            enabled: true,
            available: false,
            version: None,
            notes: None,
        },
    })
}

/// Replaces the application bundle only. Models live in the app data directory and are never touched or re-activated.
#[tauri::command]
pub async fn updater_install(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    if !state.updater {
        return Err(no_updater());
    }
    if state
        .live
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(Live::is_running)
    {
        return Err(coded(
            "update.stop_live",
            json!({}),
            "Stop the live session before updating.",
        ));
    }
    use tauri_plugin_updater::UpdaterExt;
    let update = app
        .updater()
        .map_err(|e| e.to_string())?
        .check()
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| coded("update.current", json!({}), "already up to date"))?;
    update
        .download_and_install(|_, _| {}, || {})
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn mic_permission_status() -> Result<crate::mic::MicStatus, String> {
    Ok(crate::mic::status())
}

/// Shows the system prompt if the user has not decided yet; returns the resulting status.
#[tauri::command]
pub async fn mic_permission_request() -> Result<crate::mic::MicStatus, String> {
    Ok(crate::mic::request())
}

#[tauri::command]
pub async fn mic_open_settings() -> Result<(), String> {
    crate::mic::open_settings()
}
