use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tingmix_core::Reducer;

use crate::backends::{Runtimes, Warm};
use crate::dto::{FileRef, JobState, JobView};
use crate::live::{Live, Shared};
use crate::models::Models;
use crate::store::Store;

pub struct Job {
    pub name: String,
    pub cancel: Arc<AtomicBool>,
    pub reducer: Reducer,
    pub state: JobState,
    pub audio_s: f64,
    pub message: Option<String>,
}

impl Job {
    pub fn view(&self, job_id: &str) -> JobView {
        JobView {
            job_id: job_id.into(),
            name: self.name.clone(),
            state: self.state.clone(),
            audio_s: self.audio_s,
            snapshot: self.reducer.snapshot(),
            message: self.message.clone(),
        }
    }
}

pub struct AppState {
    pub store: Store,
    pub models: Arc<Models>,
    pub runtimes: Runtimes,
    pub warm: Warm,
    /// The updater plugin is registered only when the build carries an update signing key.
    pub updater: bool,
    pub live: Mutex<Option<Live>>,
    /// The latest session, kept after stop so the transcript is preserved and a reloaded window can recover it.
    pub last: Mutex<Option<Arc<Shared>>>,
    pub jobs: Mutex<Vec<(String, Job)>>,
    /// Files the user chose or dropped; the frontend refers to them by id only.
    pub files: Mutex<HashMap<String, PathBuf>>,
    pub errors: Mutex<VecDeque<String>>,
    seq: AtomicU64,
}

impl AppState {
    pub fn new(store: Store, models: Models, runtimes: Runtimes, updater: bool) -> Self {
        Self {
            store,
            models: Arc::new(models),
            runtimes,
            warm: Warm::default(),
            updater,
            live: Mutex::default(),
            last: Mutex::default(),
            jobs: Mutex::default(),
            files: Mutex::default(),
            errors: Mutex::default(),
            seq: AtomicU64::new(0),
        }
    }

    /// Unique within and across runs: `<prefix>-<unix ms>-<counter>`.
    pub fn next_id(&self, prefix: &str) -> String {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        format!("{prefix}-{ms}-{}", self.seq.fetch_add(1, Ordering::Relaxed))
    }

    pub fn register_file(&self, path: PathBuf) -> FileRef {
        let id = self.next_id("file");
        let name = path
            .file_name()
            .map_or_else(|| "audio".into(), |n| n.to_string_lossy().into_owned());
        self.files.lock().unwrap().insert(id.clone(), path);
        FileRef { id, name }
    }

    pub fn note_error(&self, msg: impl Into<String>) {
        let mut e = self.errors.lock().unwrap();
        if e.len() >= 50 {
            e.pop_front();
        }
        e.push_back(msg.into());
    }
}
