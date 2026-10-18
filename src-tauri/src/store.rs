//! Settings and transcript history on disk. Writes are atomic (temp file + rename); ids are validated because the
//! frontend supplies them.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::dto::{HistoryEntry, HistorySummary, Settings};

pub struct Store {
    config_dir: PathBuf,
    data_dir: PathBuf,
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    fs::create_dir_all(path.parent().expect("file path has a parent"))?;
    let tmp = path.with_extension("tmp");
    let mut f = fs::File::create(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    fs::rename(&tmp, path)
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

impl Store {
    pub fn new(config_dir: PathBuf, data_dir: PathBuf) -> Self {
        Self {
            config_dir,
            data_dir,
        }
    }

    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Missing or corrupt settings fall back to defaults; the app must start either way.
    pub fn settings(&self) -> Settings {
        fs::read(self.config_dir.join("settings.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Settings>(&b).ok())
            .unwrap_or_default()
            .sanitized()
    }

    pub fn save_settings(&self, s: Settings) -> Result<Settings, String> {
        let s = s.sanitized();
        let json = serde_json::to_vec_pretty(&s).map_err(|e| e.to_string())?;
        write_atomic(&self.config_dir.join("settings.json"), &json).map_err(|e| e.to_string())?;
        Ok(s)
    }

    fn history_path(&self, id: &str) -> Result<PathBuf, String> {
        if !valid_id(id) {
            return Err(format!("invalid history id {id:?}"));
        }
        Ok(self.data_dir.join("history").join(format!("{id}.json")))
    }

    pub fn save_history(&self, e: &HistoryEntry) -> Result<(), String> {
        let json = serde_json::to_vec(e).map_err(|e| e.to_string())?;
        write_atomic(&self.history_path(&e.id)?, &json).map_err(|e| e.to_string())
    }

    pub fn load_history(&self, id: &str) -> Result<HistoryEntry, String> {
        let b = fs::read(self.history_path(id)?).map_err(|e| e.to_string())?;
        serde_json::from_slice(&b)
            .map(legacy_name)
            .map_err(|e| e.to_string())
    }

    pub fn delete_history(&self, id: &str) -> Result<(), String> {
        fs::remove_file(self.history_path(id)?).map_err(|e| e.to_string())
    }

    /// Newest first. Unreadable entries are skipped, not fatal.
    pub fn list_history(&self) -> Vec<HistorySummary> {
        let Ok(dir) = fs::read_dir(self.data_dir.join("history")) else {
            return Vec::new();
        };
        let mut out: Vec<HistorySummary> = dir
            .flatten()
            .filter(|f| f.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|f| serde_json::from_slice::<HistoryEntry>(&fs::read(f.path()).ok()?).ok())
            .map(legacy_name)
            .map(|e| HistorySummary {
                segments: e.finals.len(),
                id: e.id,
                source: e.source,
                name: e.name,
                started_at: e.started_at,
                duration_s: e.duration_s,
                backend: e.backend,
            })
            .collect();
        out.sort_by(|a, b| b.started_at.cmp(&a.started_at).then(b.id.cmp(&a.id)));
        out
    }
}

/// Live entries saved by earlier builds are named "Live session" in English; live entries now have no name and the UI titles
/// them from the start time in the user's language.
fn legacy_name(mut e: HistoryEntry) -> HistoryEntry {
    if e.source == "live" && e.name == "Live session" {
        e.name.clear();
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use tingmix_core::FinalEvent;

    fn store(tag: &str) -> (Store, PathBuf) {
        let dir = std::env::temp_dir().join(format!("tingmix-store-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        (Store::new(dir.join("config"), dir.join("data")), dir)
    }

    fn entry(id: &str, at: u64) -> HistoryEntry {
        HistoryEntry {
            id: id.into(),
            source: "live".into(),
            name: "t".into(),
            started_at: at,
            duration_s: 1.0,
            backend: "mock".into(),
            finals: vec![FinalEvent {
                segment_id: 0,
                revision: 1,
                text: "w1".into(),
                start: 0.0,
                end: 1.0,
            }],
        }
    }

    #[test]
    fn settings_roundtrip_default_and_sanitize() {
        let (s, dir) = store("settings");
        assert_eq!(s.settings().theme, "system");
        let saved = s
            .save_settings(Settings {
                theme: "neon".into(),
                caption_size: 1000,
                ..Settings::default()
            })
            .unwrap();
        assert_eq!((saved.theme.as_str(), saved.caption_size), ("system", 96));
        assert_eq!(s.settings().caption_size, 96);
        fs::write(s.config_dir().join("settings.json"), b"{broken").unwrap();
        assert_eq!(s.settings().caption_size, 28);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn live_entries_saved_with_the_old_english_name_read_back_nameless() {
        let (s, dir) = store("legacy");
        let mut old = entry("live-9", 9);
        old.name = "Live session".into();
        let mut file = entry("job-9", 8);
        file.source = "file".into();
        file.name = "Live session".into(); // a file really named like that keeps its name
        s.save_history(&old).unwrap();
        s.save_history(&file).unwrap();
        assert_eq!(s.load_history("live-9").unwrap().name, "");
        let names: Vec<_> = s
            .list_history()
            .into_iter()
            .map(|h| (h.id, h.name))
            .collect();
        assert_eq!(
            names,
            [
                ("live-9".to_string(), String::new()),
                ("job-9".to_string(), "Live session".to_string())
            ]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn history_roundtrip_order_delete_and_id_validation() {
        let (s, dir) = store("history");
        s.save_history(&entry("live-1", 10)).unwrap();
        s.save_history(&entry("live-2", 20)).unwrap();
        let ids: Vec<_> = s.list_history().into_iter().map(|h| h.id).collect();
        assert_eq!(ids, ["live-2", "live-1"]);
        assert_eq!(s.load_history("live-1").unwrap().finals[0].text, "w1");
        s.delete_history("live-1").unwrap();
        assert_eq!(s.list_history().len(), 1);
        for bad in ["../x", "a/b", "", "a.b"] {
            assert!(s.load_history(bad).is_err(), "{bad}");
        }
        fs::remove_dir_all(dir).unwrap();
    }
}
