//! Compatibility probe runner. A backend implements [`ProbeTarget`]; [`run_probe`] loads the installed model through it,
//! decodes one short fixture twice-plus (load/first decode, then timed runs), compares the text with the expected text
//! and persists a [`ProbeRecord`]. [`consult`] returns the record only while everything it was measured against is
//! unchanged (backend version, model manifest hash, OS and version, arch, device), so a driver, OS or model change forces
//! a new probe instead of trusting stale numbers.
//!
//! A backend that can crash the process (native GPU runtimes) should implement `decode` by spawning a child process:
//! a panic is caught here, a segfault is not.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::manifest::{sha256_hex, Manifest};
use crate::select::Probe;
use crate::store::{write_atomic, InstalledModel};

pub trait ProbeTarget {
    /// Matches `Manifest::backend`.
    fn backend(&self) -> &str;
    /// Version of the runtime/driver stack actually loaded; `None` = unavailable.
    fn backend_version(&self) -> Option<String>;
    /// Device the model runs on, as reported by the backend (`gpu:Apple M4 Max`, `cpu`). Never inferred from a vendor.
    fn device(&self) -> String;
    /// Load `model` if not loaded yet and decode mono f32 PCM at the model's sample rate.
    fn decode(
        &mut self,
        model: &InstalledModel,
        wav: &[f32],
    ) -> std::result::Result<String, String>;
}

pub struct ProbeFixture<'a> {
    pub name: &'a str,
    pub wav: &'a [f32],
    pub sample_rate: u32,
    pub expected_text: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeRecord {
    pub backend: String,
    pub backend_version: String,
    pub model_id: String,
    /// sha256 of the canonical manifest bytes: binds every file hash, so any model change invalidates the record.
    pub model_sha256: String,
    pub os: String,
    pub os_version: String,
    pub arch: String,
    pub device: String,
    pub fixture: String,
    pub passed: bool,
    pub rtf_x1000: Option<u32>,
    pub first_run_ms: u64,
    pub detail: String,
    pub unix_time: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Consulted {
    Fresh(Probe),
    Missing,
    /// A record exists but was measured against something else; the string says what changed.
    Stale(String),
}

fn host_os_version() -> String {
    sysinfo::System::long_os_version().unwrap_or_default()
}

fn model_sha(m: &Manifest) -> String {
    sha256_hex(&m.canonical_bytes())
}

fn record_path(root: &Path, backend: &str, m: &Manifest) -> PathBuf {
    let b: String = backend
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    root.join("probes")
        .join(format!("{b}--{}.json", &model_sha(m)[..16]))
}

/// Run the probe and persist the result (a failed probe is persisted too, so selection rejects it with a reason).
pub fn run_probe(
    root: &Path,
    target: &mut dyn ProbeTarget,
    model: &InstalledModel,
    fx: &ProbeFixture,
    timed_runs: usize,
) -> Result<ProbeRecord> {
    let audio_s = fx.wav.len() as f64 / fx.sample_rate as f64;
    let run = |t: &mut dyn ProbeTarget| -> (std::result::Result<String, String>, f64) {
        let t0 = Instant::now();
        let r = catch_unwind(AssertUnwindSafe(|| t.decode(model, fx.wav)))
            .unwrap_or_else(|_| Err("backend panicked".into()));
        (r, t0.elapsed().as_secs_f64())
    };
    let (first, first_s) = run(target);
    let (mut passed, mut detail, mut times) = (false, String::new(), Vec::new());
    match first {
        Err(e) => detail = format!("load/first decode failed: {e}"),
        Ok(text) if text.trim() != fx.expected_text.trim() => {
            detail = format!(
                "first decode text {text:?} != expected {:?}",
                fx.expected_text
            )
        }
        Ok(_) => {
            passed = true;
            for _ in 0..timed_runs.max(1) {
                let (r, s) = run(target);
                match r {
                    Ok(t) if t.trim() == fx.expected_text.trim() => times.push(s),
                    Ok(t) => {
                        (passed, detail) = (
                            false,
                            format!("timed decode text {t:?} != expected {:?}", fx.expected_text),
                        )
                    }
                    Err(e) => (passed, detail) = (false, format!("timed decode failed: {e}")),
                }
                if !passed {
                    break;
                }
            }
        }
    }
    times.sort_by(f64::total_cmp);
    let rtf_x1000 = (passed && !times.is_empty())
        .then(|| ((times[times.len() / 2] / audio_s) * 1000.0).ceil() as u32);
    let rec = ProbeRecord {
        backend: target.backend().into(),
        backend_version: target.backend_version().unwrap_or_default(),
        model_id: model.manifest.id.clone(),
        model_sha256: model_sha(&model.manifest),
        os: std::env::consts::OS.into(),
        os_version: host_os_version(),
        arch: std::env::consts::ARCH.into(),
        device: target.device(),
        fixture: fx.name.into(),
        passed,
        rtf_x1000,
        first_run_ms: (first_s * 1000.0) as u64,
        detail,
        unix_time: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    write_atomic(
        &record_path(root, &rec.backend, &model.manifest),
        &serde_json::to_vec_pretty(&rec).unwrap(),
    )?;
    Ok(rec)
}

/// Look up the stored probe for (`backend`, `m`) and check it still applies to the current `backend_version` and `device`
/// on this OS build. `backend_version == None` (runtime unavailable) is always stale.
pub fn consult(
    root: &Path,
    backend: &str,
    m: &Manifest,
    backend_version: Option<&str>,
    device: &str,
) -> Consulted {
    let path = record_path(root, backend, m);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Consulted::Missing,
        Err(e) => return Consulted::Stale(format!("unreadable record: {e}")),
    };
    let Ok(r) = serde_json::from_slice::<ProbeRecord>(&bytes) else {
        return Consulted::Stale("unparsable record".into());
    };
    let checks = [
        ("backend", r.backend == backend),
        ("model", r.model_sha256 == model_sha(m)),
        (
            "backend version",
            backend_version == Some(r.backend_version.as_str()),
        ),
        ("device", r.device == device),
        (
            "OS",
            r.os == std::env::consts::OS && r.os_version == host_os_version(),
        ),
        ("architecture", r.arch == std::env::consts::ARCH),
    ];
    match checks.iter().find(|c| !c.1) {
        Some((what, _)) => Consulted::Stale(format!("{what} changed since the probe ran")),
        None => Consulted::Fresh(Probe {
            passed: r.passed,
            rtf_x1000: r.rtf_x1000,
        }),
    }
}

/// Convenience for [`crate::select::select_probed`]: the fresh probe result for a manifest, or `None`.
pub fn fresh(
    root: &Path,
    backend: &str,
    m: &Manifest,
    backend_version: Option<&str>,
    device: &str,
) -> Option<Probe> {
    match consult(root, backend, m, backend_version, device) {
        Consulted::Fresh(p) => Some(p),
        _ => None,
    }
}
