//! Model management for the UI, on top of `tingmix-models`: list, import, verified download, activate, remove, rollback.
//! Nothing here loads a model; no inference backend is registered in this build, which the overview says explicitly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tingmix_models::caps::Capabilities;
use tingmix_models::fetch::Downloader;
use tingmix_models::import::{import_dir, import_zip};
use tingmix_models::{Registry, VerifiedManifest, Verifier};

#[derive(Debug, Clone, Serialize)]
pub struct InstalledDto {
    pub id: String,
    pub family: String,
    pub revision: String,
    pub backend: String,
    pub precision: String,
    pub size_bytes: u64,
    pub serial: u64,
    /// The manifest's own validation status word (VERIFIED / EXPERIMENTAL / BLOCKED ...).
    pub compat_status: String,
    pub compat_notes: String,
    pub licenses: Vec<String>,
    pub active: bool,
    pub previous: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CapsDto {
    pub os: String,
    pub arch: String,
    pub ram_bytes: u64,
    pub apple_chip: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SelectionDto {
    pub chosen: Option<String>,
    pub reason: String,
    pub rejected: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpView {
    pub kind: String,
    pub name: String,
    /// `running`, `done`, `failed` or `cancelled`.
    pub state: String,
    pub message: Option<String>,
    pub bytes_done: u64,
    pub bytes_total: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelsOverview {
    pub registry_available: bool,
    pub note: String,
    pub root: String,
    pub capabilities: Option<CapsDto>,
    pub installed: Vec<InstalledDto>,
    pub unreadable: Vec<(String, String)>,
    pub selection: Option<SelectionDto>,
    pub op: Option<OpView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LicenseDto {
    pub id: String,
    pub name: String,
    pub url: String,
    pub spdx: Option<String>,
    pub acceptance_required: bool,
    pub accepted: bool,
    pub notes: String,
}

/// A verified manifest waiting for the user's licence decision. The frontend holds only the token.
#[derive(Debug, Clone, Serialize)]
pub struct PendingInstall {
    pub token: String,
    pub id: String,
    pub family: String,
    pub backend: String,
    pub precision: String,
    pub download_bytes: u64,
    pub compat_status: String,
    pub licenses: Vec<LicenseDto>,
}

struct Op {
    view: OpView,
    cancel: Arc<AtomicBool>,
}

pub struct Models {
    registry: Result<Registry, String>,
    root: PathBuf,
    caps: Capabilities,
    /// Tests only: manifests may name `http://127.0.0.1:<port>/` file URLs.
    loopback_http: bool,
    pending: Mutex<HashMap<String, Pending>>,
    op: Mutex<Option<Op>>,
}

/// A verified manifest awaiting the licence decision, and the folder/zip to import it from (`None` = download).
type Pending = (VerifiedManifest, Option<(PathBuf, bool)>);

pub type Notify = Arc<dyn Fn(OpView) + Send + Sync>;

impl Models {
    pub fn caps(&self) -> &Capabilities {
        &self.caps
    }

    pub fn open(root: PathBuf) -> Self {
        Self::open_with(root, Verifier::embedded(), false)
    }

    /// `verifier` and `loopback_http` are for tests (a throwaway key and a local fixture server).
    pub fn open_with(root: PathBuf, verifier: Verifier, loopback_http: bool) -> Self {
        let registry = std::fs::create_dir_all(&root)
            .map_err(|e| e.to_string())
            .and_then(|()| Registry::open(&root, verifier).map_err(|e| e.to_string()))
            .map(|r| r.allow_loopback_http(loopback_http));
        Self {
            registry,
            root,
            caps: Capabilities::detect(),
            loopback_http,
            pending: Mutex::default(),
            op: Mutex::default(),
        }
    }

    pub fn reg(&self) -> Result<&Registry, String> {
        self.registry
            .as_ref()
            .map_err(|e| format!("model registry unavailable: {e}"))
    }

    pub fn overview(&self) -> ModelsOverview {
        let caps = CapsDto {
            os: self.caps.os.clone(),
            arch: self.caps.arch.clone(),
            ram_bytes: self.caps.ram_bytes,
            apple_chip: self.caps.apple_chip.as_ref().map(|c| c.name.clone()),
        };
        let op = self.op.lock().unwrap().as_ref().map(|o| o.view.clone());
        let reg = match &self.registry {
            Ok(r) => r,
            Err(e) => {
                return ModelsOverview {
                    registry_available: false,
                    note: crate::coded::coded(
                        "models.registry_error",
                        serde_json::json!({ "detail": e }),
                        format!("The model registry could not be opened: {e}"),
                    ),
                    root: self.root.display().to_string(),
                    capabilities: Some(caps),
                    installed: vec![],
                    unreadable: vec![],
                    selection: None,
                    op,
                }
            }
        };
        let (active, previous) = (reg.active().ok().flatten(), reg.previous().ok().flatten());
        let (mut installed, mut unreadable) = (Vec::new(), Vec::new());
        for id in reg.list().unwrap_or_default() {
            match reg.open_model(&id) {
                Ok(m) => {
                    let mf = m.manifest;
                    installed.push(InstalledDto {
                        active: active.as_deref() == Some(&mf.id),
                        previous: previous.as_deref() == Some(&mf.id),
                        size_bytes: mf.download_size(),
                        licenses: mf.licenses.iter().map(|l| l.name.clone()).collect(),
                        id: mf.id.clone(),
                        family: mf.family.clone(),
                        revision: mf.revision.clone(),
                        backend: mf.backend.clone(),
                        precision: mf.precision.clone(),
                        serial: mf.serial,
                        compat_status: mf.compat.status.clone(),
                        compat_notes: mf.compat.notes.clone(),
                    });
                }
                Err(e) => unreadable.push((id, e.to_string())),
            }
        }
        ModelsOverview {
            registry_available: true,
            note: crate::coded::coded(
                "models.note",
                serde_json::json!({}),
                "Installed models are stored and verified here; whether one can run is decided by runtime selection below.",
            ),
            root: self.root.display().to_string(),
            capabilities: Some(caps),
            installed,
            unreadable,
            selection: None,
            op,
        }
    }

    /// Parse and verify a signed manifest file and report what the user must accept. Nothing is downloaded yet.
    pub fn begin_install(&self, bytes: &[u8], token: String) -> Result<PendingInstall, String> {
        let m = self
            .reg()?
            .verifier()
            .parse(bytes)
            .map_err(|e| e.to_string())?;
        if let Some(f) = m.manifest().files.iter().find(|f| f.url.is_empty()) {
            return Err(crate::coded::coded(
                "models.import_only",
                serde_json::json!({ "file": f.path }),
                format!("{} has no download URL: this version can only be imported from a folder or zip", f.path),
            ));
        }
        self.pending_for(m, token, None)
    }

    /// Same for an offline import: the signed manifest is read from the folder or zip the user chose, so the licence
    /// step happens before anything is copied. `install` then imports from that source.
    pub fn begin_import(
        &self,
        path: PathBuf,
        zip: bool,
        token: String,
    ) -> Result<PendingInstall, String> {
        let bytes = if zip {
            let mut z =
                zip::ZipArchive::new(std::fs::File::open(&path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            let mut b = Vec::new();
            std::io::Read::read_to_end(
                &mut std::io::Read::take(
                    z.by_name(tingmix_models::store::MANIFEST_FILE)
                        .map_err(|e| e.to_string())?,
                    1 << 20,
                ),
                &mut b,
            )
            .map_err(|e| e.to_string())?;
            b
        } else {
            read_manifest_file(&path.join(tingmix_models::store::MANIFEST_FILE))?
        };
        let m = self
            .reg()?
            .verifier()
            .parse(&bytes)
            .map_err(|e| e.to_string())?;
        self.pending_for(m, token, Some((path, zip)))
    }

    /// Licence step for a manifest that already passed signature verification (a catalog entry).
    pub fn begin_verified(
        &self,
        m: VerifiedManifest,
        token: String,
    ) -> Result<PendingInstall, String> {
        self.pending_for(m, token, None)
    }

    fn pending_for(
        &self,
        m: VerifiedManifest,
        token: String,
        source: Option<(PathBuf, bool)>,
    ) -> Result<PendingInstall, String> {
        let reg = self.reg()?;
        let mf = m.manifest();
        mf.validate(self.loopback_http).map_err(|e| e.to_string())?;
        let missing = reg.missing_acceptances(mf);
        let out = PendingInstall {
            token: token.clone(),
            id: mf.id.clone(),
            family: mf.family.clone(),
            backend: mf.backend.clone(),
            precision: mf.precision.clone(),
            download_bytes: mf.download_size(),
            compat_status: mf.compat.status.clone(),
            licenses: mf
                .licenses
                .iter()
                .map(|l| LicenseDto {
                    id: l.id.clone(),
                    name: l.name.clone(),
                    url: l.url.clone(),
                    spdx: l.spdx.clone(),
                    acceptance_required: l.acceptance_required,
                    accepted: l.acceptance_required && !missing.contains(&l.id),
                    notes: l.notes.clone(),
                })
                .collect(),
        };
        self.pending.lock().unwrap().insert(token, (m, source));
        Ok(out)
    }

    /// Record the acceptances the user ticked (all required ones must be among them), then download in the background.
    pub fn install(
        self: &Arc<Self>,
        token: &str,
        accepted: &[String],
        notify: Notify,
    ) -> Result<(), String> {
        let (m, source) = self
            .pending
            .lock()
            .unwrap()
            .get(token)
            .cloned()
            .ok_or_else(|| {
                crate::coded::coded(
                    "models.expired",
                    serde_json::json!({}),
                    "this install request has expired",
                )
            })?;
        let reg = self.reg()?;
        let missing = reg.missing_acceptances(m.manifest());
        if let Some(l) = missing.iter().find(|l| !accepted.contains(l)) {
            return Err(crate::coded::coded(
                "models.licence",
                serde_json::json!({ "licence": l }),
                format!("licence {l:?} must be accepted before installing"),
            ));
        }
        let kind = if source.is_some() {
            "import"
        } else {
            "download"
        };
        let cancel = self.start_op(kind, &m.manifest().id, notify.clone())?;
        for l in &missing {
            if let Err(e) = reg.record_acceptance(&m, l) {
                self.finish_op(Err(e.to_string()), &notify);
                return Err(e.to_string());
            }
        }
        self.pending.lock().unwrap().remove(token);
        let this = self.clone();
        if let Some((path, zip)) = source {
            return this.spawn_import(path, zip, notify);
        }
        std::thread::Builder::new()
            .name("tingmix-model-download".into())
            .spawn(move || {
                let total = m.manifest().download_size();
                let mut last = Instant::now() - Duration::from_secs(1);
                let r = this.reg().and_then(|reg| {
                    Downloader::new()
                        .install(reg, &m, &cancel, |p| {
                            if last.elapsed() >= Duration::from_millis(150) {
                                last = Instant::now();
                                this.progress(p.bytes_done, total, &notify);
                            }
                        })
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                });
                this.finish_op(r, &notify);
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Verified import of a prepared version folder or zip (offline path; also the only path for import-only files).
    /// The operation was already started by `install`.
    fn spawn_import(
        self: Arc<Self>,
        path: PathBuf,
        zip: bool,
        notify: Notify,
    ) -> Result<(), String> {
        let this = self;
        std::thread::Builder::new()
            .name("tingmix-model-import".into())
            .spawn(move || {
                let r = this.reg().and_then(|reg| {
                    let r = if zip {
                        import_zip(reg, &path)
                    } else {
                        import_dir(reg, &path)
                    };
                    r.map(|_| ()).map_err(|e| e.to_string())
                });
                this.finish_op(r, &notify);
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn cancel(&self) {
        if let Some(op) = self
            .op
            .lock()
            .unwrap()
            .as_ref()
            .filter(|o| o.view.state == "running")
        {
            op.cancel.store(true, Ordering::SeqCst);
        }
    }

    pub fn activate(&self, id: &str) -> Result<(), String> {
        self.reg()?.activate(id).map_err(|e| e.to_string())
    }

    pub fn rollback(&self) -> Result<String, String> {
        self.reg()?.rollback().map_err(|e| e.to_string())
    }

    pub fn remove(&self, id: &str) -> Result<(), String> {
        self.reg()?.remove(id).map_err(|e| e.to_string())
    }

    /// Re-hash every file of an installed version (slow for large models).
    pub fn verify(&self, id: &str) -> Result<(), String> {
        self.reg()?.verify_full(id).map_err(|e| e.to_string())
    }

    pub(crate) fn start_op(
        &self,
        kind: &str,
        name: &str,
        notify: Notify,
    ) -> Result<Arc<AtomicBool>, String> {
        let mut slot = self.op.lock().unwrap();
        if slot.as_ref().is_some_and(|o| o.view.state == "running") {
            return Err(crate::coded::coded(
                "models.busy",
                serde_json::json!({}),
                "another model operation is still running",
            ));
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let view = OpView {
            kind: kind.into(),
            name: name.into(),
            state: "running".into(),
            message: None,
            bytes_done: 0,
            bytes_total: 0,
        };
        notify(view.clone());
        *slot = Some(Op {
            view,
            cancel: cancel.clone(),
        });
        Ok(cancel)
    }

    fn progress(&self, done: u64, total: u64, notify: &Notify) {
        if let Some(op) = self.op.lock().unwrap().as_mut() {
            (op.view.bytes_done, op.view.bytes_total) = (done, total);
            notify(op.view.clone());
        }
    }

    pub(crate) fn finish_op(&self, r: Result<(), String>, notify: &Notify) {
        let view = self.op.lock().unwrap().as_mut().map(|op| {
            (op.view.state, op.view.message) = match r {
                Ok(()) => ("done".into(), None),
                Err(e) if e == "cancelled" => (
                    "cancelled".into(),
                    Some(crate::coded::coded(
                        "models.cancelled",
                        serde_json::json!({}),
                        "Cancelled. Downloaded bytes are kept so the next install resumes.",
                    )),
                ),
                Err(e) => ("failed".into(), Some(e)),
            };
            op.view.clone()
        });
        // outside the lock: the callback may start the next operation (import -> probe)
        if let Some(v) = view {
            notify(v);
        }
    }
}

pub fn read_manifest_file(path: &Path) -> Result<Vec<u8>, String> {
    let md = std::fs::metadata(path).map_err(|e| e.to_string())?;
    if md.len() > 1 << 20 {
        return Err("manifest file is larger than 1 MiB".into());
    }
    std::fs::read(path).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGNED: &[u8] =
        include_bytes!("../../crates/tingmix-models/fixtures/tingmix-large-source.signed.json");

    fn models(tag: &str) -> (Arc<Models>, PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("tingmix-models-ui-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (Arc::new(Models::open(dir.join("models"))), dir)
    }

    #[test]
    fn fresh_registry_is_empty_and_says_nothing_can_run() {
        let (m, dir) = models("fresh");
        let o = m.overview();
        assert!(o.registry_available && o.installed.is_empty() && o.selection.is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn signed_manifest_lists_licences_and_install_demands_acceptance() {
        let (m, dir) = models("lic");
        let p = m.begin_install(SIGNED, "t1".into()).unwrap();
        assert_eq!(p.family, "tingmix-large");
        assert!(p.download_bytes > 1 << 20);
        let required: Vec<_> = p
            .licenses
            .iter()
            .filter(|l| l.acceptance_required)
            .map(|l| l.id.clone())
            .collect();
        if !required.is_empty() {
            let err = m.install("t1", &[], Arc::new(|_| {})).unwrap_err();
            assert!(err.contains("must be accepted"), "{err}");
        }
        assert!(m
            .install("unknown", &[], Arc::new(|_| {}))
            .unwrap_err()
            .contains("expired"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tampered_or_unsigned_manifest_is_rejected() {
        let (m, dir) = models("bad");
        let tampered = String::from_utf8(SIGNED.to_vec()).unwrap().replacen(
            "\"serial\": 1",
            "\"serial\": 2",
            1,
        );
        assert_ne!(tampered.as_bytes(), SIGNED);
        let err = m
            .begin_install(tampered.as_bytes(), "t".into())
            .unwrap_err();
        assert!(err.contains("signature"), "{err}");
        assert!(m.begin_install(b"{}", "t".into()).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
