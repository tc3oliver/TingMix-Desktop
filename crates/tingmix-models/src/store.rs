//! On-disk registry layout (all under `root`):
//!
//! ```text
//! versions/<id>/...        installed, fully verified versions (immutable once renamed in)
//!   manifest.signed.json   the signed manifest the version was verified against
//! tmp/<id>/...             staging for download / import (kept for resume)
//! acceptances/<license>.json
//! active, previous         one-line pointer files, replaced only by atomic rename
//! selection.json           last known-good runtime choice (see select.rs)
//! ```

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::{io, Error, Result};
use crate::manifest::{hex, FileEntry, Manifest, VerifiedManifest, Verifier};

pub const MANIFEST_FILE: &str = "manifest.signed.json";

pub struct Registry {
    root: PathBuf,
    verifier: Verifier,
    allow_loopback_http: bool,
}

/// A version on disk whose manifest signature and file sizes were re-checked when it was opened.
#[derive(Debug)]
pub struct InstalledModel {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

impl InstalledModel {
    pub fn file(&self, component: &str) -> impl Iterator<Item = PathBuf> + '_ {
        let c = component.to_string();
        self.manifest
            .files
            .iter()
            .filter(move |f| f.component == c)
            .map(|f| self.dir.join(&f.path))
    }
}

pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = File::create(&tmp).map_err(io(&tmp))?;
    f.write_all(bytes).map_err(io(&tmp))?;
    f.sync_all().map_err(io(&tmp))?;
    fs::rename(&tmp, path).map_err(io(path))?;
    if let Some(dir) = path.parent() {
        if let Ok(d) = File::open(dir) {
            let _ = d.sync_all(); // best effort: not supported on every platform
        }
    }
    Ok(())
}

pub(crate) fn hash_file(path: &Path) -> Result<(u64, String)> {
    let mut f = File::open(path).map_err(io(path))?;
    let mut h = Sha256::new();
    let (mut buf, mut n) = (vec![0u8; 1 << 20], 0u64);
    loop {
        let k = f.read(&mut buf).map_err(io(path))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((n, hex(&h.finalize())))
}

fn files_under(dir: &Path, base: &Path, out: &mut Vec<String>) -> Result<()> {
    for e in fs::read_dir(dir).map_err(io(dir))? {
        let e = e.map_err(io(dir))?;
        let p = e.path();
        let ft = e.file_type().map_err(io(&p))?;
        if ft.is_dir() {
            files_under(&p, base, out)?;
        } else if ft.is_file() {
            out.push(
                p.strip_prefix(base)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        } else {
            return Err(Error::Unlisted(format!(
                "{} (symlink or special file)",
                p.display()
            )));
        }
    }
    Ok(())
}

impl Registry {
    pub fn open(root: impl Into<PathBuf>, verifier: Verifier) -> Result<Self> {
        let root = root.into();
        for d in ["versions", "tmp", "acceptances", "serials", "probes"] {
            fs::create_dir_all(root.join(d)).map_err(io(root.join(d)))?;
        }
        // An interrupted pointer write leaves only a *.tmp next to the intact old pointer.
        for p in ["active.tmp", "previous.tmp", "selection.tmp"] {
            let _ = fs::remove_file(root.join(p));
        }
        Ok(Self {
            root,
            verifier,
            allow_loopback_http: false,
        })
    }

    /// Tests only: accept `http://127.0.0.1:<port>/` file URLs in manifests.
    pub fn allow_loopback_http(mut self, yes: bool) -> Self {
        self.allow_loopback_http = yes;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn verifier(&self) -> &Verifier {
        &self.verifier
    }
    fn version_dir(&self, id: &str) -> PathBuf {
        self.root.join("versions").join(id)
    }

    // ---- licence acceptance -------------------------------------------------------------------------------------

    /// Record that the user accepted licence `license_id` of this manifest (shown to them with its name and URL).
    pub fn record_acceptance(&self, m: &VerifiedManifest, license_id: &str) -> Result<()> {
        let l = m
            .manifest()
            .licenses
            .iter()
            .find(|l| l.id == license_id)
            .ok_or_else(|| Error::Invalid(format!("no licence {license_id:?} in manifest")))?;
        let body = serde_json::json!({
            "license": l.id,
            "name": l.name,
            "url": l.url,
            "manifest": m.manifest().id,
            "accepted_unix": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        })
        .to_string()
            + "\n";
        write_atomic(
            &self
                .root
                .join("acceptances")
                .join(format!("{}.json", safe_name(&l.id))),
            body.as_bytes(),
        )
    }

    /// Licences not yet accepted at their current URL: changed terms are asked again.
    pub fn missing_acceptances(&self, m: &Manifest) -> Vec<String> {
        m.licenses_requiring_acceptance()
            .filter(|l| {
                let path = self
                    .root
                    .join("acceptances")
                    .join(format!("{}.json", safe_name(&l.id)));
                let accepted_url = fs::read_to_string(path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v.get("url").and_then(|u| u.as_str().map(str::to_owned)));
                accepted_url.as_deref() != Some(l.url.as_str())
            })
            .map(|l| l.id.clone())
            .collect()
    }

    // ---- install ------------------------------------------------------------------------------------------------

    /// Check structure and licence acceptance, return the (possibly partly filled) staging directory.
    pub fn prepare(&self, m: &VerifiedManifest) -> Result<PathBuf> {
        m.manifest().validate(self.allow_loopback_http)?;
        if let Some(l) = self.missing_acceptances(m.manifest()).into_iter().next() {
            return Err(Error::LicenseNotAccepted(l));
        }
        if self.version_dir(&m.manifest().id).exists() {
            return Err(Error::AlreadyInstalled(m.manifest().id.clone()));
        }
        let man = m.manifest();
        let high_water = self.high_water(&man.lineage())?;
        if man.serial < high_water.0
            || (man.serial == high_water.0
                && high_water.1.as_deref().is_some_and(|id| id != man.id))
        {
            return Err(Error::Replay {
                id: man.id.clone(),
                lineage: man.lineage(),
                serial: man.serial,
                high_water: high_water.0,
            });
        }
        let d = self.root.join("tmp").join(&m.manifest().id);
        fs::create_dir_all(&d).map_err(io(&d))?;
        Ok(d)
    }

    /// Verify the staged tree against the manifest (exact file set, sizes, sha256) and rename it into `versions/`.
    pub fn commit(&self, m: &VerifiedManifest) -> Result<InstalledModel> {
        let man = m.manifest();
        let stage = self.root.join("tmp").join(&man.id);
        verify_tree(&stage, man)?;
        write_atomic(
            &stage.join(MANIFEST_FILE),
            &serde_json::to_vec_pretty(m.signed()).unwrap(),
        )?;
        let dest = self.version_dir(&man.id);
        if dest.exists() {
            return Err(Error::AlreadyInstalled(man.id.clone()));
        }
        fs::rename(&stage, &dest).map_err(io(&dest))?;
        self.raise_high_water(man)?;
        Ok(InstalledModel {
            dir: dest,
            manifest: man.clone(),
        })
    }

    // ---- anti-rollback ------------------------------------------------------------------------------------------

    fn serial_file(&self, lineage: &str) -> PathBuf {
        self.root.join("serials").join(safe_name(lineage))
    }

    /// Highest serial ever installed for `lineage` and the id that carried it. It is the maximum of the persisted mark
    /// (survives `remove`) and the versions currently on disk (covers a crash between the rename and the mark write).
    /// Explicit rollback via [`Registry::rollback`] never consults it: it only moves between installed versions.
    pub fn high_water(&self, lineage: &str) -> Result<(u64, Option<String>)> {
        let mut best = (0u64, None);
        if let Ok(s) = fs::read_to_string(self.serial_file(lineage)) {
            let mut it = s.split_whitespace();
            if let (Some(n), Some(id)) = (it.next().and_then(|n| n.parse().ok()), it.next()) {
                best = (n, Some(id.to_string()));
            }
        }
        for id in self.list()? {
            if let Ok(m) = self.open_model(&id) {
                if m.manifest.lineage() == lineage && m.manifest.serial > best.0 {
                    best = (m.manifest.serial, Some(m.manifest.id));
                }
            }
        }
        Ok(best)
    }

    fn raise_high_water(&self, m: &Manifest) -> Result<()> {
        if self.high_water(&m.lineage())?.0 > m.serial {
            return Ok(());
        }
        write_atomic(
            &self.serial_file(&m.lineage()),
            format!("{} {}\n", m.serial, m.id).as_bytes(),
        )
    }

    // ---- query --------------------------------------------------------------------------------------------------

    pub fn list(&self) -> Result<Vec<String>> {
        let mut v: Vec<String> = fs::read_dir(self.root.join("versions"))
            .map_err(io(self.root.join("versions")))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().join(MANIFEST_FILE).is_file())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        Ok(v)
    }

    /// Open an installed version from the local cache only: re-verify the manifest signature, then check that every
    /// listed file exists with the right size. (Full re-hashing is [`Registry::verify_full`].) No network is touched.
    pub fn open_model(&self, id: &str) -> Result<InstalledModel> {
        let dir = self.version_dir(id);
        let bytes =
            fs::read(dir.join(MANIFEST_FILE)).map_err(|_| Error::NotInstalled(id.into()))?;
        let m = self.verifier.parse(&bytes)?;
        if m.manifest().id != id {
            return Err(Error::Invalid(format!(
                "directory {id:?} holds manifest {:?}",
                m.manifest().id
            )));
        }
        for f in &m.manifest().files {
            let md = fs::symlink_metadata(dir.join(&f.path))
                .map_err(|_| Error::Missing(f.path.clone()))?;
            if !md.is_file() {
                return Err(Error::Missing(f.path.clone()));
            }
            if md.len() != f.size {
                return Err(Error::SizeMismatch {
                    path: f.path.clone(),
                    expected: f.size,
                    actual: md.len(),
                });
            }
        }
        Ok(InstalledModel {
            dir,
            manifest: m.manifest().clone(),
        })
    }

    pub fn verify_full(&self, id: &str) -> Result<()> {
        let m = self.open_model(id)?;
        verify_tree(&m.dir, &m.manifest)
    }

    pub fn remove(&self, id: &str) -> Result<()> {
        if self.active()?.as_deref() == Some(id) || self.previous()?.as_deref() == Some(id) {
            return Err(Error::Invalid(format!(
                "{id} is the active or previous version"
            )));
        }
        let d = self.version_dir(id);
        if !d.join(MANIFEST_FILE).is_file() {
            return Err(Error::NotInstalled(id.into()));
        }
        fs::remove_dir_all(&d).map_err(io(&d))
    }

    // ---- pointers -----------------------------------------------------------------------------------------------

    fn pointer(&self, name: &str) -> Result<Option<String>> {
        match fs::read_to_string(self.root.join(name)) {
            Ok(s) => Ok(Some(s.trim().to_string()).filter(|s| !s.is_empty())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io(self.root.join(name))(e)),
        }
    }
    fn set_pointer(&self, name: &str, id: Option<&str>) -> Result<()> {
        match id {
            Some(id) => write_atomic(&self.root.join(name), format!("{id}\n").as_bytes()),
            None => match fs::remove_file(self.root.join(name)) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(io(self.root.join(name))(e)),
            },
        }
    }

    pub fn active(&self) -> Result<Option<String>> {
        self.pointer("active")
    }
    pub fn previous(&self) -> Result<Option<String>> {
        self.pointer("previous")
    }

    /// The active version, opened from the local cache.
    pub fn open_active(&self) -> Result<Option<InstalledModel>> {
        self.active()?.map(|id| self.open_model(&id)).transpose()
    }

    /// Switch `active` to `id`. Order matters for crash safety: the new version is validated first, `previous` is
    /// written next, `active` last. Each write is an atomic rename, so a crash leaves either the old or the new
    /// pointer, never a torn one; a crash between the two writes leaves `previous == active`, which is harmless.
    pub fn activate(&self, id: &str) -> Result<()> {
        self.open_model(id)?;
        let old = self.active()?;
        if old.as_deref() == Some(id) {
            return Ok(());
        }
        if let Some(old) = &old {
            self.set_pointer("previous", Some(old))?;
        }
        self.set_pointer("active", Some(id))
    }

    /// Make `previous` active again (keeping the replaced version as `previous`, so the swap can be undone).
    pub fn rollback(&self) -> Result<String> {
        let prev = self.previous()?.ok_or(Error::NoPrevious)?;
        self.open_model(&prev)?;
        let cur = self.active()?;
        if cur.as_deref() == Some(prev.as_str()) {
            return Err(Error::NoPrevious);
        }
        if let Some(cur) = &cur {
            self.set_pointer("previous", Some(cur))?;
        }
        self.set_pointer("active", Some(&prev))?;
        Ok(prev)
    }
}

fn safe_name(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Exact file set + size + sha256, symlinks and extras rejected. The manifest file itself is the only allowed extra.
pub(crate) fn verify_tree(dir: &Path, m: &Manifest) -> Result<()> {
    let mut found = Vec::new();
    files_under(dir, dir, &mut found)?;
    found.retain(|f| f != MANIFEST_FILE && !f.ends_with(".part"));
    if let Some(extra) = found
        .iter()
        .find(|f| !m.files.iter().any(|e| &e.path == *f))
    {
        return Err(Error::Unlisted(extra.clone()));
    }
    for f in &m.files {
        verify_file(&dir.join(&f.path), f)?;
    }
    Ok(())
}

pub(crate) fn verify_file(path: &Path, f: &FileEntry) -> Result<()> {
    if !fs::symlink_metadata(path)
        .map(|m| m.is_file())
        .unwrap_or(false)
    {
        return Err(Error::Missing(f.path.clone()));
    }
    let (n, h) = hash_file(path)?;
    if n != f.size {
        return Err(Error::SizeMismatch {
            path: f.path.clone(),
            expected: f.size,
            actual: n,
        });
    }
    if h != f.sha256 {
        return Err(Error::HashMismatch {
            path: f.path.clone(),
            expected: f.sha256.clone(),
            actual: h,
        });
    }
    Ok(())
}
