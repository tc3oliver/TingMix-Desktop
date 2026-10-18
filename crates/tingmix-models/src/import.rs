//! Offline import from a folder or a zip with exactly the layout of an installed version
//! (`manifest.signed.json` + the files named in it). Same verification as a download; no network code is involved.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{io, Error, Result};
use crate::manifest::{hex, FileEntry, VerifiedManifest};
use crate::store::{InstalledModel, Registry, MANIFEST_FILE};

/// Copy `src` into `dest`, never reading more than `f.size + 1` bytes, and check size and hash on the way.
fn copy_checked(mut src: impl Read, dest: &Path, f: &FileEntry) -> Result<()> {
    if let Some(p) = dest.parent() {
        fs::create_dir_all(p).map_err(io(p))?;
    }
    let mut out = File::create(dest).map_err(io(dest))?;
    let (mut h, mut n, mut buf) = (Sha256::new(), 0u64, vec![0u8; 1 << 20]);
    let mut lim = (&mut src).take(f.size + 1);
    loop {
        let k = lim.read(&mut buf).map_err(io(dest))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        out.write_all(&buf[..k]).map_err(io(dest))?;
        n += k as u64;
    }
    out.sync_all().map_err(io(dest))?;
    let got = hex(&h.finalize());
    let bad = if n != f.size {
        Some(Error::SizeMismatch {
            path: f.path.clone(),
            expected: f.size,
            actual: n,
        })
    } else if got != f.sha256 {
        Some(Error::HashMismatch {
            path: f.path.clone(),
            expected: f.sha256.clone(),
            actual: got,
        })
    } else {
        None
    };
    if let Some(e) = bad {
        let _ = fs::remove_file(dest);
        return Err(e);
    }
    Ok(())
}

fn reset_stage(stage: &Path) -> Result<()> {
    // Import is all-or-nothing from a complete source: never mix with leftovers of an earlier attempt.
    fs::remove_dir_all(stage).map_err(io(stage))?;
    fs::create_dir_all(stage).map_err(io(stage))
}

pub fn import_dir(reg: &Registry, dir: &Path) -> Result<InstalledModel> {
    let m = reg
        .verifier()
        .parse(&fs::read(dir.join(MANIFEST_FILE)).map_err(io(dir.join(MANIFEST_FILE)))?)?;
    let stage = reg.prepare(&m)?;
    reset_stage(&stage)?;
    check_listed_dir(dir, dir, &m)?;
    for f in &m.manifest().files {
        let p = dir.join(&f.path);
        if !fs::symlink_metadata(&p)
            .map(|md| md.is_file())
            .unwrap_or(false)
        {
            return Err(Error::Missing(f.path.clone()));
        }
        copy_checked(File::open(&p).map_err(io(&p))?, &stage.join(&f.path), f)?;
    }
    reg.commit(&m)
}

fn check_listed_dir(root: &Path, dir: &Path, m: &VerifiedManifest) -> Result<()> {
    for e in fs::read_dir(dir).map_err(io(dir))? {
        let e = e.map_err(io(dir))?;
        let p = e.path();
        let ft = e.file_type().map_err(io(&p))?;
        let rel = p
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if ft.is_dir() {
            check_listed_dir(root, &p, m)?;
        } else if !ft.is_file()
            || (rel != MANIFEST_FILE && !m.manifest().files.iter().any(|f| f.path == rel))
        {
            return Err(Error::Unlisted(rel));
        }
    }
    Ok(())
}

pub fn import_zip(reg: &Registry, zip_path: &Path) -> Result<InstalledModel> {
    let mut z = zip::ZipArchive::new(File::open(zip_path).map_err(io(zip_path))?)
        .map_err(|e| Error::Archive(e.to_string()))?;
    let mut mbytes = Vec::new();
    z.by_name(MANIFEST_FILE)
        .map_err(|_| Error::Missing(MANIFEST_FILE.into()))?
        .take(1 << 20)
        .read_to_end(&mut mbytes)
        .map_err(io(zip_path))?;
    let m = reg.verifier().parse(&mbytes)?;
    let stage = reg.prepare(&m)?;
    reset_stage(&stage)?;
    // Entries are looked up by manifest path, never extracted under their own names (no zip-slip); extras are rejected.
    for i in 0..z.len() {
        let e = z.by_index(i).map_err(|e| Error::Archive(e.to_string()))?;
        // a name zip cannot decode (invalid UTF-8 under zip 9) is never a listed file
        let name = e
            .name()
            .map_err(|e| Error::Archive(e.to_string()))?
            .to_string();
        if !e.is_dir()
            && name != MANIFEST_FILE
            && !m.manifest().files.iter().any(|f| f.path == name)
        {
            return Err(Error::Unlisted(name));
        }
    }
    for f in &m.manifest().files {
        let e = z
            .by_name(&f.path)
            .map_err(|_| Error::Missing(f.path.clone()))?;
        copy_checked(e, &stage.join(&f.path), f)?;
    }
    reg.commit(&m)
}
