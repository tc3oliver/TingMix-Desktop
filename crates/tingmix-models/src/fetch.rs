//! Network install: resumable (HTTP Range), cancellable download into the staging directory, then verify + commit.
//! This is the only module that builds an HTTP client; loading an installed model never touches it.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::error::{io, Error, Result};
use crate::manifest::VerifiedManifest;
use crate::store::{verify_file, InstalledModel, Registry};

#[derive(Debug, Clone)]
pub struct Progress<'a> {
    pub file: &'a str,
    pub bytes_done: u64,
    pub bytes_total: u64,
}

pub struct Downloader {
    agent: ureq::Agent,
}

impl Default for Downloader {
    fn default() -> Self {
        Self::new()
    }
}

impl Downloader {
    pub fn new() -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(15))
                .timeout_read(Duration::from_secs(60))
                .redirects(5)
                .build(),
        }
    }

    /// Download every file of `m` into `tmp/<id>/`, resuming `.part` files, then verify and commit. Cancelling (or any
    /// error) leaves the staged bytes in place so the next call resumes. Licence acceptance is checked before any
    /// request is made.
    pub fn install(
        &self,
        reg: &Registry,
        m: &VerifiedManifest,
        cancel: &AtomicBool,
        mut progress: impl FnMut(Progress),
    ) -> Result<InstalledModel> {
        let stage = reg.prepare(m)?;
        let total = m.manifest().download_size();
        let mut done = 0u64;
        for f in &m.manifest().files {
            let dest = stage.join(&f.path);
            if dest.is_file() && verify_file(&dest, f).is_ok() {
                done += f.size;
                continue;
            }
            let _ = fs::remove_file(&dest); // present but wrong: refetch
            if let Some(p) = dest.parent() {
                fs::create_dir_all(p).map_err(io(p))?;
            }
            let part = dest.with_file_name(format!(
                "{}.part",
                dest.file_name().unwrap().to_string_lossy()
            ));
            self.fetch_one(&f.url, &part, f.size, cancel, |n| {
                progress(Progress {
                    file: &f.path,
                    bytes_done: done + n,
                    bytes_total: total,
                })
            })?;
            if let Err(e) = verify_file(&part, f) {
                let _ = fs::remove_file(&part); // never resume from corrupt bytes
                return Err(e);
            }
            fs::rename(&part, &dest).map_err(io(&dest))?;
            done += f.size;
        }
        reg.commit(m)
    }

    fn fetch_one(
        &self,
        url: &str,
        part: &std::path::Path,
        size: u64,
        cancel: &AtomicBool,
        mut on: impl FnMut(u64),
    ) -> Result<()> {
        let fail = |reason: String| Error::Download {
            url: url.into(),
            reason,
        };
        let mut have = fs::metadata(part).map(|m| m.len()).unwrap_or(0);
        if have > size {
            fs::remove_file(part).map_err(io(part))?;
            have = 0;
        }
        while have < size {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            let mut req = self.agent.get(url);
            if have > 0 {
                req = req.set("Range", &format!("bytes={have}-"));
            }
            let resp = match req.call() {
                Ok(r) => r,
                Err(ureq::Error::Status(416, _)) => {
                    fs::remove_file(part).map_err(io(part))?; // our partial no longer matches the server: start over
                    have = 0;
                    continue;
                }
                Err(e) => return Err(fail(e.to_string())),
            };
            let append = match resp.status() {
                206 => true,
                200 => false, // server ignored Range: restart from zero
                s => return Err(fail(format!("unexpected status {s}"))),
            };
            let mut out = OpenOptions::new()
                .create(true)
                .write(true)
                .append(append)
                .truncate(!append)
                .open(part)
                .map_err(io(part))?;
            if !append {
                have = 0;
            }
            let mut r = resp.into_reader();
            let mut buf = vec![0u8; 256 * 1024];
            loop {
                if cancel.load(Ordering::Relaxed) {
                    return Err(Error::Cancelled);
                }
                let n = match r.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) => return Err(fail(format!("connection lost after {have} bytes: {e}"))),
                };
                if have + n as u64 > size {
                    return Err(fail("server sent more bytes than the manifest size".into()));
                }
                out.write_all(&buf[..n]).map_err(io(part))?;
                have += n as u64;
                on(have);
            }
            out.sync_all().map_err(io(part))?;
            if have < size && !append {
                return Err(fail(format!("server closed after {have} of {size} bytes")));
            }
        }
        Ok(())
    }
}
