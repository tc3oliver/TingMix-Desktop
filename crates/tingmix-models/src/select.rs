//! Capability-based runtime selection (BRIEF section 12). Only backends registered as VALIDATED are eligible, and a
//! backend needs a passing compatibility probe. GPU vendor never enters the decision.

use std::path::Path;

use crate::caps::Capabilities;
use crate::error::{io, Error, Result};
use crate::manifest::{version_at_least, Manifest};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validation {
    Validated,
    Experimental,
    Blocked,
}

/// Result of a compatibility probe run (or consulted from cache) for a backend on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Probe {
    pub passed: bool,
    /// Measured real-time factor x1000 on a representative clip; lower is faster.
    pub rtf_x1000: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendReg {
    /// Matches `Manifest::backend`.
    pub backend: String,
    pub validation: Validation,
    pub is_cpu: bool,
    /// `None`: the runtime / provider is not installed or not loadable.
    pub runtime_version: Option<String>,
    pub probe: Option<Probe>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub manifest_id: String,
    pub backend: String,
    pub runtime_version: String,
    /// Human-readable explanation, including why the preferred choice was not used when it was not.
    pub reason: String,
    pub rejected: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownGood {
    pub manifest_id: String,
    pub backend: String,
    pub runtime_version: String,
    pub fingerprint: String,
}

/// `Ok(rtf)` (lower is better, `u32::MAX` = unknown) or the human-readable reason the manifest is ineligible.
fn check(
    m: &Manifest,
    caps: &Capabilities,
    regs: &[BackendReg],
) -> std::result::Result<(u32, BackendReg), String> {
    if (m.os != "any" && m.os != caps.os) || (m.arch != "any" && m.arch != caps.arch) {
        return Err(format!(
            "built for {}/{}, this machine is {}/{}",
            m.os, m.arch, caps.os, caps.arch
        ));
    }
    let reg = regs
        .iter()
        .find(|r| r.backend == m.backend)
        .ok_or_else(|| format!("backend {} is not registered in this build", m.backend))?;
    match reg.validation {
        Validation::Validated => {}
        Validation::Experimental => {
            return Err(format!(
                "backend {} is EXPERIMENTAL; only VALIDATED backends are selected",
                m.backend
            ))
        }
        Validation::Blocked => return Err(format!("backend {} is BLOCKED", m.backend)),
    }
    let have = reg
        .runtime_version
        .as_deref()
        .ok_or_else(|| format!("runtime {} is not available", m.runtime.name))?;
    if !version_at_least(have, &m.runtime.min_version) {
        return Err(format!(
            "runtime {} {} is older than the required {}",
            m.runtime.name, have, m.runtime.min_version
        ));
    }
    if let Some(f) = m
        .requirements
        .cpu_features
        .iter()
        .find(|f| !caps.cpu_features.contains(f))
    {
        return Err(format!("CPU lacks required feature {f}"));
    }
    if let Some(min) = m.requirements.apple_chip_min_generation {
        match &caps.apple_chip {
            Some(c) if c.generation >= min => {}
            Some(c) => return Err(format!("needs Apple M{min} or newer, found {}", c.name)),
            None => {
                return Err(format!(
                    "needs Apple M{min} or newer, no Apple Silicon found"
                ))
            }
        }
    }
    if caps.ram_bytes < m.requirements.min_ram_bytes {
        return Err(format!(
            "needs {} GiB RAM, machine has {} GiB",
            m.requirements.min_ram_bytes >> 30,
            caps.ram_bytes >> 30
        ));
    }
    let probe = reg
        .probe
        .ok_or_else(|| "no verified compatibility probe result for this backend".to_string())?;
    if !probe.passed {
        return Err("compatibility probe failed on this machine".into());
    }
    Ok((
        probe.rtf_x1000.or(m.compat.rtf_x1000).unwrap_or(u32::MAX),
        reg.clone(),
    ))
}

/// Pick the fastest eligible installed manifest. If `known_good` is still eligible on an unchanged machine it is kept
/// (stable behaviour, no re-benchmark); otherwise the reason says what changed. Err carries every rejection reason.
pub fn select(
    installed: &[Manifest],
    caps: &Capabilities,
    regs: &[BackendReg],
    known_good: Option<&KnownGood>,
) -> std::result::Result<Selection, Vec<(String, String)>> {
    let mut rejected = Vec::new();
    let mut ok: Vec<(u32, bool, &Manifest, BackendReg)> = Vec::new();
    for m in installed {
        match check(m, caps, regs) {
            Ok((rtf, reg)) => ok.push((rtf, reg.is_cpu, m, reg)),
            Err(why) => rejected.push((m.id.clone(), why)),
        }
    }
    let mk = |m: &Manifest, reg: &BackendReg, reason: String, rejected: Vec<(String, String)>| {
        Selection {
            manifest_id: m.id.clone(),
            backend: m.backend.clone(),
            runtime_version: reg.runtime_version.clone().unwrap_or_default(),
            reason,
            rejected,
        }
    };
    let mut prefix = String::new();
    if let Some(kg) = known_good {
        let same_machine = kg.fingerprint == caps.fingerprint();
        if let Some((_, _, m, reg)) = ok.iter().find(|(_, _, m, _)| m.id == kg.manifest_id) {
            if same_machine && reg.runtime_version.as_deref() == Some(kg.runtime_version.as_str()) {
                return Ok(mk(
                    m,
                    reg,
                    format!(
                        "last known-good choice {} ({}) is still valid",
                        m.id, m.backend
                    ),
                    rejected,
                ));
            }
        }
        prefix = match rejected.iter().find(|(id, _)| *id == kg.manifest_id) {
            Some((_, why)) => format!(
                "last known-good {} is no longer usable ({why}); ",
                kg.manifest_id
            ),
            None if !same_machine => format!(
                "hardware or OS changed since last known-good ({} -> {}); ",
                kg.fingerprint,
                caps.fingerprint()
            ),
            None => "runtime version changed since last known-good; ".to_string(),
        };
    }
    ok.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.id.cmp(&b.2.id)));
    let Some((rtf, _, m, reg)) = ok.first() else {
        return Err(rejected);
    };
    let cpu_best = ok.iter().filter(|x| x.1).map(|x| x.0).min();
    let vs_cpu = match (reg.is_cpu, cpu_best) {
        (true, _) => "CPU path".to_string(),
        (false, Some(c)) if *rtf < c => format!(
            "faster than the best CPU path (RTF {:.3} vs {:.3})",
            *rtf as f64 / 1000.0,
            c as f64 / 1000.0
        ),
        (false, None) => "no CPU path installed to compare against".to_string(),
        (false, Some(_)) => "no faster than CPU".to_string(),
    };
    let rtf_txt = if *rtf == u32::MAX {
        "unmeasured".to_string()
    } else {
        format!("RTF {:.3}", *rtf as f64 / 1000.0)
    };
    Ok(mk(
        m,
        reg,
        format!(
            "{prefix}selected {} on backend {} ({rtf_txt}; {vs_cpu})",
            m.id, m.backend
        ),
        rejected,
    ))
}

// ---- persistence of the last known-good choice (plain `key=value` lines: no parser surface, no secrets) -----------

pub fn save_known_good(root: &Path, sel: &Selection, caps: &Capabilities) -> Result<()> {
    let body = format!(
        "manifest_id={}\nbackend={}\nruntime_version={}\nfingerprint={}\n",
        sel.manifest_id,
        sel.backend,
        sel.runtime_version,
        caps.fingerprint()
    );
    crate::store::write_atomic(&root.join("selection.json"), body.as_bytes())
}

pub fn load_known_good(root: &Path) -> Result<Option<KnownGood>> {
    let s = match std::fs::read_to_string(root.join("selection.json")) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io(root.join("selection.json"))(e)),
    };
    let get = |k: &str| {
        s.lines()
            .find_map(|l| l.strip_prefix(&format!("{k}=")))
            .map(str::to_string)
    };
    match (
        get("manifest_id"),
        get("backend"),
        get("runtime_version"),
        get("fingerprint"),
    ) {
        (Some(manifest_id), Some(backend), Some(runtime_version), Some(fingerprint)) => {
            Ok(Some(KnownGood {
                manifest_id,
                backend,
                runtime_version,
                fingerprint,
            }))
        }
        _ => Err(Error::Malformed("selection.json is incomplete".into())),
    }
}
