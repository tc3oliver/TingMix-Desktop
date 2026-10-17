use std::path::Path;
use std::process::Command;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::run::{RunOut, Sample, UttRow};
use crate::stats::{median, slope};

fn cmd(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

#[derive(Serialize)]
pub struct Env {
    pub cpu: String,
    pub ram_gb: f64,
    pub os: String,
    pub git_sha: String,
    pub git_dirty: bool,
    pub rustc: String,
    pub profile: &'static str,
}

pub fn env() -> Env {
    let sys = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    );
    Env {
        cpu: cmd("sysctl", &["-n", "machdep.cpu.brand_string"]),
        ram_gb: sys.total_memory() as f64 / 2f64.powi(30),
        os: format!(
            "{} {}",
            sysinfo::System::name().unwrap_or_default(),
            sysinfo::System::long_os_version().unwrap_or_default()
        ),
        git_sha: cmd("git", &["rev-parse", "HEAD"]),
        git_dirty: !cmd(
            "git",
            &["status", "--porcelain", "--", "crates", "Cargo.toml"],
        )
        .is_empty(),
        rustc: cmd("rustc", &["--version"]),
        profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    }
}

pub fn sha256_file(p: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let (mut h, mut f, mut buf) = (Sha256::new(), std::fs::File::open(p)?, vec![0u8; 1 << 20]);
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

#[derive(Serialize)]
pub struct Check {
    pub name: String,
    pub pass: bool,
    pub detail: String,
}

fn check(name: &str, pass: bool, detail: String) -> Check {
    Check {
        name: name.into(),
        pass,
        detail,
    }
}

/// Soak assertions: no backlog growth, no latency drift, bounded RSS, no lost utterances.
/// "Early" and "late" windows are the first and last third of the run (or 10 minutes each for runs over 30 minutes).
pub fn soak_checks(o: &RunOut) -> Vec<Check> {
    let end = o.samples.last().map_or(0.0, |s| s.t_s);
    let win = if end > 1800.0 { 600.0 } else { end / 3.0 };
    let (early, late) = (|t: f64| t < win, |t: f64| t > end - win);
    let lat = |f: &dyn Fn(f64) -> bool| -> Vec<f64> {
        o.utterances
            .iter()
            .filter(|u| f(u.end_s))
            .filter_map(|u| u.final_from_speech_end_s)
            .collect()
    };
    let stale = |f: &dyn Fn(f64) -> bool| -> Vec<f64> {
        o.samples
            .iter()
            .filter(|s| f(s.t_s))
            .filter_map(|s| s.partial_staleness_s)
            .collect()
    };
    let warm = (end * 0.1).clamp(30.0, 600.0);
    let rss: Vec<&Sample> = o.samples.iter().filter(|s| s.t_s > warm).collect();
    let (xs, ys): (Vec<f64>, Vec<f64>) = rss.iter().map(|s| (s.t_s / 3600.0, s.rss_mb)).unzip();
    let first_third: Vec<f64> = ys.iter().take(ys.len() / 3).cloned().collect();
    let max_rss = ys.iter().cloned().fold(0.0, f64::max);
    let max_pending = o
        .samples
        .iter()
        .map(|s| s.pending_finals)
        .max()
        .unwrap_or(0);
    let max_inflight = o
        .samples
        .iter()
        .filter_map(|s| s.inflight_age_s)
        .fold(0.0, f64::max);
    let missed = o
        .utterances
        .iter()
        .filter(|u| u.finals == 0 && u.end_s + 5.0 < o.audio_s)
        .count();
    let (l0, l1) = (median(&lat(&early)), median(&lat(&late)));
    let (s0, s1) = (median(&stale(&early)), median(&stale(&late)));
    let tail_pending = o.samples.last().map_or(0, |s| s.pending_finals);
    vec![
        check(
            "pending finals stay bounded (max <= 3)",
            max_pending <= 3,
            format!("max {max_pending}"),
        ),
        check(
            "no pending finals at the end",
            tail_pending == 0,
            format!("{tail_pending}"),
        ),
        check(
            "final latency does not drift (late median - early median <= 0.25 s)",
            l1 - l0 <= 0.25,
            format!("early {l0:.3} s, late {l1:.3} s over {win:.0} s windows"),
        ),
        check(
            "partial staleness does not drift (<= 0.25 s)",
            s1 - s0 <= 0.25,
            format!("early {s0:.3} s, late {s1:.3} s"),
        ),
        check(
            "RSS bounded (max <= first-third median + 64 MB)",
            max_rss <= median(&first_third) + 64.0,
            format!(
                "max {max_rss:.1} MB, first-third median {:.1} MB",
                median(&first_third)
            ),
        ),
        check(
            "RSS slope <= 50 MB/h after warm-up",
            slope(&xs, &ys) <= 50.0,
            format!("{:.2} MB/h", slope(&xs, &ys)),
        ),
        check(
            "no inference call stuck (max in-flight age <= 10 s)",
            max_inflight <= 10.0,
            format!("{max_inflight:.3} s"),
        ),
        check(
            "every utterance got a final",
            missed == 0,
            format!("{missed} without a final"),
        ),
        check(
            "no backend errors",
            o.errors.is_empty(),
            format!("{}", o.errors.len()),
        ),
    ]
}

#[derive(Serialize)]
pub struct Gate {
    pub metric: &'static str,
    pub target_large: &'static str,
    pub measured_p95: f64,
    pub met: bool,
}

/// BRIEF section 13 initial targets for Large, applied to this run. Only meaningful with a real backend.
pub fn gates(o: &RunOut, silence_s: f64) -> Vec<Gate> {
    let after_endpoint: Vec<f64> = o
        .utterances
        .iter()
        .filter_map(|u: &UttRow| u.final_from_speech_end_s)
        .map(|x| x - silence_s)
        .collect();
    let fa = crate::stats::dist(&after_endpoint).p95;
    vec![
        Gate {
            metric: "first partial P95 from speech onset (s)",
            target_large: "<= 1.5",
            measured_p95: o.first_partial_from_onset_s.p95,
            met: o.first_partial_from_onset_s.p95 <= 1.5,
        },
        Gate {
            metric: "final latency P95 after endpoint = speech end + silence_ms (s)",
            target_large: "<= 1.0",
            measured_p95: fa,
            met: fa <= 1.0,
        },
        Gate {
            metric: "streaming RTF (inference busy time / audio time)",
            target_large: "< 1.0",
            measured_p95: o.streaming_rtf,
            met: o.streaming_rtf < 1.0,
        },
    ]
}
