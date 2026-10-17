//! `tingmix-bench`: replay fixture speech at real-time pace through `SessionWorker` and measure what the user would see.
//!
//! Scope: inference-side latency (audio in -> inference done -> event received). UI rendering is not measured here; the
//! desktop app measures subtitle render time separately.

mod probe;
mod report;
mod run;
mod source;
mod stats;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Serialize;
use tingmix_core::AsrBackend;
use tingmix_streaming::{MockBackend, SessionConfig};

use probe::{Rec, Sim};
use run::{RunCfg, RunOut};

const USAGE: &str = "usage: tingmix-bench [--backend mock] [--duration-s 120] [--chunk-ms 20] [--seed 1] [--gap-min 1.0] [--gap-max 4.0]
  [--noise-dbfs -70] [--clips id,id] [--tail-s 4] [--mock-native] [--mock-delay-ms 0] [--sim-base-ms 0] [--sim-rtf 0]
  [--artifact PATH] [--scenario NAME] [--soak] --out FILE.json";

struct Args(std::collections::HashMap<String, String>);

impl Args {
    fn parse() -> Result<Self, String> {
        let mut m = std::collections::HashMap::new();
        let mut it = std::env::args().skip(1).peekable();
        while let Some(k) = it.next() {
            let k = k
                .strip_prefix("--")
                .ok_or(format!("unexpected argument {k}\n{USAGE}"))?
                .to_string();
            let flag = matches!(k.as_str(), "soak" | "mock-native");
            let v = if flag {
                "1".into()
            } else {
                it.next().ok_or(format!("--{k} needs a value\n{USAGE}"))?
            };
            m.insert(k, v);
        }
        Ok(Self(m))
    }
    fn f(&self, k: &str, d: f64) -> Result<f64, String> {
        self.0.get(k).map_or(Ok(d), |v| {
            v.parse().map_err(|_| format!("--{k}: not a number: {v}"))
        })
    }
    fn s(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(String::as_str)
    }
}

#[derive(Serialize)]
struct Report<'a> {
    command: String,
    scenario: String,
    backend: String,
    backend_info: serde_json::Value,
    artifacts_sha256: std::collections::BTreeMap<String, String>,
    env: report::Env,
    config: serde_json::Value,
    note: &'static str,
    gates_large_initial_targets: Vec<report::Gate>,
    soak_assertions: Option<Vec<report::Check>>,
    #[serde(flatten)]
    out: &'a RunOut,
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}

fn real_main() -> Result<(), String> {
    let a = Args::parse()?;
    let out_path = PathBuf::from(a.s("out").ok_or(format!("--out is required\n{USAGE}"))?);
    let only: Option<Vec<String>> = a
        .s("clips")
        .map(|c| c.split(',').map(String::from).collect());
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/audio");
    let clips = source::load_clips(&fixtures, only.as_deref())?;
    let (duration_s, chunk_ms) = (a.f("duration-s", 120.0)?, a.f("chunk-ms", 20.0)? as u64);
    let noise =
        a.0.contains_key("noise-dbfs")
            .then(|| a.f("noise-dbfs", -70.0))
            .transpose()?
            .or(Some(-70.0));
    let src = source::Source::new(
        clips,
        a.f("seed", 1.0)? as u64,
        (a.f("gap-min", 1.0)?, a.f("gap-max", 4.0)?),
        noise,
        duration_s,
    );
    let cfg = RunCfg {
        chunk_ms,
        tail_s: a.f("tail-s", 4.0)?,
        sim: Sim {
            base_s: a.f("sim-base-ms", 0.0)? / 1000.0,
            rtf: a.f("sim-rtf", 0.0)?,
        },
        session: SessionConfig::default(),
    };
    let backend_name = a.s("backend").unwrap_or("mock");
    let (t0, soak) = (Instant::now(), a.0.contains_key("soak"));
    let rec = Rec::new(t0);
    let mut artifacts = std::collections::BTreeMap::new();
    if let Some(p) = a.s("artifact") {
        artifacts.insert(
            p.to_string(),
            report::sha256_file(&PathBuf::from(p)).map_err(|e| format!("{p}: {e}"))?,
        );
    }
    eprintln!("running {backend_name}: {duration_s:.0} s of audio in real time (chunks of {chunk_ms} ms)...");
    let (info, out) = match backend_name {
        "mock" => {
            let b = MockBackend::new(a.0.contains_key("mock-native"))
                .with_delay(Duration::from_millis(a.f("mock-delay-ms", 0.0)? as u64));
            let info = serde_json::json!({ "name": b.info().name, "streaming_native": b.info().streaming_native, "max_window_s": b.info().max_window_s });
            (info, run::run(b, rec, t0, src, &cfg))
        }
        other => return Err(format!("unknown backend {other:?} (built: mock)")),
    };
    let silence_s = cfg.session.silence_ms / 1000.0;
    let checks = soak.then(|| report::soak_checks(&out));
    let r = Report {
        command: std::env::args().collect::<Vec<_>>().join(" "),
        scenario: a.s("scenario").unwrap_or(if soak { "soak" } else { "replay" }).into(),
        backend: backend_name.into(),
        backend_info: info,
        artifacts_sha256: artifacts,
        env: report::env(),
        config: serde_json::json!({ "duration_s": duration_s, "chunk_ms": chunk_ms, "sim_base_ms": cfg.sim.base_s * 1000.0, "sim_rtf": cfg.sim.rtf,
            "gap_s": [a.f("gap-min", 1.0)?, a.f("gap-max", 4.0)?], "noise_dbfs": noise, "seed": a.f("seed", 1.0)?,
            "session": format!("{:?}", cfg.session) }),
        note: "Wall-clock real-time replay. Times are inference-side (audio in -> event received); UI render time is not included. \
               Mock backends recognise nothing and emit text even for silence, so mock first-partial latency is optimistic: with --backend mock these numbers characterise the harness, scheduler and queue only. \
               Other processes ran on the machine; see env. Gates apply to real backends only.",
        gates_large_initial_targets: report::gates(&out, silence_s),
        soak_assertions: checks,
        out: &out,
    };
    std::fs::write(&out_path, serde_json::to_vec_pretty(&r).unwrap())
        .map_err(|e| format!("{}: {e}", out_path.display()))?;
    eprintln!("wrote {}", out_path.display());
    eprintln!(
        "utterances {}  first partial P50/P95 {:.3}/{:.3} s  final(speech end) P50/P95 {:.3}/{:.3} s  rtf {:.4}  coalesced {}  calls {}",
        out.utterances.len(), out.first_partial_from_onset_s.p50, out.first_partial_from_onset_s.p95, out.final_from_speech_end_s.p50, out.final_from_speech_end_s.p95, out.streaming_rtf, out.coalesced, out.calls
    );
    if let Some(c) = &r.soak_assertions {
        for x in c {
            eprintln!(
                "  [{}] {} ({})",
                if x.pass { "pass" } else { "FAIL" },
                x.name,
                x.detail
            );
        }
        if c.iter().any(|x| !x.pass) {
            std::process::exit(1);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_mock_run_maps_events_to_utterances() {
        let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/audio");
        let clips =
            source::load_clips(&fixtures, Some(&["zh_02s".into(), "en_02s".into()])).unwrap();
        let src = source::Source::new(clips, 3, (1.0, 1.5), Some(-70.0), 9.0);
        let t0 = Instant::now();
        let rec = Rec::new(t0);
        let cfg = RunCfg {
            chunk_ms: 20,
            tail_s: 2.5,
            sim: Sim {
                base_s: 0.01,
                rtf: 0.0,
            },
            session: SessionConfig::default(),
        };
        let o = run::run(MockBackend::new(false), rec, t0, src, &cfg);
        assert!(o.utterances.len() >= 2, "{} utterances", o.utterances.len());
        for u in &o.utterances {
            assert!(u.finals >= 1, "{u:?}");
            let (fp, fin) = (
                u.first_partial_from_onset_s.unwrap(),
                u.final_from_speech_end_s.unwrap(),
            );
            // The mock "recognises" words from the 0.8 s preroll silence, so its first partial arrives right after VAD onset;
            // a real model emits nothing until speech is decodable. A final cannot precede the endpoint silence.
            assert!((0.0..2.0).contains(&fp), "first partial {fp}");
            assert!((0.5..2.5).contains(&fin), "final {fin}");
        }
        assert_eq!(o.unmatched_events, 0);
        assert!(o.samples.iter().all(|s| s.rss_mb > 0.0));
    }
}
