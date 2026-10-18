//! `tingmix-bench`: replay fixture speech at real-time pace through `SessionWorker` and measure what the user would see.
//!
//! Scope: inference-side latency (audio in -> inference done -> event received). UI rendering is not measured here; the
//! desktop app measures subtitle render time separately.

mod probe;
mod report;
mod run;
mod source;
mod stats;
mod vadeval;

#[cfg(feature = "mlx")]
mod lazy;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Serialize;
use tingmix_core::AsrBackend;
use tingmix_streaming::{MockBackend, SessionConfig};

use probe::{Rec, Sim};
use run::{RunCfg, RunOut};

const USAGE: &str = "usage: tingmix-bench [--backend mock|onnx|mlx] [--duration-s 120] [--chunk-ms 20] [--seed 1] [--gap-min 1.0] [--gap-max 4.0]
  [--noise-dbfs -70] [--clips id,id] [--tail-s 4] [--mock-native] [--tail-partial on|off|live]\n  [--noise none|white|pink|fan|babble] [--snr-db N | --noise-dbfs N] [--no-speech] [--vad energy|adaptive] [--suppress on|off]\n  [--sup-min C] [--sup-partial-min C] [--sup-short-chars N] [--sup-short-min C] [--speed X] [--fixtures DIR (manifest.jsonl + wavs)] [--vad-only] [--wav FILE (segment a recording, no model)] [--cut-search S] [--oracle FILE|prompt] [--avad key=value,...] [--oracle FILE] [--discard-first N] [--mock-delay-ms 0] [--sim-base-ms 0] [--sim-rtf 0]
  [--artifact PATH] [--model onnx] [--units units.json --dtype f16 --lm charlm.json.gz] [--scenario NAME] [--soak] --out FILE.json";

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
            let flag = matches!(
                k.as_str(),
                "soak" | "mock-native" | "vad-only" | "no-speech"
            );
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

/// `--avad key=value,key=value` overrides of the adaptive VAD defaults (for tuning).
fn avad_params(spec: Option<&str>) -> Result<tingmix_streaming::AdaptiveVadParams, String> {
    let mut p = tingmix_streaming::AdaptiveVadParams::default();
    for kv in spec.unwrap_or("").split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv
            .split_once('=')
            .ok_or(format!("--avad: expected key=value, got {kv}"))?;
        let f: f64 = v
            .parse()
            .map_err(|_| format!("--avad {k}: not a number: {v}"))?;
        match k {
            "floor_percentile" => p.floor_percentile = f as f32,
            "window_frames" => p.window_frames = f as usize,
            "rise_db_per_s" => p.rise_db_per_s = f as f32,
            "smooth_frames" => p.smooth_frames = f as usize,
            "start_margin_db" => p.start_margin_db = f as f32,
            "stay_margin_db" => p.stay_margin_db = f as f32,
            "abs_min_db" => p.abs_min_db = f as f32,
            "start_window" => p.start_window = f as usize,
            "start_votes" => p.start_votes = f as usize,
            "hang_frames" => p.hang_frames = f as u32,
            "warmup_frames" => p.warmup_frames = f as usize,
            "settle_frames" => p.settle_frames = f as usize,
            "settle_spread_db" => p.settle_spread_db = f as f32,
            "settle_loud_db" => p.settle_loud_db = f as f32,
            "settle_loud_frames" => p.settle_loud_frames = f as usize,
            _ => return Err(format!("--avad: unknown parameter {k}")),
        }
    }
    Ok(p)
}

fn real_main() -> Result<(), String> {
    let a = Args::parse()?;
    let out_path = PathBuf::from(a.s("out").ok_or(format!("--out is required\n{USAGE}"))?);
    let only: Option<Vec<String>> = a
        .s("clips")
        .map(|c| c.split(',').map(String::from).collect());
    let fixtures = a.s("fixtures").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/audio")
    });
    let clips = source::load_clips(&fixtures, only.as_deref())?;
    let (duration_s, chunk_ms) = (a.f("duration-s", 120.0)?, a.f("chunk-ms", 20.0)? as u64);
    let prompts: std::collections::HashMap<String, String> = clips
        .iter()
        .map(|c| (c.id.clone(), c.text.clone()))
        .collect();
    let kind = a.s("noise").unwrap_or("white");
    let noise = if kind == "none" {
        None
    } else {
        let level = match (a.0.contains_key("snr-db"), a.0.contains_key("noise-dbfs")) {
            (true, false) => source::Level::SnrDb(a.f("snr-db", 20.0)?),
            (false, _) => source::Level::Dbfs(a.f("noise-dbfs", -70.0)?),
            (true, true) => return Err("use either --snr-db or --noise-dbfs".into()),
        };
        Some(source::NoiseSpec {
            kind: source::NoiseKind::parse(kind).ok_or(format!(
                "--noise {kind}: expected none|white|pink|fan|babble|music"
            ))?,
            level,
        })
    };
    let noise_desc = format!("{noise:?}");
    let mut src = source::Source::new(
        clips,
        a.f("seed", 1.0)? as u64,
        (a.f("gap-min", 1.0)?, a.f("gap-max", 4.0)?),
        noise,
        duration_s,
    );
    if a.0.contains_key("no-speech") {
        src = src.without_speech();
    }
    let session = SessionConfig {
        vad: match a.s("vad") {
            None | Some("energy") => tingmix_streaming::VadKind::Energy,
            Some("adaptive") => tingmix_streaming::VadKind::Adaptive,
            Some(v) => return Err(format!("--vad {v}: expected energy|adaptive")),
        },
        suppression: match a.s("suppress") {
            None | Some("off") => None,
            Some("on") => Some(tingmix_streaming::Suppression {
                min_confidence: a.f("sup-min", 0.5)? as f32,
                partial_min_confidence: a.f("sup-partial-min", 0.3)? as f32,
                short_chars: a.f("sup-short-chars", 2.0)? as usize,
                short_min_confidence: a.f("sup-short-min", 0.9)? as f32,
            }),
            Some(v) => return Err(format!("--suppress {v}: expected on|off")),
        },
        adaptive_vad: avad_params(a.s("avad"))?,
        max_cut_search_s: a.f("cut-search", 0.0)?,
        ..SessionConfig::default()
    };
    if let Some(path) = a.s("wav") {
        let mut r = hound::WavReader::open(path).map_err(|e| format!("{path}: {e}"))?;
        let spec = r.spec();
        if spec.sample_rate != 16_000 || spec.channels != 1 {
            return Err(format!("{path}: expected 16 kHz mono, got {spec:?}"));
        }
        let wav: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => r
                .samples::<i16>()
                .map(|x| x.map(|v| f32::from(v) / 32768.0))
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?,
            hound::SampleFormat::Float => r
                .samples::<f32>()
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?,
        };
        let out = vadeval::evaluate_wav(&wav, &session);
        std::fs::write(&out_path, serde_json::to_vec_pretty(&serde_json::json!({ "command": std::env::args().collect::<Vec<_>>().join(" "), "vad": a.s("vad").unwrap_or("energy"), "cut_search_s": a.f("cut-search", 0.0)?, "result": out })).unwrap()).map_err(|e| e.to_string())?;
        let o = &out;
        eprintln!(
            "{} s: {} segments, {} ended at the cap, median {:.1} s, longest {:.1} s",
            o.audio_s as u64, o.segments, o.ended_at_cap, o.median_segment_s, o.longest_segment_s
        );
        return Ok(());
    }
    if a.0.contains_key("vad-only") {
        let out = vadeval::evaluate(src, &session, chunk_ms as usize);
        let body = serde_json::json!({ "command": std::env::args().collect::<Vec<_>>().join(" "), "env": report::env(),
            "config": { "noise": noise_desc, "vad": a.s("vad").unwrap_or("energy"), "avad": a.s("avad"), "cut_search_s": a.f("cut-search", 0.0)?, "oracle": a.s("oracle"), "seed": a.f("seed", 1.0)?, "duration_s": duration_s, "gap_s": [a.f("gap-min", 1.0)?, a.f("gap-max", 4.0)?], "no_speech": a.0.contains_key("no-speech") },
            "result": out });
        std::fs::write(&out_path, serde_json::to_vec_pretty(&body).unwrap())
            .map_err(|e| format!("{}: {e}", out_path.display()))?;
        eprintln!("{}", serde_json::to_string(&body["result"]).unwrap());
        return Ok(());
    }
    let oracle = if a.s("oracle") == Some("prompt") {
        Some(prompts)
    } else {
        None
    };
    let oracle = oracle.or((a.s("backend").unwrap_or("mock") != "mock")
        .then(|| {
            let p = a.s("oracle").map(PathBuf::from).unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/fixtures/large/expected_texts.json")
            });
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).ok()?).ok()?;
            Some(
                v["clips"]
                    .as_object()?
                    .iter()
                    .filter_map(|(k, c)| Some((k.clone(), c["beam_lm"].as_str()?.to_string())))
                    .collect::<std::collections::HashMap<_, _>>(),
            )
        })
        .flatten());
    #[allow(unused_mut)] // only the mlx arm adjusts it
    let mut cfg = RunCfg {
        discard_first: a.f("discard-first", 0.0)? as usize,
        chunk_ms,
        tail_partial: match a.s("tail-partial") {
            None | Some("live") => None,
            Some("on") => Some(true),
            Some("off") => Some(false),
            Some(v) => return Err(format!("--tail-partial {v}: expected on|off|live")),
        },
        tail_s: a.f("tail-s", 4.0)?,
        speed: a.f("speed", 1.0)?,
        oracle,
        sim: Sim {
            base_s: a.f("sim-base-ms", 0.0)? / 1000.0,
            rtf: a.f("sim-rtf", 0.0)?,
        },
        session,
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
        #[cfg(feature = "onnx")]
        "onnx" => {
            // Token ids are printed as numbers: the harness measures compute and scheduling, not text.
            let model = PathBuf::from(
                a.s("model")
                    .ok_or("--model PATH (step-graph .onnx) is required for --backend onnx")?,
            );
            artifacts.insert(
                model.display().to_string(),
                report::sha256_file(&model).map_err(|e| format!("{}: {e}", model.display()))?,
            );
            let dec: tingmix_runtime_onnx::TokenDecoder = std::sync::Arc::new(|ids: &[u32]| {
                ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
            });
            let b =
                tingmix_runtime_onnx::OnnxSmall::load(&model, a.f("threads", 4.0)? as usize, dec)
                    .map_err(|e| e.to_string())?;
            let info = serde_json::json!({ "name": b.info().name, "streaming_native": b.info().streaming_native, "max_window_s": b.info().max_window_s });
            (info, run::run(b, rec, t0, src, &cfg))
        }
        #[cfg(feature = "mlx")]
        "mlx" => {
            use lazy::Lazy;
            use tingmix_core::BackendInfo;
            use tingmix_runtime_mlx::{Decode, Dtype, MlxLargeBackend};
            let (art, units) = (
                PathBuf::from(
                    a.s("artifact")
                        .ok_or("--artifact (fp32 safetensors) is required")?,
                ),
                PathBuf::from(a.s("units").ok_or("--units units.json is required")?),
            );
            let dtype = match a.s("dtype").unwrap_or("f16") {
                "f32" => Dtype::F32,
                "f16" => Dtype::F16,
                "bf16" => Dtype::Bf16,
                d => return Err(format!("--dtype {d}: expected f32|f16|bf16")),
            };
            let lm_path = a.s("lm").map(PathBuf::from);
            let info = BackendInfo {
                name: "tingmix-large-mlx".into(),
                sample_rate: 16_000,
                max_window_s: 29.5,
                streaming_native: false,
            };
            // MLX streams are per thread: build (and warm) the model on the inference thread, on its first call.
            let make = move || -> Result<MlxLargeBackend, String> {
                let decode = match &lm_path {
                    Some(p) => Decode::BeamLm(
                        Box::new(tingmix_decoder::CharLm::load(p).map_err(|e| e.to_string())?),
                        tingmix_decoder::BeamConfig::default(),
                    ),
                    None => Decode::Greedy,
                };
                let mut b =
                    MlxLargeBackend::new(&art, &units, dtype, decode).map_err(|e| e.to_string())?;
                b.transcribe_batch(&[&vec![0.0f32; 16_000]])
                    .map_err(|e| e.to_string())?;
                Ok(b)
            };
            let desc = serde_json::json!({ "name": info.name, "streaming_native": false, "max_window_s": 29.5, "dtype": a.s("dtype").unwrap_or("f16"),
                "decode": if a.s("lm").is_some() { "beam+lm (alpha 0.35 beta 1.5 beam 16 defaults)" } else { "greedy" },
                "note": "model loaded on the first inference call, so the first utterance includes load time and is excluded from the distributions (discard_first >= 1)" });
            cfg.discard_first = cfg.discard_first.max(1);
            (
                desc,
                run::run(
                    Lazy {
                        info,
                        make: Some(make),
                        inner: None,
                    },
                    rec,
                    t0,
                    src,
                    &cfg,
                ),
            )
        }
        other => {
            return Err(format!(
                "unknown backend {other:?} (built in: mock{}{})",
                if cfg!(feature = "onnx") { ", onnx" } else { "" },
                if cfg!(feature = "mlx") { ", mlx" } else { "" }
            ))
        }
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
            "gap_s": [a.f("gap-min", 1.0)?, a.f("gap-max", 4.0)?], "noise": noise_desc, "speed": cfg.speed, "latency_valid": cfg.speed == 1.0, "vad": a.s("vad").unwrap_or("energy"), "suppress": a.s("suppress").unwrap_or("off"), "no_speech": a.0.contains_key("no-speech"), "seed": a.f("seed", 1.0)?,
            "tail_partial": a.s("tail-partial").unwrap_or("live"), "session": format!("{:?}", cfg.session) }),
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
        let src = source::Source::new(
            clips,
            3,
            (1.0, 1.5),
            Some(source::NoiseSpec {
                kind: source::NoiseKind::White,
                level: source::Level::Dbfs(-70.0),
            }),
            9.0,
        );
        let t0 = Instant::now();
        let rec = Rec::new(t0);
        let cfg = RunCfg {
            speed: 1.0,
            oracle: None,
            discard_first: 0,
            tail_partial: None,
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
