//! Does the live tail-partial policy (SessionConfig::tail_partial) change final texts or latency with the real MLX Large backend?
//! Replays every fixture clip in real time through SessionWorker with tail_partial off and on, beam+LM decoding, and compares
//! the joined final texts with tests/fixtures/large/expected_texts.json (CPU fp32 oracle).
//!
//!   cargo run --release -p tingmix-runtime-mlx --example tail_partial_accuracy -- \
//!       <artifact.safetensors> <TingMix-Large dir> <out.json>
//! (needs MLX_PREFIX at build time). Fixtures are synthetic TTS, so this says nothing about real-speech accuracy.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::json;
use tingmix_core::{AsrBackend, Event};
use tingmix_decoder::{BeamConfig, CharLm};
use tingmix_runtime_mlx::{Decode, Dtype, MlxLargeBackend};
use tingmix_streaming::{SessionConfig, SessionMsg, SessionWorker};

#[derive(Deserialize)]
struct Clip {
    greedy: String,
    beam_lm: String,
}
#[derive(Deserialize)]
struct Expected {
    clips: BTreeMap<String, Clip>,
}

fn wav(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    let mut p = 12;
    while &b[p..p + 4] != b"data" {
        p += 8 + u32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()) as usize;
    }
    let n = u32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()) as usize;
    b[p + 8..p + 8 + n].chunks_exact(2).map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0).collect()
}

/// Same joining rule as `Transcript::text`: CJK-only output joins without spaces.
fn join(finals: &[String]) -> String {
    let parts: Vec<&str> = finals.iter().map(String::as_str).filter(|t| !t.is_empty()).collect();
    parts.join(if parts.concat().contains(' ') { " " } else { "" })
}

struct Run {
    text: String,
    finals: usize,
    partials: usize,
    /// per final: seconds from the audio-clock endpoint (speech end + silence_ms) to the final's arrival
    final_latency_s: Vec<f64>,
    coalesced: u64,
}

fn replay(backend: MlxLargeBackend, audio: &[f32], tail_partial: bool) -> Run {
    let cfg = SessionConfig { tail_partial, ..SessionConfig::default() };
    let silence_s = cfg.silence_ms / 1000.0;
    let mut w = SessionWorker::spawn_with(backend, &cfg);
    let t0 = Instant::now();
    let (mut finals, mut latency, mut partials) = (Vec::new(), Vec::new(), 0);
    let mut take = |w: &SessionWorker| {
        for m in w.events().try_iter() {
            let SessionMsg::Event(e) = m else { panic!("backend error: {m:?}") };
            match e {
                Event::Partial(_) => partials += 1,
                Event::Final(f) => {
                    latency.push(t0.elapsed().as_secs_f64() - (f.end + silence_s));
                    finals.push(f.text);
                }
            }
        }
    };
    // 20 ms chunks paced against the wall clock, then 1.5 s of silence to let the last endpoint fire
    let tail = vec![0.0f32; 24_000];
    for (i, c) in audio.chunks(320).chain(tail.chunks(320)).enumerate() {
        let due = Duration::from_secs_f64(i as f64 * 0.02);
        if let Some(wait) = due.checked_sub(t0.elapsed()) {
            std::thread::sleep(wait);
        }
        w.feed(c);
        take(&w);
    }
    // the stream ends: flush, wait for every queued job, then collect (arrival time = now)
    let coalesced = w.coalesced_partials();
    for m in w.finish() {
        let SessionMsg::Event(e) = m else { panic!("backend error: {m:?}") };
        match e {
            Event::Partial(_) => partials += 1,
            Event::Final(f) => {
                latency.push(t0.elapsed().as_secs_f64() - (f.end + silence_s));
                finals.push(f.text);
            }
        }
    }
    Run { text: join(&finals), finals: finals.len(), partials, final_latency_s: latency, coalesced }
}

fn sh(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd).args(args).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned()).unwrap_or_default()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (art, large, out) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]), &a[3]);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let exp: Expected = serde_json::from_slice(&std::fs::read(root.join("tests/fixtures/large/expected_texts.json")).unwrap()).unwrap();
    let lm_path = large.join("lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz");
    let load = || {
        let lm = CharLm::load(&lm_path).unwrap();
        let mut b = MlxLargeBackend::new(&art, &large.join("units.json"), Dtype::F32, Decode::BeamLm(Box::new(lm), BeamConfig::default())).unwrap();
        // MLX streams are per thread and the weights are lazy: evaluate them once on this thread before the backend moves to
        // the inference thread, otherwise its first decode fails with "There is no Stream(gpu, 1) in current thread".
        b.transcribe_batch(&[&vec![0.0f32; 16_000]]).unwrap();
        b
    };
    let start_load = sh("uptime", &[]);
    let mut clips = Vec::new();
    let (mut differs_only_with_tail, mut tail_differs_from_off) = (Vec::new(), Vec::new());
    for (id, want) in &exp.clips {
        let audio = wav(&root.join(format!("tests/fixtures/audio/{id}.wav")));
        eprintln!("{id}: {:.1} s", audio.len() as f64 / 16e3);
        let off = replay(load(), &audio, false);
        let on = replay(load(), &audio, true);
        let (off_ok, on_ok) = (off.text == want.beam_lm, on.text == want.beam_lm);
        if off_ok && !on_ok {
            differs_only_with_tail.push(id.clone());
        }
        if off.text != on.text {
            tail_differs_from_off.push(id.clone());
        }
        let run = |r: &Run| json!({"text": r.text, "finals": r.finals, "partials": r.partials, "final_latency_s": r.final_latency_s, "coalesced_partials": r.coalesced});
        clips.push(json!({
            "clip": id, "audio_s": audio.len() as f64 / 16e3,
            "oracle_beam_lm": want.beam_lm, "oracle_greedy": want.greedy,
            "tail_off": run(&off), "tail_on": run(&on),
            "off_equals_oracle": off_ok, "on_equals_oracle": on_ok, "on_equals_off": off.text == on.text,
        }));
    }
    let lat = |key: &str| -> Vec<f64> { clips.iter().flat_map(|c| c[key]["final_latency_s"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap())).collect() };
    let stats = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        json!({"n": v.len(), "p50": q(0.5), "p95": q(0.95), "max": v[v.len() - 1]})
    };
    let doc = json!({
        "command": a.join(" "),
        "what": "SessionWorker replay at real time (20 ms chunks + 1.5 s trailing silence), MlxLargeBackend fp32 + beam+LM, one fresh backend per run; synthetic TTS fixtures",
        "artifact_sha256": sh("shasum", &["-a", "256", &a[1]]).split_whitespace().next().unwrap_or(""),
        "hardware": {"cpu": sh("sysctl", &["-n", "machdep.cpu.brand_string"]), "os": sh("sw_vers", &["-productVersion"]), "uptime_at_start": start_load, "uptime_at_end": sh("uptime", &[])},
        "git_sha": sh("git", &["rev-parse", "HEAD"]),
        "session_config": format!("{:?}", SessionConfig::default()),
        "final_latency_after_endpoint_s": {"tail_off": stats(lat("tail_off")), "tail_on": stats(lat("tail_on"))},
        "clips_where_only_tail_on_differs_from_oracle": differs_only_with_tail,
        "clips_where_tail_on_differs_from_tail_off": tail_differs_from_off,
        "clips": clips,
    });
    std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
    println!("{}", serde_json::to_string_pretty(&doc["final_latency_after_endpoint_s"]).unwrap());
    println!("only-tail-on differs from oracle: {differs_only_with_tail:?}; on != off: {tail_differs_from_off:?}");
}
