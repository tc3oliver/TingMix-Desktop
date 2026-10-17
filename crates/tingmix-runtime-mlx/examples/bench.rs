//! Rust-side timings of the MLX Large backend: load, warm latency per stage and duration, peak RSS, and buffered streaming
//! through tingmix-streaming's SessionWorker replaying fixtures at real time.
//! cargo run --release -p tingmix-runtime-mlx --example bench -- <artifact.safetensors> <TingMix-Large dir> <out.json>
//! (needs MLX_PREFIX at build time). Raw output is written verbatim to <out.json>.
use std::path::{Path, PathBuf};
use std::sync::{mpsc::RecvTimeoutError, Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::json;
use tingmix_core::{AsrBackend, BackendError, BackendInfo, Event};
use tingmix_decoder::{BeamConfig, CharLm};
use tingmix_frontend_gemma::Gemma4Frontend;
use tingmix_runtime_mlx::{Decode, Dtype, MlxLargeBackend};
use tingmix_streaming::{SessionConfig, SessionMsg, SessionWorker};

fn wav(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    let mut p = 12;
    while &b[p..p + 4] != b"data" {
        p += 8 + u32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()) as usize;
    }
    let n = u32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()) as usize;
    b[p + 8..p + 8 + n]
        .chunks_exact(2)
        .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0)
        .collect()
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn stats(mut v: Vec<f64>) -> serde_json::Value {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    json!({"n": v.len(), "mean_ms": mean * 1e3, "p50_ms": pct(&mut v, 0.5) * 1e3, "p95_ms": pct(&mut v, 0.95) * 1e3, "p99_ms": pct(&mut v, 0.99) * 1e3, "max_ms": v[v.len() - 1] * 1e3})
}

fn peak_rss_mib() -> f64 {
    let mut ru = unsafe { std::mem::zeroed::<libc::rusage>() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    ru.ru_maxrss as f64 / (1024.0 * 1024.0) // bytes on macOS
}

/// Shares one loaded backend between runs and records every decode (audio seconds, wall seconds).
struct Timed {
    inner: Arc<Mutex<MlxLargeBackend>>,
    info: BackendInfo,
    log: Arc<Mutex<Vec<(f64, f64)>>>,
}

impl AsrBackend for Timed {
    fn info(&self) -> &BackendInfo {
        &self.info
    }
    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        let t = Instant::now();
        let r = self.inner.lock().unwrap().transcribe_batch(wavs);
        self.log.lock().unwrap().push((
            wavs.iter().map(|w| w.len()).sum::<usize>() as f64 / 16_000.0,
            t.elapsed().as_secs_f64(),
        ));
        r
    }
}

fn replay(backend: &Arc<Mutex<MlxLargeBackend>>, audio: &[f32]) -> serde_json::Value {
    let info = backend.lock().unwrap().info().clone();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut w = SessionWorker::spawn(
        Timed {
            inner: backend.clone(),
            info,
            log: log.clone(),
        },
        &SessionConfig::default(),
    );
    let t0 = Instant::now();
    let mut seen: Vec<(f64, Event)> = Vec::new();
    let drain = |w: &SessionWorker, seen: &mut Vec<(f64, Event)>| {
        for m in w.events().try_iter() {
            if let SessionMsg::Event(e) = m {
                seen.push((t0.elapsed().as_secs_f64(), e));
            }
        }
    };
    // 100 ms chunks at real time, then 1.5 s of silence so the endpointer closes, then flush
    let tail = vec![0f32; 24_000];
    for (i, c) in audio.chunks(1_600).chain(tail.chunks(1_600)).enumerate() {
        let due = Duration::from_secs_f64(i as f64 * 0.1);
        if let Some(d) = due.checked_sub(t0.elapsed()) {
            std::thread::sleep(d);
        }
        w.feed(c);
        drain(&w, &mut seen);
    }
    let end_wall = t0.elapsed().as_secs_f64();
    w.flush();
    loop {
        match w.events().recv_timeout(Duration::from_secs(4)) {
            Ok(SessionMsg::Event(e)) => seen.push((t0.elapsed().as_secs_f64(), e)),
            Ok(_) => {}
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
        }
    }
    let coalesced = w.coalesced_partials();
    drop(w);
    let first_partial = seen
        .iter()
        .find_map(|(t, e)| matches!(e, Event::Partial(_)).then_some(*t));
    let stale: Vec<f64> = seen
        .iter()
        .filter_map(|(t, e)| {
            if let Event::Partial(p) = e {
                Some(t - p.end)
            } else {
                None
            }
        })
        .collect();
    let finals: Vec<serde_json::Value> = seen
        .iter()
        .filter_map(|(t, e)| if let Event::Final(f) = e { Some(json!({"segment": f.segment_id, "end_s": f.end, "arrived_s": t, "after_end_s": t - f.end, "text": f.text})) } else { None })
        .collect();
    let log = log.lock().unwrap();
    let cost: Vec<f64> = log.iter().map(|x| x.1).collect();
    json!({
        "audio_s": audio.len() as f64 / 16_000.0, "end_of_feed_wall_s": end_wall,
        "first_partial_arrived_s": first_partial, "partials": stale.len(), "coalesced": coalesced,
        "partial_staleness_s": if stale.is_empty() { serde_json::Value::Null } else { stats(stale) },
        "finals": finals, "decode_cost": if cost.is_empty() { serde_json::Value::Null } else { stats(cost) },
        "decode_rtf_max": log.iter().map(|(a, c)| c / a).fold(0.0, f64::max),
    })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (art, large, out) = (
        PathBuf::from(&a[1]),
        PathBuf::from(&a[2]),
        PathBuf::from(&a[3]),
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fx = |id: &str| wav(&root.join(format!("tests/fixtures/audio/{id}.wav")));
    let units = large.join("units.json");

    let t = Instant::now();
    let mut greedy = MlxLargeBackend::new(&art, &units, Dtype::F32, Decode::Greedy).unwrap();
    let load_s = t.elapsed().as_secs_f64();
    let rss_after_load = peak_rss_mib();
    let t = Instant::now();
    let lm = CharLm::load(&large.join("lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz")).unwrap();
    let lm_load_s = t.elapsed().as_secs_f64();
    let rss_after_lm = peak_rss_mib();
    let mut beam = MlxLargeBackend::new(
        &art,
        &units,
        Dtype::F32,
        Decode::BeamLm(Box::new(lm), BeamConfig::default()),
    )
    .unwrap();

    let src = fx("mix_29s");
    let fe = Gemma4Frontend::new();
    let mut rows = Vec::new();
    for d in [2usize, 4, 8, 16, 29] {
        let w = &src[..d * 16_000];
        for _ in 0..3 {
            greedy.transcribe_batch(&[w]).unwrap();
            beam.transcribe_batch(&[w]).unwrap();
        }
        let (mut mel, mut logits, mut e2e_g, mut e2e_b) = (vec![], vec![], vec![], vec![]);
        for _ in 0..30 {
            let t = Instant::now();
            let m = fe.extract(w);
            mel.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            greedy.chunk_logits(w).unwrap(); // mel + MLX (mel counted again, subtracted below in the report)
            logits.push(t.elapsed().as_secs_f64());
            assert!(m.valid > 0);
            let t = Instant::now();
            greedy.transcribe_batch(&[w]).unwrap();
            e2e_g.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            beam.transcribe_batch(&[w]).unwrap();
            e2e_b.push(t.elapsed().as_secs_f64());
        }
        let p50 = |v: &Vec<f64>| pct(&mut v.clone(), 0.5);
        rows.push(json!({"audio_s": d, "rtf_p50_greedy": p50(&e2e_g) / d as f64, "rtf_p50_beam": p50(&e2e_b) / d as f64,
            "mel": stats(mel), "mel_plus_mlx": stats(logits), "e2e_greedy": stats(e2e_g), "e2e_beam_lm": stats(e2e_b)}));
        eprintln!("{d} s done");
    }
    let rss_bench = peak_rss_mib();

    let shared = Arc::new(Mutex::new(greedy));
    let shared_beam = Arc::new(Mutex::new(beam));
    let mut stream = serde_json::Map::new();
    for (name, b) in [("greedy", &shared), ("beam_lm", &shared_beam)] {
        let mut v = Vec::new();
        for id in ["zh_02s", "en_08s", "mix_08s", "zh_16s"] {
            let mut r = replay(b, &fx(id));
            r["id"] = json!(id);
            eprintln!(
                "stream {name} {id}: first partial {:?}",
                r["first_partial_arrived_s"]
            );
            v.push(r);
        }
        stream.insert(name.into(), v.into());
    }
    let res = json!({
        "cmd": format!("cargo run --release -p tingmix-runtime-mlx --example bench -- {}", a[1..].join(" ")),
        "note": "fp32; other agents were running work on this machine; buffered streaming (re-decode of the open segment), SessionConfig::default()",
        "load_s": load_s, "lm_load_s": lm_load_s, "peak_rss_mib": {"after_model_load": rss_after_load, "after_lm_load": rss_after_lm, "after_latency_bench": rss_bench, "final": peak_rss_mib()},
        "latency": rows, "streaming": stream,
    });
    std::fs::write(out, serde_json::to_string_pretty(&res).unwrap()).unwrap();
}
