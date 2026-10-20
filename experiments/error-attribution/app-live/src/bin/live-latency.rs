//! Caption latency of Qwen3-ASR through the app's SessionWorker (`SessionConfig::live_recommended()`, the tail-partial
//! policy the app uses), with the recording fed in real time. Prints one line per event: wall time, segment, audio end,
//! lag and text length (no text), then a summary.
//!
//! usage: [EA_MAX_LOAD=0.8] live-latency EXPORT_DIR WAV THREADS   (TINGMIX_ORT_DYLIB: absolute path)
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tingmix_core::Event;
use tingmix_streaming::{SessionConfig, SessionMsg, SessionWorker};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    tingmix_runtime_onnx::set_runtime_library(PathBuf::from(
        std::env::var("TINGMIX_ORT_DYLIB").unwrap(),
    ));
    let b =
        tingmix_runtime_onnx::OnnxQwen3Asr::load(Path::new(&a[1]), a[3].parse().unwrap()).unwrap();
    let w = tingmix_audio::decode_file(Path::new(&a[2])).unwrap();
    let mut cfg = SessionConfig::live_recommended();
    if let Some(v) = std::env::var("EA_MAX_LOAD")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        cfg.max_load = v;
    }
    let mut s = SessionWorker::spawn(b, &cfg);
    let t0 = Instant::now();
    let (mut lags, mut finals) = (Vec::new(), Vec::new());
    for (i, c) in w[..w.len().min(40 * 16_000)].chunks(320).enumerate() {
        if let Some(d) = Duration::from_millis(20 * i as u64).checked_sub(t0.elapsed()) {
            std::thread::sleep(d);
        }
        s.feed(c);
        while let Ok(m) = s.events().try_recv() {
            let now = t0.elapsed().as_secs_f64();
            match m {
                SessionMsg::Event(Event::Partial(p)) => {
                    lags.push(now - p.end);
                    println!(
                        "{now:6.2}s P seg{} end {:5.2} lag {:4.2} chars {}",
                        p.segment_id,
                        p.end,
                        now - p.end,
                        p.text.chars().count()
                    );
                }
                SessionMsg::Event(Event::Final(f)) => {
                    finals.push(now - f.end);
                    println!(
                        "{now:6.2}s F seg{} end {:5.2} delay {:4.2} chars {}",
                        f.segment_id,
                        f.end,
                        now - f.end,
                        f.text.chars().count()
                    );
                }
                SessionMsg::Error { message, .. } => println!("error {message}"),
            }
        }
    }
    let max = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!(
        "partials {} mean lag {:.2}s max {:.2}s; finals {} mean delay {:.2}s max {:.2}s",
        lags.len(),
        mean(&lags),
        max(&lags),
        finals.len(),
        mean(&finals),
        max(&finals)
    );
}
