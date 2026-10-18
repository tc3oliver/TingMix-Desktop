//! Latency of the Large ONNX backend, wav in -> text out and the model part alone, for an 8 s and a 29.5 s window.
//! PIPELINE numbers on the fixtures (synthetic TTS): they say nothing about real-speech accuracy.
//!
//!   cargo run --release -p tingmix-runtime-onnx --example bench_large -- \
//!       <tingmix-large.onnx> <TingMix-Large dir> <out.json> [cpu|directml|cuda] [threads]
//! With `--no-default-features --features load-dynamic[,directml|cuda]` set TINGMIX_ORT_DYLIB to the onnxruntime library.
use std::path::{Path, PathBuf};
use std::time::Instant;

use tingmix_core::AsrBackend;
use tingmix_decoder::{BeamConfig, CharLm};
use tingmix_runtime_onnx::{Decode, OnnxLarge, Provider};

fn wav(name: &str) -> Vec<f32> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/audio")
        .join(name);
    let mut r = hound::WavReader::open(p).unwrap();
    r.samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect()
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn stats(mut v: Vec<f64>) -> serde_json::Value {
    serde_json::json!({"n": v.len(), "p50_ms": pct(&mut v, 0.5) * 1e3, "p95_ms": pct(&mut v, 0.95) * 1e3, "min_ms": v[0] * 1e3})
}

#[cfg(unix)]
fn peak_rss_mb() -> f64 {
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    let bytes = if cfg!(target_os = "macos") {
        u.ru_maxrss as f64
    } else {
        u.ru_maxrss as f64 * 1024.0
    };
    bytes / 1e6
}

#[cfg(windows)]
fn peak_rss_mb() -> f64 {
    #[repr(C)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        rest: [usize; 7],
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, cb: u32) -> i32;
    }
    let mut c = Counters {
        cb: std::mem::size_of::<Counters>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        rest: [0; 7],
    };
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.peak_working_set_size as f64 / 1e6
}

fn sh(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (model, large, out) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]), &a[3]);
    let provider = match a.get(4).map(String::as_str) {
        Some("directml") => Provider::DirectMl,
        Some("cuda") => Provider::Cuda,
        _ => Provider::Cpu,
    };
    let threads: usize = a.get(5).map_or(8, |s| s.parse().unwrap());
    let units = large.join("units.json");
    let lm_path = large.join("lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz");
    let lm = CharLm::load(&lm_path).unwrap();
    let mut beam = OnnxLarge::load(
        &model,
        &units,
        Decode::BeamLm(Box::new(lm), BeamConfig::default()),
        provider,
        threads,
    )
    .unwrap();
    let mut rows = serde_json::Map::new();
    for (label, clip, max) in [
        ("8s", "mix_08s.wav", usize::MAX),
        ("29.5s", "mix_29s.wav", 472_000),
    ] {
        let mut audio = wav(clip);
        audio.truncate(max);
        let secs = audio.len() as f64 / 16e3;
        let (mut logits_t, mut beam_t) = (vec![], vec![]);
        for i in 0..12 {
            let t = Instant::now();
            beam.chunk_logits(&audio).unwrap();
            let l = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let b = beam.transcribe_batch(&[&audio]).unwrap();
            let bt = t.elapsed().as_secs_f64();
            if i >= 2 {
                logits_t.push(l);
                beam_t.push(bt);
            } else if i == 0 {
                eprintln!("{label}: {} chars beam", b[0].chars().count());
            }
        }
        let rtf = |v: &Vec<f64>| pct(&mut v.clone(), 0.5) / secs;
        rows.insert(
            label.into(),
            serde_json::json!({"audio_s": secs, "mel_plus_model": stats(logits_t.clone()), "wav_to_text_beam_lm": stats(beam_t.clone()),
                "rtf_p50_model": rtf(&logits_t), "rtf_p50_beam_lm": rtf(&beam_t)}),
        );
        eprintln!("{label}: {}", rows[label]);
    }
    let doc = serde_json::json!({
        "what": "OnnxLarge wav->text latency (Rust Gemma4 mel + ORT + Rust CTC decode), fixtures mix_08s and the first 29.5 s of mix_29s",
        "command": a.join(" "), "provider": format!("{provider:?}"), "threads": threads, "peak_rss_mb": peak_rss_mb(), "note_memory": "peak_rss_mb = peak resident/working set of this process, which holds ONE backend (beam+LM, the production decode)",
        "model_sha256": if cfg!(windows) { sh("powershell", &["-NoProfile", "-Command", &format!("(Get-FileHash -Algorithm SHA256 -LiteralPath '{}').Hash.ToLower()", model.display())]) } else { sh("shasum", &["-a", "256", &model.display().to_string()]).split_whitespace().next().unwrap_or("").to_owned() },
        "ort_library": std::env::var("TINGMIX_ORT_DYLIB").unwrap_or_else(|_| "linked (download-binaries)".into()),
        "cpu": if cfg!(windows) { sh("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"]) } else { sh("sysctl", &["-n", "machdep.cpu.brand_string"]) },
        "results": rows,
    });
    std::fs::write(
        Path::new(out),
        serde_json::to_string_pretty(&doc).unwrap() + "\n",
    )
    .unwrap();
}
