//! App-free CLI: stream a 16 kHz mono wav through a Small step graph and print the collapsed CTC ids.
//! Pipeline check only; with random-init weights the ids mean nothing.
//!
//!   cargo run --release -p tingmix-runtime-onnx --example run_wav -- <model.onnx> <audio.wav> [ort-library]
//!
//! With `--no-default-features --features load-dynamic` the ONNX Runtime library comes from the third
//! argument, `TINGMIX_ORT_DYLIB`, or `ORT_DYLIB_PATH`, and nothing is downloaded or linked at build time.
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tingmix_runtime_onnx::{set_runtime_library, StepModel, Streamer};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: run_wav <model.onnx> <audio.wav> [ort-library]");
        std::process::exit(2);
    }
    if let Some(lib) = a.get(3) {
        set_runtime_library(PathBuf::from(lib).canonicalize().expect("ort library path"));
    }
    let wav: Vec<f32> = {
        let mut r = hound::WavReader::open(&a[2]).expect("wav");
        assert_eq!(
            (r.spec().sample_rate, r.spec().channels),
            (16_000, 1),
            "need 16 kHz mono"
        );
        r.samples::<i16>()
            .map(|s| f32::from(s.unwrap()) / 32768.0)
            .collect()
    };
    let t0 = Instant::now();
    let model = match StepModel::load(Path::new(&a[1]), 1) {
        Ok(m) => Arc::new(Mutex::new(m)),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let mut s = Streamer::new(model);
    for c in wav.chunks(5120) {
        s.push(c).expect("step");
    }
    println!(
        "{:.1}s audio, {} ids, {:.0} ms total, ids: {:?}",
        wav.len() as f32 / 16e3,
        s.ids().len(),
        t0.elapsed().as_secs_f64() * 1e3,
        &s.ids()[..s.ids().len().min(24)]
    );
}
