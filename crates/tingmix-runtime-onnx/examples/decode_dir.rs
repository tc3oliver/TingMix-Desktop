//! Decode every 16 kHz mono wav in a directory with SenseVoice-Small or Qwen3-ASR and write `{stem: text}` JSON plus timing,
//! for scoring by another harness (`experiments/error-attribution`, `ea.other`). Texts are the backend's own output (the app
//! converts Simplified characters to Traditional afterwards).
//!
//!   cargo run --release -p tingmix-runtime-onnx --example decode_dir -- <sensevoice|qwen3> <export dir> <wav dir> <out.json> [threads]
//! With `--no-default-features --features load-dynamic` set TINGMIX_ORT_DYLIB to the onnxruntime library.
use std::path::Path;
use std::time::Instant;

use tingmix_core::AsrBackend;
use tingmix_runtime_onnx::{runtime_version, OnnxQwen3Asr, OnnxSenseVoice};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(
        a.len() >= 5,
        "usage: decode_dir <sensevoice|qwen3> <export dir> <wav dir> <out.json> [threads]"
    );
    let (engine, export, dir, out) = (
        a[1].as_str(),
        Path::new(&a[2]),
        Path::new(&a[3]),
        Path::new(&a[4]),
    );
    let threads: usize = a.get(5).map_or(4, |t| t.parse().unwrap());
    let t0 = Instant::now();
    let mut b: Box<dyn AsrBackend> = match engine {
        "sensevoice" => Box::new(
            OnnxSenseVoice::load(
                &export.join("model.int8.onnx"),
                &export.join("tokens.txt"),
                threads,
            )
            .unwrap(),
        ),
        "qwen3" => Box::new(OnnxQwen3Asr::load(export, threads).unwrap()),
        e => panic!("unknown engine {e}"),
    };
    let load_s = t0.elapsed().as_secs_f64();
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    files.sort();
    let (mut texts, mut audio_s, mut decode_s) = (serde_json::Map::new(), 0.0, 0.0);
    for p in &files {
        let mut r = hound::WavReader::open(p).unwrap();
        let spec = r.spec();
        assert_eq!(
            (spec.sample_rate, spec.channels),
            (16_000, 1),
            "{}",
            p.display()
        );
        let w: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => r
                .samples::<i16>()
                .map(|s| f32::from(s.unwrap()) / 32768.0)
                .collect(),
            hound::SampleFormat::Float => r.samples::<f32>().map(Result::unwrap).collect(),
        };
        let t = Instant::now();
        let text = b.transcribe_batch(&[&w]).unwrap().remove(0);
        decode_s += t.elapsed().as_secs_f64();
        audio_s += w.len() as f64 / 16_000.0;
        texts.insert(p.file_stem().unwrap().to_string_lossy().into(), text.into());
    }
    let doc = serde_json::json!({
        "backend": b.info().name,
        "onnxruntime": runtime_version(),
        "threads": threads,
        "load_s": load_s,
        "clips": files.len(),
        "audio_s": audio_s,
        "decode_s": decode_s,
        "rtf": decode_s / audio_s,
        "texts": texts,
    });
    std::fs::write(out, serde_json::to_vec_pretty(&doc).unwrap()).unwrap();
    eprintln!(
        "{} clips, {audio_s:.1} s audio, decode {decode_s:.2} s, RTF {:.4}",
        files.len(),
        decode_s / audio_s
    );
}
