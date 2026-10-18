//! Child process for compatibility probes (and any caller that wants GPU-driver crash isolation): loads the Large model once,
//! then serves decode requests on stdin/stdout. Request: `u64 LE n_samples` + `n_samples` f32 LE (16 kHz mono). Reply: one line
//! `OK\t<text>` or `ERR\t<message>`. First line after startup: `READY\t<onnxruntime version>\t<device>`.
//! Usage: tingmix-onnx-worker <tingmix-large.onnx> <units.json> <cpu|directml|cuda> <threads> [lm.charlm.json.gz]
//! The ONNX Runtime library comes from `TINGMIX_ORT_DYLIB` / `ORT_DYLIB_PATH` in load-dynamic builds.
use std::io::{Read, Write};
use std::path::Path;

use tingmix_core::AsrBackend;
use tingmix_decoder::{BeamConfig, CharLm};
use tingmix_runtime_onnx::{runtime_version, Decode, OnnxLarge, Provider};

fn clean(s: &str) -> String {
    s.replace(['\n', '\t', '\r'], " ")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = std::io::stdout();
    let mut out = out.lock();
    let fail = |out: &mut dyn Write, m: String| {
        let _ = writeln!(out, "ERR\t{}", clean(&m));
        std::process::exit(2);
    };
    if a.len() < 5 {
        fail(&mut out, "usage: tingmix-onnx-worker <model.onnx> <units.json> <cpu|directml|cuda> <threads> [lm]".into());
    }
    let (provider, device) = match a[3].as_str() {
        "cpu" => (Provider::Cpu, "cpu"),
        "directml" => (Provider::DirectMl, "gpu:directml"),
        "cuda" => (Provider::Cuda, "gpu:cuda"),
        p => return fail(&mut out, format!("unknown provider {p}")),
    };
    let threads: usize = a[4].parse().unwrap_or(4);
    let decode = match a.get(5) {
        Some(p) => match CharLm::load(Path::new(p)) {
            Ok(lm) => Decode::BeamLm(Box::new(lm), BeamConfig::default()),
            Err(e) => return fail(&mut out, format!("lm: {e}")),
        },
        None => Decode::Greedy,
    };
    let mut backend = match OnnxLarge::load(
        Path::new(&a[1]),
        Path::new(&a[2]),
        decode,
        provider,
        threads,
    ) {
        Ok(b) => b,
        Err(e) => return fail(&mut out, e.to_string()),
    };
    let _ = writeln!(
        out,
        "READY\t{}\t{device}",
        runtime_version().unwrap_or_default()
    );
    let _ = out.flush();
    let mut stdin = std::io::stdin().lock();
    loop {
        let mut n = [0u8; 8];
        if stdin.read_exact(&mut n).is_err() {
            return; // parent closed the pipe
        }
        let mut raw = vec![0u8; u64::from_le_bytes(n) as usize * 4];
        if stdin.read_exact(&mut raw).is_err() {
            return;
        }
        let wav: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        match backend.transcribe_batch(&[&wav]) {
            Ok(mut t) => writeln!(out, "OK\t{}", clean(&t.remove(0))),
            Err(e) => writeln!(out, "ERR\t{}", clean(&e.to_string())),
        }
        .and_then(|_| out.flush())
        .ok();
    }
}
