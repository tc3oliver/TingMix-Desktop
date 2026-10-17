//! Child process for compatibility probes (and any caller that wants GPU-crash isolation): loads the model once, then serves
//! decode requests on stdin/stdout. Request: `u64 LE n_samples` + `n_samples` f32 LE (16 kHz mono). Reply: one line
//! `OK\t<text>` or `ERR\t<message>`. First line after startup: `READY\t<mlx version>\t<device>`.
//! Usage: tingmix-mlx-worker <artifact.safetensors> <units.json> <f32|f16|bf16> [lm.charlm.json.gz]
use std::io::{Read, Write};
use std::path::Path;

use tingmix_core::AsrBackend;
use tingmix_decoder::{BeamConfig, CharLm};
use tingmix_runtime_mlx::{runtime_info, Decode, Dtype, MlxLargeBackend};

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
    if a.len() < 4 {
        fail(
            &mut out,
            "usage: tingmix-mlx-worker <artifact> <units.json> <f32|f16|bf16> [lm]".into(),
        );
    }
    let dtype = match a[3].as_str() {
        "f32" => Dtype::F32,
        "f16" => Dtype::F16,
        "bf16" => Dtype::Bf16,
        d => return fail(&mut out, format!("unknown dtype {d}")),
    };
    let decode = match a.get(4) {
        Some(p) => match CharLm::load(Path::new(p)) {
            Ok(lm) => Decode::BeamLm(Box::new(lm), BeamConfig::default()),
            Err(e) => return fail(&mut out, format!("lm: {e}")),
        },
        None => Decode::Greedy,
    };
    let mut backend = match MlxLargeBackend::new(Path::new(&a[1]), Path::new(&a[2]), dtype, decode)
    {
        Ok(b) => b,
        Err(e) => return fail(&mut out, e.to_string()),
    };
    let (ver, dev) = runtime_info().unwrap_or_default();
    let _ = writeln!(out, "READY\t{ver}\t{dev}");
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
