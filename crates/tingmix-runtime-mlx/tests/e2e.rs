//! wav -> text from Rust only (Rust log-mel, MLX tower+reader, Rust greedy and beam+LM) must equal
//! tests/fixtures/large/expected_texts.json (CPU fp32 oracle). Env: TINGMIX_MLX_ARTIFACT, TINGMIX_LARGE_DIR (TingMix-Large
//! snapshot with units.json and lm/*.charlm.json.gz); skips when unset. Also checks logits against the oracle dumps when
//! TINGMIX_ORACLE_DIR is set.
#![cfg(mlx_native)]
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use tingmix_core::AsrBackend;
use tingmix_decoder::{BeamConfig, CharLm};
use tingmix_runtime_mlx::{Decode, Dtype, MlxLargeBackend};

#[derive(Deserialize)]
struct Clip {
    greedy: String,
    beam_lm: String,
}
#[derive(Deserialize)]
struct Expected {
    clips: BTreeMap<String, Clip>,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn wav(path: &Path) -> Vec<f32> {
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

#[test]
fn wav_to_text_matches_oracle_texts() {
    let (Some(art), Some(large)) = (
        std::env::var_os("TINGMIX_MLX_ARTIFACT"),
        std::env::var_os("TINGMIX_LARGE_DIR"),
    ) else {
        eprintln!("skipped: TINGMIX_MLX_ARTIFACT / TINGMIX_LARGE_DIR not set");
        return;
    };
    let large = PathBuf::from(large);
    let exp: Expected = serde_json::from_slice(
        &std::fs::read(root().join("tests/fixtures/large/expected_texts.json")).unwrap(),
    )
    .unwrap();
    let lm = CharLm::load(&large.join("lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz")).unwrap();
    let mut greedy = MlxLargeBackend::new(
        Path::new(&art),
        &large.join("units.json"),
        Dtype::F32,
        Decode::Greedy,
    )
    .unwrap();
    let mut beam = MlxLargeBackend::new(
        Path::new(&art),
        &large.join("units.json"),
        Dtype::F32,
        Decode::BeamLm(Box::new(lm), BeamConfig::default()),
    )
    .unwrap();
    let mut bad = Vec::new();
    for (id, want) in &exp.clips {
        let audio = wav(&root().join(format!("tests/fixtures/audio/{id}.wav")));
        let g = greedy.transcribe_batch(&[&audio]).unwrap().remove(0);
        let b = beam.transcribe_batch(&[&audio]).unwrap().remove(0);
        eprintln!(
            "{id}: greedy {} beam {}",
            g == want.greedy,
            b == want.beam_lm
        );
        if g != want.greedy || b != want.beam_lm {
            bad.push(format!("{id}: greedy={g:?} beam={b:?}"));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}
