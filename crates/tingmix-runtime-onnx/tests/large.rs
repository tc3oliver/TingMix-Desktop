//! wav -> text through the Large ONNX graph must equal tests/fixtures/large/expected_texts.json (CPU fp32 oracle), greedy and
//! beam+LM, on all 8 fixtures. Env: TINGMIX_LARGE_ONNX (tingmix-large.onnx from experiments/windows/large-onnx) and
//! TINGMIX_LARGE_DIR (TingMix-Large snapshot with units.json and lm/*.charlm.json.gz); skips when unset. Optional:
//! TINGMIX_ONNX_PROVIDER=cpu|directml|cuda (needs the matching cargo feature).
mod common;
use std::collections::BTreeMap;
use std::path::PathBuf;

use common::*;
use serde::Deserialize;
use tingmix_core::AsrBackend;
use tingmix_decoder::{BeamConfig, CharLm};
use tingmix_runtime_onnx::{Decode, OnnxLarge, Provider};

#[derive(Deserialize)]
struct Clip {
    greedy: String,
    beam_lm: String,
}
#[derive(Deserialize)]
struct Expected {
    clips: BTreeMap<String, Clip>,
}

fn provider() -> Provider {
    match std::env::var("TINGMIX_ONNX_PROVIDER").as_deref() {
        Ok("directml") => Provider::DirectMl,
        Ok("cuda") => Provider::Cuda,
        _ => Provider::Cpu,
    }
}

#[test]
fn wav_to_text_matches_oracle_texts() {
    let (Some(model), Some(large)) = (
        std::env::var_os("TINGMIX_LARGE_ONNX"),
        std::env::var_os("TINGMIX_LARGE_DIR"),
    ) else {
        eprintln!("skipped: TINGMIX_LARGE_ONNX / TINGMIX_LARGE_DIR not set");
        return;
    };
    init_ort();
    let (model, large) = (PathBuf::from(model), PathBuf::from(large));
    let exp: Expected = serde_json::from_slice(
        &std::fs::read(root().join("tests/fixtures/large/expected_texts.json")).unwrap(),
    )
    .unwrap();
    let lm = CharLm::load(&large.join("lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz")).unwrap();
    let mut greedy = OnnxLarge::load(
        &model,
        &large.join("units.json"),
        Decode::Greedy,
        provider(),
        4,
    )
    .unwrap();
    let mut beam = OnnxLarge::load(
        &model,
        &large.join("units.json"),
        Decode::BeamLm(Box::new(lm), BeamConfig::default()),
        provider(),
        4,
    )
    .unwrap();
    let mut bad = Vec::new();
    for (id, want) in &exp.clips {
        let audio = load_wav(&format!("{id}.wav"));
        let g = greedy.transcribe_batch(&[&audio]).unwrap().remove(0);
        let b = beam.transcribe_batch(&[&audio]).unwrap().remove(0);
        eprintln!(
            "{id}: greedy {} beam {}",
            g == want.greedy,
            b == want.beam_lm
        );
        if g != want.greedy || b != want.beam_lm {
            bad.push(format!(
                "{id}: greedy {g:?} want {:?}; beam {b:?} want {:?}",
                want.greedy, want.beam_lm
            ));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
    assert_eq!(exp.clips.len(), 8);
}
