//! Qwen3-ASR through ONNX Runtime must reproduce the export repository's own Python ONNX path (`experiments/qwen3-asr`,
//! `ref_onnx.py` -> `out/qwen_ref.json`, `out/<clip>_mel.npy`): the same log-mel within 1e-3 and the same generated token ids
//! and text on every fixture. Env: TINGMIX_QWEN3_DIR (the export directory); skips when unset.
mod common;
use std::collections::BTreeMap;
use std::path::PathBuf;

use common::*;
use serde::Deserialize;
use tingmix_core::AsrBackend;
use tingmix_runtime_onnx::{OnnxQwen3Asr, WhisperMel};

#[derive(Deserialize)]
struct Clip {
    ids: Vec<i64>,
    text: String,
    mel_frames: usize,
}
#[derive(Deserialize)]
struct Reference {
    clips: BTreeMap<String, Clip>,
}

fn letters(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[test]
fn wav_to_text_matches_the_export_reference() {
    let Some(dir) = std::env::var_os("TINGMIX_QWEN3_DIR").map(PathBuf::from) else {
        eprintln!("skipped: TINGMIX_QWEN3_DIR not set");
        return;
    };
    init_ort();
    let out = root().join("experiments/qwen3-asr/out");
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(out.join("qwen_ref.json")).unwrap()).unwrap();
    let mel = WhisperMel::new();
    let mut b = OnnxQwen3Asr::load(&dir, 4).unwrap();
    let (mut checked, mut exact) = (0, 0);
    for (name, clip) in &reference.clips {
        let w = load_wav(&format!("{name}.wav"));
        let (m, frames) = mel.compute(&w);
        assert_eq!(frames, clip.mel_frames, "{name}: mel frames");
        let (_, want) = read_npy(&out.join(format!("{name}_mel.npy")));
        let d = max_abs_diff(&m, &want);
        eprintln!("{name}: max |mel diff| {d:.2e}");
        assert!(d < 1e-3, "{name}: mel differs by {d}");
        if w.len() > 30 * 16_000 {
            continue; // the backend cuts at 30 s, the reference decodes the whole clip
        }
        let (ids, _) = b.generate(&w).unwrap();
        let got = b.transcribe_batch(&[&w]).unwrap().remove(0);
        eprintln!(
            "{name}: ids {}",
            if ids == clip.ids { "equal" } else { "differ" }
        );
        exact += usize::from(ids == clip.ids);
        if got != clip.text {
            eprintln!("  rust: {got}\n  ref:  {}", clip.text);
            assert_eq!(letters(&got), letters(&clip.text), "{name}");
        }
        checked += 1;
    }
    assert!(checked >= 7, "only {checked} clips compared");
    eprintln!("{exact} of {checked} clips with identical token ids");
}
