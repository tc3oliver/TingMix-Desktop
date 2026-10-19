//! SenseVoice-Small through ONNX Runtime must reproduce the sherpa-onnx oracle (`experiments/sensevoice`, `out/reference.json`):
//! the same text on every clip of at most 30 s (punctuation may differ on at most one clip, see below), the same LFR frame count and the first LFR+CMVN frames within 1e-2.
//! Env: TINGMIX_SENSEVOICE_DIR (the sherpa-onnx export directory with model.int8.onnx, tokens.txt, test_wavs/); skips when
//! unset. Regenerate the oracle with `cd experiments/sensevoice && uv run python ref.py <dir> <dir>/test_wavs/*.wav
//! ../../tests/fixtures/audio/*.wav`.
mod common;
use std::collections::BTreeMap;
use std::path::PathBuf;

use common::*;
use serde::Deserialize;
use tingmix_core::AsrBackend;
use tingmix_runtime_onnx::OnnxSenseVoice;

#[derive(Deserialize)]
struct Clip {
    text: String,
    lfr_frames: usize,
    lfr_head: Vec<Vec<f32>>,
}
#[derive(Deserialize)]
struct Reference {
    clips: BTreeMap<String, Clip>,
}

fn wav(path: &std::path::Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).unwrap();
    assert_eq!(r.spec().sample_rate, 16_000, "{}", path.display());
    let ch = r.spec().channels as usize;
    r.samples::<i16>()
        .step_by(ch)
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect()
}

fn letters(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).collect()
}

#[test]
fn wav_to_text_matches_sherpa_onnx() {
    let Some(dir) = std::env::var_os("TINGMIX_SENSEVOICE_DIR").map(PathBuf::from) else {
        eprintln!("skipped: TINGMIX_SENSEVOICE_DIR not set");
        return;
    };
    init_ort();
    let reference: Reference = serde_json::from_slice(
        &std::fs::read(root().join("experiments/sensevoice/out/reference.json")).unwrap(),
    )
    .unwrap();
    let mut b =
        OnnxSenseVoice::load(&dir.join("model.int8.onnx"), &dir.join("tokens.txt"), 4).unwrap();
    let (mut checked, mut punct_only) = (0, 0);
    for (name, clip) in &reference.clips {
        let path = [
            dir.join("test_wavs").join(format!("{name}.wav")),
            root()
                .join("tests/fixtures/audio")
                .join(format!("{name}.wav")),
        ]
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| panic!("no wav for {name}"));
        let w = wav(&path);
        let x = b.features(&w);
        let dim = clip.lfr_head[0].len();
        assert_eq!(x.len() / dim, clip.lfr_frames, "{name}: LFR frames");
        for (t, row) in clip.lfr_head.iter().enumerate() {
            let d = max_abs_diff(&x[t * dim..(t + 1) * dim], row);
            assert!(d < 1e-2, "{name}: LFR frame {t} differs by {d}");
        }
        // every frame, when the oracle's full dump is there (`<clip>_lfr.npy`, written by ref.py)
        let npy = root().join(format!("experiments/sensevoice/out/{name}_lfr.npy"));
        if npy.is_file() {
            let (_, all) = read_npy(&npy);
            let d = max_abs_diff(&x, &all);
            eprintln!("{name}: max |LFR diff| over all frames {d:.2e}");
            assert!(d < 1e-2, "{name}: LFR features differ by {d}");
        }
        if w.len() > 30 * 16_000 {
            continue; // the backend cuts at 30 s, sherpa-onnx decodes the whole clip
        }
        let got = b.transcribe_batch(&[&w]).unwrap().remove(0);
        if got != clip.text {
            assert_eq!(letters(&got), letters(&clip.text), "{name}");
            eprintln!(
                "{name}: punctuation differs\n  rust:   {got}\n  sherpa: {}",
                clip.text
            );
            punct_only += 1;
        }
        checked += 1;
    }
    assert!(checked >= 8, "only {checked} clips compared");
    // a comma whose logit leads by ~0.1 flips with 1e-4 feature noise (zh_16s); more than one such clip is a real difference
    assert!(punct_only <= 1, "{punct_only} clips differ in punctuation");
}
