//! The app's live captioning path on recordings: SessionConfig::live_recommended() (adaptive VAD, suppression, quiet-point
//! cut, gap bridge). Engine: MLX Large fp32 with beam 16 + char LM, or SenseVoice-Small on ONNX Runtime when PACKAGE_DIR holds
//! `model.int8.onnx` (set TINGMIX_ORT_DYLIB). Decoding is synchronous (StreamingSession), so finals do not depend on machine
//! speed; the app's SessionWorker only coalesces partials, which never become finals.
//!
//! usage: [EA_MAX_SEG_S=12] ea-app-live PACKAGE_DIR OUT.json WAV...   (MLX: needs MLX_PREFIX at build time,
//! DYLD_LIBRARY_PATH=$MLX_PREFIX/lib)
use std::path::{Path, PathBuf};
use tingmix_core::{AsrBackend, Reducer};
use tingmix_runtime_mlx::{Decode, Dtype, MlxLargeBackend};
use tingmix_runtime_onnx::OnnxSenseVoice;
use tingmix_streaming::{SessionConfig, SessionMsg, StreamingSession};

fn finals<B: AsrBackend>(b: B, wav: &Path) -> Vec<serde_json::Value> {
    let mut cfg = SessionConfig::live_recommended();
    // EA_MAX_SEG_S: segment-cap sweep (the app uses the preset value)
    if let Some(v) = std::env::var("EA_MAX_SEG_S").ok().and_then(|v| v.parse().ok()) {
        cfg.max_segment_s = v;
    }
    let mut s = StreamingSession::new(b, &cfg);
    let mut r = Reducer::new();
    let apply = |msgs: Vec<SessionMsg>, r: &mut Reducer| {
        for m in msgs {
            if let SessionMsg::Event(e) = m {
                let _ = r.apply(e);
            }
        }
    };
    tingmix_audio::decode_file_chunks(wav, |c| {
        apply(s.feed(c), &mut r);
        true
    })
    .unwrap();
    apply(s.flush(), &mut r);
    r.finals().map(|f| serde_json::json!([f.start, f.end, f.text])).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let pkg = PathBuf::from(&a[1]);
    let mut out = serde_json::Map::new();
    for wav in &a[3..] {
        let wav = Path::new(wav);
        let f = if pkg.join("model.int8.onnx").is_file() {
            finals(OnnxSenseVoice::load(&pkg.join("model.int8.onnx"), &pkg.join("tokens.txt"), 4).unwrap(), wav)
        } else {
            let lm = tingmix_decoder::CharLm::load(&pkg.join("lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz")).unwrap();
            let decode = Decode::BeamLm(Box::new(lm), tingmix_decoder::BeamConfig::default());
            finals(MlxLargeBackend::new(&pkg.join("tingmix-large-mlx-fp32.safetensors"), &pkg.join("units.json"), Dtype::F32, decode).unwrap(), wav)
        };
        out.insert(wav.file_stem().unwrap().to_string_lossy().into_owned(), f.into());
    }
    std::fs::write(&a[2], serde_json::to_vec_pretty(&out).unwrap()).unwrap();
}
