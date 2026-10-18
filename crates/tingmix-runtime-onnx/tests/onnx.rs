//! Pipeline-only checks on a seeded random-init tiny model: they prove the export, the Rust
//! frontend and the state carry are exact, not that anything is recognized correctly.
mod common;
use std::sync::{Arc, Mutex};

use common::*;
use tingmix_core::AsrBackend;
use tingmix_runtime_onnx::{Collapser, OnnxSmall, StepModel, Streamer, TokenDecoder, N_MELS};
use tingmix_streaming::{SessionConfig, SessionMsg, StreamingSession};

fn model() -> Arc<Mutex<StepModel>> {
    Arc::new(Mutex::new(StepModel::load(&tiny_model(), 1).unwrap()))
}

fn stream_logp(m: &Arc<Mutex<StepModel>>, wav: &[f32], chunk: usize) -> (Vec<f32>, Vec<u32>) {
    let mut s = Streamer::new(m.clone());
    let logp = wav.chunks(chunk).flat_map(|c| s.push(c).unwrap()).collect();
    (logp, s.ids().to_vec())
}

#[test]
fn onnx_matches_torch_logits_offline_and_at_every_chunk_size() {
    let m = model();
    for name in ["en_02s", "mix_04s"] {
        let wav = load_wav(&format!("{name}.wav"));
        let (shape, want) =
            read_npy(&root().join(format!("tests/fixtures/small/tiny_logp_{name}.npy")));
        let mut whole_ids = Collapser::new();
        let (offline, ids) = stream_logp(&m, &wav, wav.len());
        assert_eq!(offline.len(), want.len(), "{shape:?}");
        let err = max_abs_diff(&offline, &want);
        println!("{name}: onnx(Rust frontend) vs torch offline max|dlogp| = {err:.3e}");
        assert!(err < 5e-3, "{err}");
        whole_ids.push(&offline.chunks(40).map(argmax).collect::<Vec<_>>());
        assert_eq!(whole_ids.ids(), ids);
        for ms in [160, 320, 640] {
            let (chunked, cids) = stream_logp(&m, &wav, 16 * ms);
            let e = max_abs_diff(&chunked, &offline);
            println!("{name}: chunk {ms} ms vs offline max|dlogp| = {e:.3e}");
            // ORT kernels may round differently per matrix shape (seen on x86 Windows: 1.3e-4, Apple: 1e-6); ids must still be identical
            assert!(e < 5e-4, "{ms} ms: {e}");
            assert_eq!(cids, ids, "{ms} ms");
        }
        // odd chunk sizes exercise the sample remainder and the sub-4-frame mel buffer
        for odd in [1usize, 777, 3999] {
            let (chunked, cids) = stream_logp(&m, &wav, odd);
            assert!(max_abs_diff(&chunked, &offline) < 5e-4, "chunk {odd}");
            assert_eq!(cids, ids);
        }
    }
}

fn argmax(row: &[f32]) -> u32 {
    let mut b = 0;
    for (i, &x) in row.iter().enumerate() {
        if x > row[b] {
            b = i;
        }
    }
    b as u32
}

#[test]
fn state_carry_matters() {
    // Running the second half with a fresh state must differ from continuing the stream.
    let m = model();
    let wav = load_wav("mix_04s.wav");
    let (all, _) = stream_logp(&m, &wav, wav.len());
    let half = wav.len() / 2 / 160 * 160;
    let (_, _) = stream_logp(&m, &wav[..half], half);
    let (fresh_tail, _) = stream_logp(&m, &wav[half..], wav.len());
    let tail = &all[all.len() - fresh_tail.len()..];
    assert!(
        max_abs_diff(tail, &fresh_tail) > 1e-2,
        "state carry had no effect"
    );
}

#[test]
fn backend_streams_text_and_offline_decode_agree() {
    let decode: TokenDecoder = Arc::new(|ids| {
        ids.iter()
            .map(|i| format!("t{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    });
    let mut b = OnnxSmall::load(&tiny_model(), 1, decode).unwrap();
    assert!(b.info().streaming_native);
    let wav = load_wav("mix_04s.wav");
    let mut s = b.new_stream().unwrap();
    let mut last = String::new();
    for c in wav.chunks(5120) {
        last = s.accept(c).unwrap();
    }
    assert_eq!(s.finalize().unwrap(), last);
    assert!(!last.is_empty(), "random tiny model emitted nothing");
    assert_eq!(b.transcribe_batch(&[&wav]).unwrap(), [last]);
}

#[test]
fn streaming_session_runs_on_the_onnx_backend() {
    let decode: TokenDecoder = Arc::new(|ids| {
        ids.iter()
            .map(|i| format!("t{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    });
    let b = OnnxSmall::load(&tiny_model(), 1, decode).unwrap();
    let mut session = StreamingSession::new(b, &SessionConfig::default());
    let mut wav = load_wav("mix_04s.wav");
    wav.extend(std::iter::repeat_n(0.0, 16_000));
    let mut msgs: Vec<_> = wav.chunks(1600).flat_map(|c| session.feed(c)).collect();
    msgs.extend(session.flush());
    assert!(
        msgs.iter().all(|m| matches!(m, SessionMsg::Event(_))),
        "{msgs:?}"
    );
    assert!(msgs
        .iter()
        .any(|m| matches!(m, SessionMsg::Event(tingmix_core::Event::Final(_)))));
    let _ = N_MELS;
}
