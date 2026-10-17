mod common;
use common::*;
use tingmix_runtime_onnx::{LogMel, N_MELS};

#[test]
fn log_mel_matches_torch() {
    for name in ["en_02s", "mix_04s", "zh_08s"] {
        let wav = load_wav(&format!("{name}.wav"));
        let (shape, want) = read_npy(&root().join(format!("tests/fixtures/small/mel_{name}.npy")));
        let got = LogMel::new().compute(&wav);
        assert_eq!(shape, [LogMel::frames(wav.len()), N_MELS]);
        let err = max_abs_diff(&got, &want);
        println!("{name}: {} frames, max|dlogmel| = {err:.3e}", shape[0]);
        assert!(err < 1e-3, "{name}: {err}");
    }
}
