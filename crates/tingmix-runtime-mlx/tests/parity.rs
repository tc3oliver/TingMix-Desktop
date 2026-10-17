//! Parity against the PyTorch CPU fp32 oracle dumped by experiments/macos/mlx (dump_bin.py). Needs MLX_PREFIX at build time,
//! TINGMIX_MLX_ARTIFACT (fp32 safetensors) and TINGMIX_MLX_REF (out/ref-bin) at run time; skips otherwise.
#![cfg(mlx_native)]
use std::path::PathBuf;
use tingmix_runtime_mlx::{Dtype, MlxLarge};

fn f32s(p: PathBuf) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn logits_match_oracle() {
    let (Some(art), Some(refd)) = (
        std::env::var_os("TINGMIX_MLX_ARTIFACT"),
        std::env::var_os("TINGMIX_MLX_REF"),
    ) else {
        eprintln!("skipped: TINGMIX_MLX_ARTIFACT / TINGMIX_MLX_REF not set");
        return;
    };
    let refd = PathBuf::from(refd);
    let mut m = MlxLarge::load(&PathBuf::from(art), Dtype::F32).expect("load");
    for id in ["zh_02s", "en_02s", "mix_04s", "mix_29s"] {
        let mel = f32s(refd.join(format!("{id}.mel.f32")));
        let want = f32s(refd.join(format!("{id}.logits.f32")));
        let n_valid = f32s(refd.join(format!("{id}.nvalid.f32")))[0] as usize;
        let got = m.logits(&mel, n_valid).expect("run");
        assert_eq!(got.len(), want.len(), "{id}");
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "{id}: frames {} logits max-abs {max:.3e}",
            got.len() / m.vocab()
        );
        assert!(max < 2e-3, "{id}: {max}");
    }
}
