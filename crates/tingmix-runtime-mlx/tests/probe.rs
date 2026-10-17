//! Real probe: spawn tingmix-mlx-worker, decode en_02s, compare with the oracle text, record RTF. Skips unless
//! TINGMIX_MLX_ARTIFACT and TINGMIX_LARGE_DIR are set (and MLX_PREFIX at build time). Run with `--features probe`.
#![cfg(all(mlx_native, feature = "probe"))]
use std::path::{Path, PathBuf};

use tingmix_models::manifest::Manifest;
use tingmix_models::probe::{consult, run_probe, Consulted, ProbeFixture};
use tingmix_models::InstalledModel;
use tingmix_runtime_mlx::probe::MlxProbeTarget;
use tingmix_runtime_mlx::Dtype;

fn wav(path: &Path) -> Vec<f32> {
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
fn mlx_probe_passes_in_child_process() {
    let (Some(art), Some(large)) = (
        std::env::var_os("TINGMIX_MLX_ARTIFACT"),
        std::env::var_os("TINGMIX_LARGE_DIR"),
    ) else {
        eprintln!("skipped: TINGMIX_MLX_ARTIFACT / TINGMIX_LARGE_DIR not set");
        return;
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest: Manifest = serde_json::from_slice(
        &std::fs::read(
            root.join("crates/tingmix-models/fixtures/tingmix-large-mlx-fp32.manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    // version dir with symlinks to the real files, named as the manifest says
    let dir = tempfile::tempdir().unwrap();
    for f in &manifest.files {
        let src = if f.component == "model" {
            PathBuf::from(&art)
        } else {
            Path::new(&large).join(&f.path)
        };
        let dst = dir.path().join("v").join(&f.path);
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(src, dst).unwrap();
    }
    let model = InstalledModel {
        dir: dir.path().join("v"),
        manifest,
    };
    let w = wav(&root.join("tests/fixtures/audio/en_02s.wav"));
    let fx = ProbeFixture {
        name: "en_02s",
        wav: &w,
        sample_rate: 16_000,
        expected_text: "good morning everyone",
    };
    let mut t = MlxProbeTarget::new(
        PathBuf::from(env!("CARGO_BIN_EXE_tingmix-mlx-worker")),
        Dtype::F32,
        true,
    );
    std::fs::create_dir_all(dir.path().join("probes")).unwrap(); // the registry root normally has it
    let rec = run_probe(dir.path(), &mut t, &model, &fx, 5).unwrap();
    eprintln!("{rec:#?}");
    assert!(rec.passed, "{}", rec.detail);
    assert!(rec.rtf_x1000.is_some());
    assert!(rec.device.starts_with("gpu:"), "{}", rec.device);
    assert!(matches!(
        consult(
            dir.path(),
            "mlx",
            &model.manifest,
            Some(&rec.backend_version),
            &rec.device
        ),
        Consulted::Fresh(_)
    ));
}
