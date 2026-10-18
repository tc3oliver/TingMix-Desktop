//! The shipped signed manifest of the Large ONNX package, and the real compatibility probe in a child process.
//! Run with `--features probe`. The probe skips unless TINGMIX_LARGE_ONNX (tingmix-large.onnx) and TINGMIX_LARGE_DIR (TingMix-Large
//! snapshot with units.json and lm/*.charlm.json.gz) are set; load-dynamic builds also need TINGMIX_ORT_DYLIB.
//! TINGMIX_ONNX_PROVIDER=cpu|directml|cuda probes that provider (needs the matching cargo feature and an ORT library that has it).
#![cfg(feature = "probe")]
mod common;
use std::path::{Path, PathBuf};

use common::*;
use tingmix_models::manifest::{sha256_hex, Manifest};
use tingmix_models::probe::{consult, run_probe, Consulted, ProbeFixture};
use tingmix_models::{InstalledModel, Verifier};
use tingmix_runtime_onnx::probe::OnnxProbeTarget;
use tingmix_runtime_onnx::Provider;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name),
    )
    .unwrap()
}

#[test]
fn shipped_onnx_manifest_verifies_validates_and_says_what_it_is() {
    let signed = fixture("tingmix-large-onnx-fp32.signed.json");
    let v = Verifier::embedded().parse(&signed).unwrap();
    let m = v.manifest();
    m.validate(false).unwrap();
    assert_eq!(
        (
            m.backend.as_str(),
            m.os.as_str(),
            m.arch.as_str(),
            m.precision.as_str()
        ),
        ("onnx-cpu", "any", "any", "fp32")
    );
    assert_eq!(
        (
            m.runtime.name.as_str(),
            m.runtime.min_version.as_str(),
            m.compat.status.as_str()
        ),
        ("onnxruntime", "1.31.0", "EXPERIMENTAL")
    );
    let p = m.provenance.as_ref().unwrap();
    assert!(
        p.modified
            && p.converter_commit.len() == 40
            && p.converter == "experiments/windows/large-onnx/export.py"
    );
    let model = m.files.iter().find(|f| f.component == "model").unwrap();
    assert!(
        model.path.ends_with(".onnx") && model.url.is_empty(),
        "offline import only"
    );
    assert!(m
        .licenses
        .iter()
        .any(|l| l.id == "google-embeddinggemma-2" && l.acceptance_required));
    // the unsigned source of the signature is the same manifest
    let unsigned: Manifest =
        serde_json::from_slice(&fixture("tingmix-large-onnx-fp32.manifest.json")).unwrap();
    assert_eq!(&unsigned, m);
    // tampering with any signed field breaks verification
    let mut bad: serde_json::Value = serde_json::from_slice(&signed).unwrap();
    bad["manifest"]["files"][0]["sha256"] = "0".repeat(64).into();
    assert!(Verifier::embedded()
        .parse(&serde_json::to_vec(&bad).unwrap())
        .is_err());
}

#[test]
fn manifest_matches_the_exported_graph_when_it_is_available() {
    let Some(onnx) = std::env::var_os("TINGMIX_LARGE_ONNX") else {
        eprintln!("skipped: TINGMIX_LARGE_ONNX not set");
        return;
    };
    let m = Verifier::embedded()
        .parse(&fixture("tingmix-large-onnx-fp32.signed.json"))
        .unwrap();
    let model = m
        .manifest()
        .files
        .iter()
        .find(|f| f.component == "model")
        .unwrap()
        .clone();
    let bytes = std::fs::read(onnx).unwrap();
    assert_eq!(
        (bytes.len() as u64, sha256_hex(&bytes)),
        (model.size, model.sha256),
        "re-export changed the graph: re-sign the manifest"
    );
}

fn link(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(src, dst).unwrap();
    #[cfg(not(unix))]
    if std::fs::hard_link(src, dst).is_err() {
        std::fs::copy(src, dst).unwrap();
    }
}

#[test]
fn onnx_probe_passes_in_child_process() {
    let (Some(onnx), Some(large)) = (
        std::env::var_os("TINGMIX_LARGE_ONNX"),
        std::env::var_os("TINGMIX_LARGE_DIR"),
    ) else {
        eprintln!("skipped: TINGMIX_LARGE_ONNX / TINGMIX_LARGE_DIR not set");
        return;
    };
    let manifest = Verifier::embedded()
        .parse(&fixture("tingmix-large-onnx-fp32.signed.json"))
        .unwrap()
        .manifest()
        .clone();
    let dir = tempfile::tempdir().unwrap();
    for f in &manifest.files {
        let src = if f.component == "model" {
            PathBuf::from(&onnx)
        } else {
            Path::new(&large).join(&f.path)
        };
        if src.exists() {
            link(&src, &dir.path().join("v").join(&f.path));
        }
    }
    let model = InstalledModel {
        dir: dir.path().join("v"),
        manifest,
    };
    let w = load_wav("en_02s.wav");
    let fx = ProbeFixture {
        name: "en_02s",
        wav: &w,
        sample_rate: 16_000,
        expected_text: "good morning everyone",
    };
    let mut t = OnnxProbeTarget::new(
        PathBuf::from(env!("CARGO_BIN_EXE_tingmix-onnx-worker")),
        Provider::Cpu,
        4,
        true,
    );
    std::fs::create_dir_all(dir.path().join("probes")).unwrap(); // the registry root normally has it
    let rec = run_probe(dir.path(), &mut t, &model, &fx, 5).unwrap();
    eprintln!("{rec:#?}");
    assert!(rec.passed, "{}", rec.detail);
    assert_eq!(
        (rec.backend.as_str(), rec.device.as_str()),
        ("onnx-cpu", "cpu")
    );
    assert!(
        rec.backend_version.starts_with("1.3"),
        "{}",
        rec.backend_version
    );
    assert!(rec.rtf_x1000.is_some());
    assert!(matches!(
        consult(
            dir.path(),
            "onnx-cpu",
            &model.manifest,
            Some(&rec.backend_version),
            &rec.device
        ),
        Consulted::Fresh(_)
    ));
}
