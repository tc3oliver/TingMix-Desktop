mod common;
use std::sync::atomic::{AtomicBool, Ordering};

use common::*;
use tingmix_models::fetch::Downloader;
use tingmix_models::import::{import_dir, import_zip};
use tingmix_models::manifest::*;
use tingmix_models::Error;

fn accept(reg: &tingmix_models::Registry, m: &VerifiedManifest) {
    reg.record_acceptance(m, "model-terms").unwrap();
}

fn no_cancel() -> AtomicBool {
    AtomicBool::new(false)
}

#[test]
fn download_verify_activate_and_offline_load() {
    let srv = serve(&[("v1", 1)]);
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(dir.path());
    let m = signed(&manifest("v1", &srv.url, 1));
    accept(&reg, &m);
    let installed = Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap();
    assert!(installed.dir.join("model.safetensors").is_file());
    reg.activate("v1").unwrap();
    reg.verify_full("v1").unwrap();

    // Offline: the server is gone and no Downloader exists; a fresh Registry loads from the cache alone.
    srv.state.lock().unwrap().files.clear();
    let before = srv.state.lock().unwrap().requests.len();
    let reg2 = registry(dir.path());
    let active = reg2.open_active().unwrap().unwrap();
    assert_eq!(active.manifest.id, "v1");
    assert_eq!(srv.state.lock().unwrap().requests.len(), before);
}

#[test]
fn licence_must_be_accepted_before_any_request() {
    let srv = serve(&[("v1", 1)]);
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(dir.path());
    let m = signed(&manifest("v1", &srv.url, 1));
    let e = Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap_err();
    assert!(
        matches!(e, Error::LicenseNotAccepted(ref l) if l == "model-terms"),
        "{e}"
    );
    assert!(srv.state.lock().unwrap().requests.is_empty());
    assert!(reg.list().unwrap().is_empty());
}

#[test]
fn truncated_download_resumes_with_range() {
    let srv = serve(&[("v1", 1)]);
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(dir.path());
    let m = signed(&manifest("v1", &srv.url, 1));
    accept(&reg, &m);
    srv.state.lock().unwrap().cut_next = Some(100_000);
    let e = Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap_err();
    assert!(matches!(e, Error::Download { .. }), "{e}");
    let part = dir.path().join("tmp/v1/model.safetensors.part");
    assert_eq!(std::fs::metadata(&part).unwrap().len(), 100_000);
    assert!(
        reg.list().unwrap().is_empty(),
        "nothing is installed from a partial download"
    );

    Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap();
    let ranges: Vec<_> = srv
        .state
        .lock()
        .unwrap()
        .requests
        .iter()
        .filter(|r| r.0.ends_with("model.safetensors"))
        .map(|r| r.1)
        .collect();
    assert_eq!(ranges, vec![None, Some(100_000)]);
    reg.verify_full("v1").unwrap();
}

#[test]
fn server_ignoring_range_restarts_cleanly() {
    let srv = serve(&[("v1", 1)]);
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(dir.path());
    let m = signed(&manifest("v1", &srv.url, 1));
    accept(&reg, &m);
    srv.state.lock().unwrap().cut_next = Some(50_000);
    Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap_err();
    srv.state.lock().unwrap().ignore_range = true;
    Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap();
    reg.verify_full("v1").unwrap();
}

#[test]
fn corrupt_partial_is_rejected_and_discarded() {
    let srv = serve(&[("v1", 1)]);
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(dir.path());
    let m = signed(&manifest("v1", &srv.url, 1));
    accept(&reg, &m);
    srv.state.lock().unwrap().cut_next = Some(10_000);
    Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap_err();
    let part = dir.path().join("tmp/v1/model.safetensors.part");
    let mut b = std::fs::read(&part).unwrap();
    b[5] ^= 0xff; // bit rot in the resumed prefix
    std::fs::write(&part, b).unwrap();
    let e = Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap_err();
    assert!(matches!(e, Error::HashMismatch { .. }), "{e}");
    assert!(!part.exists());
    assert!(reg.list().unwrap().is_empty());
    Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap();
}

#[test]
fn tampered_server_content_is_rejected() {
    let srv = serve(&[("v1", 1)]);
    srv.state
        .lock()
        .unwrap()
        .files
        .get_mut("/v1/model.safetensors")
        .unwrap()[1000] ^= 1;
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(dir.path());
    let m = signed(&manifest("v1", &srv.url, 1));
    accept(&reg, &m);
    let e = Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap_err();
    assert!(
        matches!(e, Error::HashMismatch { ref path, .. } if path == "model.safetensors"),
        "{e}"
    );
    assert!(reg.list().unwrap().is_empty());
}

#[test]
fn cancel_keeps_partial_for_resume() {
    let srv = serve(&[("v1", 1)]);
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(dir.path());
    let m = signed(&manifest("v1", &srv.url, 1));
    accept(&reg, &m);
    let cancel = no_cancel();
    let e = Downloader::new()
        .install(&reg, &m, &cancel, |p| {
            if p.bytes_done > 1000 {
                cancel.store(true, Ordering::Relaxed)
            }
        })
        .unwrap_err();
    assert!(matches!(e, Error::Cancelled));
    assert!(reg.list().unwrap().is_empty());
    Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap();
}

#[test]
fn tampered_manifest_wrong_key_and_unknown_fields_are_rejected() {
    let m = manifest("v1", "http://127.0.0.1:1", 1);
    let good = sign(&m, "test", &key(1));
    assert!(verifier().verify(good.clone()).is_ok());

    let mut t = good.clone();
    t.manifest.files[0].sha256 = sha256_hex(b"other"); // attacker swaps a hash
    assert!(matches!(verifier().verify(t), Err(Error::BadSignature(_))));
    let mut t = good.clone();
    t.manifest.files[0].url = "https://evil.example/resolve/".to_string() + &"a".repeat(40) + "/x";
    assert!(matches!(verifier().verify(t), Err(Error::BadSignature(_))));

    assert!(
        matches!(
            verifier().verify(sign(&m, "test", &key(2))),
            Err(Error::BadSignature(_))
        ),
        "wrong key, right key id"
    );
    assert!(matches!(
        verifier().verify(sign(&m, "other", &key(1))),
        Err(Error::UnknownKey(_))
    ));

    let mut v: serde_json::Value = serde_json::to_value(&good).unwrap();
    v["manifest"]["extra"] = serde_json::json!(true);
    assert!(matches!(
        verifier().parse(&serde_json::to_vec(&v).unwrap()),
        Err(Error::Malformed(_))
    ));
    let mut v: serde_json::Value = serde_json::to_value(&good).unwrap();
    v["signature"] = serde_json::json!("00");
    assert!(matches!(
        verifier().parse(&serde_json::to_vec(&v).unwrap()),
        Err(Error::BadSignature(_))
    ));
}

#[test]
fn structural_validation_rejects_unsafe_manifests() {
    let base = manifest("v1", "http://127.0.0.1:1", 1);
    let reject = |f: &dyn Fn(&mut Manifest)| {
        let mut m = base.clone();
        f(&mut m);
        assert!(m.validate(true).is_err());
    };
    reject(&|m| m.files[0].path = "../escape.json".into());
    reject(&|m| m.files[0].path = "/abs.json".into());
    reject(&|m| m.files[0].path = "weights.pt".into()); // pickle checkpoints are never installable
    reject(&|m| m.files[0].path = "run.py".into());
    reject(&|m| m.files[1].path = m.files[0].path.to_uppercase());
    reject(&|m| m.files[0].sha256 = "xyz".into());
    reject(&|m| m.id = "../x".into());
    reject(&|m| m.licenses.clear());
    // Branch / http URLs are refused outside the loopback test allowance.
    assert!(base.validate(false).is_err());
    let mut m = base.clone();
    m.files
        .iter_mut()
        .for_each(|f| f.url = format!("https://huggingface.co/a/b/resolve/main/{}", f.path));
    assert!(m.validate(false).is_err(), "unpinned revision");
    m.files.iter_mut().for_each(|f| {
        f.url = format!(
            "https://huggingface.co/a/b/resolve/{}/{}",
            "a".repeat(40),
            f.path
        )
    });
    m.validate(false).unwrap();
}

#[test]
fn import_folder_verifies_like_a_download() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir.path().join("reg"));
    let m = signed(&manifest("v1", "http://127.0.0.1:1", 1));
    let bundle = dir.path().join("bundle");
    write_bundle(&bundle, &m, 1);
    assert!(matches!(
        import_dir(&reg, &bundle),
        Err(Error::LicenseNotAccepted(_))
    ));
    accept(&reg, &m);

    // tampered file
    let victim = bundle.join("model.safetensors");
    let orig = std::fs::read(&victim).unwrap();
    let mut bad = orig.clone();
    bad[7] ^= 1;
    std::fs::write(&victim, &bad).unwrap();
    assert!(matches!(
        import_dir(&reg, &bundle),
        Err(Error::HashMismatch { .. })
    ));
    // truncated file
    std::fs::write(&victim, &orig[..orig.len() - 1]).unwrap();
    assert!(matches!(
        import_dir(&reg, &bundle),
        Err(Error::SizeMismatch { .. })
    ));
    // missing file and extra file
    std::fs::remove_file(&victim).unwrap();
    assert!(matches!(import_dir(&reg, &bundle), Err(Error::Missing(_))));
    std::fs::write(&victim, &orig).unwrap();
    std::fs::write(bundle.join("payload.sh"), b"#!/bin/sh").unwrap();
    assert!(matches!(import_dir(&reg, &bundle), Err(Error::Unlisted(_))));
    std::fs::remove_file(bundle.join("payload.sh")).unwrap();
    assert!(reg.list().unwrap().is_empty());

    import_dir(&reg, &bundle).unwrap();
    reg.activate("v1").unwrap();
    assert!(matches!(
        import_dir(&reg, &bundle),
        Err(Error::AlreadyInstalled(_))
    ));
}

#[test]
fn import_rejects_tampered_manifest_and_wrong_key() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir.path().join("reg"));
    let m = signed(&manifest("v1", "http://127.0.0.1:1", 1));
    accept(&reg, &m);
    let bundle = dir.path().join("bundle");
    write_bundle(&bundle, &m, 1);
    let mut v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(bundle.join("manifest.signed.json")).unwrap())
            .unwrap();
    v["manifest"]["precision"] = serde_json::json!("fp32");
    std::fs::write(
        bundle.join("manifest.signed.json"),
        serde_json::to_vec(&v).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        import_dir(&reg, &bundle),
        Err(Error::BadSignature(_))
    ));
    let other = sign(m.manifest(), "test", &key(9));
    std::fs::write(
        bundle.join("manifest.signed.json"),
        serde_json::to_vec(&other).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        import_dir(&reg, &bundle),
        Err(Error::BadSignature(_))
    ));
}

fn write_zip(
    path: &std::path::Path,
    m: &VerifiedManifest,
    tag: u8,
    extra: Option<(&str, &[u8])>,
    corrupt: bool,
) {
    use std::io::Write;
    let mut z = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    let o = zip::write::SimpleFileOptions::default();
    z.start_file("manifest.signed.json", o).unwrap();
    z.write_all(&serde_json::to_vec(m.signed()).unwrap())
        .unwrap();
    for (p, _, mut b) in contents(tag) {
        if corrupt && p == "units.json" {
            b[2] ^= 1;
        }
        z.start_file(p, o).unwrap();
        z.write_all(&b).unwrap();
    }
    if let Some((n, b)) = extra {
        z.start_file(n, o).unwrap();
        z.write_all(b).unwrap();
    }
    z.finish().unwrap();
}

#[test]
fn import_zip_checks_hashes_and_refuses_extra_or_escaping_entries() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir.path().join("reg"));
    let m = signed(&manifest("v1", "http://127.0.0.1:1", 1));
    accept(&reg, &m);
    let z = dir.path().join("m.zip");
    write_zip(&z, &m, 1, None, true);
    assert!(matches!(
        import_zip(&reg, &z),
        Err(Error::HashMismatch { .. })
    ));
    write_zip(&z, &m, 1, Some(("../../evil.json", b"{}")), false);
    assert!(matches!(import_zip(&reg, &z), Err(Error::Unlisted(_))));
    assert!(!dir.path().join("evil.json").exists());
    write_zip(&z, &m, 1, None, false);
    import_zip(&reg, &z).unwrap();
    reg.verify_full("v1").unwrap();
}

fn install_dir(reg: &tingmix_models::Registry, root: &std::path::Path, id: &str, tag: u8) {
    install_serial(reg, root, id, tag as u64, tag).unwrap();
}

#[test]
fn activate_keeps_previous_and_rollback_swaps() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir.path().join("reg"));
    install_dir(&reg, dir.path(), "v1", 1);
    install_dir(&reg, dir.path(), "v2", 2);
    assert!(matches!(reg.rollback(), Err(Error::NoPrevious)));
    assert!(matches!(reg.activate("nope"), Err(Error::NotInstalled(_))));
    reg.activate("v1").unwrap();
    reg.activate("v2").unwrap();
    assert_eq!(
        (
            reg.active().unwrap().as_deref(),
            reg.previous().unwrap().as_deref()
        ),
        (Some("v2"), Some("v1"))
    );
    assert!(reg.remove("v1").is_err(), "previous version is protected");
    assert_eq!(reg.rollback().unwrap(), "v1");
    assert_eq!(
        (
            reg.active().unwrap().as_deref(),
            reg.previous().unwrap().as_deref()
        ),
        (Some("v1"), Some("v2"))
    );
    assert_eq!(reg.rollback().unwrap(), "v2");
}

#[test]
fn rollback_refuses_a_damaged_previous_version() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir.path().join("reg"));
    install_dir(&reg, dir.path(), "v1", 1);
    install_dir(&reg, dir.path(), "v2", 2);
    reg.activate("v1").unwrap();
    reg.activate("v2").unwrap();
    std::fs::remove_file(dir.path().join("reg/versions/v1/units.json")).unwrap();
    assert!(reg.rollback().is_err());
    assert_eq!(
        reg.active().unwrap().as_deref(),
        Some("v2"),
        "failed rollback must not move the pointer"
    );
}

#[test]
fn interrupted_activation_leaves_a_consistent_pointer() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("reg");
    let reg = registry(&root);
    install_dir(&reg, dir.path(), "v1", 1);
    install_dir(&reg, dir.path(), "v2", 2);
    reg.activate("v1").unwrap();

    // Crash while writing the new pointer: a torn temp file next to the intact old pointer.
    std::fs::write(root.join("active.tmp"), "v2").unwrap();
    std::fs::write(root.join("previous.tmp"), "").unwrap();
    let reg = registry(&root);
    assert_eq!(reg.active().unwrap().as_deref(), Some("v1"));
    assert!(
        !root.join("active.tmp").exists(),
        "stale temp pointers are cleaned on open"
    );
    assert_eq!(reg.open_active().unwrap().unwrap().manifest.id, "v1");

    // Crash after `previous` was written but before `active`: previous == active, activation simply repeats.
    std::fs::write(root.join("previous"), "v1\n").unwrap();
    reg.activate("v2").unwrap();
    assert_eq!(
        (
            reg.active().unwrap().as_deref(),
            reg.previous().unwrap().as_deref()
        ),
        (Some("v2"), Some("v1"))
    );

    // Crash during install leaves only staging, never a half-visible version.
    let m3 = signed(&manifest("v3", "http://127.0.0.1:1", 3));
    accept(&reg, &m3);
    std::fs::create_dir_all(root.join("tmp/v3")).unwrap();
    std::fs::write(root.join("tmp/v3/units.json"), b"half").unwrap();
    assert_eq!(reg.list().unwrap(), vec!["v1", "v2"]);
    assert!(reg.open_model("v3").is_err());
}

#[test]
fn installed_files_are_rechecked_when_opened() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("reg");
    let reg = registry(&root);
    install_dir(&reg, dir.path(), "v1", 1);
    reg.activate("v1").unwrap();
    std::fs::write(root.join("versions/v1/model.safetensors"), b"short").unwrap();
    assert!(matches!(reg.open_active(), Err(Error::SizeMismatch { .. })));
    // same-size flip is caught by the full check
    install_dir(&reg, dir.path(), "v2", 2);
    let p = root.join("versions/v2/model.safetensors");
    let mut b = std::fs::read(&p).unwrap();
    b[0] ^= 1;
    std::fs::write(&p, b).unwrap();
    assert!(reg.open_model("v2").is_ok());
    assert!(matches!(
        reg.verify_full("v2"),
        Err(Error::HashMismatch { .. })
    ));
}

#[test]
fn shipped_fixture_is_signed_by_the_embedded_key_and_matches_hf_sha256sums() {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/tingmix-large-source.signed.json"
    ))
    .unwrap();
    let v = Verifier::embedded().parse(&bytes).unwrap();
    let m = v.manifest();
    m.validate(false).unwrap();
    assert_eq!(m.revision, "75071a1260fb21874f150df6ebd53716ac423f13");
    assert_eq!(
        m.base_model.as_ref().unwrap().revision,
        "914f7f89142e33e77833254d9c9b90c3cef7303b"
    );
    // tc3oliver/TingMix-Large@75071a1 SHA256SUMS
    for (p, h) in [
        (
            "config.json",
            "eac5da8ce273202417906279698130b09c810111d75f82882ac323102ceb7431",
        ),
        (
            "encoder_delta.safetensors",
            "1b155cfeec560a07b975eea14eee515aa06d75074256989fc510129fab51d9cb",
        ),
        (
            "model.safetensors",
            "67a06c21d757fdfa0b347a84942097e7bf6315fc4b9d3437b58b008cc5b7e271",
        ),
        (
            "tingmix.json",
            "5b818f0dcd7752539a449f49d834a684518e3feac006ceb929f8d6548d6beb6e",
        ),
        (
            "tokenizer_bpe.model",
            "959e7af76bb54482be709575e7ee3da49dce7f293f1e6e18cc99baaf2eeb0c7a",
        ),
        (
            "units.json",
            "dbc1aba97abeeae49dc7556531a2f670d4dc5f39705b54ca28e84b8de9187756",
        ),
    ] {
        assert_eq!(
            m.files.iter().find(|f| f.path == p).unwrap().sha256,
            h,
            "{p}"
        );
    }
    let base: Vec<_> = m.files.iter().filter(|f| f.component == "base").collect();
    assert!(base.iter().all(|f| f
        .url
        .contains("google/embeddinggemma-2/resolve/914f7f89142e33e77833254d9c9b90c3cef7303b/")));
    assert!(m
        .licenses
        .iter()
        .any(|l| l.id == "google-embeddinggemma-2" && l.acceptance_required));
}

fn install_serial(
    reg: &tingmix_models::Registry,
    root: &std::path::Path,
    id: &str,
    serial: u64,
    tag: u8,
) -> tingmix_models::Result<tingmix_models::InstalledModel> {
    let mut m = manifest(id, "http://127.0.0.1:1", tag);
    m.serial = serial;
    let m = signed(&m);
    accept(reg, &m);
    let b = root.join(format!("bundle-{id}"));
    write_bundle(&b, &m, tag);
    import_dir(reg, &b)
}

#[test]
fn replayed_older_release_is_rejected_even_after_removal() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir.path().join("reg"));
    install_serial(&reg, dir.path(), "r1", 1, 1).unwrap();
    install_serial(&reg, dir.path(), "r2", 2, 2).unwrap();
    assert_eq!(
        reg.high_water("tingmix-large.mlx.fp16").unwrap(),
        (2, Some("r2".into()))
    );

    // A validly signed old manifest (serial 1, new id) is a replay.
    let e = install_serial(&reg, dir.path(), "r1-again", 1, 1).unwrap_err();
    assert!(
        matches!(
            e,
            Error::Replay {
                serial: 1,
                high_water: 2,
                ..
            }
        ),
        "{e}"
    );
    // Same serial under a different id is ambiguous, also rejected.
    assert!(matches!(
        install_serial(&reg, dir.path(), "r2-fork", 2, 3),
        Err(Error::Replay { .. })
    ));
    assert!(!dir.path().join("reg/versions/r1-again").exists());

    // The mark survives removal of the newest version and a new Registry object (restart).
    reg.activate("r1").unwrap();
    reg.remove("r2").unwrap();
    let reg = registry(&dir.path().join("reg"));
    assert!(matches!(
        install_serial(&reg, dir.path(), "r1-again", 1, 1),
        Err(Error::Replay { .. })
    ));
    assert!(install_serial(&reg, dir.path(), "r3", 3, 4).is_ok());

    // Explicit user rollback to a locally installed older version is not a replay.
    reg.activate("r3").unwrap();
    assert_eq!(reg.rollback().unwrap(), "r1");
    assert_eq!(reg.open_active().unwrap().unwrap().manifest.serial, 1);
}

#[test]
fn other_lineages_have_independent_serials() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir.path().join("reg"));
    install_serial(&reg, dir.path(), "a", 5, 1).unwrap();
    let mut m = manifest("b", "http://127.0.0.1:1", 2);
    m.backend = "coreml".into();
    m.serial = 1;
    let m = signed(&m);
    accept(&reg, &m);
    write_bundle(&dir.path().join("bb"), &m, 2);
    import_dir(&reg, &dir.path().join("bb")).unwrap();
    let mut z = manifest("zero", "http://127.0.0.1:1", 1);
    z.serial = 0;
    assert!(z.validate(true).is_err());
}

#[test]
fn import_only_files_cannot_be_downloaded_but_can_be_imported() {
    let srv = serve(&[("v1", 1)]);
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir.path().join("reg"));
    let mut m = manifest("v1", &srv.url, 1);
    m.files[0].url = String::new(); // unpublished converted artifact
    let m = signed(&m);
    accept(&reg, &m);
    let e = Downloader::new()
        .install(&reg, &m, &no_cancel(), |_| {})
        .unwrap_err();
    assert!(
        matches!(e, Error::ImportOnly(ref p) if p == "model.safetensors"),
        "{e}"
    );
    assert!(srv.state.lock().unwrap().requests.is_empty());
    write_bundle(&dir.path().join("b"), &m, 1);
    import_dir(&reg, &dir.path().join("b")).unwrap();
}

#[test]
fn shipped_mlx_manifest_has_provenance_and_licence_acceptance() {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/tingmix-large-mlx-fp32.signed.json"
    ))
    .unwrap();
    let v = Verifier::embedded().parse(&bytes).unwrap();
    let m = v.manifest();
    m.validate(false).unwrap();
    assert_eq!(
        (
            m.backend.as_str(),
            m.precision.as_str(),
            m.compat.status.as_str()
        ),
        ("mlx", "fp32", "EXPERIMENTAL")
    );
    let p = m.provenance.as_ref().unwrap();
    assert!(
        p.modified
            && p.converter_commit.len() == 40
            && p.sources
                .iter()
                .any(|s| s.revision == "914f7f89142e33e77833254d9c9b90c3cef7303b")
    );
    let model = m.files.iter().find(|f| f.component == "model").unwrap();
    assert_eq!((model.size, model.url.as_str()), (1_308_067_132, ""));
    assert_eq!(
        model.sha256,
        "9737153e4631bc773481ec458f34f242cc7800ccc6ae6ea1c509d895181fc7ec"
    );
    let g = m
        .licenses
        .iter()
        .find(|l| l.id == "google-embeddinggemma-2")
        .unwrap();
    assert!(g.acceptance_required && g.notes.contains("Prohibited Use Policy"));
}

#[test]
fn changed_licence_url_is_asked_again() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(dir.path());
    let mut raw = manifest("v1", "http://127.0.0.1:1", 1);
    let m = signed(&raw);
    assert_eq!(reg.missing_acceptances(m.manifest()), ["model-terms"]);
    accept(&reg, &m);
    assert!(reg.missing_acceptances(m.manifest()).is_empty());
    raw.licenses[0].url = "https://example.invalid/new-terms".into();
    let changed = signed(&raw);
    assert_eq!(reg.missing_acceptances(changed.manifest()), ["model-terms"]);
}
