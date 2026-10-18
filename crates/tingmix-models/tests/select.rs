mod common;
use common::manifest;
use tingmix_models::caps::*;
use tingmix_models::select::*;

fn caps() -> Capabilities {
    Capabilities {
        os: "macos".into(),
        arch: "aarch64".into(),
        cpu_features: vec!["neon".into()],
        ram_bytes: 64 << 30,
        apple_chip: parse_apple_chip("Apple M4 Max"),
        gpu_vendor_hint: Some("Apple".into()),
    }
}

fn reg(
    backend: &str,
    validation: Validation,
    is_cpu: bool,
    ver: Option<&str>,
    rtf: Option<u32>,
) -> BackendReg {
    BackendReg {
        backend: backend.into(),
        validation,
        is_cpu,
        runtime_version: ver.map(Into::into),
        probe: rtf.map(|r| Probe {
            passed: true,
            rtf_x1000: Some(r),
        }),
    }
}

fn mf(id: &str, backend: &str) -> tingmix_models::Manifest {
    let mut m = manifest(id, "http://127.0.0.1:1", 1);
    m.backend = backend.into();
    m.runtime.min_version = "1.0".into();
    m
}

#[test]
fn picks_fastest_validated_and_explains_the_cpu_comparison() {
    let ms = [mf("cpu", "onnx-cpu"), mf("gpu", "mlx"), mf("exp", "coreml")];
    let regs = [
        reg(
            "onnx-cpu",
            Validation::Validated,
            true,
            Some("1.20"),
            Some(300),
        ),
        reg("mlx", Validation::Validated, false, Some("1.2"), Some(12)),
        reg(
            "coreml",
            Validation::Experimental,
            false,
            Some("1.0"),
            Some(5),
        ),
    ];
    let s = select(&ms, &caps(), &regs, None).unwrap();
    assert_eq!(s.manifest_id, "gpu");
    assert!(
        s.reason.contains("faster than the best CPU path"),
        "{}",
        s.reason
    );
    assert!(s
        .rejected
        .iter()
        .any(|(id, why)| id == "exp" && why.contains("EXPERIMENTAL")));
}

#[test]
fn rejections_are_specific() {
    let mut big = mf("big", "mlx");
    big.requirements.min_ram_bytes = 128 << 30;
    let mut old = mf("oldchip", "mlx");
    old.requirements.apple_chip_min_generation = Some(5);
    let mut feat = mf("feat", "mlx");
    feat.requirements.cpu_features = vec!["avx2".into()];
    let mut win = mf("win", "mlx");
    win.os = "windows".into();
    let nover = mf("noruntime", "ort");
    let tooold = mf("tooold", "old");
    let noprobe = mf("noprobe", "np");
    let regs = [
        reg("mlx", Validation::Validated, false, Some("1.2"), Some(10)),
        reg("ort", Validation::Validated, true, None, Some(10)),
        reg("old", Validation::Validated, true, Some("0.9"), Some(10)),
        reg("np", Validation::Validated, true, Some("2.0"), None),
    ];
    let Err(rej) = select(
        &[big, old, feat, win, nover, tooold, noprobe],
        &caps(),
        &regs,
        None,
    ) else {
        panic!("nothing is eligible")
    };
    let why = |id: &str| rej.iter().find(|r| r.0 == id).unwrap().1.clone();
    assert!(why("big").contains("RAM"));
    assert!(why("oldchip").contains("M5"));
    assert!(why("feat").contains("avx2"));
    assert!(why("win").contains("built for windows"));
    assert!(why("noruntime").contains("not available"));
    assert!(why("tooold").contains("older than"));
    assert!(why("noprobe").contains("probe"));
}

#[test]
fn failed_probe_and_gpu_vendor_do_not_decide() {
    let ms = [mf("gpu", "mlx"), mf("cpu", "onnx-cpu")];
    let mut regs = [
        reg("mlx", Validation::Validated, false, Some("1.2"), Some(10)),
        reg(
            "onnx-cpu",
            Validation::Validated,
            true,
            Some("1.0"),
            Some(300),
        ),
    ];
    regs[0].probe = Some(Probe {
        passed: false,
        rtf_x1000: None,
    });
    let s = select(&ms, &caps(), &regs, None).unwrap();
    assert_eq!(s.manifest_id, "cpu");
    // Same result with a different vendor hint: the hint is display-only.
    let mut c = caps();
    c.gpu_vendor_hint = Some("SomeOtherVendor".into());
    assert_eq!(select(&ms, &c, &regs, None).unwrap().manifest_id, "cpu");
}

#[test]
fn known_good_is_kept_until_the_machine_or_runtime_changes() {
    let ms = [mf("gpu", "mlx"), mf("cpu", "onnx-cpu")];
    let regs = [
        reg("mlx", Validation::Validated, false, Some("1.2"), Some(10)),
        reg(
            "onnx-cpu",
            Validation::Validated,
            true,
            Some("1.0"),
            Some(5),
        ),
    ];
    let dir = tempfile::tempdir().unwrap();
    // CPU is "faster" here, but the persisted known-good gpu choice stays while nothing changed.
    let first = select(&ms, &caps(), &regs, None).unwrap();
    assert_eq!(first.manifest_id, "cpu");
    let kg = KnownGood {
        manifest_id: "gpu".into(),
        backend: "mlx".into(),
        runtime_version: "1.2".into(),
        fingerprint: caps().fingerprint(),
    };
    let s = select(&ms, &caps(), &regs, Some(&kg)).unwrap();
    assert_eq!(
        (
            s.manifest_id.as_str(),
            s.reason.starts_with("last known-good")
        ),
        ("gpu", true)
    );

    let mut other = caps();
    other.apple_chip = parse_apple_chip("Apple M2");
    let s = select(&ms, &other, &regs, Some(&kg)).unwrap();
    assert!(s.reason.contains("hardware or OS changed"), "{}", s.reason);

    let mut regs2 = regs.clone();
    regs2[0].runtime_version = None; // driver / runtime vanished
    let s = select(&ms, &caps(), &regs2, Some(&kg)).unwrap();
    assert_eq!(s.manifest_id, "cpu");
    assert!(
        s.reason.contains("no longer usable") && s.reason.contains("not available"),
        "{}",
        s.reason
    );

    save_known_good(dir.path(), &first, &caps()).unwrap();
    let back = load_known_good(dir.path()).unwrap().unwrap();
    assert_eq!(
        (back.manifest_id.as_str(), back.fingerprint.as_str()),
        ("cpu", caps().fingerprint().as_str())
    );
    assert!(load_known_good(&dir.path().join("none")).unwrap().is_none());
}

#[test]
fn detect_and_parse() {
    assert_eq!(parse_apple_chip("Apple M4 Max\n").unwrap().generation, 4);
    assert!(parse_apple_chip("Intel(R) Core(TM) i9").is_none());
    let c = Capabilities::detect();
    assert!(c.ram_bytes > 0 && !c.os.is_empty());
}

#[test]
fn requirements_met_rejects_a_newer_app_version_a_small_machine_and_a_missing_chip() {
    let mut m = mf("a", "mlx");
    assert!(requirements_met(&m, &caps(), "0.1.0").is_ok());
    m.min_app_version = "0.2.0".into();
    assert!(requirements_met(&m, &caps(), "0.1.9")
        .unwrap_err()
        .contains("0.2.0"));
    assert!(requirements_met(&m, &caps(), "0.2.0").is_ok());
    m.requirements.min_ram_bytes = 128 << 30;
    assert!(requirements_met(&m, &caps(), "0.2.0")
        .unwrap_err()
        .contains("RAM"));
    m.requirements.min_ram_bytes = 0;
    m.requirements.apple_chip_min_generation = Some(5);
    assert!(requirements_met(&m, &caps(), "0.2.0")
        .unwrap_err()
        .contains("M5"));
}
