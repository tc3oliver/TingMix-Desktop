mod common;
use std::time::Duration;

use common::*;
use tingmix_models::import::import_dir;
use tingmix_models::manifest::*;
use tingmix_models::probe::*;
use tingmix_models::select::*;
use tingmix_models::{InstalledModel, Registry};

struct Fake {
    version: Option<String>,
    device: String,
    text: String,
    panic: bool,
    loads: u32,
}

impl ProbeTarget for Fake {
    fn backend(&self) -> &str {
        "mlx"
    }
    fn backend_version(&self) -> Option<String> {
        self.version.clone()
    }
    fn device(&self) -> String {
        self.device.clone()
    }
    fn decode(&mut self, _m: &InstalledModel, wav: &[f32]) -> Result<String, String> {
        assert_eq!(wav.len(), 16_000);
        self.loads += 1;
        if self.panic {
            panic!("native library blew up");
        }
        std::thread::sleep(Duration::from_millis(5));
        Ok(self.text.clone())
    }
}

fn fake(text: &str) -> Fake {
    Fake {
        version: Some("1.2".into()),
        device: "gpu:test".into(),
        text: text.into(),
        panic: false,
        loads: 0,
    }
}

fn setup() -> (tempfile::TempDir, Registry, InstalledModel) {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir.path().join("reg"));
    let m = signed(&manifest("v1", "http://127.0.0.1:1", 1));
    reg.record_acceptance(&m, "model-terms").unwrap();
    write_bundle(&dir.path().join("b"), &m, 1);
    let model = import_dir(&reg, &dir.path().join("b")).unwrap();
    (dir, reg, model)
}

static WAV: [f32; 16_000] = [0.0; 16_000];

fn fx<'a>() -> ProbeFixture<'a> {
    ProbeFixture {
        name: "zh_02s",
        wav: &WAV,
        sample_rate: 16_000,
        expected_text: "今天天氣很好",
    }
}

#[test]
fn passing_probe_is_recorded_and_consulted() {
    let (_d, reg, model) = setup();
    let mut t = fake("今天天氣很好 ");
    let r = run_probe(reg.root(), &mut t, &model, &fx(), 3).unwrap();
    assert!(r.passed && r.detail.is_empty(), "{r:?}");
    assert_eq!(t.loads, 4, "one load/first decode + 3 timed");
    assert!(r.rtf_x1000.unwrap() >= 5, "5 ms for 1 s of audio");
    assert_eq!(
        (
            r.backend_version.as_str(),
            r.device.as_str(),
            r.model_id.as_str()
        ),
        ("1.2", "gpu:test", "v1")
    );
    let p = fresh(reg.root(), "mlx", &model.manifest, Some("1.2"), "gpu:test").unwrap();
    assert!(p.passed && p.rtf_x1000 == r.rtf_x1000);
}

#[test]
fn probe_is_invalidated_when_anything_it_measured_changes() {
    let (_d, reg, model) = setup();
    run_probe(reg.root(), &mut fake("今天天氣很好"), &model, &fx(), 1).unwrap();
    let c = |v: Option<&str>, dev: &str, m: &Manifest| consult(reg.root(), "mlx", m, v, dev);
    assert!(matches!(
        c(Some("1.2"), "gpu:test", &model.manifest),
        Consulted::Fresh(_)
    ));
    assert!(
        matches!(c(Some("1.3"), "gpu:test", &model.manifest), Consulted::Stale(ref w) if w.contains("backend version"))
    );
    assert!(
        matches!(c(None, "gpu:test", &model.manifest), Consulted::Stale(_)),
        "runtime vanished"
    );
    assert!(
        matches!(c(Some("1.2"), "gpu:other", &model.manifest), Consulted::Stale(ref w) if w.contains("device"))
    );
    let mut other = model.manifest.clone();
    other.files[0].sha256 = sha256_hex(b"different model");
    assert_eq!(
        c(Some("1.2"), "gpu:test", &other),
        Consulted::Missing,
        "different model hash = different record"
    );
    // OS / arch changes are caught by editing the stored record.
    let f = std::fs::read_dir(reg.root().join("probes"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let s = std::fs::read_to_string(&f).unwrap().replace(
        &format!("\"os_version\": \"{}\"", sysinfo_os()),
        "\"os_version\": \"older\"",
    );
    std::fs::write(&f, s).unwrap();
    assert!(
        matches!(c(Some("1.2"), "gpu:test", &model.manifest), Consulted::Stale(ref w) if w.contains("OS"))
    );
}

fn sysinfo_os() -> String {
    let (_d, reg, model) = setup();
    run_probe(reg.root(), &mut fake("x"), &model, &fx(), 1)
        .unwrap()
        .os_version
}

#[test]
fn wrong_text_error_and_panic_fail_the_probe() {
    let (_d, reg, model) = setup();
    let r = run_probe(reg.root(), &mut fake("something else"), &model, &fx(), 1).unwrap();
    assert!(
        !r.passed && r.rtf_x1000.is_none() && r.detail.contains("expected"),
        "{r:?}"
    );
    let mut t = fake("今天天氣很好");
    t.panic = true;
    let r = run_probe(reg.root(), &mut t, &model, &fx(), 1).unwrap();
    assert!(!r.passed && r.detail.contains("panicked"));
    let p = fresh(reg.root(), "mlx", &model.manifest, Some("1.2"), "gpu:test").unwrap();
    assert!(
        !p.passed,
        "a failed probe is persisted so selection can reject with a reason"
    );
}

#[test]
fn select_uses_the_per_model_probe() {
    let (_d, reg, model) = setup();
    let caps = tingmix_models::caps::Capabilities {
        os: "macos".into(),
        arch: "aarch64".into(),
        cpu_features: vec![],
        ram_bytes: 64 << 30,
        apple_chip: Some(tingmix_models::caps::AppleChip {
            name: "Apple M4 Max".into(),
            generation: 4,
        }),
        gpu_vendor_hint: None,
    };
    let regs = [BackendReg {
        backend: "mlx".into(),
        validation: Validation::Validated,
        is_cpu: false,
        runtime_version: Some("1.2".into()),
        probe: None,
    }];
    let mut m = model.manifest.clone();
    m.runtime.min_version = "1.0".into();
    let lookup = |m: &Manifest| fresh(reg.root(), "mlx", m, Some("1.2"), "gpu:test");
    // No probe yet: rejected for that reason. The manifest hash changed (min_version edit), so use a stored probe for it.
    assert!(select_probed(&[m.clone()], &caps, &regs, None, &lookup).is_err());
    let mm = InstalledModel {
        dir: model.dir.clone(),
        manifest: m.clone(),
    };
    run_probe(reg.root(), &mut fake("今天天氣很好"), &mm, &fx(), 1).unwrap();
    let s = select_probed(&[m], &caps, &regs, None, &lookup).unwrap();
    assert_eq!(s.manifest_id, "v1");
}
