//! Host-level end-to-end test of the MLX Large path, ignored unless `TINGMIX_E2E_MLX_PACKAGE` points at a package folder:
//! signed manifest + the files it lists (see docs/architecture/app.md). Steps: offline import with licence acceptance ->
//! automatic activate + probe after the import -> selection picks `mlx` on its own
//! -> file transcription equals the oracle text -> live session on replayed audio emits partial and final events.
//!
//! cargo test needs `--features mlx`, `MLX_PREFIX`, `TINGMIX_MLX_DEV_RPATH=1`, and `TINGMIX_MLX_WORKER=<tingmix-mlx-worker>`.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tingmix_core::{Event, Reducer};

use crate::backends::{self, Runtimes};
use crate::dto::{FileBody, JobState, LiveMsg};
use crate::live::{Live, PcmSource, Persist, Sink};
use crate::models::{Models, Notify};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Waits until the latest operation of `kind` is finished (a probe starts only after the import's callback ran).
fn wait_op(models: &Models, kind: &str, what: &str) {
    let t = Instant::now();
    loop {
        let op = models.overview().op.expect("an operation was started");
        match (op.kind.as_str() == kind, op.state.as_str()) {
            (true, "done") => return,
            (_, "failed" | "cancelled") => panic!("{what}: {} {:?}", op.state, op.message),
            _ if t.elapsed() < Duration::from_secs(600) => {
                std::thread::sleep(Duration::from_millis(200))
            }
            _ => panic!("{what}: timed out in {} ({})", op.kind, op.state),
        }
    }
}

struct Replay {
    data: Vec<f32>,
    pos: usize,
}

impl PcmSource for Replay {
    fn sample_rate(&self) -> u32 {
        16_000
    }
    fn device(&self) -> String {
        "replayed wav".into()
    }
    fn read(&mut self, out: &mut [f32]) -> usize {
        // ~4x real time, so partials are produced while audio is still arriving
        std::thread::sleep(Duration::from_millis(250 * out.len() as u64 / 64_000));
        let n = out.len().min(self.data.len() - self.pos);
        out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        n
    }
    fn overflowed(&self) -> u64 {
        0
    }
    fn poll(&mut self) -> Result<Vec<String>, String> {
        Ok(vec![])
    }
}

fn join(parts: Vec<String>) -> String {
    if parts.concat().contains(' ') {
        parts.join(" ")
    } else {
        parts.concat()
    }
}

/// The directory a test registry lives in. `TINGMIX_E2E_HOME` places it where an app run with HOME=<it> would look
/// (`<home>/Library/Application Support/app.tingmix.desktop`), otherwise it is a temp directory.
/// Text without spacing or case: engines that write punctuation differ there between decodes of the same words.
fn squash(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn scratch_dir() -> (PathBuf, Option<PathBuf>) {
    let keep = std::env::var_os("TINGMIX_E2E_HOME").map(PathBuf::from);
    let dir = keep
        .as_ref()
        .map(|h| h.join("Library/Application Support/app.tingmix.desktop"))
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("tingmix-e2e-{}", std::process::id()))
        });
    // never wipe a directory this test did not create: it must carry our marker (a mistyped TINGMIX_E2E_HOME that points at
    // the real home, or any other existing directory, is refused)
    let marker = dir.join(".tingmix-e2e-scratch");
    if dir.exists() {
        assert!(
            marker.is_file(),
            "refusing to wipe {}: it exists and was not created by this test",
            dir.display()
        );
        assert!(
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .as_ref()
                .and_then(|h| h.canonicalize().ok())
                != keep.as_ref().and_then(|k| k.canonicalize().ok()),
            "TINGMIX_E2E_HOME must not be the real home directory"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        &marker,
        b"scratch directory of the e2e test; safe to delete\n",
    )
    .unwrap();
    (dir, keep)
}

#[test]
#[ignore = "needs the MLX package folder, a --features mlx build with MLX_PREFIX, and ~2 GB of RAM"]
fn mlx_import_probe_select_transcribe_live() {
    let package =
        PathBuf::from(std::env::var("TINGMIX_E2E_MLX_PACKAGE").expect("TINGMIX_E2E_MLX_PACKAGE"));
    let (dir, keep) = scratch_dir();
    let models = Arc::new(Models::open(dir.join("models")));
    let rt = Runtimes::default();
    let notify: Notify = Arc::new(|_| {});

    // 1. offline import, licence step first
    let pending = models
        .begin_import(package, false, "e2e".into())
        .expect("verify manifest");
    let required: Vec<String> = pending
        .licenses
        .iter()
        .filter(|l| l.acceptance_required)
        .map(|l| l.id.clone())
        .collect();
    assert_eq!(required.len(), 2, "{:?}", pending.licenses);
    let err = models.install("e2e", &[], notify.clone()).unwrap_err();
    assert!(err.contains("must be accepted"), "{err}");
    // the host's notify callback does this when an import finishes: activate (nothing is active yet) and test
    let (m2, rt2) = (models.clone(), rt.clone());
    let on_op: Notify = Arc::new(move |v| {
        if v.state == "done" && v.kind == "import" {
            backends::onboard_core(&m2, &rt2, &v.name, Arc::new(|_| {})).expect("onboard");
        }
    });
    models
        .install("e2e", &required, on_op)
        .expect("start import");
    // the import callback starts the probe before a poll could see the import as done, so only the probe is awaited
    wait_op(&models, "probe", "automatic probe");
    let ov = models.overview();
    assert_eq!(ov.installed.len(), 1);
    assert!(
        ov.installed[0].active,
        "the first imported model becomes the active one"
    );

    // 2. selection: the best probed engine is chosen on its own, no setting involved
    let on = backends::decide(&models, &rt);
    assert_eq!(
        on.chosen.as_ref().map(|c| c.1.as_str()),
        Some("mlx"),
        "{}",
        on.reason
    );
    let info = backends::info(&models, &rt);
    assert!(!info.mock && info.usable, "{}", info.label);
    println!("selection: {}", on.reason);

    // 4. file transcription == oracle text
    let expected: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root().join("tests/fixtures/large/expected_texts.json")).unwrap(),
    )
    .unwrap();
    let want = expected["clips"]["mix_08s"]["beam_lm"]
        .as_str()
        .unwrap()
        .to_string();
    let wav = root().join("tests/fixtures/audio/mix_08s.wav");
    let resolved = backends::resolve(&models, &rt).expect("load mlx backend");
    let mut reducer = Reducer::new();
    let (mut partials, mut finals_seen) = (0, 0);
    let out = crate::files::run(
        &wav,
        resolved.backend,
        &AtomicBool::new(false),
        &mut reducer,
        &mut |b| {
            if let FileBody::Event { event } = b {
                match event {
                    Event::Partial(_) => partials += 1,
                    Event::Final(_) => finals_seen += 1,
                }
            }
        },
    );
    assert_eq!(out.state, JobState::Done, "{:?}", out.message);
    let got = join(reducer.finals().map(|f| f.text.clone()).collect());
    println!("file: {partials} partials, {finals_seen} finals\n  got : {got}\n  want: {want}");
    assert_eq!(got, want);

    // 5. live session on replayed audio
    live_session(&models, &rt, &wav);
    if keep.is_none() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// A live session on `wav` replayed at about 4x real time must emit partials and at least one non-empty final.
fn live_session(models: &Models, rt: &Runtimes, wav: &Path) -> Vec<String> {
    let resolved = backends::resolve(models, rt).expect("load backend");
    let log: Arc<Mutex<Vec<LiveMsg>>> = Arc::default();
    let l = log.clone();
    let sink: Sink = Arc::new(move |m| l.lock().unwrap().push(m));
    let persist: Persist = Arc::new(|_, _| {});
    let data = tingmix_audio::decode_file(wav).unwrap();
    let mut live = Live::start(
        "live-e2e".into(),
        resolved.label,
        resolved.backend,
        move || Ok(Box::new(Replay { data, pos: 0 }) as Box<dyn PcmSource>),
        sink,
        persist,
    )
    .expect("start live");
    let t = Instant::now();
    while live.shared.snapshot().finals.is_empty() && t.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(100));
    }
    live.stop();
    let msgs = log.lock().unwrap();
    let n_partial = msgs
        .iter()
        .filter(|m| {
            matches!(
                m,
                LiveMsg::Event {
                    event: Event::Partial(_),
                    ..
                }
            )
        })
        .count();
    let snap = live.shared.snapshot();
    let finals: Vec<String> = snap.finals.iter().map(|f| f.text.clone()).collect();
    let st = live.shared.status();
    println!(
        "live: {n_partial} partials, {} finals: {finals:?}; status {:?} message {:?} samples {}",
        finals.len(),
        st.state,
        st.message,
        st.samples_in
    );
    assert!(n_partial > 0 && !finals.is_empty());
    assert!(finals.iter().any(|f| !f.is_empty()));
    finals
}

/// An ONNX family through the host, from the export directory `src` and the catalog manifest in
/// `crates/tingmix-runtime-onnx/fixtures/<manifest>` signed with a throwaway key: offline import -> activate + probe -> selection
/// picks it -> file transcription of mix_08s (must contain `want`, in Traditional characters) -> live session.
fn onnx_family_e2e(manifest: &str, src: &Path, want: &str) {
    let ort = PathBuf::from(std::env::var("TINGMIX_ORT_DYLIB").expect("TINGMIX_ORT_DYLIB"));
    tingmix_runtime_onnx::set_runtime_library(ort.clone());
    let (dir, keep) = scratch_dir();
    let key = ed25519_dalek::SigningKey::from_bytes(&[7; 32]);
    let manifest: tingmix_models::Manifest = serde_json::from_slice(
        &std::fs::read(
            root()
                .join("crates/tingmix-runtime-onnx/fixtures")
                .join(manifest),
        )
        .unwrap(),
    )
    .unwrap();
    let pkg = dir.join("package");
    std::fs::create_dir_all(&pkg).unwrap();
    for f in &manifest.files {
        // same volume: a hard link, so the model is not copied twice
        std::fs::hard_link(src.join(&f.path), pkg.join(&f.path))
            .or_else(|_| std::fs::copy(src.join(&f.path), pkg.join(&f.path)).map(drop))
            .unwrap();
    }
    std::fs::write(
        pkg.join(tingmix_models::store::MANIFEST_FILE),
        serde_json::to_vec(&tingmix_models::manifest::sign(&manifest, "e2e", &key)).unwrap(),
    )
    .unwrap();
    let verifier =
        tingmix_models::Verifier::new(&[("e2e", key.verifying_key().to_bytes())]).unwrap();
    let models = Arc::new(Models::open_with(dir.join("models"), verifier, false));
    let rt = Runtimes {
        ort_library: Some(ort),
        onnx_worker: None,
    };

    let pending = models
        .begin_import(pkg, false, "e2e".into())
        .expect("verify manifest");
    let required: Vec<String> = pending
        .licenses
        .iter()
        .filter(|l| l.acceptance_required)
        .map(|l| l.id.clone())
        .collect();
    let (m2, rt2) = (models.clone(), rt.clone());
    let on_op: Notify = Arc::new(move |v| {
        if v.state == "done" && v.kind == "import" {
            backends::onboard_core(&m2, &rt2, &v.name, Arc::new(|_| {})).expect("onboard");
        }
    });
    models
        .install("e2e", &required, on_op)
        .expect("start import");
    wait_op(&models, "probe", "automatic probe");
    let on = backends::decide(&models, &rt);
    assert_eq!(
        on.chosen.as_ref().map(|c| c.0.as_str()),
        Some(manifest.id.as_str()),
        "{}",
        on.reason
    );

    let wav = root().join("tests/fixtures/audio/mix_08s.wav");
    let resolved = backends::resolve(&models, &rt).expect("load backend");
    let mut reducer = Reducer::new();
    let t = Instant::now();
    let out = crate::files::run(
        &wav,
        resolved.backend,
        &AtomicBool::new(false),
        &mut reducer,
        &mut |_| {},
    );
    assert_eq!(out.state, JobState::Done, "{:?}", out.message);
    let got = join(reducer.finals().map(|f| f.text.clone()).collect());
    println!("file ({:.2}s): {got}", t.elapsed().as_secs_f64());
    assert!(squash(&got).contains(&squash(want)), "{got}");

    let finals = live_session(&models, &rt, &wav);
    assert!(finals.iter().all(|f| !f.contains('这')), "{finals:?}");
    if keep.is_none() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// SenseVoice-Small. Env: TINGMIX_E2E_SENSEVOICE_DIR (the sherpa-onnx export: model.int8.onnx, tokens.txt), TINGMIX_ORT_DYLIB.
#[test]
#[ignore = "needs the SenseVoice export and the ONNX Runtime library (see the doc comment)"]
fn sensevoice_import_probe_select_transcribe_live() {
    let src = std::env::var("TINGMIX_E2E_SENSEVOICE_DIR").expect("TINGMIX_E2E_SENSEVOICE_DIR");
    // the oracle writes 我觉得这个deadline有点太赶了; the app converts it to Traditional characters
    onnx_family_e2e(
        "sensevoice-small-onnx-int8.manifest.json",
        Path::new(&src),
        "我覺得這個deadline有點太趕了",
    );
}

/// Qwen3-ASR-0.6B int4. Env: TINGMIX_E2E_QWEN3_DIR (the andrewleech/qwen3-asr-0.6b-onnx files the manifest lists),
/// TINGMIX_ORT_DYLIB.
#[test]
#[ignore = "needs the Qwen3-ASR export and the ONNX Runtime library (see the doc comment)"]
fn qwen3_import_probe_select_transcribe_live() {
    let src = std::env::var("TINGMIX_E2E_QWEN3_DIR").expect("TINGMIX_E2E_QWEN3_DIR");
    onnx_family_e2e(
        "qwen3-asr-0.6b-onnx-int4.manifest.json",
        Path::new(&src),
        "我覺得這個 deadline 有點太趕了",
    );
}

/// The real first-run flow against the published catalog: fetch, accept the licences (what the user's tick does), download
/// the model first run offers (Qwen3-ASR, about 2 GB, on a machine with 8 GB of RAM), verify, activate, test, select and transcribe `mix_08s.wav`. Prints a timing line per stage.
/// Needs the network and TINGMIX_E2E_REAL_FIRST_RUN=1. macOS: `--features mlx`, MLX_PREFIX, TINGMIX_MLX_WORKER. Windows (ONNX):
/// TINGMIX_ORT_DYLIB (absolute path of onnxruntime.dll) and TINGMIX_ONNX_WORKER (tingmix-onnx-worker.exe).
#[test]
#[ignore = "downloads up to 2 GB from the published catalog; needs the platform engine (see the doc comment)"]
fn real_first_run_download_probe_transcribe() {
    assert_eq!(
        std::env::var("TINGMIX_E2E_REAL_FIRST_RUN").as_deref(),
        Ok("1"),
        "set TINGMIX_E2E_REAL_FIRST_RUN=1"
    );
    let (dir, keep) = scratch_dir();
    let models = Arc::new(Models::open(dir.join("models"))); // embedded keys: the release key signs the catalog
                                                             // ONNX builds: the pinned ONNX Runtime library by absolute path (the app does this from its bundle at start-up)
    let ort = std::env::var_os("TINGMIX_ORT_DYLIB").map(PathBuf::from);
    if let Some(p) = &ort {
        tingmix_runtime_onnx::set_runtime_library(p.clone());
    }
    let rt = Runtimes {
        ort_library: ort,
        ..Default::default()
    };
    let t0 = Instant::now();
    let stamp = |what: &str| println!("[{:7.1}s] {what}", t0.elapsed().as_secs_f64());

    let cfg = crate::setup::Catalog {
        url: crate::setup::catalog_url(),
        allow_loopback_http: false,
    };
    let view = crate::setup::check(&models, &rt, &cfg, "first-run".into());
    let crate::setup::SetupView::Download { pending } = view else {
        panic!("expected a download offer, got {view:?}")
    };
    stamp(&format!(
        "catalog fetched and verified: {} ({} bytes to download)",
        pending.id, pending.download_bytes
    ));

    let required: Vec<String> = pending
        .licenses
        .iter()
        .filter(|l| l.acceptance_required && !l.accepted)
        .map(|l| l.id.clone())
        .collect();
    let last = Arc::new(Mutex::new((0u64, 0u64)));
    let l2 = last.clone();
    let (m2, rt2) = (models.clone(), rt.clone());
    let on_op: Notify = Arc::new(move |v| {
        if v.kind == "download" && v.state == "running" {
            *l2.lock().unwrap() = (v.bytes_done, v.bytes_total);
        }
        if v.state == "done" && v.kind == "download" {
            backends::onboard_core(&m2, &rt2, &v.name, Arc::new(|_| {})).expect("onboard");
        }
    });
    models
        .install("first-run", &required, on_op)
        .expect("start download");
    let mut next_report = 0u64;
    loop {
        let op = models.overview().op.expect("operation started");
        let (done, total) = *last.lock().unwrap();
        if done >= next_report && total > 0 {
            stamp(&format!(
                "download {:.0}% ({} of {} MB)",
                done as f64 * 100.0 / total as f64,
                done >> 20,
                total >> 20
            ));
            next_report = done + (200 << 20);
        }
        match (op.kind.as_str(), op.state.as_str()) {
            ("probe", "done") => break,
            (_, "failed" | "cancelled") => panic!("{} {}: {:?}", op.kind, op.state, op.message),
            _ if t0.elapsed() > Duration::from_secs(3600) => panic!("timed out in {}", op.kind),
            _ => std::thread::sleep(Duration::from_millis(500)),
        }
    }
    stamp("downloaded, verified, activated and tested on this computer");

    let info = backends::info(&models, &rt);
    assert!(!info.mock && info.usable, "{}", info.label);
    let expected: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root().join("tests/fixtures/large/expected_texts.json")).unwrap(),
    )
    .unwrap();
    let want = expected["clips"]["mix_08s"]["beam_lm"]
        .as_str()
        .unwrap()
        .to_string();
    let t_load = Instant::now();
    let resolved = backends::resolve(&models, &rt).expect("load the downloaded model");
    stamp(&format!(
        "model and LM loaded in {:.1}s",
        t_load.elapsed().as_secs_f64()
    ));
    let mut reducer = Reducer::new();
    let t_run = Instant::now();
    let out = crate::files::run(
        &root().join("tests/fixtures/audio/mix_08s.wav"),
        resolved.backend,
        &AtomicBool::new(false),
        &mut reducer,
        &mut |_| {},
    );
    assert_eq!(out.state, JobState::Done, "{:?}", out.message);
    let got = join(reducer.finals().map(|f| f.text.clone()).collect());
    stamp(&format!(
        "mix_08s transcribed in {:.1}s: {got}",
        t_run.elapsed().as_secs_f64()
    ));
    if pending.id.starts_with(backends::LARGE) {
        assert_eq!(got, want);
    } else {
        // engines that write punctuation and spacing: the words of the same sentence (see the ONNX family tests)
        assert!(
            squash(&got).contains("我覺得這個deadline有點太趕了"),
            "{got}"
        );
    }
    if keep.is_none() {
        let _ = std::fs::remove_dir_all(dir);
    }
}
