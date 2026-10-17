//! Host-level end-to-end test of the MLX Large path, ignored unless `TINGMIX_E2E_MLX_PACKAGE` points at a package folder:
//! signed manifest + the files it lists (see docs/architecture/app.md). Steps: offline import with licence acceptance ->
//! automatic activate + probe after the import -> selection refuses while experimental backends are off -> picks `mlx`
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

#[test]
#[ignore = "needs the MLX package folder, a --features mlx build with MLX_PREFIX, and ~2 GB of RAM"]
fn mlx_import_probe_select_transcribe_live() {
    let package =
        PathBuf::from(std::env::var("TINGMIX_E2E_MLX_PACKAGE").expect("TINGMIX_E2E_MLX_PACKAGE"));
    // TINGMIX_E2E_HOME keeps the registry where an app run with HOME=<it> finds it (used to check preloading by hand)
    let keep = std::env::var_os("TINGMIX_E2E_HOME").map(PathBuf::from);
    let dir = keep
        .as_ref()
        .map(|h| h.join("Library/Application Support/app.tingmix.desktop"))
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("tingmix-e2e-{}", std::process::id()))
        });
    let _ = std::fs::remove_dir_all(&dir);
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

    // 2. selection: refused while experimental backends are off, picks mlx once they are on
    let off = backends::decide(&models, false, &rt);
    assert!(
        off.chosen.is_none() && off.reason.contains("experimental engine"),
        "{}",
        off.reason
    );
    let on = backends::decide(&models, true, &rt);
    assert_eq!(
        on.chosen.as_ref().map(|c| c.1.as_str()),
        Some("mlx"),
        "{}",
        on.reason
    );
    let info = backends::info(&models, true, &rt);
    assert!(
        !info.mock && info.experimental && info.label.contains("EXPERIMENTAL"),
        "{}",
        info.label
    );
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
    let resolved = backends::resolve(&models, true, &rt).expect("load mlx backend");
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
    let resolved = backends::resolve(&models, true, &rt).expect("load mlx backend");
    let log: Arc<Mutex<Vec<LiveMsg>>> = Arc::default();
    let l = log.clone();
    let sink: Sink = Arc::new(move |m| l.lock().unwrap().push(m));
    let persist: Persist = Arc::new(|_, _| {});
    let data = tingmix_audio::decode_file(&wav).unwrap();
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
    let events: Vec<&Event> = msgs
        .iter()
        .filter_map(|m| {
            if let LiveMsg::Event { event, .. } = m {
                Some(event)
            } else {
                None
            }
        })
        .collect();
    let n_partial = events
        .iter()
        .filter(|e| matches!(e, Event::Partial(_)))
        .count();
    let snap = live.shared.snapshot();
    println!(
        "live: {n_partial} partials, {} finals: {:?}",
        snap.finals.len(),
        snap.finals
            .iter()
            .map(|f| f.text.as_str())
            .collect::<Vec<_>>()
    );
    let st = live.shared.status();
    println!(
        "live status: {:?} message {:?} samples {}",
        st.state, st.message, st.samples_in
    );
    assert!(n_partial > 0 && !snap.finals.is_empty());
    assert!(snap.finals.iter().any(|f| !f.text.is_empty()));
    if keep.is_none() {
        let _ = std::fs::remove_dir_all(dir);
    }
}
