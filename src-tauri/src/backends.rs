//! Runtime selection for the host: which `AsrBackend` the next session uses.
//!
//! Policy: a real backend is used when an installed model is active AND `tingmix-models` selection accepts it (backend
//! registered, runtime present, requirements met, a fresh passing compatibility probe); the best accepted engine wins
//! automatically, there is no user-facing gate. The placeholder engine ("Mock — not ASR") exists in debug builds only: a
//! release build with no usable model has nothing to run and the UI shows the setup (download) screen. A real backend that
//! fails to load is an error, never a silent switch to the placeholder.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;
use tingmix_core::{AsrBackend, AsrStream, BackendError, BackendInfo};
use tingmix_models::probe::{fresh, run_probe, ProbeFixture, ProbeTarget};
use tingmix_models::select::{
    load_known_good, save_known_good, select_probed, BackendReg, Validation,
};
use tingmix_models::{InstalledModel, Manifest};
use tingmix_streaming::MockBackend;

use crate::dto::BACKEND_LABEL;
use crate::models::{Models, Notify};

pub const ONNX: &str = "onnx-cpu";
pub const MLX: &str = "mlx";
/// ONNX Runtime pinned in `tools/ort-fetch/MANIFEST.tsv`.
pub const ORT_VERSION: &str = "1.31.0";

/// What each real backend honestly is. Change to `VALIDATED` only when its parity and probe evidence says so.
fn status_word(backend: &str) -> &'static str {
    match backend {
        ONNX | MLX => "EXPERIMENTAL",
        _ => "BLOCKED",
    }
}

/// A backend usable from any thread. Real backends are built and kept on one dedicated thread: MLX streams are
/// thread-local ("There is no Stream(gpu, 1) in current thread" when a model loaded on one thread decodes on another), so
/// the model never moves. Calls are forwarded over a channel; the mock runs in place.
pub struct Dyn {
    info: BackendInfo,
    inner: Inner,
    /// Set by `take`: counts this backend as held by a session or job until it is dropped, however that session ends.
    held: Option<Held>,
}

struct Held(std::sync::Arc<std::sync::Mutex<u32>>);

impl Drop for Held {
    fn drop(&mut self) {
        let mut n = self.0.lock().unwrap();
        *n = n.saturating_sub(1);
    }
}

enum Inner {
    Local(Box<dyn AsrBackend>),
    Remote(Remote),
}

type Reply<T> = std::sync::mpsc::Sender<Result<T, BackendError>>;

enum Req {
    Batch(Vec<Vec<f32>>, Reply<Vec<String>>),
    NewStream(Reply<u64>),
    Accept(u64, Vec<f32>, Reply<String>),
    Finalize(u64, Reply<String>),
    Close(u64),
}

struct Remote {
    tx: std::sync::mpsc::Sender<Req>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn gone() -> BackendError {
    BackendError::Inference("the inference thread stopped".into())
}

impl Remote {
    fn call<T>(&self, f: impl FnOnce(Reply<T>) -> Req) -> Result<T, BackendError> {
        let (rtx, rrx) = std::sync::mpsc::channel();
        self.tx.send(f(rtx)).map_err(|_| gone())?;
        rrx.recv().map_err(|_| gone())?
    }
}

impl Dyn {
    pub fn mock() -> Self {
        let b = MockBackend::new(false);
        Dyn {
            info: b.info().clone(),
            inner: Inner::Local(Box::new(b)),
            held: None,
        }
    }

    /// Runs `build` on a new thread, which then owns the backend. A build error is returned here, not at the first decode.
    pub fn spawn(
        build: impl FnOnce() -> Result<Box<dyn AsrBackend>, String> + Send + 'static,
    ) -> Result<Self, String> {
        let (tx, rx) = std::sync::mpsc::channel::<Req>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("tingmix-backend".into())
            .spawn(move || {
                let mut backend = match build() {
                    Ok(b) => b,
                    Err(e) => return drop(ready_tx.send(Err(e))),
                };
                let _ = ready_tx.send(Ok(backend.info().clone()));
                let mut streams: HashMap<u64, Box<dyn AsrStream>> = HashMap::new();
                let mut next = 0u64;
                while let Ok(req) = rx.recv() {
                    match req {
                        Req::Batch(wavs, r) => {
                            let refs: Vec<&[f32]> = wavs.iter().map(Vec::as_slice).collect();
                            let _ = r.send(backend.transcribe_batch(&refs));
                        }
                        Req::NewStream(r) => {
                            let _ = r.send(backend.new_stream().map(|s| {
                                next += 1;
                                streams.insert(next, s);
                                next
                            }));
                        }
                        Req::Accept(id, pcm, r) => {
                            let _ = r.send(
                                streams
                                    .get_mut(&id)
                                    .ok_or_else(gone)
                                    .and_then(|s| s.accept(&pcm)),
                            );
                        }
                        Req::Finalize(id, r) => {
                            let _ = r.send(
                                streams
                                    .get_mut(&id)
                                    .ok_or_else(gone)
                                    .and_then(|s| s.finalize()),
                            );
                        }
                        Req::Close(id) => drop(streams.remove(&id)),
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        match ready_rx.recv() {
            Ok(Ok(info)) => Ok(Dyn {
                info,
                inner: Inner::Remote(Remote {
                    tx,
                    thread: Some(thread),
                }),
                held: None,
            }),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => Err("the backend thread died while loading".into()),
        }
    }
}

impl Drop for Dyn {
    fn drop(&mut self) {
        if let Inner::Remote(r) = &mut self.inner {
            let (dead, _) = std::sync::mpsc::channel();
            drop(std::mem::replace(&mut r.tx, dead)); // closes the request channel, ending the thread's loop
            if let Some(t) = r.thread.take() {
                let _ = t.join();
            }
        }
    }
}

struct RemoteStream {
    tx: std::sync::mpsc::Sender<Req>,
    id: u64,
}

impl AsrStream for RemoteStream {
    fn accept(&mut self, pcm: &[f32]) -> Result<String, BackendError> {
        let (rtx, rrx) = std::sync::mpsc::channel();
        self.tx
            .send(Req::Accept(self.id, pcm.to_vec(), rtx))
            .map_err(|_| gone())?;
        rrx.recv().map_err(|_| gone())?
    }
    fn finalize(&mut self) -> Result<String, BackendError> {
        let (rtx, rrx) = std::sync::mpsc::channel();
        self.tx
            .send(Req::Finalize(self.id, rtx))
            .map_err(|_| gone())?;
        rrx.recv().map_err(|_| gone())?
    }
}

impl Drop for RemoteStream {
    fn drop(&mut self) {
        let _ = self.tx.send(Req::Close(self.id));
    }
}

impl AsrBackend for Dyn {
    fn info(&self) -> &BackendInfo {
        &self.info
    }
    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        match &mut self.inner {
            Inner::Local(b) => b.transcribe_batch(wavs),
            Inner::Remote(r) => {
                r.call(|t| Req::Batch(wavs.iter().map(|w| w.to_vec()).collect(), t))
            }
        }
    }
    fn new_stream(&mut self) -> Result<Box<dyn AsrStream>, BackendError> {
        match &mut self.inner {
            Inner::Local(b) => b.new_stream(),
            Inner::Remote(r) => {
                let id = r.call(Req::NewStream)?;
                Ok(Box::new(RemoteStream {
                    tx: r.tx.clone(),
                    id,
                }))
            }
        }
    }
}

/// Where the bundled ONNX Runtime library is (resolved once at start-up), if anywhere.
#[derive(Clone, Default)]
pub struct Runtimes {
    pub ort_library: Option<PathBuf>,
    /// Probe worker for the ONNX engine; `None` means `TINGMIX_ONNX_WORKER`, then the sibling of the executable.
    pub onnx_worker: Option<PathBuf>,
}

impl Runtimes {
    /// The version string the bundled library reports about itself, the same one the probe worker records, so a stored
    /// probe stays valid. Read once per process.
    fn ort_version(&self) -> Option<String> {
        static VERSION: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        self.ort_library
            .as_ref()
            .filter(|p| p.is_file())
            .and_then(|_| {
                VERSION
                    .get_or_init(|| {
                        tingmix_runtime_onnx::runtime_version()
                            .or_else(|| Some(ORT_VERSION.to_string()))
                    })
                    .clone()
            })
    }
}

fn mlx_compiled() -> bool {
    cfg!(all(
        feature = "mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))
}

/// `(mlx version, device)` read from the linked libmlx; `None` without the `mlx` feature or when MLX cannot initialise.
fn mlx_runtime() -> Option<(String, String)> {
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    return tingmix_runtime_mlx::runtime_info();
    #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
    None
}

/// Every backend compiled into this build is eligible: the best one that passes its compatibility test is used
/// automatically. The EXPERIMENTAL status word survives only as a label for Advanced and Diagnostics.
pub fn regs(rt: &Runtimes) -> Vec<BackendReg> {
    let validation = Validation::Validated;
    let mut v = vec![BackendReg {
        backend: ONNX.into(),
        validation,
        is_cpu: true,
        runtime_version: rt.ort_version(),
        probe: None,
    }];
    if mlx_compiled() {
        v.push(BackendReg {
            backend: MLX.into(),
            validation,
            is_cpu: false,
            runtime_version: mlx_runtime().map(|r| r.0),
            probe: None,
        });
    }
    v
}

/// Whether this build can run `backend` at all (compiled in, and its runtime library present).
pub fn engine_available(backend: &str, rt: &Runtimes) -> bool {
    version_of(backend, rt).is_some()
}

/// `engine_available` and the probe worker it is checked by exists: a build without the sidecar can install a model but
/// never activate it.
pub fn engine_ready(backend: &str, rt: &Runtimes) -> bool {
    engine_available(backend, rt) && worker_exe(backend, rt).is_some_and(|p| p.is_file())
}

fn worker_exe(backend: &str, rt: &Runtimes) -> Option<std::path::PathBuf> {
    match backend {
        ONNX => rt
            .onnx_worker
            .clone()
            .or_else(|| std::env::var_os("TINGMIX_ONNX_WORKER").map(Into::into))
            .or_else(|| tingmix_runtime_onnx::probe::OnnxProbeTarget::sibling_worker().ok()),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        MLX => std::env::var_os("TINGMIX_MLX_WORKER")
            .map(Into::into)
            .or_else(|| tingmix_runtime_mlx::probe::MlxProbeTarget::sibling_worker().ok()),
        _ => None,
    }
}

fn version_of(backend: &str, rt: &Runtimes) -> Option<String> {
    match backend {
        ONNX => rt.ort_version(),
        MLX => mlx_runtime().map(|r| r.0),
        _ => None,
    }
}

fn device_of(backend: &str) -> String {
    match backend {
        MLX => mlx_runtime().map_or_else(|| "gpu".into(), |r| r.1),
        _ => "cpu".into(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendInfoDto {
    /// Text shown wherever the backend's name appears.
    pub label: String,
    pub mock: bool,
    pub experimental: bool,
    /// Why this backend (and not another) is in use.
    pub reason: String,
    pub rejected: Vec<(String, String)>,
    pub ort_library: Option<String>,
    /// Debug builds may run on the placeholder engine; release builds never do.
    pub demo_allowed: bool,
    /// A session can start: a real engine is chosen, or the demo engine is allowed.
    pub usable: bool,
}

pub struct Decision {
    /// `(manifest id, backend)` when a real backend was accepted.
    pub chosen: Option<(String, String)>,
    pub reason: String,
    pub rejected: Vec<(String, String)>,
}

/// Plain-language version of a `tingmix-models` rejection reason or a backend error.
/// The placeholder engine exists for development only: release builds never run on it.
pub fn demo_allowed() -> bool {
    cfg!(debug_assertions)
}

pub fn plain(why: &str, backend: &str) -> String {
    // already coded (for example backend.worker_missing): keep its code and params
    if serde_json::from_str::<serde_json::Value>(why)
        .is_ok_and(|v| v.get("code").is_some_and(|c| c.is_string()))
    {
        return why.to_string();
    }
    let engine = match backend {
        MLX => "MLX",
        ONNX => "ONNX Runtime",
        other => other,
    };
    let sentence = |s: &str| {
        let mut c = s.trim().chars();
        c.next().map_or_else(String::new, |f| {
            f.to_uppercase().collect::<String>() + c.as_str()
        })
    };
    use crate::coded::{coded, computer};
    use serde_json::json;
    let (computer, mac) = (computer(), cfg!(target_os = "macos"));
    if why.contains("not registered in this build") || why.contains("not available in this build") {
        let hint = if backend == MLX {
            " Use the MLX build of TingMix (tools/bundle-mlx.sh)."
        } else {
            ""
        };
        coded(
            "backend.missing_engine",
            json!({ "engine": engine, "mlx": backend == MLX }),
            format!(
                "This build of TingMix doesn't include the {engine} engine this model needs.{hint}"
            ),
        )
    } else if why.contains("no verified compatibility probe") {
        coded(
            "backend.untested",
            json!({ "mac": mac }),
            format!("This model hasn't been tested on {computer} yet. Press “Test on this Mac”."),
        )
    } else if why.contains("compatibility probe failed") {
        coded(
            "backend.test_failed",
            json!({ "mac": mac }),
            format!("The test run on {computer} failed, so the model is not used. See the Models tab for the details."),
        )
    } else if why.contains("is not available") && why.contains("runtime") {
        coded(
            "backend.runtime_missing",
            json!({ "engine": engine, "mac": mac }),
            format!(
                "The {engine} runtime is not available on {computer} (its library is missing)."
            ),
        )
    } else if let Some(d) = why.strip_prefix("probe failed: ") {
        coded(
            "backend.test_failed_detail",
            json!({ "detail": sentence(d), "mac": mac }),
            format!("The test run on {computer} failed: {}", sentence(d)),
        )
    } else if let Some((need, have)) = parse_ram(why) {
        coded(
            "backend.ram",
            json!({ "need": need, "have": have }),
            format!("This model needs {need} GiB of memory; {computer} has {have} GiB."),
        )
    } else if why.starts_with("built for ") || why.contains("this machine is") {
        coded(
            "backend.platform",
            json!({ "mac": mac }),
            format!("This model was built for a different system than {computer}."),
        )
    } else if let Some(f) = why.strip_prefix("CPU lacks required feature ") {
        coded(
            "backend.cpu",
            json!({ "feature": f.trim() }),
            format!(
                "The processor of {computer} lacks a feature this model needs ({}).",
                f.trim()
            ),
        )
    } else if why.starts_with("needs Apple M") {
        coded(
            "backend.chip",
            json!({}),
            "This model needs a newer Apple chip than this one.",
        )
    } else if why.contains("is older than the required") {
        coded(
            "backend.runtime_old",
            json!({ "engine": engine }),
            format!("The {engine} runtime on {computer} is too old for this model."),
        )
    } else {
        coded(
            "backend.unknown",
            json!({ "detail": sentence(why) }),
            format!("This model can't be used: {}", sentence(why)),
        )
    }
}

/// "needs 16 GiB RAM, machine has 8 GiB" -> (16, 8)
fn parse_ram(why: &str) -> Option<(u64, u64)> {
    let rest = why.strip_prefix("needs ")?;
    let (need, rest) = rest.split_once(" GiB RAM, machine has ")?;
    Some((
        need.trim().parse().ok()?,
        rest.trim().strip_suffix(" GiB")?.trim().parse().ok()?,
    ))
}

pub fn decide(models: &Models, rt: &Runtimes) -> Decision {
    let none = |reason: String, rejected| Decision {
        chosen: None,
        reason,
        rejected,
    };
    let reg = match models.reg() {
        Ok(r) => r,
        Err(e) => return none(format!("{e}; using the mock backend."), vec![]),
    };
    let Some(active) = reg.active().ok().flatten() else {
        return none(
            crate::coded::coded(
                "backend.no_model",
                serde_json::json!({}),
                "No model is active; using the mock backend.",
            ),
            vec![],
        );
    };
    let model = match reg.open_model(&active) {
        Ok(m) => m,
        Err(e) => {
            return none(
                crate::coded::coded(
                    "backend.unreadable",
                    serde_json::json!({ "model": active, "detail": e.to_string() }),
                    format!(
                        "Active model {active} cannot be opened ({e}); using the mock backend."
                    ),
                ),
                vec![],
            )
        }
    };
    let regs = regs(rt);
    let probe_of = |m: &Manifest| {
        let (v, d) = (version_of(&m.backend, rt), device_of(&m.backend));
        fresh(reg.root(), &m.backend, m, v.as_deref(), &d)
    };
    let known_good = load_known_good(reg.root()).ok().flatten();
    match select_probed(
        std::slice::from_ref(&model.manifest),
        models.caps(),
        &regs,
        known_good.as_ref(),
        &probe_of,
    ) {
        Ok(sel) => {
            let _ = save_known_good(reg.root(), &sel, models.caps());
            Decision {
                chosen: Some((sel.manifest_id, sel.backend)),
                reason: sel.reason,
                rejected: sel.rejected,
            }
        }
        Err(rejected) => {
            let rejected: Vec<(String, String)> = rejected
                .into_iter()
                .map(|(id, why)| (id, plain(&why, &model.manifest.backend)))
                .collect();
            let why = rejected.first().map(|r| r.1.clone()).unwrap_or_default();
            none(
                crate::coded::coded(
                    "backend.cant_use",
                    serde_json::json!({ "model": active, "why": why }),
                    format!(
                        "{active} can't be used yet. {} The mock backend (not real ASR) is in use.",
                        crate::coded::english(&why)
                    ),
                ),
                rejected,
            )
        }
    }
}

pub fn info(models: &Models, rt: &Runtimes) -> BackendInfoDto {
    let d = decide(models, rt);
    let (label, mock, experimental) = match &d.chosen {
        None => (BACKEND_LABEL.to_string(), true, false),
        Some((id, backend)) => (
            format!("{id} ({backend}) — {}", status_word(backend)),
            false,
            status_word(backend) != "VALIDATED",
        ),
    };
    BackendInfoDto {
        label,
        mock,
        experimental,
        reason: d.reason,
        rejected: d.rejected,
        ort_library: rt.ort_library.as_ref().map(|p| p.display().to_string()),
        demo_allowed: demo_allowed(),
        usable: !mock || demo_allowed(),
    }
}

pub struct Resolved {
    pub backend: Dyn,
    pub label: String,
}

/// The backend for the next session, loaded. A failing real backend is an error.
pub fn resolve(models: &Models, rt: &Runtimes) -> Result<Resolved, String> {
    let i = info(models, rt);
    let d = decide(models, rt);
    let Some((id, backend)) = d.chosen else {
        if !demo_allowed() {
            return Err(crate::coded::coded(
                "setup.required",
                serde_json::json!({}),
                "TingMix needs its speech model before it can transcribe. Open the setup screen to download it.",
            ));
        }
        return Ok(Resolved {
            backend: Dyn::mock(),
            label: i.label,
        });
    };
    let model = models.reg()?.open_model(&id).map_err(|e| e.to_string())?;
    let backend = Dyn::spawn(move || build(&model, &backend)).map_err(|e| {
        crate::coded::coded(
            "backend.load_failed",
            serde_json::json!({ "model": id, "detail": e }),
            format!("could not load {id}: {e}"),
        )
    })?;
    Ok(Resolved {
        backend,
        label: i.label,
    })
}

fn component(
    model: &InstalledModel,
    component: &str,
    name: Option<&str>,
) -> Result<PathBuf, String> {
    model
        .file(component)
        .find(|p| name.is_none_or(|n| p.file_name().is_some_and(|f| f == n)))
        .ok_or_else(|| {
            format!(
                "manifest has no {component} file{}",
                name.map(|n| format!(" named {n}")).unwrap_or_default()
            )
        })
}

/// Construct the backend for an installed version. The ONNX Runtime library must already be set (`start-up`).
pub fn build(model: &InstalledModel, backend: &str) -> Result<Box<dyn AsrBackend>, String> {
    match backend {
        ONNX => {
            let graph = model
                .manifest
                .files
                .iter()
                .find(|f| f.path.ends_with(".onnx"))
                .map(|f| model.dir.join(&f.path))
                .ok_or("manifest lists no .onnx file")?;
            let units = component(model, "tokenizer", Some("units.json"))?;
            if is_large(model) {
                onnx_large_backend(&graph, &units, &component(model, "lm", None)?)
            } else {
                onnx_backend(&graph, &units)
            }
        }
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        MLX => {
            use tingmix_runtime_mlx::{Decode, Dtype, MlxLargeBackend};
            let lm = tingmix_decoder::CharLm::load(&component(model, "lm", None)?)
                .map_err(|e| e.to_string())?;
            let decode = Decode::BeamLm(Box::new(lm), tingmix_decoder::BeamConfig::default());
            let b = MlxLargeBackend::new(
                &component(model, "model", None)?,
                &component(model, "tokenizer", Some("units.json"))?,
                Dtype::F32,
                decode,
            )
            .map_err(|e| e.to_string())?;
            Ok(Box::new(b))
        }
        other => Err(format!("backend {other} is not available in this build")),
    }
}

/// TingMix-Large ships as an ONNX graph under the same backend id as Small; the family tells them apart.
fn is_large(model: &InstalledModel) -> bool {
    model.manifest.family == "tingmix-large"
}

/// Large on ONNX Runtime, CPU provider, with the character LM (beam search). One copy only: it holds about 2 to 2.5 GB.
fn onnx_large_backend(
    graph: &std::path::Path,
    units_json: &std::path::Path,
    lm: &std::path::Path,
) -> Result<Box<dyn AsrBackend>, String> {
    use tingmix_runtime_onnx::{Decode, OnnxLarge, Provider};
    let lm = tingmix_decoder::CharLm::load(lm).map_err(|e| e.to_string())?;
    let decode = Decode::BeamLm(Box::new(lm), tingmix_decoder::BeamConfig::default());
    let b = OnnxLarge::load(graph, units_json, decode, Provider::Cpu, onnx_threads())
        .map_err(|e| e.to_string())?;
    Ok(Box::new(b))
}

/// Intra-op threads for the CPU provider: all cores up to 8 (the measured sweet spot in docs/research/large-onnx.md).
fn onnx_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get().min(8))
}

fn onnx_backend(
    graph: &std::path::Path,
    units_json: &std::path::Path,
) -> Result<Box<dyn AsrBackend>, String> {
    let vocab =
        tingmix_decoder::Vocab::from_units_json(units_json, true).map_err(|e| e.to_string())?;
    let decode: tingmix_runtime_onnx::TokenDecoder = Arc::new(move |ids| vocab.decode(ids));
    let b = tingmix_runtime_onnx::OnnxSmall::load(graph, 1, decode).map_err(|e| e.to_string())?;
    Ok(Box::new(b))
}

// ---- preloading ---------------------------------------------------------------------------------------------------

enum WarmState {
    Empty,
    Loading(String),
    Ready(String, Resolved),
    Failed(String, String),
}

/// One preloaded backend (model and LM already in memory) so Start does not wait for the load. A session takes it; the next
/// one is loaded after the session ends, never in parallel (a second copy of Large is ~2 GiB).
pub struct Warm {
    slot: std::sync::Mutex<WarmState>,
    cv: std::sync::Condvar,
    /// Backends handed out by `take` and still alive in a live session or file job (see `Held`).
    taken: std::sync::Arc<std::sync::Mutex<u32>>,
}

impl Default for Warm {
    fn default() -> Self {
        Self {
            slot: std::sync::Mutex::new(WarmState::Empty),
            cv: std::sync::Condvar::new(),
            taken: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadView {
    pub loading: bool,
    pub label: Option<String>,
    pub error: Option<String>,
}

pub const BACKEND_EVENT: &str = "tingmix://backend";

/// Everything the UI shows about the backend, pushed on every change (activation, probe, settings, preload progress).
#[derive(Debug, Clone, Serialize)]
pub struct BackendState {
    pub info: BackendInfoDto,
    pub load: LoadView,
}

pub fn state_of(st: &crate::state::AppState) -> BackendState {
    BackendState {
        info: info(&st.models, &st.runtimes),
        load: st.warm.view(),
    }
}

/// Tell the UI the backend state changed.
pub fn publish(app: &tauri::AppHandle) {
    use tauri::{Emitter, Manager};
    let _ = app.emit(
        BACKEND_EVENT,
        state_of(&app.state::<crate::state::AppState>()),
    );
}

fn key_of(d: &Decision) -> Option<String> {
    d.chosen.as_ref().map(|(id, b)| format!("{id}|{b}"))
}

impl Warm {
    pub fn view(&self) -> LoadView {
        match &*self.slot.lock().unwrap() {
            WarmState::Loading(k) => LoadView {
                loading: true,
                label: Some(k.split('|').next().unwrap_or(k).to_string()),
                error: None,
            },
            WarmState::Failed(_, e) => LoadView {
                loading: false,
                label: None,
                error: Some(e.clone()),
            },
            _ => LoadView {
                loading: false,
                label: None,
                error: None,
            },
        }
    }
}

/// Start loading the backend the next session would use, in the background. No-op for the mock or when it is already
/// loading / loaded. Call after anything that can change the choice (start-up, activation, settings, probe, session end).
pub fn preload(app: &tauri::AppHandle) {
    use tauri::Manager;
    let st = app.state::<crate::state::AppState>();
    if *st.warm.taken.lock().unwrap() > 0 {
        publish(app);
        return; // a session or job holds a copy; loading a second Large now would double the memory
    }
    let d = decide(&st.models, &st.runtimes);
    let notify = publish;
    let Some(key) = key_of(&d) else {
        *st.warm.slot.lock().unwrap() = WarmState::Empty;
        notify(app);
        return;
    };
    {
        let mut slot = st.warm.slot.lock().unwrap();
        match &*slot {
            WarmState::Loading(k) | WarmState::Ready(k, _) if *k == key => {
                drop(slot);
                publish(app); // the choice itself may have changed (for example a fresh probe)
                return;
            }
            _ => *slot = WarmState::Loading(key.clone()),
        }
    }
    notify(app);
    let app = app.clone();
    let _ = std::thread::Builder::new()
        .name("tingmix-preload".into())
        .spawn(move || {
            let st = app.state::<crate::state::AppState>();
            let r = resolve(&st.models, &st.runtimes);
            {
                let mut slot = st.warm.slot.lock().unwrap();
                // the choice may have changed while loading; then this copy is simply dropped
                if matches!(&*slot, WarmState::Loading(k) if *k == key) {
                    *slot = match r {
                        Ok(res) => WarmState::Ready(key, res),
                        Err(e) => WarmState::Failed(key, e),
                    };
                }
                st.warm.cv.notify_all();
            }
            notify(&app);
        });
}

/// After a successful import or download: make the model active when none is, and test it on this Mac. Progress and the
/// result reach the UI through the models operation events and `publish`.
pub fn onboard(app: &tauri::AppHandle, id: &str, notify: Notify) {
    use tauri::Manager;
    let st = app.state::<crate::state::AppState>();
    if let Err(e) = onboard_core(&st.models, &st.runtimes, id, notify) {
        st.note_error(format!("onboarding {id}: {e}"));
    }
    publish(app);
}

/// The steps without the UI plumbing: activate when nothing is active, then start the compatibility test.
pub fn onboard_core(
    models: &Arc<Models>,
    rt: &Runtimes,
    id: &str,
    notify: Notify,
) -> Result<(), String> {
    let reg = models.reg()?;
    if reg.active().ok().flatten().is_none() {
        reg.activate(id).map_err(|e| format!("activate: {e}"))?;
    }
    if reg.active().ok().flatten().as_deref() == Some(id) {
        start_probe(models, rt.clone(), id.to_string(), notify)?;
    }
    Ok(())
}

/// The backend for a new session: the preloaded one if it matches the current choice (waiting for it while it loads),
/// otherwise a fresh load. A preload failure is reported once and then retried fresh.
pub fn take(app: &tauri::AppHandle) -> Result<Resolved, String> {
    use tauri::Manager;
    let mut r = take_inner(app)?;
    let taken = &app.state::<crate::state::AppState>().warm.taken;
    *taken.lock().unwrap() += 1;
    r.backend.held = Some(Held(taken.clone()));
    Ok(r)
}

fn take_inner(app: &tauri::AppHandle) -> Result<Resolved, String> {
    use tauri::Manager;
    let st = app.state::<crate::state::AppState>();
    let key = key_of(&decide(&st.models, &st.runtimes));
    if let Some(key) = key {
        let mut slot = st.warm.slot.lock().unwrap();
        loop {
            match std::mem::replace(&mut *slot, WarmState::Empty) {
                WarmState::Ready(k, r) if k == key => return Ok(r),
                WarmState::Loading(k) if k == key => {
                    *slot = WarmState::Loading(k);
                    slot = st.warm.cv.wait(slot).unwrap();
                }
                WarmState::Failed(k, e) if k == key => return Err(e),
                other => {
                    *slot = other;
                    break;
                }
            }
        }
    }
    resolve(&st.models, &st.runtimes)
}

// ---- compatibility probe ------------------------------------------------------------------------------------------

struct AppProbe<'a> {
    backend: &'static str,
    rt: &'a Runtimes,
    loaded: Option<(String, Box<dyn AsrBackend>)>,
}

impl ProbeTarget for AppProbe<'_> {
    fn backend(&self) -> &str {
        self.backend
    }
    fn backend_version(&self) -> Option<String> {
        version_of(self.backend, self.rt)
    }
    fn device(&self) -> String {
        device_of(self.backend)
    }
    fn decode(&mut self, model: &InstalledModel, wav: &[f32]) -> Result<String, String> {
        if self
            .loaded
            .as_ref()
            .is_none_or(|(id, _)| *id != model.manifest.id)
        {
            self.loaded = Some((model.manifest.id.clone(), build(model, self.backend)?));
        }
        let (_, b) = self.loaded.as_mut().expect("loaded above");
        b.transcribe_batch(&[wav])
            .map_err(|e| e.to_string())?
            .into_iter()
            .next()
            .ok_or_else(|| "no text returned".into())
    }
}

/// Short fixtures that ship inside the app, per model family: audio plus the text the reference oracle produced.
const ZH_02S: &[u8] = include_bytes!("../../tests/fixtures/audio/zh_02s.wav");

/// 16 kHz mono 16-bit PCM wav -> f32. The shipped fixtures are exactly this; anything else is rejected.
fn wav_samples(b: &[u8]) -> Result<Vec<f32>, String> {
    let u16le = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let u32le = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let (mut o, mut fmt_ok) = (12, false);
    while o + 8 <= b.len() {
        let (id, len) = (&b[o..o + 4], u32le(o + 4) as usize);
        let body = o + 8;
        if id == b"fmt " && body + 16 <= b.len() {
            fmt_ok = u16le(body) == 1
                && u16le(body + 2) == 1
                && u32le(body + 4) == 16_000
                && u16le(body + 14) == 16;
        } else if id == b"data" {
            if !fmt_ok {
                break;
            }
            return Ok(b[body..(body + len).min(b.len())]
                .chunks_exact(2)
                .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                .collect());
        }
        o = body + len + (len & 1);
    }
    Err("probe fixture is not 16 kHz mono 16-bit PCM".into())
}

pub fn probe_fixture(family: &str) -> Option<(&'static str, Vec<f32>, &'static str)> {
    match family {
        // text from tests/fixtures/large/expected_texts.json (CPU fp32 oracle), also what the MLX port reproduces
        "tingmix-large" => wav_samples(ZH_02S)
            .ok()
            .map(|w| ("zh_02s", w, "今天天氣很好")),
        _ => None,
    }
}

/// MLX/Metal can abort the process, so MLX is probed in a child (`tingmix-mlx-worker`, next to the app binary).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn probe_mlx(
    root: &std::path::Path,
    model: &InstalledModel,
    fx: &ProbeFixture,
) -> Option<Result<tingmix_models::probe::ProbeRecord, String>> {
    use tingmix_runtime_mlx::{probe::MlxProbeTarget, Dtype};
    // TINGMIX_MLX_WORKER overrides the sibling lookup (tests, development)
    let worker = std::env::var_os("TINGMIX_MLX_WORKER")
        .map(|p| Ok(p.into()))
        .unwrap_or_else(MlxProbeTarget::sibling_worker);
    Some(match worker {
        Ok(exe) => run_probe(
            root,
            &mut MlxProbeTarget::new(exe, Dtype::F32, true),
            model,
            fx,
            2,
        )
        .map_err(|e| e.to_string()),
        Err(e) => Err(crate::coded::failure(
            "backend.worker_missing",
            "The MLX test helper is missing",
            e,
        )),
    })
}

#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn probe_mlx(
    _: &std::path::Path,
    _: &InstalledModel,
    _: &ProbeFixture,
) -> Option<Result<tingmix_models::probe::ProbeRecord, String>> {
    None
}

/// The Large ONNX graph is probed in a child (`tingmix-onnx-worker`, next to the app executable), so an ONNX Runtime or driver
/// crash cannot take the app down. The child gets the library path as `TINGMIX_ORT_DYLIB` from the target.
fn probe_onnx_large(
    rt: &Runtimes,
    root: &std::path::Path,
    model: &InstalledModel,
    fx: &ProbeFixture,
) -> Option<Result<tingmix_models::probe::ProbeRecord, String>> {
    use tingmix_runtime_onnx::{probe::OnnxProbeTarget, Provider};
    if !is_large(model) {
        return None;
    }
    // TINGMIX_ONNX_WORKER overrides the sibling lookup (tests, development)
    let worker = rt
        .onnx_worker
        .clone()
        .or_else(|| std::env::var_os("TINGMIX_ONNX_WORKER").map(Into::into))
        .map(Ok)
        .unwrap_or_else(OnnxProbeTarget::sibling_worker);
    Some(match worker {
        Ok(exe) => run_probe(
            root,
            &mut OnnxProbeTarget::new(exe, Provider::Cpu, onnx_threads(), true)
                .with_library(rt.ort_library.clone()),
            model,
            fx,
            2,
        )
        .map_err(|e| e.to_string()),
        Err(e) => Err(crate::coded::failure(
            "backend.worker_missing",
            "The ONNX test helper is missing",
            e,
        )),
    })
}

/// A probe result describes a backend on this machine; a build that does not contain the backend (or cannot load its
/// runtime) has nothing to measure. It must report that and must not write a record: `run_probe` persists failures too, and
/// would replace a good record written by a build that has the engine (the registry is shared between builds).
pub fn probe_precheck(backend: &str, rt: &Runtimes) -> Result<(), String> {
    if version_of(backend, rt).is_some() {
        return Ok(());
    }
    Err(plain(
        &format!("backend {backend} is not available in this build"),
        backend,
    ))
}

/// Run the compatibility probe of an installed version in the background (progress via the models op).
pub fn start_probe(
    models: &Arc<Models>,
    rt: Runtimes,
    id: String,
    notify: Notify,
) -> Result<(), String> {
    let reg = models.reg()?;
    let model = reg.open_model(&id).map_err(|e| e.to_string())?;
    let backend: &'static str = match model.manifest.backend.as_str() {
        ONNX => ONNX,
        MLX => MLX,
        other => return Err(format!("backend {other} has no probe in this build")),
    };
    if let Err(e) = probe_precheck(backend, &rt) {
        // shown as a failed operation in the UI; nothing is measured, so nothing is written
        models.start_op("probe", &id, notify.clone())?;
        models.finish_op(Err(e), &notify);
        return Ok(());
    }
    let (name, wav, expected) = probe_fixture(&model.manifest.family).ok_or_else(|| {
        format!(
            "no probe fixture ships for family {}",
            model.manifest.family
        )
    })?;
    let _cancel = models.start_op("probe", &id, notify.clone())?;
    let models = models.clone();
    std::thread::Builder::new()
        .name("tingmix-probe".into())
        .spawn(move || {
            let r = models
                .reg()
                .and_then(|reg| {
                    let fx = ProbeFixture {
                        name,
                        wav: &wav,
                        sample_rate: 16_000,
                        expected_text: expected,
                    };
                    let rec = (match backend {
                        MLX => probe_mlx(reg.root(), &model, &fx),
                        ONNX => probe_onnx_large(&rt, reg.root(), &model, &fx),
                        _ => None,
                    })
                    .unwrap_or_else(|| {
                        let mut t = AppProbe {
                            backend,
                            rt: &rt,
                            loaded: None,
                        };
                        run_probe(reg.root(), &mut t, &model, &fx, 2).map_err(|e| e.to_string())
                    })?;
                    if rec.passed {
                        Ok(())
                    } else {
                        Err(format!("probe failed: {}", rec.detail))
                    }
                })
                .map_err(|e| plain(&e, backend));
            models.finish_op(r, &notify);
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models(tag: &str) -> (Models, std::path::PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("tingmix-backends-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (Models::open(dir.join("models")), dir)
    }

    /// Like MLX: fails when used on a thread other than the one that built it.
    struct ThreadBound {
        info: BackendInfo,
        born: std::thread::ThreadId,
    }

    impl ThreadBound {
        fn check(&self) -> Result<(), BackendError> {
            (std::thread::current().id() == self.born)
                .then_some(())
                .ok_or(BackendError::Inference("wrong thread".into()))
        }
    }

    impl AsrBackend for ThreadBound {
        fn info(&self) -> &BackendInfo {
            &self.info
        }
        fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
            self.check()?;
            Ok(wavs.iter().map(|w| w.len().to_string()).collect())
        }
        fn new_stream(&mut self) -> Result<Box<dyn AsrStream>, BackendError> {
            self.check()?;
            MockBackend::new(true).new_stream()
        }
    }

    #[test]
    fn spawned_backend_stays_on_its_own_thread_and_load_errors_surface_at_spawn() {
        let build = || {
            let b = MockBackend::new(true);
            Ok(Box::new(ThreadBound {
                info: b.info().clone(),
                born: std::thread::current().id(),
            }) as Box<dyn AsrBackend>)
        };
        let mut d = Dyn::spawn(build).unwrap();
        assert!(d.info().streaming_native);
        assert_eq!(d.transcribe_batch(&[&[0.0; 5]]).unwrap(), ["5"]);
        // moved to another thread (as SessionWorker does): still served by the owner thread
        let mut stream_text = std::thread::spawn(move || {
            assert_eq!(d.transcribe_batch(&[&[0.0; 7]]).unwrap(), ["7"]);
            let mut s = d.new_stream().unwrap();
            s.accept(&[0.0; 16_000]).unwrap()
        })
        .join()
        .unwrap();
        stream_text.truncate(2);
        assert_eq!(stream_text, "w1");
        let err = Dyn::spawn(|| Err("no such model".into())).err().unwrap();
        assert_eq!(err, "no such model");
    }

    #[test]
    fn a_taken_backend_is_released_whenever_it_is_dropped() {
        let n = std::sync::Arc::new(std::sync::Mutex::new(1u32));
        let mut d = Dyn::mock();
        d.held = Some(Held(n.clone()));
        drop(d); // a session that ended by itself (mic unplugged) drops its backend without any caller noticing
        assert_eq!(*n.lock().unwrap(), 0);
    }

    #[test]
    fn plain_keeps_an_already_coded_payload() {
        let c = crate::coded::failure(
            "backend.worker_missing",
            "The MLX test helper is missing",
            "x",
        );
        assert_eq!(plain(&c, MLX), c);
    }

    #[test]
    fn a_build_without_the_engine_reports_it_and_never_probes_or_persists() {
        // the stub MLX crate (no MLX_PREFIX) and a missing ONNX Runtime library both have no runtime version
        let rt = Runtimes::default();
        if mlx_runtime().is_none() {
            let e = probe_precheck(MLX, &rt).unwrap_err();
            assert!(e.contains("doesn't include the MLX engine"), "{e}");
        }
        assert!(probe_precheck(ONNX, &rt)
            .unwrap_err()
            .contains("ONNX Runtime"));
        let here = Runtimes {
            ort_library: Some(std::env::current_exe().unwrap()),
            ..Default::default()
        };
        assert!(probe_precheck(ONNX, &here).is_ok());
    }

    #[test]
    fn every_rejection_reason_gets_a_code_and_wording_that_names_the_computer() {
        let c = |why: &str| serde_json::from_str::<serde_json::Value>(&plain(why, MLX)).unwrap();
        assert_eq!(
            c("needs 16 GiB RAM, machine has 8 GiB")["params"],
            serde_json::json!({ "need": 16, "have": 8 })
        );
        assert_eq!(
            c("built for macos/aarch64, this machine is windows/x86_64")["code"],
            "backend.platform"
        );
        assert_eq!(
            c("CPU lacks required feature avx2")["params"]["feature"],
            "avx2"
        );
        assert_eq!(
            c("needs Apple M3 or newer, no Apple Silicon found")["code"],
            "backend.chip"
        );
        assert_eq!(
            c("runtime mlx 0.1 is older than the required 0.2")["code"],
            "backend.runtime_old"
        );
        assert_eq!(c("something new we never saw")["code"], "backend.unknown");
        // the English fallback follows the platform; the flag lets the UI do the same
        let v = c("no verified compatibility probe result for this backend");
        assert_eq!(v["params"]["mac"], cfg!(target_os = "macos"));
        let text = v["text"].as_str().unwrap();
        assert_eq!(
            text.contains("this Mac yet"),
            cfg!(target_os = "macos"),
            "{text}"
        );
        assert_eq!(
            text.contains("this computer yet"),
            !cfg!(target_os = "macos"),
            "{text}"
        );
    }

    #[test]
    fn selection_reasons_are_plain_language() {
        let mlx = plain("backend mlx is not registered in this build", MLX);
        assert!(
            mlx.contains("doesn't include the MLX engine") && mlx.contains("MLX build"),
            "{mlx}"
        );
        let onnx = plain("backend onnx-cpu is not available in this build", ONNX);
        assert!(
            onnx.contains("ONNX Runtime") && !onnx.contains("MLX build"),
            "{onnx}"
        );
        assert!(plain(
            "no verified compatibility probe result for this backend",
            MLX
        )
        .contains("hasn't been tested"));
        assert!(
            crate::coded::english(&plain("probe failed: load/first decode failed: boom", MLX))
                .starts_with(&format!(
                    "The test run on {} failed: Load/first",
                    crate::coded::computer()
                ))
        );
        assert!(plain("backend mlx is not registered in this build", MLX)
            .contains("\"backend.missing_engine\""));
        assert!(plain("needs 16 GiB RAM, machine has 8 GiB", MLX).contains("backend.ram"));
    }

    #[test]
    fn mock_is_the_default_and_the_label_is_honest() {
        let (m, dir) = models("default");
        let i = info(&m, &Runtimes::default());
        assert!(i.mock && !i.experimental);
        assert_eq!(i.label, BACKEND_LABEL);
        assert!(i.reason.contains("No model is active") && i.reason.contains("backend.no_model"));
        assert_eq!(
            resolve(&m, &Runtimes::default()).unwrap().label,
            BACKEND_LABEL
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn every_compiled_backend_is_eligible_and_needs_its_runtime() {
        let regs0 = regs(&Runtimes::default());
        assert!(regs0.iter().all(|r| r.validation == Validation::Validated));
        // ONNX needs the library file to exist
        assert_eq!(
            regs0
                .iter()
                .find(|r| r.backend == ONNX)
                .unwrap()
                .runtime_version,
            None
        );
        let here = Runtimes {
            ort_library: Some(std::env::current_exe().unwrap()),
            ..Default::default()
        };
        assert_eq!(
            regs(&here)
                .iter()
                .find(|r| r.backend == ONNX)
                .unwrap()
                .runtime_version
                .as_deref(),
            Some(ORT_VERSION)
        );
    }

    #[test]
    fn shipped_probe_fixture_is_valid_audio_with_the_oracle_text() {
        let (name, wav, text) = probe_fixture("tingmix-large").unwrap();
        assert_eq!((name, text), ("zh_02s", "今天天氣很好"));
        assert!(
            (wav.len() as f64 / 16_000.0 - 1.83).abs() < 0.5,
            "{} samples",
            wav.len()
        );
        assert!(wav.iter().any(|x| x.abs() > 0.05));
        assert!(probe_fixture("unknown").is_none());
        assert!(wav_samples(b"RIFFxxxxWAVE").is_err());
    }

    /// The bundled-library path of the app: `TINGMIX_ORT_DYLIB=<.app>/Contents/Resources/onnxruntime/libonnxruntime.1.31.0.dylib
    /// TINGMIX_TINY_ONNX=tools/small-export/out/tiny.onnx cargo test -p tingmix-app -- --ignored onnx_bundle_smoke --nocapture`
    /// (pipeline check only: the tiny graph has random weights, so the ids mean nothing).
    #[test]
    #[ignore = "needs the fetched ONNX Runtime and the tiny.onnx export"]
    fn onnx_bundle_smoke() {
        let lib = std::path::PathBuf::from(
            std::env::var("TINGMIX_ORT_DYLIB").expect("TINGMIX_ORT_DYLIB"),
        );
        tingmix_runtime_onnx::set_runtime_library(lib);
        let units = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/decoder/units.json");
        let mut b = onnx_backend(
            std::path::Path::new(&std::env::var("TINGMIX_TINY_ONNX").expect("TINGMIX_TINY_ONNX")),
            &units,
        )
        .unwrap();
        assert!(b.info().streaming_native);
        let wav: Vec<f32> = (0..32_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let out = b.transcribe_batch(&[&wav]).unwrap();
        println!("onnx pipeline ran: {} chars", out[0].chars().count());
    }
}
