use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

use ort::session::Session;
use ort::value::Tensor;

use crate::ctc::Collapser;
use crate::frontend::{MelStream, N_MELS};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("onnx runtime: {0}")]
    Ort(String),
    #[error("model does not match its graph contract: {0}")]
    Contract(String),
}

pub(crate) fn ort_err(e: impl std::fmt::Display) -> Error {
    Error::Ort(e.to_string())
}

static LIBRARY: OnceLock<PathBuf> = OnceLock::new();

/// Where to load the ONNX Runtime shared library from. Only used by builds with the `load-dynamic`
/// feature (release bundles); call it once before the first model load. Fallbacks, in order:
/// `TINGMIX_ORT_DYLIB`, `ORT_DYLIB_PATH`. Ignored by `download-binaries` builds, which link ORT.
pub fn set_runtime_library(path: impl Into<PathBuf>) {
    let _ = LIBRARY.set(path.into());
}

static ENVIRONMENT: OnceLock<ort::environment::Environment> = OnceLock::new();

/// The process-wide ORT environment (ORT allows exactly one; every session of every backend shares it).
pub(crate) fn environment() -> Result<ort::environment::Environment, Error> {
    if let Some(e) = ENVIRONMENT.get() {
        return Ok(e.clone());
    }
    let env = build_environment()?;
    Ok(ENVIRONMENT.get_or_init(|| env).clone())
}

#[cfg(feature = "load-dynamic")]
fn build_environment() -> Result<ort::environment::Environment, Error> {
    let path = LIBRARY
        .get()
        .cloned()
        .or_else(|| {
            ["TINGMIX_ORT_DYLIB", "ORT_DYLIB_PATH"]
                .iter()
                .find_map(|k| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from))
        })
        .ok_or_else(|| Error::Ort("no ONNX Runtime library configured: call set_runtime_library or set TINGMIX_ORT_DYLIB".into()))?;
    if !path.is_absolute() {
        return Err(Error::Ort(format!(
            "ONNX Runtime library path must be absolute: {}",
            path.display()
        )));
    }
    ort::init_from(&path)
        .map_err(|e| Error::Ort(e.to_string()))?
        .build()
        .map_err(ort_err)
}

#[cfg(not(feature = "load-dynamic"))]
fn build_environment() -> Result<ort::environment::Environment, Error> {
    ort::init().build().map_err(ort_err)
}

/// Streaming state carried between steps: last normalized mel frame, last conv1 frame, GRU hidden.
#[derive(Clone)]
pub struct StepState {
    c1: Vec<f32>,
    c2: Vec<f32>,
    h: Vec<f32>,
}

/// The ONNX streaming-step graph on the CPU execution provider.
pub struct StepModel {
    session: Session,
    d: usize,
    layers: usize,
    hidden: usize,
    vocab: usize,
}

impl StepModel {
    /// `intra_threads`: ORT intra-op thread count (the whole inference runs on the caller's thread otherwise).
    pub fn load(path: &Path, intra_threads: usize) -> Result<Self, Error> {
        let env = environment()?;
        let session = Session::builder(&env)
            .map_err(ort_err)?
            .with_intra_threads(intra_threads)
            .map_err(ort_err)?
            .with_inter_threads(1)
            .map_err(ort_err)?
            .commit_from_file(path)
            .map_err(ort_err)?;
        let dim = |name: &str, from_output: bool, axis: usize| -> Result<usize, Error> {
            let list = if from_output {
                session.outputs()
            } else {
                session.inputs()
            };
            let o = list
                .iter()
                .find(|o| o.name() == name)
                .ok_or_else(|| Error::Contract(format!("missing `{name}`")))?;
            let shape = o
                .dtype()
                .tensor_shape()
                .ok_or_else(|| Error::Contract(format!("`{name}` is not a tensor")))?;
            usize::try_from(shape[axis])
                .map_err(|_| Error::Contract(format!("`{name}` axis {axis} is dynamic")))
        };
        let (d, layers, hidden) = (
            dim("c2", false, 1)?,
            dim("h", false, 0)?,
            dim("h", false, 2)?,
        );
        let vocab = dim("logp", true, 2)?;
        Ok(Self {
            session,
            d,
            layers,
            hidden,
            vocab,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    pub fn zero_state(&self) -> StepState {
        StepState {
            c1: vec![0.0; N_MELS],
            c2: vec![0.0; self.d],
            h: vec![0.0; self.layers * self.hidden],
        }
    }

    /// Run `k` mel frames (`mel.len() == k * 80`, `k` a multiple of 4); returns `[k/4, V]` log-probs
    /// and advances `state`.
    pub fn step(&mut self, state: &mut StepState, mel: &[f32]) -> Result<Vec<f32>, Error> {
        let k = mel.len() / N_MELS;
        if k == 0 || k % 4 != 0 || mel.len() != k * N_MELS {
            return Err(Error::Contract(format!(
                "mel must be a non-empty multiple of 4 frames, got {} values",
                mel.len()
            )));
        }
        let t = |shape: [usize; 3], data: &[f32]| {
            Tensor::from_array((shape, data.to_vec())).map_err(ort_err)
        };
        let inputs = ort::inputs! {
            "mel" => t([1, k, N_MELS], mel)?,
            "c1" => t([1, N_MELS, 1], &state.c1)?,
            "c2" => t([1, self.d, 1], &state.c2)?,
            "h" => t([self.layers, 1, self.hidden], &state.h)?,
        };
        let out = self.session.run(inputs).map_err(ort_err)?;
        let get = |name: &str| -> Result<Vec<f32>, Error> {
            let (_, data) = out[name].try_extract_tensor::<f32>().map_err(ort_err)?;
            Ok(data.to_vec())
        };
        let logp = get("logp")?;
        if logp.len() != k / 4 * self.vocab {
            return Err(Error::Contract(format!(
                "logp has {} values, expected {}",
                logp.len(),
                k / 4 * self.vocab
            )));
        }
        (state.c1, state.c2, state.h) = (get("c1_out")?, get("c2_out")?, get("h_out")?);
        Ok(logp)
    }
}

/// The session shared by every stream of a backend; one step runs at a time.
pub type SharedModel = Arc<Mutex<StepModel>>;

/// Audio in, log-probs and collapsed CTC ids out, for one utterance. Samples that do not yet
/// complete 4 mel frames stay buffered; at the end of the utterance they are dropped, exactly as
/// the offline reference drops a trailing partial block.
pub struct Streamer {
    model: SharedModel,
    mel: MelStream,
    pending: Vec<f32>,
    state: StepState,
    ctc: Collapser,
}

impl Streamer {
    pub fn new(model: SharedModel) -> Self {
        let state = model.lock().unwrap().zero_state();
        Self {
            model,
            mel: MelStream::new(),
            pending: Vec::new(),
            state,
            ctc: Collapser::new(),
        }
    }

    /// Feed samples; returns the new log-prob rows `[n, V]` (possibly empty).
    pub fn push(&mut self, pcm: &[f32]) -> Result<Vec<f32>, Error> {
        self.pending.extend(self.mel.push(pcm));
        let k = self.pending.len() / N_MELS / 4 * 4;
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut model = self.model.lock().unwrap();
        let logp = model.step(&mut self.state, &self.pending[..k * N_MELS])?;
        self.pending.drain(..k * N_MELS);
        let raw: Vec<u32> = logp.chunks_exact(model.vocab_size()).map(argmax).collect();
        self.ctc.push(&raw);
        Ok(logp)
    }

    /// Collapsed CTC ids so far.
    pub fn ids(&self) -> &[u32] {
        self.ctc.ids()
    }
}

/// First maximum wins, like `np.argmax`.
fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in row.iter().enumerate() {
        if x > row[best] {
            best = i;
        }
    }
    best as u32
}

/// Version string of the loaded ONNX Runtime library (`1.31.0`), as the library itself reports it; `None` when it cannot be
/// loaded. Loads the library if it is not loaded yet (the same one `StepModel::load` / `OnnxLarge::load` would use).
pub fn runtime_version() -> Option<String> {
    environment().ok()?;
    #[cfg(feature = "load-dynamic")]
    let base = unsafe {
        let path = LIBRARY.get().cloned().or_else(|| {
            ["TINGMIX_ORT_DYLIB", "ORT_DYLIB_PATH"]
                .iter()
                .find_map(|k| {
                    std::env::var_os(k)
                        .filter(|v| !v.is_empty())
                        .map(PathBuf::from)
                })
        })?;
        let lib = libloading::Library::new(path).ok()?; // already loaded by `environment`: dlopen/LoadLibrary returns the same handle
        let get: libloading::Symbol<unsafe extern "system" fn() -> *const ort::sys::OrtApiBase> =
            lib.get(b"OrtGetApiBase").ok()?;
        let base = get();
        std::mem::forget(lib); // keep it mapped; ort holds the same module
        base
    };
    #[cfg(not(feature = "load-dynamic"))]
    let base = unsafe { ort::sys::OrtGetApiBase() };
    if base.is_null() {
        return None;
    }
    let s = unsafe { (*base).GetVersionString };
    let p = unsafe { s() };
    (!p.is_null()).then(|| {
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    })
}
