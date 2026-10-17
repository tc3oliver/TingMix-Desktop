//! Runtime selection for the host: which `AsrBackend` the next session uses.
//!
//! Policy: the mock backend ("Mock — not ASR") is the default. A real backend is used only when the user activated an
//! installed model AND `tingmix-models` selection accepts it (backend registered, runtime present, requirements met, a
//! fresh passing compatibility probe). Both real backends are EXPERIMENTAL and `select` refuses non-VALIDATED backends,
//! so they are registered as eligible only while "experimental backends" is switched on in Settings. A real backend
//! that fails to load is an error, never a silent switch to the mock.

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

/// Boxed backend that is itself an `AsrBackend`, so it can go into `SessionWorker` / `StreamingSession`.
pub struct Dyn(Box<dyn AsrBackend>);

impl Dyn {
    pub fn mock() -> Self {
        Dyn(Box::new(MockBackend::new(false)))
    }
}

impl AsrBackend for Dyn {
    fn info(&self) -> &BackendInfo {
        self.0.info()
    }
    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        self.0.transcribe_batch(wavs)
    }
    fn new_stream(&mut self) -> Result<Box<dyn AsrStream>, BackendError> {
        self.0.new_stream()
    }
}

/// Where the bundled ONNX Runtime library is (resolved once at start-up), if anywhere.
#[derive(Clone, Default)]
pub struct Runtimes {
    pub ort_library: Option<PathBuf>,
}

impl Runtimes {
    fn ort_version(&self) -> Option<String> {
        self.ort_library
            .as_ref()
            .filter(|p| p.is_file())
            .map(|_| ORT_VERSION.to_string())
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

pub fn regs(experimental_ok: bool, rt: &Runtimes) -> Vec<BackendReg> {
    let validation = if experimental_ok {
        Validation::Validated
    } else {
        Validation::Experimental
    };
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
    pub experimental_allowed: bool,
}

pub struct Decision {
    /// `(manifest id, backend)` when a real backend was accepted.
    pub chosen: Option<(String, String)>,
    pub reason: String,
    pub rejected: Vec<(String, String)>,
}

pub fn decide(models: &Models, experimental_ok: bool, rt: &Runtimes) -> Decision {
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
        return none("No model is active; using the mock backend.".into(), vec![]);
    };
    let model = match reg.open_model(&active) {
        Ok(m) => m,
        Err(e) => {
            return none(
                format!("Active model {active} cannot be opened ({e}); using the mock backend."),
                vec![],
            )
        }
    };
    let regs = regs(experimental_ok, rt);
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
            let why = rejected.first().map(|r| r.1.clone()).unwrap_or_default();
            let hint = if why.contains("EXPERIMENTAL") {
                " Enable experimental backends in Settings to allow it."
            } else {
                ""
            };
            none(
                format!("Active model {active} cannot run: {why}.{hint} Using the mock backend."),
                rejected,
            )
        }
    }
}

pub fn info(models: &Models, experimental_ok: bool, rt: &Runtimes) -> BackendInfoDto {
    let d = decide(models, experimental_ok, rt);
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
        experimental_allowed: experimental_ok,
    }
}

pub struct Resolved {
    pub backend: Dyn,
    pub label: String,
}

/// The backend for the next session, loaded. A failing real backend is an error.
pub fn resolve(models: &Models, experimental_ok: bool, rt: &Runtimes) -> Result<Resolved, String> {
    let i = info(models, experimental_ok, rt);
    let d = decide(models, experimental_ok, rt);
    let Some((id, backend)) = d.chosen else {
        return Ok(Resolved {
            backend: Dyn::mock(),
            label: i.label,
        });
    };
    let model = models.reg()?.open_model(&id).map_err(|e| e.to_string())?;
    let backend = build(&model, &backend).map_err(|e| format!("could not load {id}: {e}"))?;
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
pub fn build(model: &InstalledModel, backend: &str) -> Result<Dyn, String> {
    match backend {
        ONNX => {
            let graph = model
                .manifest
                .files
                .iter()
                .find(|f| f.path.ends_with(".onnx"))
                .map(|f| model.dir.join(&f.path))
                .ok_or("manifest lists no .onnx file")?;
            onnx_backend(&graph, &component(model, "tokenizer", Some("units.json"))?)
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
            Ok(Dyn(Box::new(b)))
        }
        other => Err(format!("backend {other} is not available in this build")),
    }
}

fn onnx_backend(graph: &std::path::Path, units_json: &std::path::Path) -> Result<Dyn, String> {
    let vocab =
        tingmix_decoder::Vocab::from_units_json(units_json, true).map_err(|e| e.to_string())?;
    let decode: tingmix_runtime_onnx::TokenDecoder = Arc::new(move |ids| vocab.decode(ids));
    let b = tingmix_runtime_onnx::OnnxSmall::load(graph, 1, decode).map_err(|e| e.to_string())?;
    Ok(Dyn(Box::new(b)))
}

// ---- compatibility probe ------------------------------------------------------------------------------------------

struct AppProbe<'a> {
    backend: &'static str,
    rt: &'a Runtimes,
    loaded: Option<(String, Dyn)>,
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
    Some(match MlxProbeTarget::sibling_worker() {
        Ok(exe) => run_probe(
            root,
            &mut MlxProbeTarget::new(exe, Dtype::F32, true),
            model,
            fx,
            2,
        )
        .map_err(|e| e.to_string()),
        Err(e) => Err(format!("MLX probe worker unavailable: {e}")),
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
            let r = models.reg().and_then(|reg| {
                let fx = ProbeFixture {
                    name,
                    wav: &wav,
                    sample_rate: 16_000,
                    expected_text: expected,
                };
                let rec = (if backend == MLX {
                    probe_mlx(reg.root(), &model, &fx)
                } else {
                    None
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
            });
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

    #[test]
    fn mock_is_the_default_and_the_label_is_honest() {
        let (m, dir) = models("default");
        let i = info(&m, false, &Runtimes::default());
        assert!(i.mock && !i.experimental);
        assert_eq!(i.label, BACKEND_LABEL);
        assert!(i.reason.contains("No model is active"));
        assert_eq!(
            resolve(&m, true, &Runtimes::default()).unwrap().label,
            BACKEND_LABEL
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn real_backends_are_experimental_unless_the_user_opts_in_and_need_their_runtime() {
        let off = regs(false, &Runtimes::default());
        assert!(off.iter().all(|r| r.validation == Validation::Experimental));
        let on = regs(true, &Runtimes::default());
        assert!(on.iter().all(|r| r.validation == Validation::Validated));
        // ONNX needs the library file to exist
        assert_eq!(
            on.iter()
                .find(|r| r.backend == ONNX)
                .unwrap()
                .runtime_version,
            None
        );
        let here = Runtimes {
            ort_library: Some(std::env::current_exe().unwrap()),
        };
        assert_eq!(
            regs(true, &here)
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
