//! TingMix-Large through ONNX Runtime: wav -> Rust Gemma4 log-mel -> `tingmix-large.onnx` (audio tower + Reader, dynamic
//! frame count, exported by `experiments/windows/large-onnx`) -> Rust CTC decode. Chunking, joining and confidence follow
//! `tingmix-runtime-mlx`'s `MlxLargeBackend` (and `embed2asr.pipeline.transcribe`): 29.5 s windows without overlap, tails under
//! 0.1 s dropped, CJK-only output joined with "" and otherwise with " ", every window encoded on its own (batch 1).

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tingmix_core::{AsrBackend, BackendError, BackendInfo, Transcription};
use tingmix_decoder::{beam_decode, greedy_ids, log_softmax_rows, BeamConfig, CharLm, Vocab};
use tingmix_frontend_gemma::{Gemma4Frontend, MEL_BINS};

use crate::model::{environment, ort_err, Error};

const CHUNK_SAMPLES: usize = 472_000; // int(29.5 * 16000)
const MIN_CHUNK_SAMPLES: usize = 1_600;

pub enum Decode {
    Greedy,
    /// CTC prefix beam search with the character 5-gram LM (model card: alpha 0.35, beta 1.5, beam 16).
    BeamLm(Box<CharLm>, BeamConfig),
}

/// Execution provider. A provider that is requested must initialise: there is no silent fallback to the CPU, so a measurement
/// or a probe can never attribute CPU work to a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Cpu,
    DirectMl,
    Cuda,
}

pub struct OnnxLarge {
    info: BackendInfo,
    session: Session,
    frontend: Gemma4Frontend,
    vocab: Vocab,
    decode: Decode,
}

impl OnnxLarge {
    /// `model`: `tingmix-large.onnx`; `units_json`: `units.json` of the release.
    pub fn load(
        model: &Path,
        units_json: &Path,
        decode: Decode,
        provider: Provider,
        intra_threads: usize,
    ) -> Result<Self, Error> {
        let env = environment()?;
        let mut builder = Session::builder(&env)
            .map_err(ort_err)?
            .with_intra_threads(intra_threads)
            .map_err(ort_err)?;
        builder = match provider {
            Provider::Cpu => builder,
            #[cfg(feature = "directml")]
            Provider::DirectMl => builder
                .with_execution_providers([ort::ep::DirectML::default().build().error_on_failure()])
                .map_err(ort_err)?,
            #[cfg(feature = "cuda")]
            Provider::Cuda => builder
                .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                .map_err(ort_err)?,
            #[allow(unreachable_patterns)]
            other => {
                return Err(Error::Contract(format!(
                    "{other:?} was not compiled in (cargo features `directml` / `cuda`)"
                )))
            }
        };
        let session = builder.commit_from_file(model).map_err(ort_err)?;
        let vocab =
            Vocab::from_units_json(units_json, true).map_err(|e| Error::Contract(e.to_string()))?;
        let out_ok = session.outputs().iter().any(|o| o.name() == "logits")
            && session.inputs().iter().any(|i| i.name() == "mel");
        if !out_ok {
            return Err(Error::Contract(
                "expected input `mel` and output `logits`".into(),
            ));
        }
        let info = BackendInfo {
            name: "tingmix-large-onnx".into(),
            sample_rate: 16_000,
            max_window_s: 29.5,
            streaming_native: false,
        };
        Ok(Self {
            info,
            session,
            frontend: Gemma4Frontend::new(),
            vocab,
            decode,
        })
    }

    /// Reader logits `[T, vocab]` of one window (at most 29.5 s).
    pub fn chunk_logits(&mut self, wav: &[f32]) -> Result<Vec<f32>, Error> {
        let mel = self.frontend.extract(wav);
        if mel.valid == 0 {
            return Ok(Vec::new());
        }
        let input = Tensor::from_array((
            [1usize, mel.valid, MEL_BINS],
            mel.data[..mel.valid * MEL_BINS].to_vec(),
        ))
        .map_err(ort_err)?;
        let out = self
            .session
            .run(ort::inputs! { "mel" => input })
            .map_err(ort_err)?;
        let (shape, data) = out["logits"].try_extract_tensor::<f32>().map_err(ort_err)?;
        if shape.len() != 3 || shape[2] as usize != self.vocab.len() {
            return Err(Error::Contract(format!(
                "logits shape {:?} does not match a vocabulary of {}",
                &**shape,
                self.vocab.len()
            )));
        }
        Ok(data.to_vec())
    }

    fn decode_chunk(&mut self, wav: &[f32]) -> Result<(String, Option<f32>), Error> {
        let logits = self.chunk_logits(wav)?;
        let v = self.vocab.len();
        let lp = log_softmax_rows(&logits, v);
        let text_conf = non_blank_confidence(&lp, v);
        let text = match &mut self.decode {
            Decode::Greedy => self.vocab.decode(&greedy_ids(&lp, v)),
            Decode::BeamLm(lm, cfg) => beam_decode(&lp, &self.vocab, lm, cfg).text,
        };
        // beam+LM can write characters although every frame's best label is blank: the signature of a hallucination
        let confidence = text_conf.or(if text.trim().is_empty() {
            None
        } else {
            Some(0.0)
        });
        Ok((text, confidence))
    }
}

/// Mean max-posterior over frames whose argmax is not the blank (id 0); `None` when every frame is blank.
fn non_blank_confidence(log_probs: &[f32], vocab: usize) -> Option<f32> {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for row in log_probs.chunks_exact(vocab) {
        let (arg, best) =
            row.iter().enumerate().fold(
                (0, f32::NEG_INFINITY),
                |a, (i, &x)| if x > a.1 { (i, x) } else { a },
            );
        if arg != 0 {
            sum += f64::from(best.exp());
            n += 1;
        }
    }
    (n > 0).then(|| (sum / n as f64) as f32)
}

fn join(parts: Vec<String>) -> String {
    if parts.concat().contains(' ') {
        parts.join(" ")
    } else {
        parts.concat()
    }
}

impl AsrBackend for OnnxLarge {
    fn info(&self) -> &BackendInfo {
        &self.info
    }

    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        Ok(self
            .transcribe_batch_scored(wavs)?
            .into_iter()
            .map(|t| t.text)
            .collect())
    }

    fn transcribe_batch_scored(
        &mut self,
        wavs: &[&[f32]],
    ) -> Result<Vec<Transcription>, BackendError> {
        let mut out = Vec::with_capacity(wavs.len());
        for w in wavs {
            if w.iter().any(|x| !x.is_finite()) {
                return Err(BackendError::Inference("non-finite samples".into()));
            }
            let (mut parts, mut conf) = (Vec::new(), Vec::new());
            for c in w
                .chunks(CHUNK_SAMPLES)
                .filter(|c| c.len() >= MIN_CHUNK_SAMPLES)
            {
                let (text, p) = self
                    .decode_chunk(c)
                    .map_err(|e| BackendError::Inference(e.to_string()))?;
                parts.push(text);
                conf.extend(p);
            }
            // several 29.5 s chunks: the least confident chunk decides
            out.push(Transcription {
                text: join(parts),
                confidence: conf.into_iter().reduce(f32::min),
            });
        }
        Ok(out)
    }
}
