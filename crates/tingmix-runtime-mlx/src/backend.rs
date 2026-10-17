//! `AsrBackend` for TingMix-Large on MLX: wav -> Rust log-mel -> MLX tower+reader -> Rust CTC decode.
//! Chunking and joining follow `embed2asr.pipeline.Embed2ASR.transcribe` (29.5 s windows without overlap, tails under
//! 0.1 s dropped, texts joined with "" for CJK-only output and " " otherwise); each chunk is encoded on its own (batch 1).

use std::path::Path;

use tingmix_core::{AsrBackend, BackendError, BackendInfo, Transcription};
use tingmix_decoder::{beam_decode, greedy_ids, log_softmax_rows, BeamConfig, CharLm, Vocab};
use tingmix_frontend_gemma::Gemma4Frontend;

use crate::{Dtype, MlxError, MlxLarge};

const CHUNK_SAMPLES: usize = 472_000; // int(29.5 * 16000)
const MIN_CHUNK_SAMPLES: usize = 1_600;

pub enum Decode {
    Greedy,
    /// CTC prefix beam search with the character 5-gram LM (model card: alpha 0.35, beta 1.5, beam 16).
    BeamLm(Box<CharLm>, BeamConfig),
}

pub struct MlxLargeBackend {
    info: BackendInfo,
    model: MlxLarge,
    frontend: Gemma4Frontend,
    vocab: Vocab,
    decode: Decode,
}

impl MlxLargeBackend {
    /// `artifact`: fp32 safetensors from `experiments/macos/mlx/convert.py`; `units_json`: `units.json` of the release.
    pub fn new(
        artifact: &Path,
        units_json: &Path,
        dtype: Dtype,
        decode: Decode,
    ) -> Result<Self, MlxError> {
        let model = MlxLarge::load(artifact, dtype)?;
        let vocab =
            Vocab::from_units_json(units_json, true).map_err(|e| MlxError::Mlx(e.to_string()))?;
        if model.vocab() != vocab.len() {
            return Err(MlxError::Mlx(format!(
                "artifact vocab {} != units.json {}",
                model.vocab(),
                vocab.len()
            )));
        }
        let info = BackendInfo {
            name: "tingmix-large-mlx".into(),
            sample_rate: 16_000,
            max_window_s: 29.5,
            streaming_native: false,
        };
        Ok(Self {
            info,
            model,
            frontend: Gemma4Frontend::new(),
            vocab,
            decode,
        })
    }

    /// Reader logits `[T, vocab]` of one window (at most 29.5 s).
    pub fn chunk_logits(&mut self, wav: &[f32]) -> Result<Vec<f32>, MlxError> {
        let mel = self.frontend.extract(wav);
        if mel.valid == 0 {
            return Ok(Vec::new());
        }
        self.model.logits(&mel.data, mel.valid)
    }

    fn decode_chunk(&mut self, wav: &[f32]) -> Result<(String, Option<f32>), MlxError> {
        let logits = self.chunk_logits(wav)?;
        let v = self.vocab.len();
        let lp = log_softmax_rows(&logits, v);
        let confidence = non_blank_confidence(&lp, v);
        let text = match &mut self.decode {
            Decode::Greedy => self.vocab.decode(&greedy_ids(&lp, v)),
            Decode::BeamLm(lm, cfg) => beam_decode(&lp, &self.vocab, lm, cfg).text,
        };
        Ok((text, confidence))
    }
}

/// Mean max-posterior over frames whose argmax is not the blank (id 0): how sure the reader is about what it wrote.
/// `None` when every frame is blank (nothing was written).
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

impl AsrBackend for MlxLargeBackend {
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
