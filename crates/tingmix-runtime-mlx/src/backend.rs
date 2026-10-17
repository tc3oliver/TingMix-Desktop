//! `AsrBackend` for TingMix-Large on MLX: wav -> Rust log-mel -> MLX tower+reader -> Rust CTC decode.
//! Chunking and joining follow `embed2asr.pipeline.Embed2ASR.transcribe` (29.5 s windows without overlap, tails under
//! 0.1 s dropped, texts joined with "" for CJK-only output and " " otherwise); each chunk is encoded on its own (batch 1).

use std::path::Path;

use tingmix_core::{AsrBackend, BackendError, BackendInfo};
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

    fn decode_chunk(&mut self, wav: &[f32]) -> Result<String, MlxError> {
        let logits = self.chunk_logits(wav)?;
        let v = self.vocab.len();
        let lp = log_softmax_rows(&logits, v);
        Ok(match &mut self.decode {
            Decode::Greedy => self.vocab.decode(&greedy_ids(&lp, v)),
            Decode::BeamLm(lm, cfg) => beam_decode(&lp, &self.vocab, lm, cfg).text,
        })
    }
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
        let mut out = Vec::with_capacity(wavs.len());
        for w in wavs {
            if w.iter().any(|x| !x.is_finite()) {
                return Err(BackendError::Inference("non-finite samples".into()));
            }
            let mut parts = Vec::new();
            for c in w
                .chunks(CHUNK_SAMPLES)
                .filter(|c| c.len() >= MIN_CHUNK_SAMPLES)
            {
                parts.push(
                    self.decode_chunk(c)
                        .map_err(|e| BackendError::Inference(e.to_string()))?,
                );
            }
            out.push(join(parts));
        }
        Ok(out)
    }
}
