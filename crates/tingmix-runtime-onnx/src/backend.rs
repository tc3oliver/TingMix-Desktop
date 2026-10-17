use std::path::Path;
use std::sync::{Arc, Mutex};

use tingmix_core::{AsrBackend, AsrStream, BackendError, BackendInfo};

use crate::model::{Error, SharedModel, StepModel, Streamer};

/// Maps collapsed CTC ids to text (vocabulary / SentencePiece handling lives with the release's
/// `units.json`, not here).
pub type TokenDecoder = Arc<dyn Fn(&[u32]) -> String + Send + Sync>;

/// TingMix-Small on ONNX Runtime CPU. Native streaming: each utterance is a [`SmallStream`]
/// that feeds audio once and keeps the conv/GRU state.
pub struct OnnxSmall {
    info: BackendInfo,
    model: SharedModel,
    decode: TokenDecoder,
}

impl OnnxSmall {
    pub fn load(
        model_path: &Path,
        intra_threads: usize,
        decode: TokenDecoder,
    ) -> Result<Self, Error> {
        let model = Arc::new(Mutex::new(StepModel::load(model_path, intra_threads)?));
        let info = BackendInfo {
            name: "tingmix-small-onnx".into(),
            sample_rate: 16_000,
            max_window_s: f64::INFINITY,
            streaming_native: true,
        };
        Ok(Self {
            info,
            model,
            decode,
        })
    }

    fn stream(&self) -> SmallStream {
        SmallStream {
            inner: Streamer::new(self.model.clone()),
            decode: self.decode.clone(),
        }
    }
}

impl AsrBackend for OnnxSmall {
    fn info(&self) -> &BackendInfo {
        &self.info
    }

    /// Offline decode through the same graph (a fresh stream per waveform).
    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        wavs.iter()
            .map(|w| {
                let mut s = self.stream();
                s.accept(w)?;
                s.finalize()
            })
            .collect()
    }

    fn new_stream(&mut self) -> Result<Box<dyn AsrStream>, BackendError> {
        Ok(Box::new(self.stream()))
    }
}

pub struct SmallStream {
    inner: Streamer,
    decode: TokenDecoder,
}

impl AsrStream for SmallStream {
    fn accept(&mut self, pcm: &[f32]) -> Result<String, BackendError> {
        self.inner
            .push(pcm)
            .map_err(|e| BackendError::Inference(e.to_string()))?;
        Ok((self.decode)(self.inner.ids()))
    }

    /// Nothing is left to flush inside the model: every complete 4-frame block was already run by
    /// `accept`, and a trailing partial block is dropped like in the offline reference.
    fn finalize(&mut self) -> Result<String, BackendError> {
        Ok((self.decode)(self.inner.ids()))
    }
}
