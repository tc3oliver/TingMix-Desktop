/// Static facts about a loaded model, used by the streaming session to configure itself.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendInfo {
    pub name: String,
    /// Sample rate of the mono `f32` PCM the backend consumes.
    pub sample_rate: u32,
    /// Longest audio one `transcribe_batch` item should cover.
    pub max_window_s: f64,
    /// `false`: offline model, live transcription is buffered re-decoding (TingMix-Large).
    /// `true`: stateful streaming through [`AsrBackend::new_stream`] (TingMix-Small).
    pub streaming_native: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("backend does not support {0}")]
    Unsupported(&'static str),
    #[error("inference failed: {0}")]
    Inference(String),
}

/// An interchangeable inference backend. Implementations must be `Send` (they run on a worker
/// thread) and may keep any model state privately; only PCM and text cross this interface.
pub trait AsrBackend: Send {
    fn info(&self) -> &BackendInfo;

    /// Offline decode: mono `f32` waveforms at `info().sample_rate` -> texts, one per input.
    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError>;

    /// Open one utterance of a natively streaming model. Only called when `info().streaming_native`.
    fn new_stream(&mut self) -> Result<Box<dyn AsrStream>, BackendError> {
        Err(BackendError::Unsupported("native streaming"))
    }
}

/// One utterance of a natively streaming model: audio is consumed once and the model keeps its own state.
pub trait AsrStream: Send {
    /// Feed samples; returns the current (revisable) hypothesis of this utterance.
    fn accept(&mut self, pcm: &[f32]) -> Result<String, BackendError>;

    /// No more audio: the final text of this utterance. Called exactly once, last.
    fn finalize(&mut self) -> Result<String, BackendError>;
}
