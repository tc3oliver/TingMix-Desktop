//! Builds a backend on the thread that first calls it (for runtimes whose handles are thread-bound, like MLX).

use tingmix_core::{AsrBackend, AsrStream, BackendError, BackendInfo, Transcription};

pub struct Lazy<B, F> {
    pub info: BackendInfo,
    pub make: Option<F>,
    pub inner: Option<B>,
}

impl<B: AsrBackend, F: FnOnce() -> Result<B, String> + Send> Lazy<B, F> {
    fn get(&mut self) -> Result<&mut B, BackendError> {
        if self.inner.is_none() {
            let make = self
                .make
                .take()
                .ok_or_else(|| BackendError::Inference("backend failed to load earlier".into()))?;
            self.inner = Some(make().map_err(BackendError::Inference)?);
        }
        Ok(self.inner.as_mut().unwrap())
    }
}

impl<B: AsrBackend, F: FnOnce() -> Result<B, String> + Send> AsrBackend for Lazy<B, F> {
    fn info(&self) -> &BackendInfo {
        &self.info
    }

    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        self.get()?.transcribe_batch(wavs)
    }

    fn transcribe_batch_scored(
        &mut self,
        wavs: &[&[f32]],
    ) -> Result<Vec<Transcription>, BackendError> {
        self.get()?.transcribe_batch_scored(wavs)
    }

    fn new_stream(&mut self) -> Result<Box<dyn AsrStream>, BackendError> {
        self.get()?.new_stream()
    }
}
