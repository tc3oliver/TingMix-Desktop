use std::time::Duration;

use tingmix_core::{AsrBackend, AsrStream, BackendError, BackendInfo};

/// Deterministic backend for tests and UI development; it recognizes nothing. The text is a pure
/// function of the audio length: `words_per_second` words `w1 w2 ...`, with a trailing `~` on an
/// odd word count so partials revise (and stable prefixes differ) as audio grows.
/// Not an ASR model: never ship it as a transcription feature.
pub struct MockBackend {
    info: BackendInfo,
    words_per_second: u32,
    delay: Duration,
}

impl MockBackend {
    pub fn new(native: bool) -> Self {
        let info = BackendInfo {
            name: "mock".into(),
            sample_rate: 16_000,
            max_window_s: 30.0,
            streaming_native: native,
        };
        Self {
            info,
            words_per_second: 2,
            delay: Duration::ZERO,
        }
    }

    /// Sleep this long per decode call, to simulate a slow model.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    fn text(&self, samples: usize) -> String {
        mock_text(samples, self.info.sample_rate, self.words_per_second)
    }
}

fn mock_text(samples: usize, sr: u32, wps: u32) -> String {
    let n = samples * wps as usize / sr as usize;
    let mut words: Vec<String> = (1..=n).map(|i| format!("w{i}")).collect();
    if n % 2 == 1 {
        words[n - 1].push('~');
    }
    words.join(" ")
}

impl AsrBackend for MockBackend {
    fn info(&self) -> &BackendInfo {
        &self.info
    }

    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        std::thread::sleep(self.delay);
        Ok(wavs.iter().map(|w| self.text(w.len())).collect())
    }

    fn new_stream(&mut self) -> Result<Box<dyn AsrStream>, BackendError> {
        if !self.info.streaming_native {
            return Err(BackendError::Unsupported("native streaming"));
        }
        Ok(Box::new(MockStream {
            total: 0,
            sr: self.info.sample_rate,
            wps: self.words_per_second,
            delay: self.delay,
        }))
    }
}

struct MockStream {
    total: usize,
    sr: u32,
    wps: u32,
    delay: Duration,
}

impl AsrStream for MockStream {
    fn accept(&mut self, pcm: &[f32]) -> Result<String, BackendError> {
        std::thread::sleep(self.delay);
        self.total += pcm.len();
        Ok(mock_text(self.total, self.sr, self.wps))
    }

    fn finalize(&mut self) -> Result<String, BackendError> {
        Ok(mock_text(self.total, self.sr, self.wps))
    }
}
