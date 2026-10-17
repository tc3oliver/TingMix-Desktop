//! Timing wrapper around any `AsrBackend`: records every inference call (start, duration, audio length), and can add a
//! simulated model cost (`base + rtf * audio`) on top of a mock so queue behaviour can be studied without a model.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tingmix_core::{AsrBackend, AsrStream, BackendError, BackendInfo};

#[derive(Debug, Clone, Copy)]
pub struct Call {
    pub dur_s: f64,
    pub audio_s: f64,
}

pub struct Rec {
    t0: Instant,
    pub calls: Mutex<Vec<Call>>,
    /// Nanoseconds since `t0` at which the running call started, plus one; 0 = idle.
    inflight: AtomicU64,
}

impl Rec {
    pub fn new(t0: Instant) -> Arc<Self> {
        Arc::new(Self {
            t0,
            calls: Mutex::new(Vec::new()),
            inflight: AtomicU64::new(0),
        })
    }

    /// Age in seconds of the call currently running, if any.
    pub fn inflight_age(&self) -> Option<f64> {
        match self.inflight.load(Ordering::Relaxed) {
            0 => None,
            n => Some(self.t0.elapsed().as_secs_f64() - (n - 1) as f64 / 1e9),
        }
    }

    fn timed<T>(&self, audio_samples: usize, sr: u32, sim: Sim, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        self.inflight.store(
            start.duration_since(self.t0).as_nanos() as u64 + 1,
            Ordering::Relaxed,
        );
        let audio_s = audio_samples as f64 / f64::from(sr);
        std::thread::sleep(Duration::from_secs_f64(sim.base_s + sim.rtf * audio_s));
        let r = f();
        let dur_s = start.elapsed().as_secs_f64();
        self.inflight.store(0, Ordering::Relaxed);
        self.calls.lock().unwrap().push(Call { dur_s, audio_s });
        r
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Sim {
    pub base_s: f64,
    pub rtf: f64,
}

pub struct Timed<B> {
    pub inner: B,
    pub rec: Arc<Rec>,
    pub sim: Sim,
}

impl<B: AsrBackend> AsrBackend for Timed<B> {
    fn info(&self) -> &BackendInfo {
        self.inner.info()
    }

    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        let (n, sr) = (
            wavs.iter().map(|w| w.len()).sum(),
            self.inner.info().sample_rate,
        );
        let (inner, rec, sim) = (&mut self.inner, &self.rec, self.sim);
        rec.timed(n, sr, sim, || inner.transcribe_batch(wavs))
    }

    fn new_stream(&mut self) -> Result<Box<dyn AsrStream>, BackendError> {
        let (rec, sim, sr) = (self.rec.clone(), self.sim, self.inner.info().sample_rate);
        Ok(Box::new(TimedStream {
            inner: self.inner.new_stream()?,
            rec,
            sim,
            sr,
        }))
    }
}

struct TimedStream {
    inner: Box<dyn AsrStream>,
    rec: Arc<Rec>,
    sim: Sim,
    sr: u32,
}

impl AsrStream for TimedStream {
    fn accept(&mut self, pcm: &[f32]) -> Result<String, BackendError> {
        let (inner, rec, sim, sr) = (&mut self.inner, &self.rec, self.sim, self.sr);
        rec.timed(pcm.len(), sr, sim, || inner.accept(pcm))
    }

    fn finalize(&mut self) -> Result<String, BackendError> {
        let (inner, rec, sim, sr) = (&mut self.inner, &self.rec, self.sim, self.sr);
        rec.timed(0, sr, sim, || inner.finalize())
    }
}
