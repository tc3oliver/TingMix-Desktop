use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

pub const SR: f64 = 16_000.0;
pub const N_FFT: usize = 512;
pub const WIN: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 80;
const BINS: usize = N_FFT / 2 + 1;
const LOG_FLOOR: f32 = 1e-6;

/// Causal log-mel, numerically the reference `LogMel`: STFT without centring (frame `t` covers
/// samples `[t*HOP, t*HOP + N_FFT)`), periodic Hann of 400 centred in the 512-point FFT, power
/// spectrum, HTK-mel triangular filters 20-7600 Hz with area normalisation, `ln(max(x, 1e-6))`.
pub struct LogMel {
    window: [f32; N_FFT],
    fb: Vec<f32>, // [N_MELS, BINS]
    fft: Arc<dyn Fft<f32>>,
}

impl Default for LogMel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogMel {
    pub fn new() -> Self {
        let mut window = [0f32; N_FFT];
        let pad = (N_FFT - WIN) / 2;
        for n in 0..WIN {
            window[pad + n] = (0.5 - 0.5 * (2.0 * PI * n as f64 / WIN as f64).cos()) as f32;
        }
        Self {
            window,
            fb: mel_filters(),
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    /// Number of frames `n_samples` samples yield.
    pub fn frames(n_samples: usize) -> usize {
        if n_samples < N_FFT {
            0
        } else {
            (n_samples - N_FFT) / HOP + 1
        }
    }

    /// Log-mel of a whole waveform, row-major `[frames, N_MELS]`.
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let n = Self::frames(samples.len());
        let mut out = Vec::with_capacity(n * N_MELS);
        let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut power = [0f32; BINS];
        for t in 0..n {
            let frame = &samples[t * HOP..t * HOP + N_FFT];
            for (b, (&s, &w)) in buf.iter_mut().zip(frame.iter().zip(&self.window)) {
                *b = Complex32::new(s * w, 0.0);
            }
            self.fft.process(&mut buf);
            for (p, c) in power.iter_mut().zip(&buf) {
                *p = c.re * c.re + c.im * c.im;
            }
            for row in self.fb.chunks_exact(BINS) {
                let e: f32 = row.iter().zip(&power).map(|(a, b)| a * b).sum();
                out.push(e.max(LOG_FLOOR).ln());
            }
        }
        out
    }
}

fn mel_filters() -> Vec<f32> {
    let hz2mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let (m0, m1) = (hz2mel(20.0), hz2mel(7600.0));
    let hz: Vec<f64> = (0..N_MELS + 2)
        .map(|i| {
            700.0 * (10f64.powf((m0 + (m1 - m0) * i as f64 / (N_MELS + 1) as f64) / 2595.0) - 1.0)
        })
        .collect();
    let mut fb = vec![0f32; N_MELS * BINS];
    for m in 0..N_MELS {
        let (lo, ce, hi) = (hz[m], hz[m + 1], hz[m + 2]);
        for k in 0..BINS {
            let f = SR / 2.0 * k as f64 / (BINS - 1) as f64;
            let v = ((f - lo) / (ce - lo)).min((hi - f) / (hi - ce)).max(0.0);
            fb[m * BINS + k] = (v * 2.0 / (hi - lo)) as f32;
        }
    }
    fb
}

/// Incremental frontend: keeps the sample remainder between calls, so any chunking yields the
/// same frames as one pass over the concatenated audio.
pub struct MelStream {
    mel: LogMel,
    samples: Vec<f32>,
}

impl Default for MelStream {
    fn default() -> Self {
        Self::new()
    }
}

impl MelStream {
    pub fn new() -> Self {
        Self {
            mel: LogMel::new(),
            samples: Vec::new(),
        }
    }

    /// Add samples; returns the newly complete frames, row-major `[n, N_MELS]`.
    pub fn push(&mut self, pcm: &[f32]) -> Vec<f32> {
        self.samples.extend_from_slice(pcm);
        let n = LogMel::frames(self.samples.len());
        if n == 0 {
            return Vec::new();
        }
        let out = self.mel.compute(&self.samples[..(n - 1) * HOP + N_FFT]);
        self.samples.drain(..n * HOP);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize) -> Vec<f32> {
        let mut s = 12345u32;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn frame_counts_match_the_reference() {
        assert_eq!(LogMel::frames(511), 0);
        assert_eq!(LogMel::frames(512), 1);
        assert_eq!(LogMel::frames(512 + 159), 1);
        assert_eq!(LogMel::frames(512 + 160), 2);
    }

    #[test]
    fn digital_silence_hits_the_log_floor() {
        let m = LogMel::new().compute(&vec![0.0; 1024]);
        assert!(m.iter().all(|&v| v == LOG_FLOOR.ln()));
    }

    #[test]
    fn any_chunking_gives_the_same_frames() {
        let x = noise(16_000);
        let whole = LogMel::new().compute(&x);
        for chunk in [1usize, 37, 160, 513, 2560] {
            let mut s = MelStream::new();
            let got: Vec<f32> = x.chunks(chunk).flat_map(|c| s.push(c)).collect();
            assert_eq!(got, whole, "chunk {chunk}");
        }
    }
}
