//! Port of `transformers.models.gemma4.feature_extraction_gemma4.Gemma4AudioFeatureExtractor` (transformers 5.19.0) with the
//! google/embeddinggemma-2@914f7f8 preprocessor settings: 16 kHz, 20 ms frame (320), 10 ms hop (160), FFT 512, periodic Hann,
//! no preemphasis/dither, 128 HTK mel bins 0-8000 Hz (magnitude, not power), `ln(mel + 1e-3)`, no per-bin normalisation.
//!
//! Input is truncated to 480 000 samples and zero-padded on the right to a multiple of 128; a frame is valid when its centre
//! sample (`i * 160 + 160`) is real audio; padded frames are zeroed. FFT and mel run in f64 (numpy computes the FFT in f32), so
//! values agree with the reference to ~1e-5 absolute on the log scale rather than bit-exactly.

use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

pub const SAMPLE_RATE: usize = 16_000;
pub const MEL_BINS: usize = 128;
const FRAME: usize = 320;
const HOP: usize = 160;
const NFFT: usize = 512;
const BINS: usize = NFFT / 2 + 1;
const MAX_SAMPLES: usize = 480_000;
const PAD_MULTIPLE: usize = 128;
const MEL_FLOOR: f64 = 1e-3;

/// Row-major `[frames, 128]` log-mel; the first `valid` rows are real, the rest are zero.
#[derive(Debug, Clone, PartialEq)]
pub struct Mel {
    pub data: Vec<f32>,
    pub frames: usize,
    pub valid: usize,
}

pub struct Gemma4Frontend {
    window: [f32; FRAME],
    /// `[BINS, MEL_BINS]` row-major, f64 like the reference filter bank.
    filters: Vec<f64>,
    /// Per mel bin the inclusive FFT-bin range holding its non-zero triangle (skipping zeros leaves the sums unchanged).
    span: Vec<(usize, usize)>,
    fft: Arc<dyn Fft<f64>>,
}

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    // numpy: start + i * step, last element forced to stop
    let step = (b - a) / (n - 1) as f64;
    (0..n)
        .map(|i| if i == n - 1 { b } else { a + i as f64 * step })
        .collect()
}

impl Default for Gemma4Frontend {
    fn default() -> Self {
        Self::new()
    }
}

impl Gemma4Frontend {
    pub fn new() -> Self {
        // np.hanning(321)[:320] == 0.5 - 0.5 cos(2 pi n / 320)
        let mut window = [0f32; FRAME];
        for (n, w) in window.iter_mut().enumerate() {
            *w = (0.5 - 0.5 * (2.0 * PI * n as f64 / FRAME as f64).cos()) as f32;
        }
        let filter_freqs: Vec<f64> = linspace(hz_to_mel(0.0), hz_to_mel(8000.0), MEL_BINS + 2)
            .into_iter()
            .map(mel_to_hz)
            .collect();
        let fft_freqs = linspace(0.0, (SAMPLE_RATE / 2) as f64, BINS);
        let mut filters = vec![0f64; BINS * MEL_BINS];
        for (b, &ff) in fft_freqs.iter().enumerate() {
            for m in 0..MEL_BINS {
                let down = -(filter_freqs[m] - ff) / (filter_freqs[m + 1] - filter_freqs[m]);
                let up = (filter_freqs[m + 2] - ff) / (filter_freqs[m + 2] - filter_freqs[m + 1]);
                filters[b * MEL_BINS + m] = down.min(up).max(0.0);
            }
        }
        let span = (0..MEL_BINS)
            .map(|m| {
                let nz: Vec<usize> = (0..BINS)
                    .filter(|&b| filters[b * MEL_BINS + m] != 0.0)
                    .collect();
                // an all-zero filter (narrow low mel bins) keeps (0, 0): its weights are 0, so it contributes 0 like the reference
                (
                    nz.first().copied().unwrap_or(0),
                    nz.last().copied().unwrap_or(0),
                )
            })
            .collect();
        Self {
            window,
            filters,
            span,
            fft: FftPlanner::new().plan_fft_forward(NFFT),
        }
    }

    /// `wav`: mono 16 kHz f32. Empty input gives an empty `Mel`.
    pub fn extract(&self, wav: &[f32]) -> Mel {
        let n_real = wav.len().min(MAX_SAMPLES);
        if n_real == 0 {
            return Mel {
                data: Vec::new(),
                frames: 0,
                valid: 0,
            };
        }
        let n_pad = n_real.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE;
        // left pad of FRAME/2 zeros, right pad to the multiple of 128
        let at = |p: usize| -> f32 {
            if p < FRAME / 2 || p - FRAME / 2 >= n_real {
                0.0
            } else {
                wav[p - FRAME / 2]
            }
        };
        let total = n_pad + FRAME / 2;
        let frames = if total > FRAME {
            (total - (FRAME + 1)) / HOP + 1
        } else {
            0
        };
        let valid = (0..frames)
            .take_while(|i| i * HOP + FRAME / 2 < n_real)
            .count();
        let mut data = vec![0f32; frames * MEL_BINS];
        let mut buf = vec![Complex::new(0f64, 0.0); NFFT];
        let mut scratch = vec![Complex::new(0f64, 0.0); self.fft.get_inplace_scratch_len()];
        let mut mel = [0f64; MEL_BINS];
        for i in 0..valid {
            for (k, c) in buf.iter_mut().enumerate() {
                *c = Complex::new(
                    if k < FRAME {
                        f64::from(at(i * HOP + k) * self.window[k])
                    } else {
                        0.0
                    },
                    0.0,
                );
            }
            self.fft.process_with_scratch(&mut buf, &mut scratch);
            let mag: Vec<f64> = buf[..BINS].iter().map(|c| c.norm()).collect();
            for (m, out) in mel.iter_mut().enumerate() {
                let (lo, hi) = self.span[m];
                *out = (lo..=hi)
                    .map(|b| mag[b] * self.filters[b * MEL_BINS + m])
                    .sum();
            }
            for (o, m) in data[i * MEL_BINS..(i + 1) * MEL_BINS].iter_mut().zip(&mel) {
                *o = (m + MEL_FLOOR).ln() as f32;
            }
        }
        Mel {
            data,
            frames,
            valid,
        }
    }
}
