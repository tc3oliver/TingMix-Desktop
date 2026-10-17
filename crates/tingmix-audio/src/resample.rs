use std::f64::consts::PI;

/// Half-width of the sinc kernel in input samples at ratio <= 1; widens when downsampling.
const HALF_TAPS: f64 = 16.0;

/// Streaming windowed-sinc resampler (mono). Chunking does not change the output: feeding the
/// same samples in any split gives the same result as feeding them at once. Latency is about
/// `HALF_TAPS * max(1, in/out)` input samples; call [`finish`](Self::finish) at end of stream.
///
/// shortcut: kernel weights are computed per output sample (a few trig calls per tap), fine for
/// 16 kHz output; switch to a per-phase table if profiling ever shows this.
pub struct Resampler {
    step: f64,   // input samples per output sample
    cutoff: f64, // fraction of the input Nyquist that is kept
    half: usize, // kernel half-width in input samples
    hist: Vec<f32>,
    pos: f64, // next output position, in `hist` coordinates
}

impl Resampler {
    pub fn new(from_hz: u32, to_hz: u32) -> Self {
        assert!(from_hz > 0 && to_hz > 0);
        let step = f64::from(from_hz) / f64::from(to_hz);
        let cutoff = (1.0 / step).min(1.0);
        let half = (HALF_TAPS / cutoff).ceil() as usize;
        // leading zeros so the first output sees a full (silent) left wing
        Self {
            step,
            cutoff,
            half,
            hist: vec![0.0; half],
            pos: half as f64,
        }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        self.hist.extend_from_slice(input);
        let mut out = Vec::with_capacity((input.len() as f64 / self.step) as usize + 1);
        while self.pos.floor() as usize + self.half < self.hist.len() {
            out.push(self.output_at(self.pos));
            self.pos += self.step;
        }
        let drop = (self.pos.floor() as usize).saturating_sub(self.half);
        self.hist.drain(..drop);
        self.pos -= drop as f64;
        out
    }

    /// Flush the tail by feeding silence; the resampler restarts afterwards.
    pub fn finish(&mut self) -> Vec<f32> {
        let tail = self.process(&vec![0.0; self.half + 1]);
        self.hist = vec![0.0; self.half];
        self.pos = self.half as f64;
        tail
    }

    fn output_at(&self, pos: f64) -> f32 {
        let c = pos.floor() as isize;
        let half = self.half as isize;
        let (mut acc, mut norm) = (0.0f64, 0.0f64);
        for j in (c - half + 1)..=(c + half) {
            let x = j as f64 - pos;
            let t = x * self.cutoff;
            let sinc = if t == 0.0 {
                1.0
            } else {
                (PI * t).sin() / (PI * t)
            };
            let w = x / self.half as f64; // Blackman window over [-1, 1]
            let win = 0.42 + 0.5 * (PI * w).cos() + 0.08 * (2.0 * PI * w).cos();
            let k = sinc * win;
            acc += k * f64::from(self.hist[j as usize]);
            norm += k;
        }
        (acc / norm) as f32
    }
}

/// One-shot resample of a whole buffer.
pub fn resample(input: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz {
        return input.to_vec();
    }
    let mut r = Resampler::new(from_hz, to_hz);
    let mut out = r.process(input);
    out.extend(r.finish());
    out.truncate((input.len() as f64 * f64::from(to_hz) / f64::from(from_hz)).round() as usize);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, rate: u32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize)
            .map(|i| (i as f32 * hz * std::f32::consts::TAU / rate as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn length_follows_the_ratio() {
        for (from, to) in [
            (48_000, 16_000),
            (44_100, 16_000),
            (8_000, 16_000),
            (16_000, 16_000),
            (96_000, 16_000),
        ] {
            let out = resample(&tone(440.0, from, 1.0), from, to);
            assert!(
                (out.len() as i64 - i64::from(to)).abs() <= 1,
                "{from}->{to}: {}",
                out.len()
            );
        }
    }

    #[test]
    fn in_band_tone_keeps_amplitude_and_frequency() {
        for from in [48_000, 44_100] {
            let out = resample(&tone(1000.0, from, 1.0), from, 16_000);
            let mid = &out[2_000..14_000];
            assert!(
                (rms(mid) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01,
                "{from}: rms {}",
                rms(mid)
            );
            let want = tone(1000.0, 16_000, 1.0);
            // same frequency and phase => small error against the ideal 16 kHz tone
            let err = rms(&mid
                .iter()
                .zip(&want[2_000..14_000])
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>());
            assert!(err < 0.02, "{from}: error {err}");
        }
    }

    #[test]
    fn out_of_band_tone_is_rejected() {
        let out = resample(&tone(12_000.0, 48_000, 1.0), 48_000, 16_000); // above the 8 kHz output Nyquist
        assert!(rms(&out[2_000..14_000]) < 0.01);
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        let x = tone(700.0, 44_100, 1.0);
        let whole = resample(&x, 44_100, 16_000);
        let mut r = Resampler::new(44_100, 16_000);
        let mut chunked = Vec::new();
        for c in x.chunks(333) {
            chunked.extend(r.process(c));
        }
        chunked.extend(r.finish());
        let n = whole.len().min(chunked.len());
        assert!(whole[..n]
            .iter()
            .zip(&chunked[..n])
            .all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn upsampling_reproduces_the_tone() {
        let out = resample(&tone(500.0, 8_000, 1.0), 8_000, 16_000);
        let want = tone(500.0, 16_000, 1.0);
        let err = rms(&out[1_000..15_000]
            .iter()
            .zip(&want[1_000..15_000])
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>());
        assert!(err < 0.01, "error {err}");
    }
}
