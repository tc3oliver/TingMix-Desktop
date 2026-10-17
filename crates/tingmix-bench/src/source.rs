//! Deterministic audio timeline: fixture clips separated by random silence gaps, generated on demand so a 60-minute
//! run never holds the whole stream in memory.

use std::path::Path;

pub const SR: usize = 16_000;

pub struct Clip {
    pub id: String,
    pub lang: String,
    pub wav: Vec<f32>,
    /// First and one-past-last sample whose 20 ms frame is above -45 dBFS (acoustic speech span).
    pub onset: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct Utt {
    pub clip: usize,
    pub onset_s: f64,
    pub end_s: f64,
}

pub fn load_clips(dir: &Path, only: Option<&[String]>) -> Result<Vec<Clip>, String> {
    let manifest = std::fs::read_to_string(dir.join("manifest.jsonl"))
        .map_err(|e| format!("manifest.jsonl: {e}"))?;
    let mut clips = Vec::new();
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let id = v["id"]
            .as_str()
            .ok_or("manifest row without id")?
            .to_string();
        if only.is_some_and(|o| !o.contains(&id)) {
            continue;
        }
        let mut r =
            hound::WavReader::open(dir.join(v["audio"].as_str().ok_or("row without audio")?))
                .map_err(|e| e.to_string())?;
        let spec = r.spec();
        if spec.sample_rate as usize != SR || spec.channels != 1 || spec.bits_per_sample != 16 {
            return Err(format!("{id}: expected 16 kHz mono 16-bit, got {spec:?}"));
        }
        let wav: Vec<f32> = r
            .samples::<i16>()
            .map(|s| s.map(|x| f32::from(x) / 32768.0))
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        let (onset, end) = speech_span(&wav);
        clips.push(Clip {
            id,
            lang: v["lang"].as_str().unwrap_or("").into(),
            wav,
            onset,
            end,
        });
    }
    if clips.is_empty() {
        return Err("no clips selected".into());
    }
    Ok(clips)
}

fn speech_span(w: &[f32]) -> (usize, usize) {
    let f = SR / 50;
    let loud: Vec<bool> = w
        .chunks(f)
        .map(|c| {
            20.0 * ((c.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / c.len() as f64).sqrt()
                + 1e-10)
                .log10()
                > -45.0
        })
        .collect();
    let first = loud.iter().position(|&l| l).unwrap_or(0);
    let last = loud
        .iter()
        .rposition(|&l| l)
        .unwrap_or(loud.len().saturating_sub(1));
    (first * f, ((last + 1) * f).min(w.len()))
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub struct Source {
    clips: Vec<Clip>,
    rng: Rng,
    gap: (f64, f64),
    noise_amp: f32,
    stop_after_s: f64,
    pending: Vec<f32>,
    at: usize,
    generated: usize,
    order: Vec<usize>,
    pub utts: Vec<Utt>,
}

impl Source {
    pub fn new(
        clips: Vec<Clip>,
        seed: u64,
        gap: (f64, f64),
        noise_dbfs: Option<f64>,
        stop_after_s: f64,
    ) -> Self {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        // Fisher-Yates over the clip order, re-shuffled each cycle so every clip recurs with varying neighbours.
        let order = (0..clips.len()).collect();
        let _ = rng.next();
        Self {
            clips,
            rng,
            gap,
            noise_amp: noise_dbfs.map_or(0.0, |d| (10f64.powf(d / 20.0) * 3f64.sqrt()) as f32),
            stop_after_s,
            pending: Vec::new(),
            at: 0,
            generated: 0,
            order,
            utts: Vec::new(),
        }
    }

    pub fn clip(&self, i: usize) -> &Clip {
        &self.clips[i]
    }

    /// Seconds of audio produced so far.
    pub fn generated_s(&self) -> f64 {
        self.generated as f64 / SR as f64
    }

    /// True once no further utterance will be started and the current block is exhausted.
    pub fn finished(&self) -> bool {
        self.at >= self.pending.len() && self.generated_s() >= self.stop_after_s
    }

    fn refill(&mut self) {
        if self.utts.len() % self.clips.len() == 0 {
            for i in (1..self.order.len()).rev() {
                let j = (self.rng.next() * (i + 1) as f64) as usize;
                self.order.swap(i, j);
            }
        }
        let clip = self.order[self.utts.len() % self.clips.len()];
        let gap = ((self.gap.0 + self.rng.next() * (self.gap.1 - self.gap.0)) * SR as f64) as usize;
        let start = self.generated + gap;
        self.utts.push(Utt {
            clip,
            onset_s: (start + self.clips[clip].onset) as f64 / SR as f64,
            end_s: (start + self.clips[clip].end) as f64 / SR as f64,
        });
        self.pending = vec![0.0; gap];
        self.pending.extend_from_slice(&self.clips[clip].wav);
        self.at = 0;
    }

    /// Next `n` samples; silence (plus noise) once the run is over.
    pub fn next(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            if self.at >= self.pending.len() {
                if self.generated_s() >= self.stop_after_s {
                    out.resize(n, 0.0);
                    break;
                }
                self.refill();
            }
            let k = (n - out.len()).min(self.pending.len() - self.at);
            out.extend_from_slice(&self.pending[self.at..self.at + k]);
            self.at += k;
        }
        if self.noise_amp > 0.0 {
            for x in &mut out {
                *x += (self.rng.next() as f32 * 2.0 - 1.0) * self.noise_amp;
            }
        }
        self.generated += n;
        out
    }
}
