//! Deterministic audio timeline: fixture clips separated by random silence gaps, generated on demand so a 60-minute
//! run never holds the whole stream in memory.

use std::path::Path;

pub const SR: usize = 16_000;

pub struct Clip {
    pub id: String,
    /// The TTS prompt: ground truth of what was said.
    pub text: String,
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
            text: v["text"].as_str().unwrap_or("").into(),
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

#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximately standard normal (sum of 12 uniforms).
    fn gauss(&mut self) -> f64 {
        (0..12).map(|_| self.next()).sum::<f64>() - 6.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseKind {
    White,
    Pink,
    /// Low-frequency rumble plus 100/200 Hz mains hum.
    Fan,
    /// Several time-reversed fixture voices summed: speech-like energy and rhythm, no intelligible words.
    Babble,
    /// Sustained chords (harmonic tones, slow chord changes, light tremolo): a music bed.
    Music,
}

impl NoiseKind {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "white" => Self::White,
            "pink" => Self::Pink,
            "fan" => Self::Fan,
            "babble" => Self::Babble,
            "music" => Self::Music,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Level {
    /// RMS in dBFS.
    Dbfs(f64),
    /// RMS relative to the mean RMS of the speech frames of the clip set.
    SnrDb(f64),
}

#[derive(Debug, Clone, Copy)]
pub struct NoiseSpec {
    pub kind: NoiseKind,
    pub level: Level,
}

/// Generated noise at a calibrated RMS. Everything is synthesised from the seed and the (synthetic TTS) fixtures; no
/// recordings of third parties are involved.
struct Noise {
    kind: NoiseKind,
    rng: Rng,
    gain: f32,
    pink: [f64; 7],
    lp: f64,
    phase: f64,
    voices: Vec<Voice>,
    bank: Vec<Vec<f32>>,
}

struct Voice {
    clip: usize,
    pos: usize,
    gain: f32,
}

impl Noise {
    fn raw(&mut self) -> f64 {
        match self.kind {
            NoiseKind::White => self.rng.gauss(),
            NoiseKind::Pink => self.pink_sample(),
            NoiseKind::Fan => {
                self.lp = 0.985 * self.lp + 0.015 * self.pink_sample();
                self.phase += 1.0 / SR as f64;
                let t = 2.0 * std::f64::consts::PI * self.phase;
                self.lp * 6.0 + 0.5 * (100.0 * t).sin() + 0.25 * (200.0 * t).sin()
            }
            NoiseKind::Music => {
                // four chord tones (A minor / F major / C major / G major, two seconds each) with 5 harmonics each
                const CHORDS: [[f64; 4]; 4] = [
                    [220.0, 261.6, 329.6, 440.0],
                    [174.6, 220.0, 261.6, 349.2],
                    [261.6, 329.6, 392.0, 523.3],
                    [196.0, 246.9, 293.7, 392.0],
                ];
                self.phase += 1.0 / SR as f64;
                let (ci, cj) = (
                    ((self.phase / 2.0) as usize) % 4,
                    ((self.phase / 2.0) as usize + 1) % 4,
                );
                let x = (self.phase / 2.0).fract();
                let (wa, wb) = ((1.0 - x).min(1.0), x); // linear crossfade between chords
                let tone = |f: f64| {
                    (1..=5)
                        .map(|h| {
                            (2.0 * std::f64::consts::PI * f * h as f64 * self.phase).sin()
                                / h as f64
                        })
                        .sum::<f64>()
                };
                let tremolo = 1.0 + 0.15 * (2.0 * std::f64::consts::PI * 4.0 * self.phase).sin();
                tremolo
                    * (CHORDS[ci].iter().map(|f| tone(*f)).sum::<f64>() * wa
                        + CHORDS[cj].iter().map(|f| tone(*f)).sum::<f64>() * wb)
            }
            NoiseKind::Babble => {
                let mut sum = 0.0;
                for v in &mut self.voices {
                    let clip = &self.bank[v.clip];
                    sum += f64::from(clip[clip.len() - 1 - v.pos] * v.gain); // reversed
                    v.pos += 1;
                    if v.pos >= clip.len() {
                        v.clip =
                            (self.rng.next() * self.bank.len() as f64) as usize % self.bank.len();
                        v.pos = (self.rng.next() * 0.5 * self.bank[v.clip].len() as f64) as usize;
                        v.gain = 0.5 + self.rng.next() as f32;
                    }
                }
                sum
            }
        }
    }

    fn pink_sample(&mut self) -> f64 {
        // Paul Kellet's economy pink filter
        let w = self.rng.gauss();
        let b = &mut self.pink;
        b[0] = 0.99886 * b[0] + w * 0.0555179;
        b[1] = 0.99332 * b[1] + w * 0.0750759;
        b[2] = 0.96900 * b[2] + w * 0.1538520;
        b[3] = 0.86650 * b[3] + w * 0.3104856;
        b[4] = 0.55000 * b[4] + w * 0.5329522;
        b[5] = -0.7616 * b[5] - w * 0.0168980;
        let out = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362;
        b[6] = w * 0.115926;
        out
    }

    fn new(spec: NoiseSpec, seed: u64, clips: &[Clip]) -> Self {
        let level_rms = match spec.level {
            Level::Dbfs(d) => 10f64.powf(d / 20.0),
            Level::SnrDb(snr) => speech_rms(clips) / 10f64.powf(snr / 20.0),
        };
        let bank: Vec<Vec<f32>> = clips
            .iter()
            .map(|c| c.wav[c.onset..c.end].to_vec())
            .collect();
        let mut n = Noise {
            kind: spec.kind,
            rng: Rng(seed.wrapping_mul(0xD6E8_FEB8_6659_FD93) | 1),
            gain: 1.0,
            pink: [0.0; 7],
            lp: 0.0,
            phase: 0.0,
            voices: Vec::new(),
            bank,
        };
        if spec.kind == NoiseKind::Babble && !n.bank.is_empty() {
            n.voices = (0..6)
                .map(|_| Voice {
                    clip: (n.rng.next() * n.bank.len() as f64) as usize % n.bank.len(),
                    pos: 0,
                    gain: 0.5 + n.rng.next() as f32,
                })
                .collect();
        }
        // Calibrate on 10 s of the generator's own output, then continue from the advanced state.
        let ms = (0..10 * SR).map(|_| n.raw().powi(2)).sum::<f64>() / (10 * SR) as f64;
        n.gain = (level_rms / ms.sqrt().max(1e-12)) as f32;
        n
    }

    fn add(&mut self, out: &mut [f32]) {
        for x in out {
            *x += self.raw() as f32 * self.gain;
        }
    }
}

/// Mean RMS over the speech spans of all clips.
fn speech_rms(clips: &[Clip]) -> f64 {
    let (sum, n) = clips.iter().fold((0.0f64, 0usize), |(s, n), c| {
        let span = &c.wav[c.onset..c.end];
        (
            s + span.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>(),
            n + span.len(),
        )
    });
    (sum / n.max(1) as f64).sqrt()
}

pub struct Source {
    clips: Vec<Clip>,
    rng: Rng,
    gap: (f64, f64),
    noise: Option<Noise>,
    speech: bool,
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
        noise: Option<NoiseSpec>,
        stop_after_s: f64,
    ) -> Self {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        // Fisher-Yates over the clip order, re-shuffled each cycle so every clip recurs with varying neighbours.
        let order = (0..clips.len()).collect();
        let _ = rng.next();
        let noise = noise.map(|spec| Noise::new(spec, seed, &clips));
        Self {
            clips,
            rng,
            gap,
            noise,
            speech: true,
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

    /// Noise-only mode: the clips still calibrate the noise level, but nothing is spoken.
    pub fn without_speech(mut self) -> Self {
        self.speech = false;
        self
    }

    fn refill(&mut self) {
        if !self.speech {
            let gap =
                ((self.gap.0 + self.rng.next() * (self.gap.1 - self.gap.0)) * SR as f64) as usize;
            self.pending = vec![0.0; gap];
            self.at = 0;
            return;
        }
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
        if let Some(n) = &mut self.noise {
            n.add(&mut out);
        }
        self.generated += n;
        out
    }
}
