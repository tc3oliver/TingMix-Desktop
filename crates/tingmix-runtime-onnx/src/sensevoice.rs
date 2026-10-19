//! SenseVoice-Small (FunAudioLLM, the sherpa-onnx export `model.int8.onnx` + `tokens.txt`) through ONNX Runtime, numerically the
//! sherpa-onnx `OfflineRecognizerSenseVoiceImpl`: Kaldi fbank (80 bins, Hamming, snip edges, samples scaled to int16 range) ->
//! LFR stacking (7 frames, shift 6, edge frames repeated) -> CMVN from the model metadata -> graph `x, x_length, language, text_norm ->
//! logits` -> greedy CTC (blank 0) -> drop the four query tokens (language, emotion, event, ITN) -> SentencePiece join.
//! Oracle: `experiments/sensevoice` (sherpa-onnx 1.13.8, kaldi-native-fbank 1.22.3).

use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;

use ort::session::Session;
use ort::value::Tensor;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use tingmix_core::{AsrBackend, BackendError, BackendInfo, Transcription};
use tingmix_decoder::{greedy_ids, log_softmax_rows, Vocab};

use crate::large::non_blank_confidence;
use crate::model::{environment, ort_err, Error};

const SR: f64 = 16_000.0;
const WIN: usize = 400;
const HOP: usize = 160;
const N_FFT: usize = 512;
const N_MELS: usize = 80;
/// Query tokens the export prepends to the encoder input; their CTC labels are not text.
const QUERY: usize = 4;
/// Longest window decoded at once; longer input is cut without overlap (as `OnnxLarge`).
const CHUNK_SAMPLES: usize = 30 * 16_000;
const MIN_CHUNK_SAMPLES: usize = 1_600;

/// Kaldi `ComputeFbank` with the options sherpa-onnx sets for SenseVoice: 25 ms / 10 ms frames with snip edges, DC removal,
/// pre-emphasis 0.97, Hamming window, 512-point power spectrum, 80 Kaldi-mel triangles 20 Hz..Nyquist, `ln(max(x, f32::EPSILON))`.
pub struct Fbank {
    window: [f32; WIN],
    /// `[N_MELS][N_FFT / 2]`: Kaldi's banks never use the Nyquist bin.
    banks: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl Default for Fbank {
    fn default() -> Self {
        Self::new()
    }
}

fn mel(f: f64) -> f64 {
    1127.0 * (1.0 + f / 700.0).ln()
}

impl Fbank {
    pub fn new() -> Self {
        let mut window = [0f32; WIN];
        for (n, w) in window.iter_mut().enumerate() {
            *w = (0.54 - 0.46 * (2.0 * PI * n as f64 / (WIN - 1) as f64).cos()) as f32;
        }
        let half = N_FFT / 2;
        let (lo, hi) = (mel(20.0), mel(SR / 2.0));
        let delta = (hi - lo) / (N_MELS + 1) as f64;
        let mut banks = vec![0f32; N_MELS * half];
        for b in 0..N_MELS {
            let (left, center, right) = (
                lo + b as f64 * delta,
                lo + (b + 1) as f64 * delta,
                lo + (b + 2) as f64 * delta,
            );
            for i in 0..half {
                let m = mel(SR / N_FFT as f64 * i as f64);
                if m > left && m < right {
                    banks[b * half + i] = if m <= center {
                        (m - left) / (center - left)
                    } else {
                        (right - m) / (right - center)
                    } as f32;
                }
            }
        }
        Self {
            window,
            banks,
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    /// `[frames, 80]` row-major; `wav` is mono f32 in `[-1, 1]` at 16 kHz.
    pub fn compute(&self, wav: &[f32]) -> Vec<f32> {
        if wav.len() < WIN {
            return Vec::new();
        }
        let frames = 1 + (wav.len() - WIN) / HOP;
        let half = N_FFT / 2;
        let mut out = Vec::with_capacity(frames * N_MELS);
        let mut buf = vec![Complex32::default(); N_FFT];
        let mut frame = [0f32; WIN];
        let mut power = vec![0f32; half];
        for t in 0..frames {
            for (d, s) in frame.iter_mut().zip(&wav[t * HOP..t * HOP + WIN]) {
                *d = s * 32768.0;
            }
            let mean = frame.iter().sum::<f32>() / WIN as f32;
            frame.iter_mut().for_each(|x| *x -= mean);
            for i in (1..WIN).rev() {
                frame[i] -= 0.97 * frame[i - 1];
            }
            frame[0] -= 0.97 * frame[0];
            for (i, c) in buf.iter_mut().enumerate() {
                *c = Complex32::new(
                    if i < WIN {
                        frame[i] * self.window[i]
                    } else {
                        0.0
                    },
                    0.0,
                );
            }
            self.fft.process(&mut buf);
            for (p, c) in power.iter_mut().zip(&buf) {
                *p = c.norm_sqr();
            }
            for b in 0..N_MELS {
                let e: f32 = self.banks[b * half..(b + 1) * half]
                    .iter()
                    .zip(&power)
                    .map(|(w, p)| w * p)
                    .sum();
                out.push(e.max(f32::EPSILON).ln());
            }
        }
        out
    }
}

/// LFR + CMVN of sherpa-onnx (`lfr.cc`, FunASR's `apply_lfr`): output frame `i` stacks `m` frames centred on input frame
/// `i*shift`, the first frame repeated `(m-1)/2` times on the left and the last frame repeated past the end; `ceil(n/shift)`
/// output frames.
fn lfr_cmvn(fb: &[f32], m: usize, shift: usize, neg_mean: &[f32], inv_std: &[f32]) -> Vec<f32> {
    let n = fb.len() / N_MELS;
    if n == 0 {
        return Vec::new();
    }
    let out_n = 1 + (n - 1) / shift;
    let left = (m - 1) / 2;
    let mut out = Vec::with_capacity(out_n * m * N_MELS);
    for i in 0..out_n {
        for j in 0..m {
            let f = (i * shift + j).saturating_sub(left).min(n - 1);
            let k = j * N_MELS;
            out.extend(
                fb[f * N_MELS..(f + 1) * N_MELS]
                    .iter()
                    .zip(neg_mean[k..k + N_MELS].iter().zip(&inv_std[k..k + N_MELS]))
                    .map(|(x, (a, b))| (x + a) * b),
            );
        }
    }
    out
}

type TextMap = Box<dyn Fn(&str) -> String + Send>;

pub struct OnnxSenseVoice {
    info: BackendInfo,
    session: Session,
    fbank: Fbank,
    vocab: Vocab,
    lfr: (usize, usize),
    neg_mean: Vec<f32>,
    inv_std: Vec<f32>,
    language: i32,
    text_norm: i32,
    /// Applied to each final text (the app converts Simplified to Traditional characters here).
    output: Option<TextMap>,
}

fn meta_list(s: &str) -> Result<Vec<f32>, Error> {
    s.split(',')
        .map(|v| v.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .map_err(|e| Error::Contract(format!("metadata list: {e}")))
}

/// `tokens.txt`: one `<piece> <id>` per line, ids dense from 0.
fn read_tokens(path: &Path) -> Result<Vocab, Error> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| Error::Contract(format!("{}: {e}", path.display())))?;
    let mut itos = Vec::new();
    for (n, line) in raw.lines().enumerate() {
        let (piece, id) = line
            .rsplit_once(' ')
            .ok_or_else(|| Error::Contract(format!("tokens.txt line {}: no id", n + 1)))?;
        if id.parse::<usize>() != Ok(itos.len()) {
            return Err(Error::Contract(format!(
                "tokens.txt line {}: ids are not dense",
                n + 1
            )));
        }
        itos.push(piece.to_string());
    }
    Ok(Vocab::new(itos, true))
}

impl OnnxSenseVoice {
    /// `model`: the sherpa-onnx export (`model.int8.onnx` or `model.onnx`); `tokens`: its `tokens.txt`. Language detection is
    /// automatic and inverse text normalisation (punctuation, digits) is on.
    pub fn load(model: &Path, tokens: &Path, intra_threads: usize) -> Result<Self, Error> {
        let session = Session::builder(&environment()?)
            .map_err(ort_err)?
            .with_intra_threads(intra_threads)
            .map_err(ort_err)?
            .commit_from_file(model)
            .map_err(ort_err)?;
        let md = session.metadata().map_err(ort_err)?;
        let get = |k: &str| {
            md.custom(k)
                .ok_or_else(|| Error::Contract(format!("model metadata lacks {k}")))
        };
        if get("model_type")? != "sense_voice_ctc" {
            return Err(Error::Contract("not a SenseVoice CTC export".into()));
        }
        let int = |k: &str| -> Result<i64, Error> {
            get(k)?
                .trim()
                .parse()
                .map_err(|e| Error::Contract(format!("metadata {k}: {e}")))
        };
        if int("normalize_samples")? != 0 {
            return Err(Error::Contract("expected normalize_samples = 0".into()));
        }
        let lfr = (
            int("lfr_window_size")? as usize,
            int("lfr_window_shift")? as usize,
        );
        let (neg_mean, inv_std) = (
            meta_list(&get("neg_mean")?)?,
            meta_list(&get("inv_stddev")?)?,
        );
        if lfr.0 == 0
            || lfr.1 == 0
            || neg_mean.len() != lfr.0 * N_MELS
            || inv_std.len() != neg_mean.len()
        {
            return Err(Error::Contract(format!(
                "LFR {lfr:?} does not match CMVN of {} values",
                neg_mean.len()
            )));
        }
        let (language, text_norm) = (int("lang_auto")? as i32, int("with_itn")? as i32);
        let vocab = read_tokens(tokens)?;
        if int("vocab_size")? as usize != vocab.len() {
            return Err(Error::Contract(format!(
                "tokens.txt has {} entries, the model {}",
                vocab.len(),
                int("vocab_size")?
            )));
        }
        drop(md);
        Ok(Self {
            info: BackendInfo {
                name: "sensevoice-small-onnx".into(),
                sample_rate: 16_000,
                max_window_s: CHUNK_SAMPLES as f64 / 16_000.0,
                streaming_native: false,
            },
            session,
            fbank: Fbank::new(),
            vocab,
            lfr,
            neg_mean,
            inv_std,
            language,
            text_norm,
            output: None,
        })
    }

    /// Rewrite every final text with `f`.
    pub fn with_output(mut self, f: impl Fn(&str) -> String + Send + 'static) -> Self {
        self.output = Some(Box::new(f));
        self
    }

    /// The encoder input of `wav`: `[frames, lfr_m * 80]` row-major.
    pub fn features(&self, wav: &[f32]) -> Vec<f32> {
        lfr_cmvn(
            &self.fbank.compute(wav),
            self.lfr.0,
            self.lfr.1,
            &self.neg_mean,
            &self.inv_std,
        )
    }

    fn decode_chunk(&mut self, wav: &[f32]) -> Result<(String, Option<f32>), Error> {
        let x = self.features(wav);
        let dim = self.neg_mean.len();
        let frames = x.len() / dim;
        if frames == 0 {
            return Ok((String::new(), None));
        }
        let one = |v: i32| Tensor::from_array(([1usize], vec![v])).map_err(ort_err);
        let out = self
            .session
            .run(ort::inputs! {
                "x" => Tensor::from_array(([1usize, frames, dim], x)).map_err(ort_err)?,
                "x_length" => one(frames as i32)?,
                "language" => one(self.language)?,
                "text_norm" => one(self.text_norm)?,
            })
            .map_err(ort_err)?;
        let (shape, logits) = out["logits"].try_extract_tensor::<f32>().map_err(ort_err)?;
        let v = self.vocab.len();
        if shape.len() != 3 || shape[2] as usize != v {
            return Err(Error::Contract(format!(
                "logits shape {:?}, vocabulary {v}",
                &**shape
            )));
        }
        let lp = log_softmax_rows(logits, v);
        // greedy_ids already merges repeats and drops blanks (id 0)
        let text: Vec<u32> = greedy_ids(&lp, v).into_iter().skip(QUERY).collect();
        // the query frames always carry a label, so they would make every window look confident
        let conf = non_blank_confidence(&lp[QUERY.min(lp.len() / v) * v..], v);
        Ok((self.vocab.decode(&text), conf))
    }
}

impl AsrBackend for OnnxSenseVoice {
    fn info(&self) -> &BackendInfo {
        &self.info
    }

    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        Ok(self
            .transcribe_batch_scored(wavs)?
            .into_iter()
            .map(|t| t.text)
            .collect())
    }

    fn transcribe_batch_scored(
        &mut self,
        wavs: &[&[f32]],
    ) -> Result<Vec<Transcription>, BackendError> {
        let mut out = Vec::with_capacity(wavs.len());
        for w in wavs {
            if w.iter().any(|x| !x.is_finite()) {
                return Err(BackendError::Inference("non-finite samples".into()));
            }
            let (mut text, mut conf) = (String::new(), Vec::new());
            for c in w
                .chunks(CHUNK_SAMPLES)
                .filter(|c| c.len() >= MIN_CHUNK_SAMPLES)
            {
                let (t, p) = self
                    .decode_chunk(c)
                    .map_err(|e| BackendError::Inference(e.to_string()))?;
                // English chunks need a space between them; CJK ones must not get one
                if !text.is_empty() && t.starts_with(|c: char| c.is_ascii_alphanumeric()) {
                    text.push(' ');
                }
                text.push_str(&t);
                conf.extend(p);
            }
            let text = match &self.output {
                Some(f) => f(&text),
                None => text,
            };
            out.push(Transcription {
                text,
                confidence: conf.into_iter().reduce(f32::min),
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbank_frame_count_and_silence_floor() {
        let f = Fbank::new();
        assert!(f.compute(&[0.0; 399]).is_empty());
        let out = f.compute(&[0.0; 400 + 160 * 9]);
        assert_eq!(out.len(), 10 * N_MELS);
        // digital silence: every bank sits on the epsilon floor
        assert!(out.iter().all(|&x| (x - f32::EPSILON.ln()).abs() < 1e-3));
    }

    #[test]
    fn lfr_repeats_edge_frames_and_applies_cmvn() {
        let fb: Vec<f32> = (0..13 * N_MELS).map(|i| (i / N_MELS) as f32).collect();
        let (neg, inv) = (vec![-1.0; 7 * N_MELS], vec![2.0; 7 * N_MELS]);
        let out = lfr_cmvn(&fb, 7, 6, &neg, &inv);
        let frame = |o: usize, j: usize| out[(o * 7 + j) * N_MELS];
        assert_eq!(out.len(), 3 * 7 * N_MELS); // ceil(13 / 6)
                                               // output 0 stacks input frames 0,0,0,0,1,2,3; CMVN maps x to (x - 1) * 2
        assert_eq!(
            (0..7).map(|j| frame(0, j)).collect::<Vec<_>>(),
            [-2.0, -2.0, -2.0, -2.0, 0.0, 2.0, 4.0]
        );
        // output 2 is centred on frame 12, the last one: 9,10,11,12,12,12,12
        assert_eq!(
            (0..7).map(|j| frame(2, j)).collect::<Vec<_>>(),
            [16.0, 18.0, 20.0, 22.0, 22.0, 22.0, 22.0]
        );
        assert!(lfr_cmvn(&[], 7, 6, &neg, &inv).is_empty());
    }
}
