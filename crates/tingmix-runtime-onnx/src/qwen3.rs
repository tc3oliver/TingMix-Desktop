//! Qwen3-ASR (Alibaba Qwen, Apache-2.0) through ONNX Runtime, using the export of `andrewleech/qwen3-asr-onnx`
//! (`encoder`, `decoder_init`, `decoder_step` + shared weights, `embed_tokens.bin`). Pipeline as that repository's
//! `src/{mel,prompt,inference}.py`: Whisper log-mel (128 bins, centred STFT, Slaney filters, dynamic range 8, last frame
//! dropped) -> encoder `[1,128,T] -> [1,N,1024]` -> chat prompt with N `<|audio_pad|>` tokens -> prefill -> greedy steps with
//! the KV cache until an end token -> byte-level BPE. The model writes `language <name><asr_text><text>`; only the text is
//! returned. Oracle: `experiments/qwen3-asr/ref_onnx.py`.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;

use ort::session::Session;
use ort::value::{DynValue, Tensor};
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use tingmix_core::{AsrBackend, BackendError, BackendInfo, Transcription};

use crate::model::{environment, ort_err, Error};
use crate::sensevoice::TextMap;

const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 128;
const BINS: usize = N_FFT / 2 + 1;
/// Longest window decoded at once (the encoder's attention windows are 8 s, the reference splits long audio at 30 s).
const CHUNK_SAMPLES: usize = 30 * 16_000;
const MIN_CHUNK_SAMPLES: usize = 1_600;

const IM_START: i64 = 151_644;
const IM_END: i64 = 151_645;
const END_OF_TEXT: i64 = 151_643;
const AUDIO_START: i64 = 151_669;
const AUDIO_END: i64 = 151_670;
const AUDIO_PAD: i64 = 151_676;
const ASR_TEXT: i64 = 151_704;
const NEWLINE: i64 = 198;
/// `language`, the first token of every answer (`language Chinese<asr_text>...`).
const LANGUAGE: i64 = 11_528;
/// Tokens at the end of the previous answer that a growing window decodes again instead of keeping (the last words of a
/// cut-off utterance are the ones most likely to change).
const ROLLBACK: usize = 10;
const SYSTEM: i64 = 9125;
const USER: i64 = 882;
const ASSISTANT: i64 = 77_091;

/// Whisper's log-mel: periodic Hann 400, reflect-padded centred STFT, power, Slaney mel filters 0-8000 Hz with Slaney area
/// normalisation, `log10(max(x, 1e-10))`, floor at max - 8, `(x + 4) / 4`, last frame dropped.
pub struct WhisperMel {
    window: [f32; N_FFT],
    /// `[N_MELS][BINS]`
    filters: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl Default for WhisperMel {
    fn default() -> Self {
        Self::new()
    }
}

/// librosa `hz_to_mel` / `mel_to_hz` with `htk=False` (Slaney): linear below 1 kHz, logarithmic above.
fn slaney_mel(f: f64) -> f64 {
    let (f_sp, min_log_hz, logstep) = (200.0 / 3.0, 1000.0, (6.4f64).ln() / 27.0);
    if f >= min_log_hz {
        min_log_hz / f_sp + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn slaney_hz(m: f64) -> f64 {
    let (f_sp, min_log_hz, logstep) = (200.0 / 3.0, 1000.0, (6.4f64).ln() / 27.0);
    let min_log_mel = min_log_hz / f_sp;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

impl WhisperMel {
    pub fn new() -> Self {
        let mut window = [0f32; N_FFT];
        for (n, w) in window.iter_mut().enumerate() {
            *w = (0.5 - 0.5 * (2.0 * PI * n as f64 / N_FFT as f64).cos()) as f32;
        }
        // librosa.filters.mel(sr=16000, n_fft=400, n_mels=128, fmin=0, fmax=8000, norm="slaney")
        let (lo, hi) = (slaney_mel(0.0), slaney_mel(8000.0));
        let hz: Vec<f64> = (0..N_MELS + 2)
            .map(|i| slaney_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64))
            .collect();
        let mut filters = vec![0f32; N_MELS * BINS];
        for m in 0..N_MELS {
            let enorm = 2.0 / (hz[m + 2] - hz[m]);
            for k in 0..BINS {
                let f = 8000.0 * k as f64 / (BINS - 1) as f64;
                let lower = (f - hz[m]) / (hz[m + 1] - hz[m]);
                let upper = (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1]);
                filters[m * BINS + k] = (lower.min(upper).max(0.0) * enorm) as f32;
            }
        }
        Self {
            window,
            filters,
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    /// `([N_MELS][frames] row-major, frames)` of mono f32 at 16 kHz; `frames = len / 160`.
    pub fn compute(&self, wav: &[f32]) -> (Vec<f32>, usize) {
        let frames = wav.len() / HOP; // 1 + len / HOP centred frames, the last one dropped
        if frames == 0 || wav.len() <= N_FFT / 2 {
            return (Vec::new(), 0);
        }
        let pad = N_FFT / 2;
        // reflect padding as torch.stft(center=True, pad_mode="reflect"): sample -i mirrors sample i
        let at = |i: isize| -> f32 {
            let n = wav.len() as isize;
            let j = if i < 0 {
                -i
            } else if i >= n {
                2 * (n - 1) - i
            } else {
                i
            };
            wav[j as usize]
        };
        let mut power = vec![0f32; frames * BINS];
        let mut buf = vec![Complex32::default(); N_FFT];
        for t in 0..frames {
            let start = (t * HOP) as isize - pad as isize;
            for (i, c) in buf.iter_mut().enumerate() {
                *c = Complex32::new(at(start + i as isize) * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for (p, c) in power[t * BINS..(t + 1) * BINS].iter_mut().zip(&buf) {
                *p = c.norm_sqr();
            }
        }
        let mut out = vec![0f32; N_MELS * frames];
        for m in 0..N_MELS {
            let f = &self.filters[m * BINS..(m + 1) * BINS];
            for t in 0..frames {
                let e: f32 = f
                    .iter()
                    .zip(&power[t * BINS..(t + 1) * BINS])
                    .map(|(a, b)| a * b)
                    .sum();
                out[m * frames + t] = e.max(1e-10).log10();
            }
        }
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for x in &mut out {
            *x = (x.max(max - 8.0) + 4.0) / 4.0;
        }
        (out, frames)
    }
}

/// Encoder tokens for `frames` mel frames (`_get_feat_extract_output_lengths`): 13 per full 100-frame window, and three
/// stride-2 convolutions over the remainder.
pub fn audio_tokens(frames: usize) -> usize {
    let half = |t: usize| t.div_ceil(2);
    half(half(half(frames % 100))) + frames / 100 * 13
}

/// The ASR chat prompt with `n` audio placeholders (`src/prompt.py`), and where the placeholders start.
fn prompt(n: usize) -> (Vec<i64>, usize) {
    let mut ids = vec![
        IM_START,
        SYSTEM,
        NEWLINE,
        IM_END,
        NEWLINE,
        IM_START,
        USER,
        NEWLINE,
        AUDIO_START,
    ];
    let start = ids.len();
    ids.extend(std::iter::repeat_n(AUDIO_PAD, n));
    ids.extend([AUDIO_END, IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE]);
    (ids, start)
}

/// GPT-2 `bytes_to_unicode`, inverted: the printable stand-in character of each byte -> the byte.
fn unicode_to_byte() -> HashMap<char, u8> {
    let mut printable: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
    printable.extend(0xA1..=0xAC);
    printable.extend(0xAE..=0xFF);
    let mut map = HashMap::new();
    let mut extra = 0;
    for b in 0..=255u32 {
        let c = if printable.contains(&b) {
            b
        } else {
            extra += 1;
            255 + extra
        };
        map.insert(char::from_u32(c).unwrap(), b as u8);
    }
    map
}

/// Byte-level BPE decoding of the Qwen tokenizer: `vocab.json` (piece -> id) and `added_tokens.json` (special tokens,
/// never written out).
struct Tokens {
    pieces: Vec<Vec<u8>>,
}

impl Tokens {
    fn load(vocab: &Path, added: &Path) -> Result<Self, Error> {
        let read = |p: &Path| -> Result<HashMap<String, usize>, Error> {
            let raw =
                std::fs::read(p).map_err(|e| Error::Contract(format!("{}: {e}", p.display())))?;
            serde_json::from_slice(&raw)
                .map_err(|e| Error::Contract(format!("{}: {e}", p.display())))
        };
        let (vocab, added) = (read(vocab)?, read(added)?);
        if added.get("<asr_text>") != Some(&(ASR_TEXT as usize))
            || added.get("<|audio_pad|>") != Some(&(AUDIO_PAD as usize))
        {
            return Err(Error::Contract(
                "special token ids differ from Qwen3-ASR's".into(),
            ));
        }
        let back = unicode_to_byte();
        let n = vocab
            .values()
            .chain(added.values())
            .max()
            .map_or(0, |m| m + 1);
        let mut pieces = vec![Vec::new(); n];
        for (piece, id) in vocab {
            pieces[id] = piece
                .chars()
                .map(|c| {
                    back.get(&c)
                        .copied()
                        .ok_or_else(|| Error::Contract(format!("vocab piece {piece:?}")))
                })
                .collect::<Result<_, _>>()?;
        }
        Ok(Self { pieces })
    }

    fn decode(&self, ids: &[i64]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter_map(|&i| self.pieces.get(i as usize))
            .flatten()
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}

/// IEEE half -> f32 (the embedding table is stored as f16).
fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, frac) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1f) as u32,
        (h & 0x3ff) as u32,
    );
    let bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, _) => {
            // subnormal: renormalise
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
        (0x1f, _) => (sign << 31) | 0x7f80_0000 | (frac << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

pub struct OnnxQwen3Asr {
    info: BackendInfo,
    encoder: Session,
    init: Session,
    step: Session,
    mel: WhisperMel,
    tokens: Tokens,
    /// `[vocab][hidden]` as f16 bits.
    embed: Vec<u16>,
    hidden: usize,
    /// Name of the encoder's only output.
    enc_out: String,
    output: Option<TextMap>,
    /// The latest window, its answer and each token's probability: a window that extends it (the next live partial or the
    /// final of the same segment) keeps that answer, minus `ROLLBACK` tokens, as a prefix and only decodes what follows.
    last: Option<(Vec<f32>, Vec<i64>, Vec<f32>)>,
}

impl OnnxQwen3Asr {
    /// `dir`: the export directory. The int4 graphs are used when present (`*.int4.onnx`), else the FP32 ones.
    pub fn load(dir: &Path, intra_threads: usize) -> Result<Self, Error> {
        let env = environment()?;
        let pick = |stem: &str| {
            let q = dir.join(format!("{stem}.int4.onnx"));
            if q.is_file() {
                q
            } else {
                dir.join(format!("{stem}.onnx"))
            }
        };
        let open = |stem: &str| -> Result<Session, Error> {
            Session::builder(&env)
                .map_err(ort_err)?
                .with_intra_threads(intra_threads)
                .map_err(ort_err)?
                .commit_from_file(pick(stem))
                .map_err(ort_err)
        };
        let (encoder, init, step) = (
            open("encoder")?,
            open("decoder_init")?,
            open("decoder_step")?,
        );
        let names = |s: &Session| {
            s.inputs()
                .iter()
                .map(|i| i.name().to_string())
                .collect::<Vec<_>>()
        };
        if names(&init)
            != [
                "input_ids",
                "position_ids",
                "audio_features",
                "audio_offset",
            ]
            || names(&step) != ["input_embeds", "position_ids", "past_keys", "past_values"]
        {
            return Err(Error::Contract(format!(
                "unexpected decoder inputs {:?} / {:?}",
                names(&init),
                names(&step)
            )));
        }
        let enc_out = encoder
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| Error::Contract("encoder has no output".into()))?;
        let tokens = Tokens::load(&dir.join("vocab.json"), &dir.join("added_tokens.json"))?;
        let raw = std::fs::read(dir.join("embed_tokens.bin"))
            .map_err(|e| Error::Contract(format!("embed_tokens.bin: {e}")))?;
        let embed: Vec<u16> = raw
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let vocab = 151_936;
        if embed.len() % vocab != 0 {
            return Err(Error::Contract(format!(
                "embed_tokens.bin holds {} values, not a multiple of {vocab}",
                embed.len()
            )));
        }
        Ok(Self {
            info: BackendInfo {
                name: "qwen3-asr-onnx".into(),
                sample_rate: 16_000,
                max_window_s: CHUNK_SAMPLES as f64 / 16_000.0,
                streaming_native: false,
            },
            encoder,
            init,
            step,
            mel: WhisperMel::new(),
            tokens,
            hidden: embed.len() / vocab,
            enc_out,
            embed,
            output: None,
            last: None,
        })
    }

    /// Rewrite every final text with `f`.
    pub fn with_output(mut self, f: impl Fn(&str) -> String + Send + 'static) -> Self {
        self.output = Some(Box::new(f));
        self
    }

    fn embed_row(&self, id: i64) -> Vec<f32> {
        let h = self.hidden;
        self.embed[id as usize * h..(id as usize + 1) * h]
            .iter()
            .map(|&x| f16_to_f32(x))
            .collect()
    }

    /// Generated token ids of one window (end token excluded) and the mean probability of the chosen tokens.
    pub fn generate(&mut self, wav: &[f32]) -> Result<(Vec<i64>, Option<f32>), Error> {
        let (gen, probs, _) = self.generate_after(wav, &[], &[])?;
        let conf = text_confidence(&gen, &probs);
        Ok((gen, conf))
    }

    /// The answer kept from the previous window when `wav` extends it, with its token probabilities.
    fn reusable_prefix(&self, wav: &[f32]) -> (Vec<i64>, Vec<f32>) {
        self.last
            .as_ref()
            .map(|(audio, ids, probs)| kept_prefix(audio, ids, probs, wav))
            .unwrap_or_default()
    }

    /// Generated ids and their probabilities, continuing after `prefix` (answer tokens fed with the prompt), and whether the
    /// answer ended before the token limit.
    fn generate_after(
        &mut self,
        wav: &[f32],
        prefix: &[i64],
        prefix_probs: &[f32],
    ) -> Result<(Vec<i64>, Vec<f32>, bool), Error> {
        let (mel, frames) = self.mel.compute(wav);
        if frames == 0 {
            return Ok((Vec::new(), Vec::new(), true));
        }
        let mut out = self
            .encoder
            .run(ort::inputs! { "mel" => Tensor::from_array(([1usize, N_MELS, frames], mel)).map_err(ort_err)? })
            .map_err(ort_err)?;
        let features: DynValue = out
            .remove(&self.enc_out)
            .ok_or_else(|| Error::Contract("encoder has no output".into()))?;
        drop(out);
        let n = features
            .shape()
            .get(1)
            .copied()
            .ok_or_else(|| Error::Contract("encoder output is not [1, N, D]".into()))?
            as usize;
        if n != audio_tokens(frames) {
            return Err(Error::Contract(format!(
                "encoder gave {n} tokens for {frames} frames"
            )));
        }
        let (mut ids, start) = prompt(n);
        ids.extend_from_slice(prefix);
        let len = ids.len();
        // ~15 tokens per second is far above any speaking rate. After a prefix only the rolled-back tokens and the new audio
        // are left to write; an answer that runs past that has started repeating itself.
        let max_new = match self.last.as_ref().filter(|_| !prefix.is_empty()) {
            Some((audio, ..)) => {
                prefix.len() + ROLLBACK + 8 + (wav.len() - audio.len()) * 15 / 16_000
            }
            None => 32 + wav.len() * 15 / 16_000,
        };
        let pos: Vec<i64> = (0..len as i64).collect();
        let mut o = self
            .init
            .run(ort::inputs! {
                "input_ids" => Tensor::from_array(([1usize, len], ids)).map_err(ort_err)?,
                "position_ids" => Tensor::from_array(([1usize, len], pos)).map_err(ort_err)?,
                "audio_features" => features,
                "audio_offset" => Tensor::from_array(([1usize], vec![start as i64])).map_err(ort_err)?,
            })
            .map_err(ort_err)?;
        // The answer always opens with "language"; the int4 decoder's choice of that first token flips on 1e-5 changes of the
        // input for some clips (experiments/qwen3-asr), and a wrong first token loses the whole window. The model still picks
        // the language name.
        let (arg, mut p) = argmax_last(&o)?;
        let mut next = if prefix.is_empty() { LANGUAGE } else { arg };
        let (mut keys, mut values) = (
            take(&mut o, "present_keys")?,
            take(&mut o, "present_values")?,
        );
        drop(o);
        let (mut gen, mut probs) = (prefix.to_vec(), prefix_probs.to_vec());
        let mut at = len as i64;
        while next != IM_END && next != END_OF_TEXT && gen.len() < max_new {
            gen.push(next);
            probs.push(p);
            let mut o = self
                .step
                .run(ort::inputs! {
                    "input_embeds" => Tensor::from_array(([1usize, 1, self.hidden], self.embed_row(next))).map_err(ort_err)?,
                    "position_ids" => Tensor::from_array(([1usize, 1], vec![at])).map_err(ort_err)?,
                    "past_keys" => keys,
                    "past_values" => values,
                })
                .map_err(ort_err)?;
            (next, p) = argmax_last(&o)?;
            (keys, values) = (
                take(&mut o, "present_keys")?,
                take(&mut o, "present_values")?,
            );
            at += 1;
        }
        let ended = next == IM_END || next == END_OF_TEXT;
        Ok((gen, probs, ended))
    }

    fn decode_chunk(&mut self, wav: &[f32]) -> Result<(String, Option<f32>), Error> {
        let (prefix, prefix_probs) = self.reusable_prefix(wav);
        let (mut gen, mut probs, ended) = self.generate_after(wav, &prefix, &prefix_probs)?;
        if !ended && !prefix.is_empty() {
            (gen, probs, _) = self.generate_after(wav, &[], &[])?;
        }
        let conf = text_confidence(&gen, &probs);
        self.last = Some((wav.to_vec(), gen.clone(), probs));
        let text = match gen.iter().position(|&t| t == ASR_TEXT) {
            Some(i) => &gen[i + 1..],
            None => &gen[..], // no header: "language None" for silence writes nothing after it
        };
        let text = if gen.contains(&ASR_TEXT) {
            self.tokens.decode(text)
        } else {
            String::new()
        };
        Ok((text, conf))
    }
}

/// What a window `wav` keeps of the previous window's answer `ids`: all but the last `ROLLBACK` tokens when `wav` extends
/// that window's `audio` sample for sample and the kept part still reaches into the text.
fn kept_prefix(audio: &[f32], ids: &[i64], probs: &[f32], wav: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let keep = ids.len().saturating_sub(ROLLBACK);
    let extends = wav.len() > audio.len() && wav[..audio.len()] == *audio;
    match extends && ids[..keep].contains(&ASR_TEXT) {
        true => (ids[..keep].to_vec(), probs[..keep].to_vec()),
        false => (Vec::new(), Vec::new()),
    }
}

/// Mean probability of the text tokens, not of the "language X<asr_text>" header.
fn text_confidence(gen: &[i64], probs: &[f32]) -> Option<f32> {
    let from = gen.iter().position(|&t| t == ASR_TEXT).map_or(0, |i| i + 1);
    (from < probs.len()).then(|| probs[from..].iter().sum::<f32>() / (probs.len() - from) as f32)
}

fn take(o: &mut ort::session::SessionOutputs<'_>, name: &str) -> Result<DynValue, Error> {
    o.remove(name)
        .ok_or_else(|| Error::Contract(format!("decoder has no output {name}")))
}

/// Argmax over the last position's logits and its softmax probability.
fn argmax_last(o: &ort::session::SessionOutputs<'_>) -> Result<(i64, f32), Error> {
    let (shape, logits) = o["logits"].try_extract_tensor::<f32>().map_err(ort_err)?;
    let v = *shape
        .last()
        .ok_or_else(|| Error::Contract("logits without shape".into()))? as usize;
    let row = &logits[logits.len() - v..];
    let (arg, best) =
        row.iter().enumerate().fold(
            (0, f32::NEG_INFINITY),
            |a, (i, &x)| if x > a.1 { (i, x) } else { a },
        );
    let z: f32 = row.iter().map(|&x| (x - best).exp()).sum();
    Ok((arg as i64, 1.0 / z))
}

impl AsrBackend for OnnxQwen3Asr {
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
    fn a_growing_window_keeps_the_previous_text_minus_the_rollback() {
        let audio = [0.1f32, 0.2, 0.3];
        let mut ids = vec![LANGUAGE, 1, ASR_TEXT];
        ids.extend(100..100 + ROLLBACK as i64 + 2);
        let probs: Vec<f32> = (0..ids.len()).map(|i| i as f32).collect();
        let (kept, p) = kept_prefix(&audio, &ids, &probs, &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(kept, ids[..5]);
        assert_eq!(p, probs[..5]);
        // other audio, the same window again, or a kept part that ends inside the header: decode from scratch
        assert!(kept_prefix(&audio, &ids, &probs, &[0.1, 0.9, 0.3, 0.4])
            .0
            .is_empty());
        assert!(kept_prefix(&audio, &ids, &probs, &audio).0.is_empty());
        assert!(
            kept_prefix(&audio, &ids[..ROLLBACK + 2], &probs, &[0.1, 0.2, 0.3, 0.4])
                .0
                .is_empty()
        );
    }

    #[test]
    fn token_counts_follow_the_reference_formula() {
        // _get_feat_extract_output_lengths: 997 frames -> 9 full windows (117) + ceil^3(97/2) = 13
        assert_eq!(audio_tokens(997), 9 * 13 + 13);
        assert_eq!(audio_tokens(100), 13);
        assert_eq!(audio_tokens(1), 1);
        assert_eq!(audio_tokens(0), 0);
    }

    #[test]
    fn prompt_places_the_audio_between_its_markers() {
        let (ids, start) = prompt(3);
        assert_eq!(
            &ids[start - 1..start + 4],
            &[AUDIO_START, AUDIO_PAD, AUDIO_PAD, AUDIO_PAD, AUDIO_END]
        );
        assert_eq!(&ids[ids.len() - 3..], &[IM_START, ASSISTANT, NEWLINE]);
        assert_eq!(ids.len(), 9 + 3 + 6);
    }

    #[test]
    fn half_floats_and_byte_table() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        let b = unicode_to_byte();
        assert_eq!(b.len(), 256);
        assert_eq!(b[&'Ġ'], b' '); // GPT-2's stand-in for the space byte
        assert_eq!(b[&'A'], b'A');
    }

    #[test]
    fn mel_frames_and_range() {
        let m = WhisperMel::new();
        let wav: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
        let (x, frames) = m.compute(&wav);
        assert_eq!(frames, 100);
        assert_eq!(x.len(), N_MELS * 100);
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = x.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            (max - min - 2.0).abs() < 1e-5,
            "dynamic range is clamped to 8 log10 units = 2 after scaling"
        );
    }
}
