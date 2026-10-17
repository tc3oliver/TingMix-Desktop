use std::fs::File;
use std::path::Path;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::{AudioError, Resampler, MODEL_SAMPLE_RATE};

fn decode_err(e: impl std::fmt::Display) -> AudioError {
    AudioError::Decode(e.to_string())
}

/// Decode an audio file (wav, flac, mp3, ogg/vorbis, aac/m4a, ...) to 16 kHz mono, handing
/// chunks to `sink` as they are produced. `sink` returns `false` to cancel; the call then
/// returns `Ok(())` with the audio delivered so far.
///
/// Undecodable packets inside an otherwise readable file are skipped; a file that yields no
/// audio at all is an error.
pub fn decode_file_chunks(
    path: &Path,
    mut sink: impl FnMut(&[f32]) -> bool,
) -> Result<(), AudioError> {
    let file = File::open(path).map_err(decode_err)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(decode_err)?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| decode_err("no audio track"))?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| decode_err("not an audio track"))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(decode_err)?;

    let mut interleaved: Vec<f32> = Vec::new();
    let mut mono: Vec<f32> = Vec::new();
    let mut resampler: Option<(u32, Resampler)> = None;
    let mut decoded_any = false;
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(decode_err(e)),
        };
        if packet.track_id != track_id {
            continue;
        }
        let buf = match decoder.decode(&packet) {
            Ok(b) => b,
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(decode_err(e)),
        };
        let (rate, channels) = (buf.spec().rate(), buf.spec().channels().count());
        if channels == 0 || buf.frames() == 0 {
            continue;
        }
        buf.copy_to_vec_interleaved(&mut interleaved);
        mono.clear();
        mono.extend(
            interleaved
                .chunks_exact(channels)
                .map(|f| f.iter().sum::<f32>() / channels as f32),
        );
        match &mut resampler {
            Some((r, _)) if *r != rate => return Err(decode_err("sample rate changes mid-stream")),
            Some(_) => {}
            None => resampler = Some((rate, Resampler::new(rate, MODEL_SAMPLE_RATE))),
        }
        let (rate, r) = resampler.as_mut().expect("set above");
        decoded_any = true;
        let out = if *rate == MODEL_SAMPLE_RATE {
            std::mem::take(&mut mono)
        } else {
            r.process(&mono)
        };
        if !sink(&out) {
            return Ok(());
        }
    }
    let Some((rate, mut r)) = resampler.filter(|_| decoded_any) else {
        return Err(decode_err("no decodable audio in file"));
    };
    if rate != MODEL_SAMPLE_RATE {
        sink(&r.finish());
    }
    Ok(())
}

/// Decode a whole file to 16 kHz mono. Prefer [`decode_file_chunks`] for long files.
pub fn decode_file(path: &Path) -> Result<Vec<f32>, AudioError> {
    let mut all = Vec::new();
    decode_file_chunks(path, |c| {
        all.extend_from_slice(c);
        true
    })?;
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_wav(path: &Path, rate: u32, channels: u16, samples: impl Iterator<Item = i16>) {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for s in samples {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
    }

    fn tone(rate: u32, secs: u32, ch: u16) -> impl Iterator<Item = i16> {
        (0..rate * secs).flat_map(move |i| {
            let v = (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 16_000.0;
            std::iter::repeat_n(v as i16, usize::from(ch))
        })
    }

    #[test]
    fn stereo_44k_wav_becomes_16k_mono() {
        let dir = std::env::temp_dir().join(format!("tingmix-audio-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.wav");
        write_wav(&p, 44_100, 2, tone(44_100, 2, 2));
        let x = decode_file(&p).unwrap();
        assert!((x.len() as i64 - 32_000).abs() <= 2, "{}", x.len());
        let rms = (x[4_000..28_000].iter().map(|v| v * v).sum::<f32>() / 24_000.0).sqrt();
        assert!((rms - 0.345).abs() < 0.01, "rms {rms}"); // 16000/32768 peak sine
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sink_can_cancel_and_garbage_is_an_error() {
        let dir = std::env::temp_dir().join(format!("tingmix-audio-test2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("b.wav");
        write_wav(&p, 16_000, 1, tone(16_000, 3, 1));
        let mut chunks = 0;
        decode_file_chunks(&p, |_| {
            chunks += 1;
            false
        })
        .unwrap();
        assert_eq!(chunks, 1);
        let g = dir.join("c.wav");
        std::fs::write(&g, b"this is not audio at all, just text").unwrap();
        assert!(matches!(decode_file(&g), Err(AudioError::Decode(_))));
        assert!(decode_file(&dir.join("missing.wav")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
