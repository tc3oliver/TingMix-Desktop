//! File transcription: decode -> `StreamingSession` (synchronous, deterministic) -> events.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use tingmix_audio::{decode_file_chunks, MODEL_SAMPLE_RATE};
use tingmix_core::Reducer;
use tingmix_streaming::{MockBackend, SessionConfig, SessionMsg, StreamingSession};

use crate::dto::{FileBody, JobState};

pub struct Outcome {
    pub state: JobState,
    pub audio_s: f64,
    pub message: Option<String>,
}

/// Decodes `path` and transcribes it, calling `emit` for progress and events. Never panics on bad input: a corrupt or
/// unsupported file ends as `Failed` with the decoder's message.
pub fn run(
    path: &Path,
    cancel: &AtomicBool,
    reducer: &mut Reducer,
    emit: &mut dyn FnMut(FileBody),
) -> Outcome {
    let mut session = StreamingSession::new(MockBackend::new(false), &SessionConfig::default());
    let mut samples = 0u64;
    let handle = |msgs: Vec<SessionMsg>, reducer: &mut Reducer, emit: &mut dyn FnMut(FileBody)| {
        for m in msgs {
            if let SessionMsg::Event(ev) = m {
                if reducer.apply(ev.clone()).is_ok() {
                    emit(FileBody::Event { event: ev });
                }
            }
        }
    };
    let decoded = decode_file_chunks(path, |chunk| {
        if cancel.load(Ordering::SeqCst) {
            return false;
        }
        samples += chunk.len() as u64;
        handle(session.feed(chunk), reducer, emit);
        emit(FileBody::Progress {
            audio_s: samples as f64 / f64::from(MODEL_SAMPLE_RATE),
        });
        true
    });
    let audio_s = samples as f64 / f64::from(MODEL_SAMPLE_RATE);
    match decoded {
        Err(e) => Outcome {
            state: JobState::Failed,
            audio_s,
            message: Some(e.to_string()),
        },
        Ok(()) if cancel.load(Ordering::SeqCst) => Outcome {
            state: JobState::Cancelled,
            audio_s,
            message: None,
        },
        Ok(()) => {
            handle(session.flush(), reducer, emit);
            Outcome {
                state: JobState::Done,
                audio_s,
                message: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(path: &Path, secs: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..16_000 * secs {
            let v = if i < 16_000 * 2 {
                (i as f32 * 440.0 * std::f32::consts::TAU / 16_000.0).sin() * 12_000.0
            } else {
                0.0
            };
            w.write_sample(v as i16).unwrap();
        }
        w.finalize().unwrap();
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("tingmix-files-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn wav_file_produces_finals_and_progress() {
        let dir = tmp("ok");
        wav(&dir.join("a.wav"), 4);
        let (mut r, mut progress, mut events) = (Reducer::new(), 0, 0);
        let out = run(
            &dir.join("a.wav"),
            &AtomicBool::new(false),
            &mut r,
            &mut |b| match b {
                FileBody::Progress { .. } => progress += 1,
                FileBody::Event { .. } => events += 1,
                _ => {}
            },
        );
        assert_eq!(out.state, JobState::Done);
        assert!((out.audio_s - 4.0).abs() < 0.01);
        assert!(progress > 0 && events > 0);
        assert_eq!(r.finals().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn garbage_and_missing_files_fail_cleanly_and_cancel_is_honoured() {
        let dir = tmp("bad");
        std::fs::write(dir.join("junk.wav"), b"this is not audio at all").unwrap();
        let mut r = Reducer::new();
        let out = run(
            &dir.join("junk.wav"),
            &AtomicBool::new(false),
            &mut r,
            &mut |_| {},
        );
        assert_eq!(out.state, JobState::Failed);
        assert!(out.message.is_some());
        assert_eq!(
            run(
                &dir.join("missing.mp3"),
                &AtomicBool::new(false),
                &mut r,
                &mut |_| {}
            )
            .state,
            JobState::Failed
        );

        wav(&dir.join("c.wav"), 3);
        let out = run(
            &dir.join("c.wav"),
            &AtomicBool::new(true),
            &mut r,
            &mut |_| {},
        );
        assert_eq!(out.state, JobState::Cancelled);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
