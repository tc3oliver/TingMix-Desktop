use tingmix_core::{AsrBackend, BackendError, BackendInfo, Event, Transcription};
use tingmix_streaming::{SessionConfig, SessionMsg, StreamingSession, Suppression};

struct Scripted {
    info: BackendInfo,
    text: &'static str,
    confidence: Option<f32>,
}

impl AsrBackend for Scripted {
    fn info(&self) -> &BackendInfo {
        &self.info
    }
    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        Ok(wavs.iter().map(|_| self.text.to_string()).collect())
    }
    fn transcribe_batch_scored(
        &mut self,
        wavs: &[&[f32]],
    ) -> Result<Vec<Transcription>, BackendError> {
        Ok(wavs
            .iter()
            .map(|_| Transcription {
                text: self.text.to_string(),
                confidence: self.confidence,
            })
            .collect())
    }
}

fn run(
    text: &'static str,
    confidence: Option<f32>,
    suppression: Option<Suppression>,
    tail_partial: bool,
) -> Vec<Event> {
    let info = BackendInfo {
        name: "scripted".into(),
        sample_rate: 16_000,
        max_window_s: 29.5,
        streaming_native: false,
    };
    let cfg = SessionConfig {
        suppression,
        tail_partial,
        ..SessionConfig::default()
    };
    let mut s = StreamingSession::new(
        Scripted {
            info,
            text,
            confidence,
        },
        &cfg,
    );
    let tone: Vec<f32> = (0..32_000).map(|i| 0.2 * (i as f32 * 0.3).sin()).collect();
    let mut out = Vec::new();
    for c in tone.chunks(320).chain(vec![0.0f32; 24_000].chunks(320)) {
        out.extend(s.feed(c));
    }
    out.extend(s.flush());
    out.into_iter()
        .filter_map(|m| {
            if let SessionMsg::Event(e) = m {
                Some(e)
            } else {
                None
            }
        })
        .collect()
}

fn final_text(ev: &[Event]) -> Option<String> {
    ev.iter().find_map(|e| {
        if let Event::Final(f) = e {
            Some(f.text.clone())
        } else {
            None
        }
    })
}

#[test]
fn off_by_default_nothing_changes() {
    assert_eq!(
        final_text(&run("m", Some(0.1), None, false)).as_deref(),
        Some("m")
    );
}

#[test]
fn low_confidence_fragment_is_retracted_at_the_final() {
    for tail in [false, true] {
        let ev = run("m", Some(0.4), Some(Suppression::RECOMMENDED), tail);
        assert!(
            ev.iter()
                .any(|e| matches!(e, Event::Partial(p) if p.text == "m")),
            "the partial is shown (tail={tail})"
        );
        assert_eq!(
            final_text(&ev).as_deref(),
            Some(""),
            "its final retracts the caption (tail={tail})"
        );
    }
}

#[test]
fn very_low_confidence_never_shows_anything() {
    let ev = run("m", Some(0.1), Some(Suppression::RECOMMENDED), false);
    assert!(ev.is_empty(), "{ev:?}");
}

#[test]
fn confident_short_words_and_unscored_backends_pass() {
    assert_eq!(
        final_text(&run("好", Some(0.97), Some(Suppression::RECOMMENDED), true)).as_deref(),
        Some("好")
    );
    assert_eq!(
        final_text(&run("OK", None, Some(Suppression::RECOMMENDED), false)).as_deref(),
        Some("OK")
    );
    assert_eq!(
        final_text(&run(
            "今天天氣很好",
            Some(0.7),
            Some(Suppression::RECOMMENDED),
            false
        ))
        .as_deref(),
        Some("今天天氣很好")
    );
}
