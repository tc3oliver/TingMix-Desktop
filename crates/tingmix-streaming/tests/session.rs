use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tingmix_core::{AsrBackend, AsrStream, BackendError, BackendInfo, Event, Reducer};
use tingmix_streaming::{MockBackend, SessionConfig, SessionMsg, SessionWorker, StreamingSession};

/// 440 Hz tone (loud enough for the VAD) of `speech_s` seconds followed by `silence_s` of zeros.
fn tone_then_silence(speech_s: f64, silence_s: f64) -> Vec<f32> {
    let n = (speech_s * 16_000.0) as usize;
    let tone = (0..n).map(|i| 0.2 * (i as f32 * 440.0 * std::f32::consts::TAU / 16_000.0).sin());
    tone.chain(std::iter::repeat_n(0.0, (silence_s * 16_000.0) as usize))
        .collect()
}

fn events(msgs: Vec<SessionMsg>) -> Vec<Event> {
    msgs.into_iter()
        .map(|m| match m {
            SessionMsg::Event(e) => e,
            SessionMsg::Error { message, .. } => panic!("backend error: {message}"),
        })
        .collect()
}

#[test]
fn sync_session_output_is_accepted_by_the_reducer() {
    let mut s = StreamingSession::new(MockBackend::new(false), &SessionConfig::default());
    let audio = tone_then_silence(3.0, 1.0);
    let mut msgs = Vec::new();
    for c in audio.chunks(1600) {
        msgs.extend(s.feed(c));
    }
    let evs = events(msgs);
    let mut r = Reducer::new();
    for e in &evs {
        r.apply(e.clone()).unwrap();
    }
    assert!(
        evs.iter()
            .filter(|e| matches!(e, Event::Partial(_)))
            .count()
            >= 4
    );
    assert_eq!(r.finals().count(), 1);
    assert_eq!(r.open().count(), 0);
}

#[test]
fn slow_backend_coalesces_partials_but_not_finals() {
    let audio = tone_then_silence(4.0, 1.0);
    let sync_final = {
        let mut s = StreamingSession::new(MockBackend::new(false), &SessionConfig::default());
        let mut msgs: Vec<_> = audio.chunks(1600).flat_map(|c| s.feed(c)).collect();
        msgs.extend(s.flush());
        events(msgs)
            .into_iter()
            .rev()
            .find(|e| matches!(e, Event::Final(_)))
            .unwrap()
    };
    let backend = MockBackend::new(false).with_delay(Duration::from_millis(150));
    let mut w = SessionWorker::spawn(backend, &SessionConfig::default());
    for c in audio.chunks(1600) {
        w.feed(c); // no sleeping: the model is far slower than this
    }
    let coalesced = w.coalesced_partials();
    let evs = events(w.finish());
    assert!(coalesced > 0, "nothing was coalesced");
    let mut r = Reducer::new();
    for e in &evs {
        r.apply(e.clone()).unwrap();
    }
    let finals: Vec<_> = evs
        .iter()
        .filter(|e| matches!(e, Event::Final(_)))
        .collect();
    assert_eq!(finals.len(), 1);
    // fewer partials were shown, so only the revision may differ
    let (Event::Final(got), Event::Final(want)) = (finals[0], &sync_final) else {
        unreachable!()
    };
    assert_eq!(
        (&got.text, got.start, got.end),
        (&want.text, want.start, want.end)
    );
    assert!(got.revision < want.revision);
}

struct CountingBackend {
    info: BackendInfo,
    fed: Arc<AtomicUsize>,
}

struct CountingStream(Arc<AtomicUsize>);

impl AsrBackend for CountingBackend {
    fn info(&self) -> &BackendInfo {
        &self.info
    }
    fn transcribe_batch(&mut self, _: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        Err(BackendError::Unsupported("batch"))
    }
    fn new_stream(&mut self) -> Result<Box<dyn AsrStream>, BackendError> {
        Ok(Box::new(CountingStream(self.fed.clone())))
    }
}

impl AsrStream for CountingStream {
    fn accept(&mut self, pcm: &[f32]) -> Result<String, BackendError> {
        std::thread::sleep(Duration::from_millis(40));
        self.0.fetch_add(pcm.len(), Ordering::SeqCst);
        Ok("x".into())
    }
    fn finalize(&mut self) -> Result<String, BackendError> {
        Ok("x".into())
    }
}

#[test]
fn native_stream_receives_every_sample_exactly_once_even_when_coalesced() {
    let fed = Arc::new(AtomicUsize::new(0));
    let info = BackendInfo {
        name: "count".into(),
        sample_rate: 16_000,
        max_window_s: 30.0,
        streaming_native: true,
    };
    let mut w = SessionWorker::spawn(
        CountingBackend {
            info,
            fed: fed.clone(),
        },
        &SessionConfig::default(),
    );
    for c in tone_then_silence(3.0, 1.0).chunks(800) {
        w.feed(c);
    }
    let coalesced = w.coalesced_partials();
    let evs = events(w.finish());
    let Some(Event::Final(f)) = evs.last() else {
        panic!("no final: {evs:?}")
    };
    assert!(coalesced > 0);
    let expected = ((f.end - f.start) * 16_000.0).round() as usize;
    // audio already fed during the trimmed trailing silence cannot be un-fed, so allow the trailing part
    let got = fed.load(Ordering::SeqCst);
    assert!(
        got >= expected && got <= expected + 16_000 * 6 / 10,
        "fed {got}, segment {expected}"
    );
}

struct FailingBackend(BackendInfo, usize);

impl AsrBackend for FailingBackend {
    fn info(&self) -> &BackendInfo {
        &self.0
    }
    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        self.1 += 1;
        if self.1 >= 3 {
            return Err(BackendError::Inference("boom".into()));
        }
        Ok(wavs.iter().map(|_| "partial text".to_owned()).collect())
    }
}

#[test]
fn failed_final_falls_back_to_the_last_partial_and_reports_the_error() {
    let info = BackendInfo {
        name: "fail".into(),
        sample_rate: 16_000,
        max_window_s: 30.0,
        streaming_native: false,
    };
    let mut s = StreamingSession::new(FailingBackend(info, 0), &SessionConfig::default());
    let msgs: Vec<_> = tone_then_silence(1.5, 1.0)
        .chunks(1600)
        .flat_map(|c| s.feed(c))
        .collect();
    assert!(msgs.iter().any(|m| matches!(m, SessionMsg::Error { .. })));
    let Some(SessionMsg::Event(Event::Final(f))) = msgs
        .iter()
        .find(|m| matches!(m, SessionMsg::Event(Event::Final(_))))
    else {
        panic!("no final in {msgs:?}")
    };
    assert_eq!(f.text, "partial text");
}
