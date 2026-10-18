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

/// Time from the audio-clock endpoint (speech end + VAD hang + silence_ms) to the final's arrival.
fn final_latency(tail_partial: bool, delay: Duration) -> Duration {
    use std::time::Instant;
    let cfg = SessionConfig {
        tail_partial,
        ..SessionConfig::default()
    };
    let mut w = SessionWorker::spawn_with(MockBackend::new(false).with_delay(delay), &cfg);
    let tone = tone_then_silence(3.0, 0.0);
    for c in tone.chunks(1600) {
        w.feed(c); // faster than real time: the worker is busy with a partial when the speech ends
    }
    let endpoint_s = 3.0 + 0.2 + 0.6 - 0.04; // VAD hang is 10 frames of 20 ms, then silence_ms (closes within one frame of it)
    let (mut fed_s, mut t_endpoint) = (3.0f64, None);
    let silence = [0.0f32; 320];
    let started = Instant::now();
    let final_at = loop {
        w.feed(&silence);
        fed_s += 0.02;
        std::thread::sleep(Duration::from_millis(20));
        if t_endpoint.is_none() && fed_s >= endpoint_s {
            t_endpoint = Some(Instant::now());
        }
        if let Some(at) = w
            .events()
            .try_iter()
            .find_map(|m| matches!(m, SessionMsg::Event(Event::Final(_))).then(Instant::now))
        {
            break at;
        }
        assert!(started.elapsed() < Duration::from_secs(10), "no final");
    };
    final_at.saturating_duration_since(t_endpoint.expect("endpoint passed before the final"))
}

#[test]
fn final_waits_at_most_one_inflight_decode_plus_its_own() {
    let delay = Duration::from_millis(300);
    let slack = Duration::from_millis(250); // paced feed granularity and a loaded machine
    let baseline = final_latency(false, delay);
    assert!(
        baseline <= 2 * delay + slack,
        "baseline final latency {baseline:?}"
    );
}

#[test]
fn tail_partial_lets_the_final_reuse_the_decode_done_during_the_silence() {
    let delay = Duration::from_millis(300);
    let slack = Duration::from_millis(250);
    let (base, tail) = (final_latency(false, delay), final_latency(true, delay));
    println!("final latency after endpoint: baseline {base:?}, tail-partial {tail:?}");
    assert!(tail <= delay + slack, "tail-partial final latency {tail:?}");
    assert!(tail <= base + slack);
}

struct CallCounter(BackendInfo, Arc<AtomicUsize>);

impl AsrBackend for CallCounter {
    fn info(&self) -> &BackendInfo {
        &self.0
    }
    fn transcribe_batch(&mut self, wavs: &[&[f32]]) -> Result<Vec<String>, BackendError> {
        self.1.fetch_add(1, Ordering::SeqCst);
        Ok(wavs
            .iter()
            .map(|w| format!("w{}", w.len() / 8000))
            .collect())
    }
}

#[test]
fn tail_partial_saves_the_final_decode_and_keeps_the_event_contract() {
    let info = BackendInfo {
        name: "count".into(),
        sample_rate: 16_000,
        max_window_s: 30.0,
        streaming_native: false,
    };
    let run = |tail_partial: bool| {
        let calls = Arc::new(AtomicUsize::new(0));
        let cfg = SessionConfig {
            tail_partial,
            ..SessionConfig::default()
        };
        let mut s = StreamingSession::new(CallCounter(info.clone(), calls.clone()), &cfg);
        let mut r = Reducer::new();
        for c in tone_then_silence(2.0, 1.5).chunks(1600) {
            for m in s.feed(c) {
                let SessionMsg::Event(e) = m else {
                    panic!("{m:?}")
                };
                r.apply(e).unwrap();
            }
        }
        (
            calls.load(Ordering::SeqCst),
            r.finals().count(),
            r.open().count(),
        )
    };
    let (base, tail) = (run(false), run(true));
    assert_eq!((base.1, base.2, tail.1, tail.2), (1, 0, 1, 0));
    assert!(
        tail.0 < base.0,
        "decodes: baseline {}, tail {}",
        base.0,
        tail.0
    );
}

fn fixture_wav(name: &str) -> Vec<f32> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/audio")
        .join(name);
    let mut r = hound::WavReader::open(p).unwrap();
    r.samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect()
}

/// Final segments `(start, end)` of a replay of `wav` (plus 1.5 s of silence) through the mock backend.
fn replay_segments(wav: &[f32], cfg: &SessionConfig) -> Vec<(f64, f64)> {
    let mut wav = wav.to_vec();
    wav.extend(std::iter::repeat_n(0.0, 24_000));
    let mut s = StreamingSession::new(MockBackend::new(false), cfg);
    let mut msgs: Vec<_> = wav.chunks(320).flat_map(|c| s.feed(c)).collect();
    msgs.extend(s.flush());
    events(msgs)
        .into_iter()
        .filter_map(|e| match e {
            Event::Final(f) => Some((f.start, f.end)),
            _ => None,
        })
        .collect()
}

/// mix_08s.wav starts with speech at sample 0. The adaptive VAD used to learn that speech as its noise floor and lose the
/// first ~3 s; the live preset must keep the utterance from the first sample: every clearly loud 20 ms frame (above -40 dBFS)
/// has to lie inside a final segment (pauses may be cut out, speech may not).
#[test]
fn live_preset_keeps_speech_that_starts_at_sample_zero() {
    let wav = fixture_wav("mix_08s.wav");
    let live = replay_segments(&wav, &SessionConfig::live_recommended());
    assert!(!live.is_empty(), "no final at all");
    assert_eq!(
        live[0].0, 0.0,
        "the first segment must start with the audio: {live:?}"
    );
    let loud: Vec<f64> = wav
        .chunks_exact(320)
        .enumerate()
        .filter(|(_, f)| {
            10.0 * (f.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / 320.0 + 1e-20)
                .log10()
                > -40.0
        })
        .map(|(i, _)| i as f64 * 0.02 + 0.01)
        .collect();
    assert!(loud.len() > 200, "the fixture should be mostly speech");
    let lost: Vec<f64> = loud
        .iter()
        .copied()
        .filter(|t| !live.iter().any(|(a, b)| a <= t && t <= b))
        .collect();
    assert!(
        lost.is_empty(),
        "{} loud frames outside the segments {live:?}, e.g. at {:?} s",
        lost.len(),
        &lost[..lost.len().min(5)]
    );
}
