//! Live transcription: capture thread -> resampler -> `SessionWorker` -> reducer + events.
//!
//! The reducer here is authoritative: a reloaded window asks for its snapshot instead of replaying events.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use tingmix_audio::{
    start_capture, CaptureEvent, CaptureGuard, CaptureReader, Resampler, MODEL_SAMPLE_RATE,
};
use tingmix_core::{Event, FinalEvent, Reducer};
use tingmix_streaming::{MockBackend, SessionConfig, SessionMsg, SessionWorker};

use crate::dto::{LiveMsg, LiveState, LiveStatus, BACKEND_LABEL};

/// Mono samples at some native rate; the capture device in production, synthetic audio in tests.
pub trait PcmSource {
    fn sample_rate(&self) -> u32;
    fn device(&self) -> String;
    fn read(&mut self, out: &mut [f32]) -> usize;
    fn overflowed(&self) -> u64;
    /// Out-of-band problems since the last call; `Err` = capture is over.
    fn poll(&mut self) -> Result<Vec<String>, String>;
}

struct MicSource {
    _guard: CaptureGuard,
    reader: CaptureReader,
}

impl MicSource {
    fn open(device_id: Option<String>) -> Result<Self, String> {
        // two seconds of ring: the pump drains every few ms, so this only absorbs a stalled thread
        let (guard, reader) =
            start_capture(device_id.as_deref(), 2.0).map_err(|e| e.to_string())?;
        Ok(Self {
            _guard: guard,
            reader,
        })
    }
}

impl PcmSource for MicSource {
    fn sample_rate(&self) -> u32 {
        self.reader.sample_rate()
    }
    fn device(&self) -> String {
        self.reader.device_name().to_owned()
    }
    fn read(&mut self, out: &mut [f32]) -> usize {
        self.reader.read(out)
    }
    fn overflowed(&self) -> u64 {
        self.reader.overflowed_samples()
    }
    fn poll(&mut self) -> Result<Vec<String>, String> {
        let mut notes = Vec::new();
        for e in self.reader.poll_events() {
            match e {
                CaptureEvent::Disconnected(m) => {
                    return Err(format!("microphone disconnected: {m}"))
                }
                CaptureEvent::Rerouted => notes.push("input rerouted to another device".into()),
                CaptureEvent::Xrun => notes.push("audio glitch (xrun)".into()),
                CaptureEvent::Error(m) => notes.push(m),
            }
        }
        Ok(notes)
    }
}

pub type Sink = Arc<dyn Fn(LiveMsg) + Send + Sync>;
/// Called with the finals so far whenever one is added, and once more when the session ends.
pub type Persist = Arc<dyn Fn(&[FinalEvent], f64) + Send + Sync>;

pub struct Shared {
    pub id: String,
    reducer: Mutex<Reducer>,
    status: Mutex<LiveStatus>,
}

impl Shared {
    fn new(id: String) -> Arc<Self> {
        let status = LiveStatus {
            state: LiveState::Starting,
            session_id: Some(id.clone()),
            level_db: -120.0,
            backend: BACKEND_LABEL.into(),
            ..LiveStatus::default()
        };
        Arc::new(Self {
            id,
            reducer: Mutex::default(),
            status: Mutex::new(status),
        })
    }

    pub fn status(&self) -> LiveStatus {
        self.status.lock().unwrap().clone()
    }

    pub fn snapshot(&self) -> tingmix_core::Snapshot {
        self.reducer.lock().unwrap().snapshot()
    }

    fn update(&self, sink: &Sink, f: impl FnOnce(&mut LiveStatus)) {
        let mut s = self.status.lock().unwrap();
        f(&mut s);
        sink(LiveMsg::Status(s.clone()));
    }
}

pub struct Live {
    pub shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Live {
    /// Opens the microphone on the capture thread and returns once it is recording (or failed to open).
    pub fn start_mic(
        id: String,
        device_id: Option<String>,
        sink: Sink,
        persist: Persist,
    ) -> Result<Self, String> {
        Self::start(
            id,
            move || Ok(Box::new(MicSource::open(device_id)?) as Box<dyn PcmSource>),
            sink,
            persist,
        )
    }

    /// `open` runs on the new thread because the OS capture stream must stay on the thread that created it.
    pub fn start(
        id: String,
        open: impl FnOnce() -> Result<Box<dyn PcmSource>, String> + Send + 'static,
        sink: Sink,
        persist: Persist,
    ) -> Result<Self, String> {
        let shared = Shared::new(id);
        let stop = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = channel();
        let (sh, st) = (shared.clone(), stop.clone());
        let thread = std::thread::Builder::new()
            .name("tingmix-live".into())
            .spawn(move || match open() {
                Ok(src) => {
                    let _ = ready_tx.send(Ok(()));
                    pump(src, &sh, &st, &sink, &persist);
                }
                Err(e) => {
                    sh.update(&sink, |s| {
                        s.state = LiveState::Error;
                        s.message = Some(e.clone());
                    });
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|e| e.to_string())?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                shared,
                stop,
                thread: Some(thread),
            }),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => Err("capture thread died while starting".into()),
        }
    }

    pub fn is_running(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }

    /// Stop capture, finalize the open segment and wait until every queued job has been reported.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        self.stop();
    }
}

fn rms_db(x: &[f32]) -> f64 {
    if x.is_empty() {
        return -120.0;
    }
    let ms = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64;
    (10.0 * ms.max(1e-12).log10()).max(-120.0)
}

fn pump(
    mut src: Box<dyn PcmSource>,
    sh: &Arc<Shared>,
    stop: &AtomicBool,
    sink: &Sink,
    persist: &Persist,
) {
    let native = src.sample_rate();
    let mut resampler =
        (native != MODEL_SAMPLE_RATE).then(|| Resampler::new(native, MODEL_SAMPLE_RATE));
    let mut worker = SessionWorker::spawn(MockBackend::new(false), &SessionConfig::default());
    let t0 = Instant::now();
    let device = src.device();
    sh.update(sink, |s| {
        s.state = LiveState::Recording;
        s.device = device;
    });

    let (mut buf, mut samples_in, mut last_status) = (vec![0f32; 4096], 0u64, Instant::now());
    let mut failure: Option<String> = None;
    let apply = |m: SessionMsg, sh: &Shared| -> bool {
        match m {
            SessionMsg::Event(ev) => {
                let is_final = matches!(ev, Event::Final(_));
                // a duplicate or stale event is rejected by the reducer and must not reach the UI
                if sh.reducer.lock().unwrap().apply(ev.clone()).is_ok() {
                    sink(LiveMsg::Event {
                        session_id: sh.id.clone(),
                        event: ev,
                    });
                    return is_final;
                }
            }
            SessionMsg::Error { message, .. } => sh.update(sink, |s| s.message = Some(message)),
        }
        false
    };
    let finals_now = |sh: &Shared| sh.reducer.lock().unwrap().snapshot().finals;

    loop {
        let stopping = stop.load(Ordering::SeqCst);
        let n = src.read(&mut buf);
        if n > 0 {
            samples_in += n as u64;
            let level = rms_db(&buf[..n]);
            let pcm = match &mut resampler {
                Some(r) => r.process(&buf[..n]),
                None => buf[..n].to_vec(),
            };
            worker.feed(&pcm);
            sh.status.lock().unwrap().level_db = level;
        }
        let mut new_final = false;
        while let Ok(m) = worker.events().try_recv() {
            new_final |= apply(m, sh);
        }
        if new_final {
            persist(&finals_now(sh), t0.elapsed().as_secs_f64());
        }
        match src.poll() {
            Ok(notes) => {
                if let Some(n) = notes.last() {
                    sh.status.lock().unwrap().message = Some(n.clone());
                }
            }
            Err(e) => failure = Some(e),
        }
        if stopping || failure.is_some() {
            break;
        }
        if last_status.elapsed() >= Duration::from_millis(250) {
            last_status = Instant::now();
            let (ov, co) = (src.overflowed(), worker.coalesced_partials());
            sh.update(sink, |s| {
                s.elapsed_s = t0.elapsed().as_secs_f64();
                s.samples_in = samples_in;
                s.overflow_samples = ov;
                s.coalesced_partials = co;
            });
        }
        if n == 0 {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    sh.update(sink, |s| s.state = LiveState::Stopping);
    let coalesced = worker.coalesced_partials();
    for m in worker.finish() {
        apply(m, sh);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    persist(&finals_now(sh), elapsed);
    let overflow = src.overflowed();
    sh.update(sink, |s| {
        s.state = if failure.is_some() {
            LiveState::Error
        } else {
            LiveState::Idle
        };
        if failure.is_some() {
            s.message = failure.clone();
        }
        s.elapsed_s = elapsed;
        s.samples_in = samples_in;
        s.overflow_samples = overflow;
        s.coalesced_partials = coalesced;
        s.level_db = -120.0;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 s of a loud tone, 1.5 s of silence, then nothing more: one segment that must end up final.
    struct Synthetic {
        data: Vec<f32>,
        pos: usize,
        disconnect_at_end: bool,
    }

    impl PcmSource for Synthetic {
        fn sample_rate(&self) -> u32 {
            48_000
        }
        fn device(&self) -> String {
            "synthetic".into()
        }
        fn read(&mut self, out: &mut [f32]) -> usize {
            let n = out.len().min(self.data.len() - self.pos);
            out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            n
        }
        fn overflowed(&self) -> u64 {
            0
        }
        fn poll(&mut self) -> Result<Vec<String>, String> {
            if self.disconnect_at_end && self.pos == self.data.len() {
                return Err("microphone disconnected: test".into());
            }
            Ok(vec![])
        }
    }

    fn synthetic(disconnect_at_end: bool) -> Box<dyn PcmSource> {
        let tone = (0..48_000 * 3)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5);
        let data = tone.chain(std::iter::repeat_n(0.0, 72_000)).collect();
        Box::new(Synthetic {
            data,
            pos: 0,
            disconnect_at_end,
        })
    }

    fn run(disconnect: bool) -> (Live, Vec<LiveMsg>, Arc<Mutex<Vec<usize>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let l = log.clone();
        let sink: Sink = Arc::new(move |m| l.lock().unwrap().push(m));
        let persisted = Arc::new(Mutex::new(Vec::new()));
        let p = persisted.clone();
        let persist: Persist = Arc::new(move |f, _| p.lock().unwrap().push(f.len()));
        let mut live = Live::start(
            "live-test".into(),
            move || Ok(synthetic(disconnect)),
            sink,
            persist,
        )
        .unwrap();
        let t = Instant::now();
        while live.is_running()
            && (disconnect || live.shared.snapshot().finals.is_empty())
            && t.elapsed() < Duration::from_secs(10)
        {
            std::thread::sleep(Duration::from_millis(20));
        }
        live.stop();
        let msgs = std::mem::take(&mut *log.lock().unwrap());
        (live, msgs, persisted)
    }

    #[test]
    fn partials_revise_in_place_then_final_is_persisted_and_snapshot_survives_stop() {
        let (live, msgs, persisted) = run(false);
        let events: Vec<&Event> = msgs
            .iter()
            .filter_map(|m| {
                if let LiveMsg::Event { event, .. } = m {
                    Some(event)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            events.iter().any(|e| matches!(e, Event::Partial(_))),
            "no partial: {msgs:?}"
        );
        assert!(matches!(events.last(), Some(Event::Final(_))));
        // revisions of a segment strictly increase
        let mut last = None;
        for e in &events {
            if let Some(prev) = last.filter(|(id, _)| *id == e.segment_id()) {
                assert!(e.revision() > prev.1);
            }
            last = Some((e.segment_id(), e.revision()));
        }
        let snap = live.shared.snapshot();
        assert_eq!(snap.finals.len(), 1);
        assert!(snap.open.is_empty());
        assert_eq!(persisted.lock().unwrap().last(), Some(&1));
        let st = live.shared.status();
        assert_eq!(
            (st.state, st.device.as_str(), st.backend.as_str()),
            (LiveState::Idle, "synthetic", BACKEND_LABEL)
        );
        assert!(st.samples_in >= 48_000 * 3);
    }

    #[test]
    fn disconnect_ends_in_error_state_and_keeps_the_transcript() {
        let (live, _, _) = run(true);
        let st = live.shared.status();
        assert_eq!(st.state, LiveState::Error);
        assert!(st.message.unwrap().contains("disconnected"));
        assert_eq!(live.shared.snapshot().finals.len(), 1);
    }

    #[test]
    fn open_failure_is_reported_to_the_caller() {
        let sink: Sink = Arc::new(|_| {});
        let persist: Persist = Arc::new(|_, _| {});
        let err = Live::start(
            "x".into(),
            || Err("microphone access denied".into()),
            sink,
            persist,
        )
        .err()
        .unwrap();
        assert!(err.contains("denied"));
    }
}
