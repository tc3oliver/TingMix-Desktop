use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use tingmix_core::{AsrBackend, Event};

use crate::decoder::Outcome;
use crate::{Decoder, Job, JobKind, Pipeline, SessionConfig};

/// What a session reports to its consumer.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionMsg {
    Event(Event),
    /// A backend call failed. The session keeps running; `segment_id` names the affected segment.
    Error {
        segment_id: u64,
        message: String,
    },
}

impl SessionMsg {
    pub(crate) fn from_outcome(job: &Job, o: Outcome) -> Vec<SessionMsg> {
        let err = o.error.map(|e| SessionMsg::Error {
            segment_id: job.segment_id,
            message: e.to_string(),
        });
        o.event
            .map(SessionMsg::Event)
            .into_iter()
            .chain(err)
            .collect()
    }
}

#[derive(Default)]
struct QueueState {
    jobs: VecDeque<Job>,
    closed: bool,
}

#[derive(Default)]
struct Queue {
    state: Mutex<QueueState>,
    ready: Condvar,
    coalesced: AtomicU64,
    cost_bits: AtomicU64, // f64 seconds of the latest decode
}

impl Queue {
    /// Queue a job. A still-waiting partial of the same segment is superseded: buffered jobs carry
    /// the whole segment, so the older one is dropped; incremental jobs carry only new audio, so
    /// theirs is prepended to the new job. Finals are never dropped.
    fn push(&self, mut job: Job) {
        let mut st = self.state.lock().unwrap();
        if let Some(i) = st
            .jobs
            .iter()
            .position(|j| j.segment_id == job.segment_id && j.kind == JobKind::Partial)
        {
            let old = st.jobs.remove(i).unwrap();
            // a tail partial that never started cannot be reused: the final decodes the audio itself
            job.reuse_tail &= !old.tail;
            if job.incremental {
                let mut audio = old.audio;
                audio.append(&mut job.audio);
                job.audio = audio;
                job.opens |= old.opens;
            }
            self.coalesced.fetch_add(1, Ordering::Relaxed);
        }
        st.jobs.push_back(job);
        self.ready.notify_one();
    }

    fn close(&self) {
        self.state.lock().unwrap().closed = true;
        self.ready.notify_one();
    }

    /// Next job; `None` once closed and drained.
    fn pop(&self) -> Option<Job> {
        let mut st = self.state.lock().unwrap();
        loop {
            if let Some(j) = st.jobs.pop_front() {
                return Some(j);
            }
            if st.closed {
                return None;
            }
            st = self.ready.wait(st).unwrap();
        }
    }
}

/// Live session: the pipeline runs in the caller's thread (cheap, never blocks on inference), the
/// decoder on its own thread. If the model falls behind, queued partials of a segment are
/// coalesced; audio and finals are never dropped. Dropping the worker finishes queued jobs first.
pub struct SessionWorker {
    pipeline: Pipeline,
    queue: Arc<Queue>,
    events: Receiver<SessionMsg>,
    thread: Option<JoinHandle<()>>,
}

impl SessionWorker {
    /// Live session. Uses the tail-partial policy (`cfg.tail_partial` is forced on) so a final
    /// costs at most the remainder of one decode.
    pub fn spawn<B: AsrBackend + 'static>(backend: B, cfg: &SessionConfig) -> Self {
        Self::spawn_with(
            backend,
            &SessionConfig {
                tail_partial: true,
                ..cfg.clone()
            },
        )
    }

    /// Like [`spawn`](Self::spawn) but uses `cfg` exactly as given (for comparing policies).
    pub fn spawn_with<B: AsrBackend + 'static>(backend: B, cfg: &SessionConfig) -> Self {
        let pipeline = Pipeline::new(backend.info(), cfg);
        let queue = Arc::new(Queue::default());
        let (tx, events) = channel();
        let (q, k, sup) = (queue.clone(), cfg.stability_k, cfg.suppression);
        let thread = std::thread::Builder::new()
            .name("tingmix-inference".into())
            .spawn(move || run_worker(Decoder::new(backend, k).with_suppression(sup), q, tx))
            .expect("spawn inference thread");
        Self {
            pipeline,
            queue,
            events,
            thread: Some(thread),
        }
    }

    /// Never blocks on inference.
    pub fn feed(&mut self, pcm: &[f32]) {
        self.pipeline
            .report_cost(f64::from_bits(self.queue.cost_bits.load(Ordering::Relaxed)));
        for job in self.pipeline.feed(pcm) {
            self.queue.push(job);
        }
    }

    /// End of stream: finalize the open segment, if any.
    pub fn flush(&mut self) {
        for job in self.pipeline.flush() {
            self.queue.push(job);
        }
    }

    pub fn events(&self) -> &Receiver<SessionMsg> {
        &self.events
    }

    /// Partial jobs superseded before they ran.
    pub fn coalesced_partials(&self) -> u64 {
        self.queue.coalesced.load(Ordering::Relaxed)
    }

    /// Flush, wait for every queued job, and return all messages not yet received.
    pub fn finish(mut self) -> Vec<SessionMsg> {
        self.flush();
        self.shutdown();
        self.events.try_iter().collect()
    }

    fn shutdown(&mut self) {
        self.queue.close();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for SessionWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run_worker<B: AsrBackend>(mut decoder: Decoder<B>, queue: Arc<Queue>, tx: Sender<SessionMsg>) {
    while let Some(job) = queue.pop() {
        let o = decoder.run(&job);
        if o.cost_s > 0.0 {
            queue.cost_bits.store(o.cost_s.to_bits(), Ordering::Relaxed);
        }
        for m in SessionMsg::from_outcome(&job, o) {
            if tx.send(m).is_err() {
                return; // consumer gone
            }
        }
    }
}
