use std::time::Instant;

use tingmix_core::{AsrBackend, AsrStream, BackendError, Event, Transcript};

use crate::{Job, JobKind, StablePrefix, Suppression};

/// Result of one job. A failed final still yields an event when a partial was shown, so a
/// segment never stays open forever; the error is reported alongside.
pub struct Outcome {
    pub event: Option<Event>,
    pub error: Option<BackendError>,
    /// Wall-clock seconds spent in the backend; feeds the scheduler. 0.0: the backend was not called.
    pub cost_s: f64,
}

/// Runs jobs against a backend and numbers the resulting events. Single-threaded; jobs must be
/// given in the order the pipeline produced them (coalescing in between is allowed).
pub struct Decoder<B: AsrBackend> {
    backend: B,
    sr: usize,
    transcript: Transcript,
    stability: StablePrefix,
    stream: Option<Box<dyn AsrStream>>,
    last: Option<(usize, usize, String, Option<f32>)>, // (start, end, text, confidence) of the latest buffered decode
    tail: Option<(u64, String, Option<f32>)>, // (segment, text, confidence) of the latest completed tail partial
    suppression: Option<Suppression>,
}

impl<B: AsrBackend> Decoder<B> {
    pub fn new(backend: B, stability_k: usize) -> Self {
        let sr = backend.info().sample_rate as usize;
        Self {
            backend,
            sr,
            transcript: Transcript::new(),
            stability: StablePrefix::new(stability_k),
            stream: None,
            last: None,
            tail: None,
            suppression: None,
        }
    }

    /// Never show text the backend is not confident about (see [`Suppression`]). Off by default.
    pub fn with_suppression(mut self, s: Option<Suppression>) -> Self {
        self.suppression = s;
        self
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn transcript(&self) -> &Transcript {
        &self.transcript
    }

    pub fn run(&mut self, job: &Job) -> Outcome {
        if job.opens {
            self.stability.reset();
            self.last = None;
            self.stream = None;
            if job.incremental {
                match self.backend.new_stream() {
                    Ok(s) => self.stream = Some(s),
                    Err(e) => return self.failed(job, e, 0.0),
                }
            }
        }
        let reused = job.kind == JobKind::Final
            && job.reuse_tail
            && self.tail.as_ref().is_some_and(|t| t.0 == job.segment_id);
        let t0 = Instant::now();
        let res = if reused {
            Ok(self.tail.take().map(|t| (t.1, t.2)).unwrap_or_default())
        } else {
            self.decode(job)
        };
        let cost_s = if reused {
            0.0
        } else {
            t0.elapsed().as_secs_f64()
        };
        if let (Ok((text, conf)), true) = (&res, job.tail) {
            self.tail = Some((job.segment_id, text.clone(), *conf));
        }
        // A suppressed decode is treated as empty text: no partial, and no final for a segment that never showed anything.
        let res = res.map(|(text, conf)| {
            match self
                .suppression
                .as_ref()
                .is_some_and(|p| p.rejects(&text, conf, job.kind == JobKind::Final))
            {
                true => String::new(),
                false => text,
            }
        });
        match (res, job.kind) {
            (Ok(text), JobKind::Partial) => Outcome {
                event: self.partial(job, text),
                error: None,
                cost_s,
            },
            (Ok(text), JobKind::Final) => Outcome {
                event: self.final_(job, text),
                error: None,
                cost_s,
            },
            (Err(e), _) => self.failed(job, e, cost_s),
        }
    }

    fn decode(&mut self, job: &Job) -> Result<(String, Option<f32>), BackendError> {
        let mut conf = None;
        let text = if job.incremental {
            let no_stream = || BackendError::Inference("no open stream for this segment".into());
            let stream = self.stream.as_mut().ok_or_else(no_stream)?;
            match job.kind {
                JobKind::Partial => stream.accept(&job.audio)?,
                JobKind::Final => {
                    if !job.audio.is_empty() {
                        stream.accept(&job.audio)?; // residual samples must reach the model before finalize
                    }
                    let text = stream.finalize();
                    self.stream = None;
                    text?
                }
            }
        } else if job.kind == JobKind::Final
            && self
                .last
                .as_ref()
                .is_some_and(|l| (l.0, l.1) == (job.start, job.end))
        {
            let (_, _, text, c) = self.last.take().unwrap_or_default(); // the last partial already covered exactly this audio
            conf = c;
            text
        } else if job.audio.len() >= self.sr / 10 {
            let t = self
                .backend
                .transcribe_batch_scored(&[&job.audio])?
                .into_iter()
                .next()
                .unwrap_or_else(|| tingmix_core::Transcription {
                    text: String::new(),
                    confidence: None,
                });
            conf = t.confidence;
            self.last = Some((job.start, job.end, t.text.trim().to_owned(), conf));
            t.text
        } else {
            String::new()
        };
        Ok((text.trim().to_owned(), conf))
    }

    fn partial(&mut self, job: &Job, text: String) -> Option<Event> {
        let stable = self.stability.update(&text);
        let prev = self.transcript.current(job.segment_id);
        if text.is_empty() || prev.is_some_and(|p| p.text == text && p.stable_prefix_len == stable)
        {
            return None;
        }
        let (a, b) = self.span(job);
        self.transcript
            .partial(job.segment_id, text, a, b, stable)
            .ok()
            .map(Event::Partial)
    }

    fn final_(&mut self, job: &Job, text: String) -> Option<Event> {
        self.last = None;
        // A suppressed final is an empty one: if the segment showed a partial it closes with empty text (the caption is
        // retracted), otherwise nothing is emitted.
        if text.is_empty() && self.transcript.current(job.segment_id).is_none() {
            return None; // false trigger, nothing was ever shown
        }
        let (a, b) = self.span(job);
        self.transcript
            .final_(job.segment_id, text, a, b)
            .ok()
            .map(Event::Final)
    }

    fn failed(&mut self, job: &Job, error: BackendError, cost_s: f64) -> Outcome {
        let event = (job.kind == JobKind::Final)
            .then(|| {
                self.stream = None;
                self.last = None;
                let text = self.transcript.current(job.segment_id)?.text.clone();
                self.final_(job, text)
            })
            .flatten();
        Outcome {
            event,
            error: Some(error),
            cost_s,
        }
    }

    fn span(&self, job: &Job) -> (f64, f64) {
        (
            job.start as f64 / self.sr as f64,
            job.end as f64 / self.sr as f64,
        )
    }
}
