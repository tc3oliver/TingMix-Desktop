use tingmix_core::AsrBackend;

use crate::{Decoder, Job, Pipeline, SessionConfig, SessionMsg};

/// Synchronous session: every job is decoded inline, so output depends only on the audio and the
/// chunking. Used for file transcription and golden tests; live capture uses [`crate::SessionWorker`].
pub struct StreamingSession<B: AsrBackend> {
    pipeline: Pipeline,
    decoder: Decoder<B>,
}

impl<B: AsrBackend> StreamingSession<B> {
    pub fn new(backend: B, cfg: &SessionConfig) -> Self {
        Self {
            pipeline: Pipeline::new(backend.info(), cfg),
            decoder: Decoder::new(backend, cfg.stability_k),
        }
    }

    pub fn feed(&mut self, pcm: &[f32]) -> Vec<SessionMsg> {
        let jobs = self.pipeline.feed(pcm);
        self.run(jobs)
    }

    /// End of stream: finalize the open segment, if any.
    pub fn flush(&mut self) -> Vec<SessionMsg> {
        let jobs = self.pipeline.flush();
        self.run(jobs)
    }

    fn run(&mut self, jobs: Vec<Job>) -> Vec<SessionMsg> {
        let mut out = Vec::new();
        for job in &jobs {
            let o = self.decoder.run(job);
            if o.cost_s > 0.0 {
                self.pipeline.report_cost(o.cost_s);
            }
            out.extend(SessionMsg::from_outcome(job, o));
        }
        out
    }
}
