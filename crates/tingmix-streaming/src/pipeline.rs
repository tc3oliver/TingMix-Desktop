use tingmix_core::BackendInfo;

use crate::{CloseReason, Endpointer, EnergyVad, SampleRing, Scheduler, Vad};

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub hop_ms: f64,
    pub silence_ms: f64,
    pub max_segment_s: f64,
    pub preroll_ms: f64,
    pub tail_pad_ms: f64,
    pub stability_k: usize,
    /// Compute budget of partial decoding, as a fraction of real time.
    pub max_load: f64,
    pub vad_threshold_db: f64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            hop_ms: 480.0,
            silence_ms: 600.0,
            max_segment_s: 25.0,
            preroll_ms: 800.0,
            tail_pad_ms: 0.0,
            stability_k: 2,
            max_load: 0.5,
            vad_threshold_db: -50.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobKind {
    Partial,
    Final,
}

/// One unit of inference work. Owns its audio, so the pipeline's ring buffer can move on.
#[derive(Debug, Clone)]
pub struct Job {
    pub segment_id: u64,
    pub kind: JobKind,
    /// Absolute sample range of the whole segment so far (`[start, end)` on the stream timeline).
    pub start: usize,
    pub end: usize,
    /// Buffered models: the whole range. Native streams: only samples not fed before (`incremental`).
    pub audio: Vec<f32>,
    /// First job of its segment: the decoder starts a fresh utterance.
    pub opens: bool,
    pub incremental: bool,
}

/// VAD, endpointing and decode scheduling over an absolute sample timeline. Feed it any chunking
/// of 16 kHz-ish mono PCM; it emits [`Job`]s. Samples are never dropped: a closed segment's job
/// carries a copy of every sample it needs.
pub struct Pipeline {
    sr: usize,
    native: bool,
    vad: Box<dyn Vad>,
    preroll: usize,
    tail_pad: usize,
    endpointer: Endpointer,
    scheduler: Scheduler,
    frame: usize,
    buf: SampleRing,
    rem: Vec<f32>, // samples of an incomplete VAD frame
    pos: usize,    // samples processed (whole frames)
    prev_end: usize,
    seg_start: usize,
    next_id: u64,
    fed: usize,
    first_job: bool,
}

impl Pipeline {
    pub fn new(info: &BackendInfo, cfg: &SessionConfig) -> Self {
        Self::with_vad(
            info,
            cfg,
            Box::new(EnergyVad::new(cfg.vad_threshold_db, 20, 2, 10)),
        )
    }

    pub fn with_vad(info: &BackendInfo, cfg: &SessionConfig, vad: Box<dyn Vad>) -> Self {
        let sr = info.sample_rate as usize;
        let ms = |m: f64| (m * sr as f64 / 1000.0) as usize;
        // the open segment (plus preroll) must fit the model window
        let max_seg = cfg
            .max_segment_s
            .min(info.max_window_s - cfg.preroll_ms / 1000.0 - 0.1);
        let frame = ms(f64::from(vad.frame_ms()));
        let ring = ((info.max_window_s.min(cfg.max_segment_s) + 30.0) * sr as f64) as usize;
        Self {
            sr,
            native: info.streaming_native,
            vad,
            preroll: ms(cfg.preroll_ms),
            tail_pad: ms(cfg.tail_pad_ms),
            endpointer: Endpointer::new(cfg.silence_ms, max_seg),
            scheduler: Scheduler::new(info.sample_rate, cfg.hop_ms, cfg.max_load),
            frame,
            buf: SampleRing::new(ring),
            rem: Vec::new(),
            pos: 0,
            prev_end: 0,
            seg_start: 0,
            next_id: 0,
            fed: 0,
            first_job: false,
        }
    }

    /// Seconds of audio processed so far (whole VAD frames).
    pub fn position_s(&self) -> f64 {
        self.pos as f64 / self.sr as f64
    }

    /// Wall-clock seconds the latest decode took; stretches the partial hop when decoding is slow.
    pub fn report_cost(&mut self, cost_s: f64) {
        self.scheduler.report_cost(cost_s);
    }

    /// Add mono samples at the backend's sample rate; returns the jobs they trigger.
    pub fn feed(&mut self, pcm: &[f32]) -> Vec<Job> {
        self.buf.append(pcm);
        let mut x = std::mem::take(&mut self.rem);
        x.extend_from_slice(pcm);
        let mut jobs = Vec::new();
        let n = x.len() / self.frame;
        for i in 0..n {
            let speech = self.vad.update(&x[i * self.frame..(i + 1) * self.frame]);
            self.pos += self.frame;
            let was_active = self.endpointer.active;
            let reason = self
                .endpointer
                .update(speech, self.frame as f64 / self.sr as f64);
            if !was_active && (self.endpointer.active || reason.is_some()) {
                self.open(self.pos - self.frame);
            }
            if let Some(r) = reason {
                jobs.push(self.close(r != CloseReason::MaxLength));
            }
        }
        self.rem = x[n * self.frame..].to_vec();
        if self.endpointer.active && self.scheduler.due(self.pos - self.seg_start) {
            jobs.push(self.partial());
        }
        jobs
    }

    /// End of stream: finalize the open segment, if any.
    pub fn flush(&mut self) -> Vec<Job> {
        if !self.endpointer.active {
            return Vec::new();
        }
        self.endpointer.active = false;
        vec![self.close(true)]
    }

    fn open(&mut self, frame_start: usize) {
        self.seg_start = frame_start
            .saturating_sub(self.preroll)
            .max(self.prev_end)
            .max(self.buf.start());
        self.fed = self.seg_start;
        self.first_job = true;
        self.scheduler.reset();
    }

    fn job(&mut self, kind: JobKind, end: usize) -> Job {
        let (start, id) = (self.seg_start, self.next_id);
        let from = if self.native {
            self.fed.min(end)
        } else {
            start
        };
        let audio = self
            .buf
            .get(from, end)
            .expect("ring capacity covers the longest segment");
        if self.native {
            self.fed = self.fed.max(end);
        }
        let opens = std::mem::take(&mut self.first_job);
        Job {
            segment_id: id,
            kind,
            start,
            end,
            audio,
            opens,
            incremental: self.native,
        }
    }

    fn partial(&mut self) -> Job {
        self.scheduler.mark(self.pos - self.seg_start);
        self.job(JobKind::Partial, self.pos)
    }

    fn close(&mut self, trim_tail: bool) -> Job {
        let mut end = self.pos;
        if trim_tail {
            // keep a little trailing silence
            let trailing = (self.endpointer.trailing_s * self.sr as f64) as usize;
            end = end.saturating_sub(trailing.saturating_sub(self.tail_pad));
        }
        let end = end.max(self.seg_start);
        self.prev_end = end;
        let job = self.job(JobKind::Final, end);
        self.next_id += 1;
        job
    }
}
