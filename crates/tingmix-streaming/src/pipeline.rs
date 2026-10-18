use tingmix_core::BackendInfo;

use crate::{
    AdaptiveVad, AdaptiveVadParams, CloseReason, Endpointer, EnergyVad, SampleRing, Scheduler,
    Suppression, Vad, VadKind,
};

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
    /// Voice activity detector. `Energy` (default) is the Python reference's fixed threshold; `Adaptive` tracks the noise floor.
    pub vad: VadKind,
    /// Tunables used when `vad` is `Adaptive`.
    pub adaptive_vad: AdaptiveVadParams,
    /// Hide low-confidence text (default off, see [`Suppression`]).
    pub suppression: Option<Suppression>,
    /// Live-latency policy for models that re-decode the open segment (off = Python-identical):
    /// when the VAD reports speech has ended, decode the whole segment once right away (the "tail
    /// partial") and issue no further partials; at the endpoint the final reuses that result
    /// instead of decoding again, so a final costs at most the remainder of one decode. The final
    /// then covers up to one VAD frame more silence than a fresh decode would.
    pub tail_partial: bool,
    /// When a segment hits `max_segment_s`, cut at the quietest point of its last this-many seconds instead of at the cap,
    /// and carry the rest into the next segment (0 = hard cut, Python-identical). Buffered models only.
    pub max_cut_search_s: f64,
    /// A segment that opens within this many seconds of the previous one's end starts at that end instead of `preroll_ms`
    /// earlier, so speech too quiet to open the VAD between two segments is still decoded (0 = off, Python-identical).
    pub bridge_s: f64,
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
            vad: VadKind::Energy,
            adaptive_vad: AdaptiveVadParams::default(),
            suppression: None,
            tail_partial: false,
            max_cut_search_s: 0.0,
            bridge_s: 0.0,
        }
    }
}

impl SessionConfig {
    /// The live-capture setup measured in docs/research/vad-noise.md: adaptive VAD, confidence suppression and a
    /// quiet-point cut at the segment cap. `Default` stays Python-identical (file mode, goldens).
    pub fn live_recommended() -> Self {
        Self {
            vad: VadKind::Adaptive,
            suppression: Some(Suppression::RECOMMENDED),
            max_cut_search_s: 3.0,
            bridge_s: 4.0,
            ..Self::default()
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
    /// Partial: the tail partial (see [`SessionConfig::tail_partial`]).
    pub tail: bool,
    /// Final: the tail partial covers this range, so its result may stand in for a fresh decode.
    pub reuse_tail: bool,
}

/// VAD, endpointing and decode scheduling over an absolute sample timeline. Feed it any chunking
/// of 16 kHz-ish mono PCM; it emits [`Job`]s. Samples are never dropped: a closed segment's job
/// carries a copy of every sample it needs.
pub struct Pipeline {
    sr: usize,
    native: bool,
    vad: Box<dyn Vad>,
    preroll: usize,
    bridge: usize,
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
    tail_partial: bool,
    cut_search: usize,                       // frames
    levels: std::collections::VecDeque<f32>, // dB of the latest VAD frames, for the quiet-point search
    tail_end: Option<usize>,                 // end of the tail partial issued for the open segment
}

impl Pipeline {
    pub fn new(info: &BackendInfo, cfg: &SessionConfig) -> Self {
        let vad: Box<dyn Vad> = match cfg.vad {
            VadKind::Energy => Box::new(EnergyVad::new(cfg.vad_threshold_db, 20, 2, 10)),
            VadKind::Adaptive => Box::new(AdaptiveVad::with_params(cfg.adaptive_vad)),
        };
        Self::with_vad(info, cfg, vad)
    }

    pub fn with_vad(info: &BackendInfo, cfg: &SessionConfig, vad: Box<dyn Vad>) -> Self {
        let sr = info.sample_rate as usize;
        let ms = |m: f64| (m * sr as f64 / 1000.0) as usize;
        // the open segment (plus preroll or bridge) must fit the model window
        let lead_s = (cfg.preroll_ms / 1000.0).max(cfg.bridge_s);
        let max_seg = cfg.max_segment_s.min(info.max_window_s - lead_s - 0.1);
        let frame_ms = f64::from(vad.frame_ms());
        let frame = ms(frame_ms);
        let ring = ((info.max_window_s.min(cfg.max_segment_s) + 30.0) * sr as f64) as usize;
        Self {
            sr,
            native: info.streaming_native,
            vad,
            preroll: ms(cfg.preroll_ms),
            bridge: ms(cfg.bridge_s * 1000.0),
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
            tail_partial: cfg.tail_partial && !info.streaming_native,
            cut_search: if info.streaming_native {
                0
            } else {
                (cfg.max_cut_search_s * 1000.0 / frame_ms) as usize
            },
            levels: Default::default(),
            tail_end: None,
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
            let fr = &x[i * self.frame..(i + 1) * self.frame];
            let speech = self.vad.update(fr);
            if self.cut_search > 0 {
                let ms =
                    fr.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / fr.len() as f64;
                self.levels.push_back((10.0 * (ms + 1e-20).log10()) as f32);
                if self.levels.len() > self.cut_search + 2 {
                    self.levels.pop_front();
                }
            }
            self.pos += self.frame;
            let was_active = self.endpointer.active;
            let reason = self
                .endpointer
                .update(speech, self.frame as f64 / self.sr as f64);
            if !was_active && (self.endpointer.active || reason.is_some()) {
                self.open(self.pos - self.frame);
            }
            if self.tail_partial && reason.is_none() && self.endpointer.active {
                if self.endpointer.trailing_s == 0.0 {
                    self.tail_end = None; // speech resumed
                } else if self.tail_end.is_none() {
                    self.tail_end = Some(self.pos);
                    let mut job = self.partial();
                    job.tail = true;
                    jobs.push(job);
                }
            }
            if let Some(r) = reason {
                match (r, self.quiet_cut()) {
                    (CloseReason::MaxLength, Some(cut)) => jobs.push(self.cut_at(cut)),
                    _ => jobs.push(self.close(r != CloseReason::MaxLength)),
                }
            }
        }
        self.rem = x[n * self.frame..].to_vec();
        if self.endpointer.active
            && self.tail_end.is_none()
            && self.scheduler.due(self.pos - self.seg_start)
        {
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
        let bridged = self.next_id > 0 && frame_start <= self.prev_end + self.bridge;
        let lead = if bridged {
            self.prev_end
        } else {
            frame_start.saturating_sub(self.preroll)
        };
        self.seg_start = lead.max(self.prev_end).max(self.buf.start());
        self.fed = self.seg_start;
        self.first_job = true;
        self.tail_end = None;
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
            tail: false,
            reuse_tail: false,
        }
    }

    fn partial(&mut self) -> Job {
        self.scheduler.mark(self.pos - self.seg_start);
        self.job(JobKind::Partial, self.pos)
    }

    /// Sample position of the quietest 3-frame stretch in the last `cut_search` frames, if it leaves a usable segment.
    fn quiet_cut(&self) -> Option<usize> {
        let n = self.levels.len();
        if self.cut_search == 0 || n < 3 {
            return None;
        }
        let lv: Vec<f32> = self.levels.iter().copied().collect();
        let (mut best, mut at) = (f32::INFINITY, 0);
        for j in 1..n - 1 {
            let m = (lv[j - 1] + lv[j] + lv[j + 1]) / 3.0;
            if m < best {
                (best, at) = (m, j);
            }
        }
        // frame j ends at pos - (n - 1 - j) frames; cut in its middle
        let cut = self.pos - (n - 1 - at) * self.frame - self.frame / 2;
        (cut > self.seg_start + self.sr).then_some(cut)
    }

    /// Close the open segment at `cut` and continue straight into the next one that starts there (speech goes on).
    fn cut_at(&mut self, cut: usize) -> Job {
        self.prev_end = cut;
        let job = self.job(JobKind::Final, cut);
        self.next_id += 1;
        self.seg_start = cut;
        self.fed = cut;
        self.first_job = true;
        self.tail_end = None;
        self.scheduler.reset();
        self.endpointer
            .resume((self.pos - cut) as f64 / self.sr as f64);
        job
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
        let mut job = self.job(JobKind::Final, end);
        job.reuse_tail = self
            .tail_end
            .is_some_and(|e| end <= e + self.tail_pad && e - end.min(e) <= self.frame);
        self.tail_end = None;
        self.next_id += 1;
        job
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(secs: f64, amp: f32) -> Vec<f32> {
        (0..(secs * 16_000.0) as usize)
            .map(|i| amp * (i as f32 * 0.3).sin())
            .collect()
    }

    #[test]
    fn a_short_gap_between_segments_is_bridged() {
        let info = BackendInfo {
            name: "t".into(),
            sample_rate: 16_000,
            max_window_s: 29.5,
            streaming_native: false,
        };
        let run = |bridge_s: f64, gap_s: f64| {
            let cfg = SessionConfig {
                bridge_s,
                ..SessionConfig::default()
            };
            let mut p = Pipeline::new(&info, &cfg);
            // speech, a pause long enough to end the segment, speech again
            let mut pcm = tone(3.0, 0.2);
            pcm.extend(tone(gap_s, 0.0));
            pcm.extend(tone(3.0, 0.2));
            let mut jobs = Vec::new();
            for c in pcm.chunks(320) {
                jobs.extend(p.feed(c));
            }
            jobs.extend(p.flush());
            let f: Vec<_> = jobs
                .into_iter()
                .filter(|j| j.kind == JobKind::Final)
                .collect();
            assert_eq!(f.len(), 2);
            (f[0].end, f[1].start)
        };
        let (end, start) = run(0.0, 3.0);
        assert!(
            start > end + 16_000,
            "without a bridge the pause is skipped"
        );
        let (end, start) = run(4.0, 3.0);
        assert_eq!(
            start, end,
            "the second segment starts where the first ended"
        );
        let (end, start) = run(4.0, 6.0);
        assert!(start > end + 16_000, "a long pause is not bridged");
    }

    #[test]
    fn max_length_cut_moves_to_the_quiet_point_and_loses_no_audio() {
        let info = BackendInfo {
            name: "t".into(),
            sample_rate: 16_000,
            max_window_s: 29.5,
            streaming_native: false,
        };
        let run = |search: f64| {
            let cfg = SessionConfig {
                max_cut_search_s: search,
                ..SessionConfig::default()
            };
            let mut p = Pipeline::new(&info, &cfg);
            // 22 s of speech-level tone, a 0.3 s dip (shorter than the endpoint silence), then 8 s more
            let mut pcm = tone(22.0, 0.2);
            pcm.extend(tone(0.3, 0.0004));
            pcm.extend(tone(8.0, 0.2));
            let mut jobs = Vec::new();
            for c in pcm.chunks(320) {
                jobs.extend(p.feed(c));
            }
            jobs.extend(p.flush());
            jobs.into_iter()
                .filter(|j| j.kind == JobKind::Final)
                .collect::<Vec<_>>()
        };
        let hard = run(0.0);
        assert!(
            hard[0].end as f64 / 16_000.0 > 25.0,
            "hard cut at the cap: {}",
            hard[0].end
        );
        let soft = run(3.0);
        let cut = soft[0].end as f64 / 16_000.0;
        assert!(
            (22.0..22.4).contains(&cut),
            "cut {cut} should sit in the dip"
        );
        assert_eq!(
            soft[1].start, soft[0].end,
            "the next segment starts exactly where the first ended"
        );
        assert!(
            soft.last().unwrap().end as f64 / 16_000.0 > 29.5,
            "the rest of the audio is not dropped"
        );
        assert!(soft.iter().all(|j| j.audio.len() == j.end - j.start));
    }
}
