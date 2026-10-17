//! Real-time replay and measurement. All times are wall clock (`Instant`) relative to `t0`, the moment the first chunk's
//! window opened; audio sample `n` becomes available at `t0 + (n + 1) / sr` because chunks are fed when they are complete.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use sysinfo::{Pid, ProcessesToUpdate, System};
use tingmix_core::{AsrBackend, Event};
use tingmix_streaming::{SessionConfig, SessionMsg, SessionWorker};

use crate::probe::{Rec, Sim, Timed};
use crate::source::{Source, SR};
use crate::stats::{dist, Dist};

pub struct RunCfg {
    pub chunk_ms: u64,
    pub tail_s: f64,
    pub sim: Sim,
    pub session: SessionConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct UttRow {
    pub n: usize,
    pub clip: String,
    pub lang: String,
    pub onset_s: f64,
    pub end_s: f64,
    pub first_partial_from_onset_s: Option<f64>,
    pub final_from_speech_end_s: Option<f64>,
    pub partials: u32,
    pub finals: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Sample {
    pub t_s: f64,
    pub rss_mb: f64,
    /// Utterances whose endpoint should have fired by now but have no closing final yet.
    pub pending_finals: usize,
    pub oldest_pending_age_s: f64,
    /// While someone is speaking: seconds between now and the audio position of the latest text shown.
    pub partial_staleness_s: Option<f64>,
    /// Age of the inference call running right now (a stuck backend shows up here).
    pub inflight_age_s: Option<f64>,
    pub calls_done: usize,
    pub coalesced: u64,
}

#[derive(Debug, Serialize)]
pub struct RunOut {
    pub wall_s: f64,
    pub audio_s: f64,
    pub utterances: Vec<UttRow>,
    pub samples: Vec<Sample>,
    pub unmatched_events: usize,
    pub events: usize,
    pub errors: Vec<String>,
    pub feed_lateness_s: Dist,
    pub compute_per_call_s: Dist,
    /// Per call: compute seconds / audio seconds handed to that call (buffered models re-decode the whole segment).
    pub per_call_rtf: Dist,
    pub busy_s: f64,
    pub calls: usize,
    pub coalesced: u64,
    pub first_partial_from_onset_s: Dist,
    pub final_from_speech_end_s: Dist,
    pub partial_staleness_s: Dist,
    pub streaming_rtf: f64,
}

/// Maps session events onto the utterances of the synthetic timeline.
#[derive(Default)]
struct Track {
    events: usize,
    seg_utt: HashMap<u64, usize>,
    unmatched: usize,
    closed: Vec<Option<f64>>, // per utterance: time of its closing final
    first_partial: Vec<Option<f64>>,
    counts: Vec<(u32, u32)>,
    last_end: Vec<f64>,
}

impl Track {
    fn handle(&mut self, m: SessionMsg, t: f64, src: &Source, errors: &mut Vec<String>) {
        let e = match m {
            SessionMsg::Event(e) => e,
            SessionMsg::Error {
                segment_id,
                message,
            } => return errors.push(format!("t={t:.2}s segment {segment_id}: {message}")),
        };
        self.events += 1;
        let (partial, seg, start, end) = match &e {
            Event::Partial(p) => (true, p.segment_id, p.start, p.end),
            Event::Final(f) => (false, f.segment_id, f.start, f.end),
        };
        // A segment belongs to the latest utterance its time span touches (gaps are longer than the preroll).
        let u = *self.seg_utt.entry(seg).or_insert_with(|| {
            src.utts
                .iter()
                .rposition(|u| start - 0.1 <= u.end_s && end + 0.1 >= u.onset_s)
                .unwrap_or(usize::MAX)
        });
        if u == usize::MAX {
            self.unmatched += 1;
            return;
        }
        if self.closed.len() <= u {
            self.closed.resize(u + 1, None);
            self.first_partial.resize(u + 1, None);
            self.counts.resize(u + 1, (0, 0));
            self.last_end.resize(u + 1, 0.0);
        }
        self.last_end[u] = self.last_end[u].max(end);
        if partial {
            self.counts[u].0 += 1;
            self.first_partial[u].get_or_insert(t);
        } else {
            self.counts[u].1 += 1;
            if end >= src.utts[u].end_s - 0.5 {
                self.closed[u] = Some(t);
            }
        }
    }
}

fn rss_mb(sys: &mut System, pid: Pid) -> f64 {
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    sys.process(pid)
        .map_or(0.0, |p| p.memory() as f64 / 1_048_576.0)
}

pub fn run<B: AsrBackend + 'static>(
    backend: B,
    rec: Arc<Rec>,
    t0: Instant,
    mut src: Source,
    cfg: &RunCfg,
) -> RunOut {
    let timed = Timed {
        inner: backend,
        rec: rec.clone(),
        sim: cfg.sim,
    };
    let mut worker = SessionWorker::spawn(timed, &cfg.session);
    let chunk = SR * cfg.chunk_ms as usize / 1000;
    let (mut sys, pid) = (System::new(), Pid::from_u32(std::process::id()));
    let (mut samples, mut lateness): (Vec<Sample>, Vec<f64>) = Default::default();
    let (mut next_sample, mut tail_start) = (0.0f64, None::<f64>);
    let mut tr = Track::default();
    let mut errors: Vec<String> = Vec::new();

    let mut i: u64 = 0;
    loop {
        let deadline = t0 + Duration::from_millis(cfg.chunk_ms * (i + 1));
        // Waiting for the next chunk doubles as the event pump, so event timestamps are not quantised to the chunk size.
        loop {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            if let Ok(m) = worker.events().recv_timeout(deadline - now) {
                tr.handle(m, t0.elapsed().as_secs_f64(), &src, &mut errors);
            }
        }
        lateness.push(Instant::now().duration_since(deadline).as_secs_f64());
        let pcm = src.next(chunk);
        worker.feed(&pcm);
        i += 1;
        let t = t0.elapsed().as_secs_f64();
        if src.finished() && tail_start.is_none() {
            tail_start = Some(t);
        }
        if t >= next_sample {
            next_sample = t + 1.0;
            let due = |u: &crate::source::Utt| u.end_s + cfg.session.silence_ms / 1000.0 < t;
            let pend: Vec<f64> = src
                .utts
                .iter()
                .enumerate()
                .filter(|(n, u)| due(u) && tr.closed.get(*n).copied().flatten().is_none())
                .map(|(_, u)| t - (u.end_s + cfg.session.silence_ms / 1000.0))
                .collect();
            let speaking = src
                .utts
                .iter()
                .rposition(|u| u.onset_s <= t && t <= u.end_s);
            let calls = rec.calls.lock().unwrap().len();
            samples.push(Sample {
                t_s: t,
                rss_mb: rss_mb(&mut sys, pid),
                pending_finals: pend.len(),
                oldest_pending_age_s: pend.iter().cloned().fold(0.0, f64::max),
                partial_staleness_s: speaking.map(|n| {
                    t - src.utts[n]
                        .onset_s
                        .max(tr.last_end.get(n).copied().unwrap_or(0.0))
                }),
                inflight_age_s: rec.inflight_age(),
                calls_done: calls,
                coalesced: worker.coalesced_partials(),
            });
        }
        if tail_start.is_some_and(|s| t >= s + cfg.tail_s) {
            break;
        }
    }
    let coalesced = worker.coalesced_partials();
    let audio_s = src.generated_s();
    for m in worker.finish() {
        tr.handle(m, t0.elapsed().as_secs_f64(), &src, &mut errors);
    }
    let wall_s = t0.elapsed().as_secs_f64();

    let rows: Vec<UttRow> = src
        .utts
        .iter()
        .enumerate()
        .map(|(n, u)| UttRow {
            n,
            clip: src.clip(u.clip).id.clone(),
            lang: src.clip(u.clip).lang.clone(),
            onset_s: u.onset_s,
            end_s: u.end_s,
            first_partial_from_onset_s: tr
                .first_partial
                .get(n)
                .copied()
                .flatten()
                .map(|t| t - u.onset_s),
            final_from_speech_end_s: tr.closed.get(n).copied().flatten().map(|t| t - u.end_s),
            partials: tr.counts.get(n).map_or(0, |c| c.0),
            finals: tr.counts.get(n).map_or(0, |c| c.1),
        })
        .collect();
    let calls = rec.calls.lock().unwrap().clone();
    let busy_s: f64 = calls.iter().map(|c| c.dur_s).sum();
    let col = |f: fn(&UttRow) -> Option<f64>| dist(&rows.iter().filter_map(f).collect::<Vec<_>>());
    RunOut {
        wall_s,
        audio_s,
        unmatched_events: tr.unmatched,
        events: tr.events,
        errors,
        feed_lateness_s: dist(&lateness),
        compute_per_call_s: dist(&calls.iter().map(|c| c.dur_s).collect::<Vec<_>>()),
        per_call_rtf: dist(
            &calls
                .iter()
                .filter(|c| c.audio_s > 0.0)
                .map(|c| c.dur_s / c.audio_s)
                .collect::<Vec<_>>(),
        ),
        busy_s,
        calls: calls.len(),
        coalesced,
        first_partial_from_onset_s: col(|r| r.first_partial_from_onset_s),
        final_from_speech_end_s: col(|r| r.final_from_speech_end_s),
        partial_staleness_s: dist(
            &samples
                .iter()
                .filter_map(|s| s.partial_staleness_s)
                .collect::<Vec<_>>(),
        ),
        streaming_rtf: busy_s / audio_s,
        utterances: rows,
        samples,
    }
}
