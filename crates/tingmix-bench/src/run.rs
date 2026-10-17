//! Real-time replay and measurement. All times are wall clock (`Instant`) relative to `t0`, the moment the first chunk's
//! window opened; audio sample `n` becomes available at `t0 + (n + 1) / sr` because chunks are fed when they are complete.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use sysinfo::{Pid, ProcessesToUpdate, System};
use tingmix_core::{AsrBackend, Event};
use tingmix_streaming::{SessionConfig, SessionMsg, SessionWorker};

use crate::probe::{Rec, Sim, Timed};
use crate::source::{Source, SR};
use crate::stats::{dist, edit_distance, norm, Dist};

pub struct RunCfg {
    /// Utterances at the start excluded from the latency distributions (cold-load warm-up). They stay in the table.
    pub discard_first: usize,
    pub chunk_ms: u64,
    /// `None`: `SessionWorker::spawn` (live policy, tail partial on). `Some(b)`: `spawn_with` and exactly this setting.
    pub tail_partial: Option<bool>,
    pub tail_s: f64,
    /// Replay faster than real time (text and segmentation metrics only; latencies are then not valid).
    pub speed: f64,
    /// Expected text per clip id (clean-audio oracle) for CER.
    pub oracle: Option<HashMap<String, String>>,
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
    /// Concatenated final texts of this utterance.
    pub text: String,
    /// Confidence of the closing final, when the backend reports one.
    pub confidence: Option<f32>,
    /// Character error rate against the oracle text (1.0 when nothing was recognised).
    pub cer: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FalseEvent {
    pub t_s: f64,
    pub kind: &'static str,
    pub text: String,
    pub confidence: Option<f32>,
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
    pub speed: f64,
    /// Distinct segments that overlap no utterance (noise triggered the VAD and the model wrote something).
    pub false_segments: usize,
    pub false_segments_per_min: f64,
    pub false_events: Vec<FalseEvent>,
    /// Utterances that got no final at all.
    pub missed_utterances: usize,
    /// Edits / reference characters over all utterances (missed ones count as fully deleted); `None` without an oracle.
    pub cer_micro: Option<f64>,
    /// CER of all final texts in time order (false finals included) against all oracle texts in order: robust to merged or split segments.
    pub stream_cer: Option<f64>,
    pub exact_match_rate: Option<f64>,
}

/// Maps session events onto the utterances of the synthetic timeline.
#[derive(Default)]
struct Track {
    events: usize,
    unmatched: usize,
    closed: Vec<Option<f64>>, // per utterance: time of its closing final
    first_partial: Vec<Option<f64>>,
    counts: Vec<(u32, u32)>,
    last_end: Vec<f64>,
    texts: Vec<Vec<(f64, String)>>,
    conf: Vec<Option<f32>>,
    false_events: Vec<FalseEvent>,
    false_segs: HashSet<u64>,
}

impl Track {
    fn handle(
        &mut self,
        m: SessionMsg,
        wall: f64,
        speed: f64,
        src: &Source,
        rec: &Rec,
        errors: &mut Vec<String>,
    ) {
        let t = wall * speed; // stream-time equivalent; equals wall time at speed 1
        let e = match m {
            SessionMsg::Event(e) => e,
            SessionMsg::Error {
                segment_id,
                message,
            } => return errors.push(format!("t={t:.2}s segment {segment_id}: {message}")),
        };
        self.events += 1;
        let (partial, seg, start, end, text) = match &e {
            Event::Partial(p) => (true, p.segment_id, p.start, p.end, p.text.as_str()),
            Event::Final(f) => (false, f.segment_id, f.start, f.end, f.text.as_str()),
        };
        // An event belongs to the latest utterance whose speech span its audio range touches. A stuck-open segment
        // (noise keeps the VAD active) therefore maps its early events to nothing: those are the false ones.
        let u = src
            .utts
            .iter()
            .rposition(|u| start - 0.1 <= u.end_s && end + 0.1 >= u.onset_s)
            .unwrap_or(usize::MAX);
        if u == usize::MAX {
            self.unmatched += 1;
            self.false_segs.insert(seg);
            self.false_events.push(FalseEvent {
                t_s: t,
                kind: if partial { "partial" } else { "final" },
                text: text.into(),
                confidence: rec.confidence_of(text),
            });
            return;
        }
        if self.closed.len() <= u {
            self.closed.resize(u + 1, None);
            self.first_partial.resize(u + 1, None);
            self.counts.resize(u + 1, (0, 0));
            self.last_end.resize(u + 1, 0.0);
            self.texts.resize(u + 1, Vec::new());
            self.conf.resize(u + 1, None);
        }
        self.last_end[u] = self.last_end[u].max(end);
        if partial {
            self.counts[u].0 += 1;
            self.first_partial[u].get_or_insert(t);
        } else {
            self.counts[u].1 += 1;
            self.texts[u].push((end, text.to_string()));
            if end >= src.utts[u].end_s - 0.5 {
                self.closed[u] = Some(t);
                self.conf[u] = rec.confidence_of(text);
            }
        }
    }
}

fn join_texts(t: Option<&Vec<(f64, String)>>) -> String {
    let parts: Vec<String> = t
        .map(|t| t.iter().map(|x| x.1.clone()).collect())
        .unwrap_or_default();
    if parts.concat().contains(' ') {
        parts.join(" ")
    } else {
        parts.concat()
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
    let mut worker = match cfg.tail_partial {
        None => SessionWorker::spawn(timed, &cfg.session),
        Some(t) => SessionWorker::spawn_with(
            timed,
            &SessionConfig {
                tail_partial: t,
                ..cfg.session.clone()
            },
        ),
    };
    let chunk = SR * cfg.chunk_ms as usize / 1000;
    let (mut sys, pid) = (System::new(), Pid::from_u32(std::process::id()));
    let (mut samples, mut lateness): (Vec<Sample>, Vec<f64>) = Default::default();
    let (mut next_sample, mut tail_start) = (0.0f64, None::<f64>);
    let mut tr = Track::default();
    let mut errors: Vec<String> = Vec::new();

    let mut i: u64 = 0;
    loop {
        let deadline =
            t0 + Duration::from_secs_f64((cfg.chunk_ms * (i + 1)) as f64 / 1000.0 / cfg.speed);
        // Waiting for the next chunk doubles as the event pump, so event timestamps are not quantised to the chunk size.
        loop {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            if let Ok(m) = worker.events().recv_timeout(deadline - now) {
                tr.handle(
                    m,
                    t0.elapsed().as_secs_f64(),
                    cfg.speed,
                    &src,
                    &rec,
                    &mut errors,
                );
            }
        }
        lateness.push(Instant::now().duration_since(deadline).as_secs_f64());
        let pcm = src.next(chunk);
        worker.feed(&pcm);
        i += 1;
        let t = t0.elapsed().as_secs_f64() * cfg.speed;
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
        tr.handle(
            m,
            t0.elapsed().as_secs_f64(),
            cfg.speed,
            &src,
            &rec,
            &mut errors,
        );
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
            text: join_texts(tr.texts.get(n)),
            confidence: tr.conf.get(n).copied().flatten(),
            cer: None,
        })
        .collect();
    let mut rows = rows;
    let (mut edits, mut ref_chars, mut exact, mut scored) = (0usize, 0usize, 0usize, 0usize);
    if let Some(oracle) = &cfg.oracle {
        for r in &mut rows {
            let Some(want) = oracle.get(&r.clip) else {
                continue;
            };
            let (a, b) = (norm(&r.text), norm(want));
            let d = edit_distance(&a, &b);
            r.cer = Some(d as f64 / b.len().max(1) as f64);
            (edits, ref_chars, scored, exact) = (
                edits + d,
                ref_chars + b.len(),
                scored + 1,
                exact + usize::from(d == 0),
            );
        }
    }
    let stream_cer = cfg.oracle.as_ref().map(|o| {
        let mut finals: Vec<(f64, &str)> = tr
            .texts
            .iter()
            .flatten()
            .map(|(e, t)| (*e, t.as_str()))
            .collect();
        finals.extend(
            tr.false_events
                .iter()
                .filter(|f| f.kind == "final")
                .map(|f| (f.t_s, f.text.as_str())),
        );
        finals.sort_by(|a, b| a.0.total_cmp(&b.0));
        let hyp: Vec<char> = finals.iter().flat_map(|f| norm(f.1)).collect();
        let want: Vec<char> = src
            .utts
            .iter()
            .filter_map(|u| o.get(&src.clip(u.clip).id))
            .flat_map(|t| norm(t))
            .collect();
        edit_distance(&hyp, &want) as f64 / want.len().max(1) as f64
    });
    let missed_utterances = rows
        .iter()
        .filter(|r| r.finals == 0 && r.end_s + 5.0 < audio_s)
        .count();
    let calls = rec.calls.lock().unwrap().clone();
    let busy_s: f64 = calls.iter().map(|c| c.dur_s).sum();
    let col = |f: fn(&UttRow) -> Option<f64>| {
        dist(
            &rows
                .iter()
                .skip(cfg.discard_first)
                .filter_map(f)
                .collect::<Vec<_>>(),
        )
    };
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
        speed: cfg.speed,
        false_segments: tr.false_segs.len(),
        false_segments_per_min: tr.false_segs.len() as f64 / (audio_s / 60.0),
        false_events: tr.false_events,
        missed_utterances,
        stream_cer,
        cer_micro: (scored > 0).then(|| edits as f64 / ref_chars.max(1) as f64),
        exact_match_rate: (scored > 0).then(|| exact as f64 / scored as f64),
        utterances: rows,
        samples,
    }
}
