//! Model-free VAD / endpointing evaluation: run the real `Pipeline` over the synthetic stream as fast as possible and
//! compare the segments it closes with the known speech spans. Needs no backend, so wide noise matrices take seconds.

use serde::Serialize;
use tingmix_core::BackendInfo;
use tingmix_streaming::{JobKind, Pipeline, SessionConfig};

use crate::source::{Source, SR};

#[derive(Serialize)]
pub struct VadOut {
    pub audio_s: f64,
    pub utterances: usize,
    pub segments: usize,
    /// Segments that overlap no utterance speech span.
    pub false_segments: usize,
    pub false_segments_per_min: f64,
    /// Seconds of segment time per minute of audio that lies outside every speech span (preroll excluded).
    pub active_non_speech_s_per_min: f64,
    /// Utterances with less than half of their speech inside segments.
    pub missed_utterances: usize,
    /// Mean fraction of speech frames inside segments.
    pub speech_coverage: f64,
    /// Seconds from acoustic onset until the segment's first job (the VAD decision delay), per utterance.
    pub detection_delay_s_mean: f64,
    pub detection_delay_s_max: f64,
    /// Segments that span more than one utterance (silence never reached the endpoint).
    pub merged_segments: usize,
    pub longest_segment_s: f64,
    /// First segments as (start_s, end_s) for inspection.
    pub first_segments: Vec<(f64, f64)>,
}

pub fn evaluate(mut src: Source, cfg: &SessionConfig, chunk_ms: usize) -> VadOut {
    let info = BackendInfo {
        name: "vad-only".into(),
        sample_rate: 16_000,
        max_window_s: 29.5,
        streaming_native: false,
    };
    let mut p = Pipeline::new(&info, cfg);
    let chunk = SR * chunk_ms / 1000;
    let (mut segs, mut first_job): (Vec<(f64, f64)>, Vec<f64>) = Default::default();
    while !src.finished() {
        let pcm = src.next(chunk);
        for j in p.feed(&pcm) {
            if j.opens {
                first_job.push(j.end as f64 / SR as f64);
            }
            if j.kind == JobKind::Final {
                segs.push((j.start as f64 / SR as f64, j.end as f64 / SR as f64));
            }
        }
    }
    for j in p.flush() {
        segs.push((j.start as f64 / SR as f64, j.end as f64 / SR as f64));
    }
    let audio_s = src.generated_s();
    let overlap = |a: (f64, f64), b: (f64, f64)| (a.1.min(b.1) - a.0.max(b.0)).max(0.0);
    let speech: Vec<(f64, f64)> = src.utts.iter().map(|u| (u.onset_s, u.end_s)).collect();
    let false_segments = segs
        .iter()
        .filter(|s| !speech.iter().any(|u| overlap(**s, *u) > 0.0))
        .count();
    let non_speech: f64 = segs
        .iter()
        .map(|s| (s.1 - s.0) - speech.iter().map(|u| overlap(*s, *u)).sum::<f64>())
        .sum();
    let cover: Vec<f64> = speech
        .iter()
        .map(|u| segs.iter().map(|s| overlap(*s, *u)).sum::<f64>() / (u.1 - u.0).max(1e-9))
        .collect();
    // detection delay: first job opening a segment that touches the utterance
    let delays: Vec<f64> = speech
        .iter()
        .filter_map(|u| {
            first_job
                .iter()
                .find(|t| **t >= u.0 - 0.3 && **t <= u.1)
                .map(|t| (t - u.0).max(0.0))
        })
        .collect();
    VadOut {
        audio_s,
        utterances: speech.len(),
        segments: segs.len(),
        false_segments,
        false_segments_per_min: false_segments as f64 / (audio_s / 60.0),
        active_non_speech_s_per_min: non_speech.max(0.0) / (audio_s / 60.0),
        missed_utterances: cover.iter().filter(|c| **c < 0.5).count(),
        speech_coverage: cover.iter().sum::<f64>() / cover.len().max(1) as f64,
        detection_delay_s_mean: delays.iter().sum::<f64>() / delays.len().max(1) as f64,
        detection_delay_s_max: delays.iter().cloned().fold(0.0, f64::max),
        merged_segments: segs
            .iter()
            .filter(|s| speech.iter().filter(|u| overlap(**s, **u) > 0.0).count() > 1)
            .count(),
        first_segments: segs.iter().take(12).cloned().collect(),
        longest_segment_s: segs.iter().map(|s| s.1 - s.0).fold(0.0, f64::max),
    }
}

#[derive(Serialize)]
pub struct WavOut {
    pub audio_s: f64,
    pub segments: usize,
    /// Segments that ended because they reached the length cap rather than at a pause.
    pub ended_at_cap: usize,
    pub median_segment_s: f64,
    pub longest_segment_s: f64,
    pub segment_lengths_s: Vec<f64>,
}

/// Segment a recording with the real pipeline (no ground truth): how many segments end at a pause vs at the cap.
pub fn evaluate_wav(wav: &[f32], cfg: &SessionConfig) -> WavOut {
    let info = BackendInfo {
        name: "vad-only".into(),
        sample_rate: 16_000,
        max_window_s: 29.5,
        streaming_native: false,
    };
    let mut p = Pipeline::new(&info, cfg);
    let mut lens = Vec::new();
    let mut take = |jobs: Vec<tingmix_streaming::Job>| {
        for j in jobs.into_iter().filter(|j| j.kind == JobKind::Final) {
            lens.push((j.end - j.start) as f64 / SR as f64);
        }
    };
    for c in wav.chunks(SR / 50) {
        take(p.feed(c));
    }
    take(p.flush());
    let cap = cfg
        .max_segment_s
        .min(29.5 - (cfg.preroll_ms / 1000.0).max(cfg.bridge_s) - 0.1);
    let mut sorted = lens.clone();
    sorted.sort_by(f64::total_cmp);
    WavOut {
        audio_s: wav.len() as f64 / SR as f64,
        segments: lens.len(),
        ended_at_cap: lens.iter().filter(|l| **l >= cap - 0.1).count(),
        median_segment_s: sorted.get(sorted.len() / 2).copied().unwrap_or(0.0),
        longest_segment_s: sorted.last().copied().unwrap_or(0.0),
        segment_lengths_s: lens,
    }
}
