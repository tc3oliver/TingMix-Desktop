//! The Rust session must reproduce the Python reference event sequence exactly
//! (fixtures: `tools/golden-session/generate.py`).

use std::path::PathBuf;

use serde::Deserialize;
use tingmix_core::Event;
use tingmix_streaming::{MockBackend, SessionConfig, SessionMsg, StreamingSession};

#[derive(Deserialize)]
struct Input {
    wav: Option<String>,
    silence_s: Option<f64>,
}

#[derive(Deserialize)]
struct Golden {
    native: bool,
    chunk: usize,
    config: serde_json::Value,
    inputs: Vec<Input>,
    events: Vec<Event>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn load_wav(name: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(root().join("audio").join(name)).unwrap();
    assert_eq!(
        (
            r.spec().sample_rate,
            r.spec().channels,
            r.spec().bits_per_sample
        ),
        (16_000, 1, 16)
    );
    r.samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect()
}

fn config(v: &serde_json::Value) -> SessionConfig {
    let mut c = SessionConfig::default();
    if let Some(x) = v.get("max_segment_s") {
        c.max_segment_s = x.as_f64().unwrap();
    }
    if let Some(x) = v.get("silence_ms") {
        c.silence_ms = x.as_f64().unwrap();
    }
    if let Some(x) = v.get("tail_pad_ms") {
        c.tail_pad_ms = x.as_f64().unwrap();
    }
    c
}

fn run(name: &str) {
    let g: Golden = serde_json::from_slice(
        &std::fs::read(root().join("session").join(format!("{name}.json"))).unwrap(),
    )
    .unwrap();
    let audio: Vec<f32> = g
        .inputs
        .iter()
        .flat_map(|i| match (&i.wav, i.silence_s) {
            (Some(w), _) => load_wav(w),
            (_, Some(s)) => vec![0.0; (s * 16_000.0) as usize],
            _ => panic!("bad input"),
        })
        .collect();
    let mut session = StreamingSession::new(MockBackend::new(g.native), &config(&g.config));
    let mut msgs = Vec::new();
    for c in audio.chunks(g.chunk) {
        msgs.extend(session.feed(c));
    }
    msgs.extend(session.flush());
    let events: Vec<Event> = msgs
        .into_iter()
        .map(|m| match m {
            SessionMsg::Event(e) => e,
            SessionMsg::Error { message, .. } => panic!("backend error: {message}"),
        })
        .collect();
    assert!(!g.events.is_empty());
    assert_eq!(events.len(), g.events.len(), "event count");
    for (i, (got, want)) in events.iter().zip(&g.events).enumerate() {
        assert_eq!(got, want, "event {i}");
    }
}

#[test]
fn buffered_two_segments() {
    run("buffered_two_segments");
}

#[test]
fn native_two_segments() {
    run("native_two_segments");
}

#[test]
fn buffered_max_length_cut() {
    run("buffered_max_length_cut");
}

#[test]
fn buffered_flush_open_segment() {
    run("buffered_flush_open_segment");
}

#[test]
fn native_flush_open_segment() {
    run("native_flush_open_segment");
}
