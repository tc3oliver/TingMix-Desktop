use serde::{Deserialize, Serialize};

/// A revisable hypothesis for an open segment. Serializes to the TingMix protocol JSON
/// (`{"type":"partial","segment_id":..,"revision":..,"text":..,"start":..,"end":..,"stable_prefix_len":..}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartialEvent {
    pub segment_id: u64,
    /// 0, 1, ... per segment; strictly increasing.
    pub revision: u32,
    pub text: String,
    /// Seconds from the start of the stream.
    pub start: f64,
    pub end: f64,
    /// `text[..stable_prefix_len]` (in chars) agreed across the last partials. Absent in old payloads.
    #[serde(default)]
    pub stable_prefix_len: usize,
}

/// The immutable result of a segment. Its revision is the last partial's revision + 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinalEvent {
    pub segment_id: u64,
    pub revision: u32,
    pub text: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Event {
    Partial(PartialEvent),
    Final(FinalEvent),
}

impl Event {
    pub fn segment_id(&self) -> u64 {
        match self {
            Event::Partial(e) => e.segment_id,
            Event::Final(e) => e.segment_id,
        }
    }

    pub fn revision(&self) -> u32 {
        match self {
            Event::Partial(e) => e.revision,
            Event::Final(e) => e.revision,
        }
    }
}
