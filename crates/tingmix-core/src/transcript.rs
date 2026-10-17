use std::collections::{HashMap, HashSet};

use crate::event::{FinalEvent, PartialEvent};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TranscriptError {
    #[error("segment {0} is final; it cannot be revised")]
    SegmentClosed(u64),
}

/// Producer-side revision manager: numbers the partials of each open segment and commits finals,
/// which can never change afterwards.
#[derive(Debug, Default)]
pub struct Transcript {
    finals: Vec<FinalEvent>,
    open: HashMap<u64, PartialEvent>,
    closed: HashSet<u64>,
}

impl Transcript {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn finals(&self) -> &[FinalEvent] {
        &self.finals
    }

    /// Latest partial of an open segment.
    pub fn current(&self, segment_id: u64) -> Option<&PartialEvent> {
        self.open.get(&segment_id)
    }

    pub fn partial(
        &mut self,
        segment_id: u64,
        text: String,
        start: f64,
        end: f64,
        stable_prefix_len: usize,
    ) -> Result<PartialEvent, TranscriptError> {
        if self.closed.contains(&segment_id) {
            return Err(TranscriptError::SegmentClosed(segment_id));
        }
        let revision = self.open.get(&segment_id).map_or(0, |p| p.revision + 1);
        let ev = PartialEvent {
            segment_id,
            revision,
            text,
            start,
            end,
            stable_prefix_len,
        };
        self.open.insert(segment_id, ev.clone());
        Ok(ev)
    }

    pub fn final_(
        &mut self,
        segment_id: u64,
        text: String,
        start: f64,
        end: f64,
    ) -> Result<FinalEvent, TranscriptError> {
        if self.closed.contains(&segment_id) {
            return Err(TranscriptError::SegmentClosed(segment_id));
        }
        let revision = self.open.remove(&segment_id).map_or(0, |p| p.revision + 1);
        self.closed.insert(segment_id);
        let ev = FinalEvent {
            segment_id,
            revision,
            text,
            start,
            end,
        };
        self.finals.push(ev.clone());
        Ok(ev)
    }

    /// Final texts joined as the Python reference does: CJK-only output joins without spaces.
    pub fn text(&self) -> String {
        let parts: Vec<&str> = self
            .finals
            .iter()
            .map(|f| f.text.as_str())
            .filter(|t| !t.is_empty())
            .collect();
        let sep = if parts.concat().contains(' ') {
            " "
        } else {
            ""
        };
        parts.join(sep)
    }
}
