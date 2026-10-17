use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::event::{Event, FinalEvent, PartialEvent};

/// Why [`Reducer::apply`] ignored an event. The state is unchanged in every case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Rejected {
    #[error("segment is already final")]
    SegmentClosed,
    #[error("revision is not newer than the one already applied")]
    StaleRevision,
}

/// Everything a UI needs to rebuild its view after a reload or window recreation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub finals: Vec<FinalEvent>,
    pub open: Vec<PartialEvent>,
}

/// Consumer-side state machine: applies events in any delivery order and keeps only the newest
/// partial per open segment. Partials after a final, and any revision that is not strictly newer,
/// are rejected, so duplicated or reordered delivery cannot corrupt the view.
#[derive(Debug, Default)]
pub struct Reducer {
    finals: BTreeMap<u64, FinalEvent>,
    open: BTreeMap<u64, PartialEvent>,
    closed: HashSet<u64>,
}

impl Reducer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_snapshot(s: Snapshot) -> Self {
        let mut r = Self::new();
        for f in s.finals {
            r.closed.insert(f.segment_id);
            r.finals.insert(f.segment_id, f);
        }
        for p in s.open {
            if !r.closed.contains(&p.segment_id) {
                r.open.insert(p.segment_id, p);
            }
        }
        r
    }

    pub fn apply(&mut self, ev: Event) -> Result<(), Rejected> {
        if self.closed.contains(&ev.segment_id()) {
            return Err(Rejected::SegmentClosed);
        }
        if let Some(cur) = self.open.get(&ev.segment_id()) {
            if ev.revision() <= cur.revision {
                return Err(Rejected::StaleRevision);
            }
        }
        match ev {
            Event::Partial(p) => {
                self.open.insert(p.segment_id, p);
            }
            Event::Final(f) => {
                self.open.remove(&f.segment_id);
                self.closed.insert(f.segment_id);
                self.finals.insert(f.segment_id, f);
            }
        }
        Ok(())
    }

    /// Finals ordered by segment id (= time order).
    pub fn finals(&self) -> impl Iterator<Item = &FinalEvent> {
        self.finals.values()
    }

    /// Open segments' latest partials, ordered by segment id.
    pub fn open(&self) -> impl Iterator<Item = &PartialEvent> {
        self.open.values()
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            finals: self.finals.values().cloned().collect(),
            open: self.open.values().cloned().collect(),
        }
    }
}
