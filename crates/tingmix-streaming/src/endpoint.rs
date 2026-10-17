#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseReason {
    Silence,
    MaxLength,
}

/// A segment opens on the first speech frame and closes after `silence_ms` of non-speech or at `max_segment_s`.
pub struct Endpointer {
    silence_s: f64,
    max_segment_s: f64,
    pub active: bool,
    length_s: f64,
    /// Non-speech at the end of the segment; kept after it closes.
    pub trailing_s: f64,
}

impl Endpointer {
    pub fn new(silence_ms: f64, max_segment_s: f64) -> Self {
        Self {
            silence_s: silence_ms / 1000.0,
            max_segment_s,
            active: false,
            length_s: 0.0,
            trailing_s: 0.0,
        }
    }

    /// Keep the segment open after a `MaxLength` close that was moved back in time: `length_s` of it is already behind us.
    pub fn resume(&mut self, length_s: f64) {
        self.active = true;
        self.length_s = length_s;
        self.trailing_s = 0.0;
    }

    /// Feed one frame. Returns the reason when the segment ended with this frame.
    pub fn update(&mut self, speech: bool, dur_s: f64) -> Option<CloseReason> {
        if !self.active {
            if !speech {
                return None;
            }
            self.active = true;
            self.length_s = 0.0;
            self.trailing_s = 0.0;
        }
        self.length_s += dur_s;
        self.trailing_s = if speech { 0.0 } else { self.trailing_s + dur_s };
        let reason = if self.trailing_s >= self.silence_s {
            Some(CloseReason::Silence)
        } else if self.length_s >= self.max_segment_s {
            Some(CloseReason::MaxLength)
        } else {
            None
        };
        if reason.is_some() {
            self.active = false;
        }
        reason
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closes_on_silence_after_speech() {
        let mut e = Endpointer::new(100.0, 10.0);
        assert_eq!(e.update(false, 0.02), None);
        assert!(!e.active);
        assert_eq!(e.update(true, 0.02), None);
        for _ in 0..4 {
            assert_eq!(e.update(false, 0.02), None);
        }
        assert_eq!(e.update(false, 0.02), Some(CloseReason::Silence));
        assert!(!e.active);
    }

    #[test]
    fn speech_resets_trailing_and_max_length_forces_a_cut() {
        let mut e = Endpointer::new(100.0, 0.1);
        assert_eq!(e.update(true, 0.02), None);
        e.update(false, 0.02);
        e.update(true, 0.02);
        assert_eq!(e.trailing_s, 0.0);
        e.update(true, 0.02);
        assert_eq!(e.update(true, 0.02), Some(CloseReason::MaxLength));
    }
}
