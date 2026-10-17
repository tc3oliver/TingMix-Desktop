/// Decides when to re-decode the open segment: every `hop_ms` of new audio, stretched when
/// decoding is slow so that compute stays below `max_load` x real time (audio that arrived
/// meanwhile is simply covered by the next decode).
pub struct Scheduler {
    sample_rate: f64,
    hop_s: f64,
    max_load: f64,
    last: usize, // segment length (samples) at the last decode
    cost_s: f64,
}

impl Scheduler {
    pub fn new(sample_rate: u32, hop_ms: f64, max_load: f64) -> Self {
        Self {
            sample_rate: f64::from(sample_rate),
            hop_s: hop_ms / 1000.0,
            max_load,
            last: 0,
            cost_s: 0.0,
        }
    }

    /// New segment: restart the hop counter and forget the measured cost.
    pub fn reset(&mut self) {
        self.last = 0;
        self.cost_s = 0.0;
    }

    pub fn due(&self, segment_samples: usize) -> bool {
        let hop = self.hop_s.max(self.cost_s / self.max_load);
        (segment_samples - self.last) as f64 >= hop * self.sample_rate
    }

    /// A decode job covering `segment_samples` was issued.
    pub fn mark(&mut self, segment_samples: usize) {
        self.last = segment_samples;
    }

    pub fn report_cost(&mut self, cost_s: f64) {
        self.cost_s = cost_s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_stretches_with_cost() {
        let mut s = Scheduler::new(16_000, 480.0, 0.5);
        assert!(!s.due(7_000) && s.due(7_680));
        s.mark(7_680);
        s.report_cost(0.6); // needs hop >= 1.2 s
        assert!(!s.due(7_680 + 19_000) && s.due(7_680 + 19_200));
        s.reset();
        assert!(s.due(7_680));
    }
}
