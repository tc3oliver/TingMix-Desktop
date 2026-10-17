/// Ring buffer of samples addressed by absolute sample index (index 0 = first sample ever appended).
pub struct SampleRing {
    buf: Vec<f32>,
    end: usize,
}

impl SampleRing {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            buf: vec![0.0; capacity],
            end: 0,
        }
    }

    /// Absolute index one past the newest sample.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Absolute index of the oldest sample still held.
    pub fn start(&self) -> usize {
        self.end.saturating_sub(self.buf.len())
    }

    pub fn append(&mut self, x: &[f32]) {
        let cap = self.buf.len();
        let kept = &x[x.len().saturating_sub(cap)..];
        let first = self.end + x.len() - kept.len();
        for (i, &s) in kept.iter().enumerate() {
            self.buf[(first + i) % cap] = s;
        }
        self.end += x.len();
    }

    /// Samples `[a, b)`; `None` if any of them was overwritten or is not yet written.
    pub fn get(&self, a: usize, b: usize) -> Option<Vec<f32>> {
        if a < self.start() || b > self.end || a > b {
            return None;
        }
        let cap = self.buf.len();
        Some((a..b).map(|i| self.buf[i % cap]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_and_tracks_absolute_range() {
        let mut r = SampleRing::new(4);
        r.append(&[1.0, 2.0, 3.0]);
        assert_eq!(r.get(0, 3).unwrap(), [1.0, 2.0, 3.0]);
        r.append(&[4.0, 5.0, 6.0]);
        assert_eq!((r.start(), r.end()), (2, 6));
        assert_eq!(r.get(2, 6).unwrap(), [3.0, 4.0, 5.0, 6.0]);
        assert!(r.get(1, 6).is_none() && r.get(2, 7).is_none());
        r.append(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]); // longer than capacity
        assert_eq!(r.get(8, 12).unwrap(), [9.0, 10.0, 11.0, 12.0]);
    }
}
