/// CTC blank id of the TingMix vocabularies.
pub const BLANK: u32 = 0;

/// Incremental greedy CTC collapse. The previous raw frame id is carried between calls, so a
/// token repeated across a chunk boundary is merged exactly like in a single pass, while a blank
/// in between keeps both occurrences.
#[derive(Debug, Default)]
pub struct Collapser {
    prev: u32, // BLANK == 0 == Default
    out: Vec<u32>,
}

impl Collapser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw per-frame argmax ids.
    pub fn push(&mut self, raw: &[u32]) {
        for &t in raw {
            if t != BLANK && t != self.prev {
                self.out.push(t);
            }
            self.prev = t;
        }
    }

    /// Collapsed ids so far.
    pub fn ids(&self) -> &[u32] {
        &self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_across_a_chunk_boundary_is_merged() {
        let mut c = Collapser::new();
        c.push(&[0, 5, 5]);
        c.push(&[5, 0]);
        assert_eq!(c.ids(), [5]);
    }

    #[test]
    fn blank_at_the_boundary_keeps_both_tokens() {
        let mut c = Collapser::new();
        c.push(&[5, 0]);
        c.push(&[5, 7, 7]);
        c.push(&[7]);
        assert_eq!(c.ids(), [5, 5, 7]);
    }

    #[test]
    fn any_split_equals_one_pass() {
        let raw = [0, 3, 3, 0, 3, 4, 4, 4, 0, 0, 4, 9, 9];
        let mut whole = Collapser::new();
        whole.push(&raw);
        for cut in 0..=raw.len() {
            let mut c = Collapser::new();
            c.push(&raw[..cut]);
            c.push(&raw[cut..]);
            assert_eq!(c.ids(), whole.ids(), "cut {cut}");
        }
        assert_eq!(whole.ids(), [3, 3, 4, 4, 9]);
    }
}
