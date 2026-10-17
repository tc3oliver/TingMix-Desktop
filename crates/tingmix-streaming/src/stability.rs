use std::collections::VecDeque;

/// Stable-prefix detection by local agreement: the prefix shared by the last `k` successive partial hypotheses.
pub struct StablePrefix {
    k: usize,
    hyps: VecDeque<Vec<char>>,
}

impl StablePrefix {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            hyps: VecDeque::new(),
        }
    }

    pub fn reset(&mut self) {
        self.hyps.clear();
    }

    /// Add a hypothesis; returns the length in chars of the prefix of `text` that all of the last
    /// `k` hypotheses share (0 until `k` seen).
    pub fn update(&mut self, text: &str) -> usize {
        if self.hyps.len() == self.k {
            self.hyps.pop_front();
        }
        self.hyps.push_back(text.chars().collect());
        if self.hyps.len() < self.k {
            return 0;
        }
        let first = &self.hyps[0];
        let shortest = self.hyps.iter().map(Vec::len).min().unwrap_or(0);
        (0..shortest)
            .take_while(|&i| self.hyps.iter().all(|h| h[i] == first[i]))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_agreement_over_k_hypotheses() {
        let mut s = StablePrefix::new(2);
        assert_eq!(s.update("今天天氣"), 0);
        assert_eq!(s.update("今天天氣很"), 4);
        assert_eq!(s.update("今天天汽很好"), 3);
        s.reset();
        assert_eq!(s.update("abc"), 0);
    }
}
