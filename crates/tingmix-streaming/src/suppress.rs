/// Drops decoded text the model is not sure about, so a lone "m" decoded from noise is never shown.
///
/// A decode is rejected when its confidence (mean max-posterior over non-blank frames, from the backend) is below
/// `min_confidence`, or when the text is a short fragment (at most `short_chars` letters/characters, spaces ignored)
/// and the confidence is below `short_min_confidence`. A backend that reports no confidence is never suppressed.
/// Thresholds are tuned in docs/research/vad-noise.md.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Suppression {
    pub min_confidence: f32,
    pub short_chars: usize,
    pub short_min_confidence: f32,
}

impl Suppression {
    pub fn rejects(&self, text: &str, confidence: Option<f32>) -> bool {
        let Some(c) = confidence else { return false };
        if text.trim().is_empty() {
            return false;
        }
        let n = text.chars().filter(|ch| !ch.is_whitespace()).count();
        c < self.min_confidence || (n <= self.short_chars && c < self.short_min_confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Suppression = Suppression {
        min_confidence: 0.4,
        short_chars: 2,
        short_min_confidence: 0.9,
    };

    #[test]
    fn short_fragments_need_high_confidence_and_everything_needs_some() {
        assert!(S.rejects("m", Some(0.6)));
        assert!(!S.rejects("OK", Some(0.95)));
        assert!(!S.rejects("好", Some(0.97)));
        assert!(S.rejects("今天天氣很好", Some(0.3)));
        assert!(!S.rejects("今天天氣很好", Some(0.7)));
        assert!(!S.rejects("m", None), "no confidence, no suppression");
        assert!(!S.rejects("", Some(0.0)));
    }
}
