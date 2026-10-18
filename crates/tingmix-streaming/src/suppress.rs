/// Drops decoded text the model is not sure about, so a lone "m" decoded from noise is never shown.
///
/// A decode is rejected when its confidence (mean max-posterior over non-blank frames, from the backend) is below
/// `min_confidence`, or when the text is a short fragment (at most `short_chars` letters/characters, spaces ignored) and
/// the confidence is below `short_min_confidence`. These two rules apply to finals; a partial is dropped only below
/// `partial_min_confidence`, so first words still appear quickly. A suppressed final retracts its segment's caption. A backend that reports no confidence is never suppressed.
/// Thresholds are tuned in docs/research/vad-noise.md.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Suppression {
    /// Finals below this are dropped.
    pub min_confidence: f32,
    /// Partials below this are dropped (lower than for finals: a half-heard word is legitimately uncertain).
    pub partial_min_confidence: f32,
    pub short_chars: usize,
    pub short_min_confidence: f32,
}

impl Suppression {
    /// Thresholds chosen on the MLX Large measurements in docs/research/vad-noise.md.
    pub const RECOMMENDED: Suppression = Suppression {
        min_confidence: 0.5,
        partial_min_confidence: 0.3,
        short_chars: 2,
        short_min_confidence: 0.9,
    };

    pub fn rejects(&self, text: &str, confidence: Option<f32>, is_final: bool) -> bool {
        let Some(c) = confidence else { return false };
        if text.trim().is_empty() {
            return false;
        }
        if !is_final {
            return c < self.partial_min_confidence; // partials are revisable: only drop clear garbage, keep first words fast
        }
        let n = text.chars().filter(|ch| !ch.is_whitespace()).count();
        c < self.min_confidence || (n <= self.short_chars && c < self.short_min_confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Suppression = Suppression::RECOMMENDED;

    #[test]
    fn short_fragments_need_high_confidence_and_everything_needs_some() {
        assert!(S.rejects("m", Some(0.6), true));
        assert!(
            !S.rejects("m", Some(0.6), false),
            "a lone partial fragment is shown; its final is what gets suppressed"
        );
        assert!(S.rejects("m", Some(0.2), false));
        assert!(!S.rejects("OK", Some(0.95), true));
        assert!(!S.rejects("好", Some(0.97), true));
        assert!(S.rejects("今天天氣很好", Some(0.4), true));
        assert!(
            !S.rejects("今天天氣很好", Some(0.4), false),
            "partials get the lower bar"
        );
        assert!(!S.rejects("今天天氣很好", Some(0.7), true));
        assert!(!S.rejects("m", None, true), "no confidence, no suppression");
        assert!(!S.rejects("", Some(0.0), true));
    }
}
