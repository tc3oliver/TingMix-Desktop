use crate::BLANK;

/// Greedy CTC collapse: merge repeats, drop blanks (`ctc.collapse_ctc`).
pub fn collapse_ctc(ids: &[u32], blank: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut prev = blank;
    for &t in ids {
        if t != blank && t != prev {
            out.push(t);
        }
        prev = t;
    }
    out
}

/// Per-frame argmax (first maximum wins, like `np.argmax`) followed by the collapse.
pub fn greedy_ids(lp: &[f32], vocab_size: usize) -> Vec<u32> {
    let argmax: Vec<u32> = lp
        .chunks_exact(vocab_size)
        .map(|row| {
            let mut best = 0;
            for (i, &x) in row.iter().enumerate() {
                if x > row[best] {
                    best = i;
                }
            }
            best as u32
        })
        .collect();
    collapse_ctc(&argmax, BLANK)
}

/// Row-wise `log_softmax` in f32 for raw reader logits `[T, vocab_size]` (the reference does
/// `logits.float().log_softmax(-1)`; summation order differs from torch, so values may differ in the last ulp).
pub fn log_softmax_rows(logits: &[f32], vocab_size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(logits.len());
    for row in logits.chunks_exact(vocab_size) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + row.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
        out.extend(row.iter().map(|&x| x - lse));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_keeps_repeats_separated_by_blank() {
        assert_eq!(collapse_ctc(&[0, 3, 3, 0, 3, 4, 4, 0], 0), vec![3, 3, 4]);
    }
}
