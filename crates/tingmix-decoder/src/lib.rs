//! CTC decoding compatible with the embed2asr reference (`embed2asr@b582757`: `ctc.py`, `tokenizer.py`, `lm.py`).
//!
//! Input is a row-major `[T, V]` slice of `f32` log-probabilities (the reference does `logits.float().log_softmax(-1)`);
//! output is unit ids and text. The crate does not depend on any other workspace crate.

mod beam;
mod ctc;
mod lm;
mod vocab;

pub use beam::{beam_decode, BeamConfig, BeamOutput};
pub use ctc::{collapse_ctc, greedy_ids, log_softmax_rows};
pub use lm::CharLm;
pub use vocab::Vocab;

/// Blank unit id (`BLANK_ID` in the reference).
pub const BLANK: u32 = 0;

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}
