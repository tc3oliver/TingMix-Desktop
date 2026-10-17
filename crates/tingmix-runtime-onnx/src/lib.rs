//! TingMix-Small on ONNX Runtime (CPU): a Rust log-mel frontend identical to the reference
//! `LogMel`, the streaming step graph, and a native-stream [`tingmix_core::AsrBackend`].
//!
//! The graph contract (`tools/small-export`): inputs `mel [1,K,80]` (K a multiple of 4), `c1 [1,80,1]`,
//! `c2 [1,d,1]`, `h [layers,1,hidden]`; outputs `logp [1,K/4,V]`, `c1_out`, `c2_out`, `h_out`.
//! State starts at zero, which reproduces the offline left padding, so chunked output equals offline output.

mod backend;
mod ctc;
mod frontend;
mod model;

pub use backend::{OnnxSmall, SmallStream, TokenDecoder};
pub use ctc::Collapser;
pub use frontend::{LogMel, MelStream, HOP, N_FFT, N_MELS, WIN};
pub use model::{set_runtime_library, Error, SharedModel, StepModel, StepState, Streamer};
