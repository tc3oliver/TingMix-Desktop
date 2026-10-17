//! Streaming session: audio in, partial/final events out. Semantics follow the Python reference
//! (`tingmix@f44cbdb src/tingmix/streaming`).
//!
//! Layers:
//! - [`Pipeline`]: VAD, endpointing, scheduling over an absolute sample timeline. It never runs
//!   inference; it emits [`Job`]s that own copies of the audio they need.
//! - [`Decoder`]: turns jobs into events using an [`AsrBackend`](tingmix_core::AsrBackend).
//! - [`StreamingSession`]: both, synchronously and deterministically (files, tests).
//! - [`SessionWorker`]: both, with the decoder on its own thread, coalescing stale partial jobs.

mod buffer;
mod decoder;
mod endpoint;
mod mock;
mod pipeline;
mod scheduler;
mod session;
mod stability;
mod suppress;
mod vad;
mod worker;

pub use buffer::SampleRing;
pub use decoder::{Decoder, Outcome};
pub use endpoint::{CloseReason, Endpointer};
pub use mock::MockBackend;
pub use pipeline::{Job, JobKind, Pipeline, SessionConfig};
pub use scheduler::Scheduler;
pub use session::StreamingSession;
pub use stability::StablePrefix;
pub use suppress::Suppression;
pub use vad::{AdaptiveVad, AdaptiveVadParams, EnergyVad, Vad, VadKind};
pub use worker::{SessionMsg, SessionWorker};
