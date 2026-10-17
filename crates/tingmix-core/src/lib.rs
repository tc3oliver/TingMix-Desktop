//! Shared types of the TingMix pipeline: the streaming event protocol, the revision manager that
//! produces it, the client-side reducer that consumes it, and the backend traits.
//!
//! Nothing here knows about model tensors; backends only exchange `f32` PCM and text.

mod backend;
mod event;
mod reducer;
mod transcript;

pub use backend::{AsrBackend, AsrStream, BackendError, BackendInfo, Transcription};
pub use event::{Event, FinalEvent, PartialEvent};
pub use reducer::{Reducer, Rejected, Snapshot};
pub use transcript::{Transcript, TranscriptError};

#[cfg(test)]
mod tests;
