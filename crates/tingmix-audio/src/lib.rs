//! Audio I/O for TingMix: microphone capture into a lock-free ring, streaming resampling to
//! 16 kHz mono, and file decoding. No inference, no model types.
//!
//! Capture path: the OS callback downmixes to mono and pushes into a bounded SPSC ring (never
//! blocking, never allocating); a consumer thread drains it with [`CaptureReader::read`] and feeds
//! [`Resampler`]. When the consumer stalls the ring overflows: the newest samples are dropped
//! and counted in [`CaptureReader::overflowed_samples`], never silently.

mod capture;
mod devices;
mod error;
mod file;
mod resample;

pub use capture::{start_capture, CaptureEvent, CaptureGuard, CaptureReader};
pub use devices::{list_input_devices, InputDevice};
pub use error::AudioError;
pub use file::{decode_file, decode_file_chunks};
pub use resample::{resample, Resampler};

/// Sample rate every TingMix model consumes.
pub const MODEL_SAMPLE_RATE: u32 = 16_000;
