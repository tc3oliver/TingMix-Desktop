use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ErrorKind, FromSample, SampleFormat, SizedSample};
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::{HeapCons, HeapRb};

use crate::AudioError;

/// Frames downmixed per push; keeps the callback allocation-free with a fixed stack buffer.
const BLOCK: usize = 1024;

/// Out-of-band stream condition, delivered through [`CaptureReader::poll_events`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureEvent {
    /// The device went away or the stream must be rebuilt. No more audio will arrive;
    /// restart capture (possibly with another device).
    Disconnected(String),
    /// The OS rerouted the stream to another device (default device changed). Audio continues,
    /// but the signal source and level may differ.
    Rerouted,
    /// The OS dropped audio before it reached the ring (driver-side glitch).
    Xrun,
    /// Any other stream error.
    Error(String),
}

#[derive(Default)]
struct Shared {
    overflow: AtomicU64,
}

/// Keeps the OS stream alive; dropping it stops capture. May be `!Send` on some platforms, so
/// keep it on the thread that created it and move the [`CaptureReader`] instead.
pub struct CaptureGuard {
    _stream: cpal::Stream,
}

/// The consumer half: `Send`, drains the ring from any thread.
pub struct CaptureReader {
    cons: HeapCons<f32>,
    shared: Arc<Shared>,
    events: Receiver<CaptureEvent>,
    sample_rate: u32,
    device_name: String,
}

impl CaptureReader {
    /// Native sample rate of the mono samples returned by [`read`](Self::read); feed it to [`crate::Resampler`].
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Copy up to `out.len()` mono samples; returns how many were available.
    pub fn read(&mut self, out: &mut [f32]) -> usize {
        self.cons.pop_slice(out)
    }

    /// Samples dropped so far because the ring was full (consumer too slow).
    pub fn overflowed_samples(&self) -> u64 {
        self.shared.overflow.load(Ordering::Relaxed)
    }

    pub fn poll_events(&self) -> Vec<CaptureEvent> {
        self.events.try_iter().collect()
    }
}

/// Start capturing from `device_id` (as listed by [`crate::list_input_devices`]) or the default
/// input. `ring_seconds` of mono audio can queue up before samples are dropped.
pub fn start_capture(
    device_id: Option<&str>,
    ring_seconds: f32,
) -> Result<(CaptureGuard, CaptureReader), AudioError> {
    let host = cpal::default_host();
    let device = match device_id {
        Some(id) => {
            let parsed = id
                .parse()
                .map_err(|_| AudioError::DeviceNotFound(id.to_owned()))?;
            host.device_by_id(&parsed)
                .ok_or_else(|| AudioError::DeviceNotFound(id.to_owned()))?
        }
        None => host
            .default_input_device()
            .ok_or(AudioError::NoInputDevice)?,
    };
    let device_name = device
        .description()
        .map_or_else(|_| "unknown".into(), |d| d.name().to_owned());
    let cfg = device.default_input_config()?;
    let (rate, channels, format) = (
        cfg.sample_rate(),
        usize::from(cfg.channels()),
        cfg.sample_format(),
    );
    let capacity = ((rate as f32 * ring_seconds) as usize).max(BLOCK);
    let (prod, cons) = HeapRb::<f32>::new(capacity).split();
    let shared = Arc::new(Shared::default());
    let (tx, events) = channel();
    let stream_cfg: cpal::StreamConfig = cfg.into();

    let stream = match format {
        SampleFormat::F32 => build::<f32>(&device, &stream_cfg, channels, prod, &shared, tx),
        SampleFormat::I16 => build::<i16>(&device, &stream_cfg, channels, prod, &shared, tx),
        SampleFormat::I32 => build::<i32>(&device, &stream_cfg, channels, prod, &shared, tx),
        other => {
            return Err(AudioError::Device(format!(
                "unsupported sample format {other}"
            )))
        }
    }?;
    stream.play()?;
    Ok((
        CaptureGuard { _stream: stream },
        CaptureReader {
            cons,
            shared,
            events,
            sample_rate: rate,
            device_name,
        },
    ))
}

fn build<T>(
    device: &cpal::Device,
    cfg: &cpal::StreamConfig,
    channels: usize,
    mut prod: ringbuf::HeapProd<f32>,
    shared: &Arc<Shared>,
    tx: Sender<CaptureEvent>,
) -> Result<cpal::Stream, AudioError>
where
    T: SizedSample + Send + 'static,
    f32: FromSample<T>,
{
    let shared = shared.clone();
    let on_data = move |data: &[T], _: &cpal::InputCallbackInfo| {
        let mut mono = [0f32; BLOCK];
        for block in data.chunks(BLOCK * channels) {
            let frames = block.len() / channels;
            for (m, frame) in mono.iter_mut().zip(block.chunks_exact(channels)) {
                *m = frame.iter().map(|&s| f32::from_sample_(s)).sum::<f32>() / channels as f32;
            }
            let written = prod.push_slice(&mono[..frames]);
            if written < frames {
                shared
                    .overflow
                    .fetch_add((frames - written) as u64, Ordering::Relaxed);
            }
        }
    };
    let on_error = move |e: cpal::Error| {
        let ev = match e.kind() {
            ErrorKind::DeviceChanged => CaptureEvent::Rerouted,
            ErrorKind::Xrun => CaptureEvent::Xrun,
            ErrorKind::DeviceNotAvailable | ErrorKind::StreamInvalidated => {
                CaptureEvent::Disconnected(e.to_string())
            }
            _ => CaptureEvent::Error(e.to_string()),
        };
        let _ = tx.send(ev);
    };
    Ok(device.build_input_stream(*cfg, on_data, on_error, None)?)
}

#[cfg(test)]
mod tests {
    #[test]
    fn reader_can_move_to_another_thread() {
        fn is_send<T: Send>() {}
        is_send::<super::CaptureReader>();
    }
}
