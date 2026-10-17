//! Capture 3 s from the default (or given) input and print the 16 kHz RMS every 100 ms.
//! `cargo run -p tingmix-audio --example capture_rms [-- <device id>]`
use std::time::{Duration, Instant};

use tingmix_audio::{start_capture, Resampler, MODEL_SAMPLE_RATE};

fn main() {
    let device = std::env::args().nth(1);
    let (_guard, mut reader) = match start_capture(device.as_deref(), 2.0) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "capturing from {:?} at {} Hz",
        reader.device_name(),
        reader.sample_rate()
    );
    let mut resampler = Resampler::new(reader.sample_rate(), MODEL_SAMPLE_RATE);
    let (mut pending, mut buf, mut total) = (Vec::<f32>::new(), vec![0f32; 4096], 0usize);
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(3) {
        std::thread::sleep(Duration::from_millis(20));
        loop {
            let n = reader.read(&mut buf);
            if n == 0 {
                break;
            }
            pending.extend(resampler.process(&buf[..n]));
        }
        while pending.len() >= 1600 {
            let block: Vec<f32> = pending.drain(..1600).collect();
            total += block.len();
            let rms = (block.iter().map(|v| v * v).sum::<f32>() / block.len() as f32).sqrt();
            println!(
                "{:5.2}s rms {rms:.5} ({:.1} dBFS)",
                total as f32 / 16_000.0,
                20.0 * (rms + 1e-10).log10()
            );
        }
        for e in reader.poll_events() {
            eprintln!("event: {e:?}");
        }
    }
    println!(
        "{total} samples at 16 kHz, {} dropped",
        reader.overflowed_samples()
    );
}
