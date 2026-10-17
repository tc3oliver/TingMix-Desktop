/// Frame-based voice activity detection. A VAD has a fixed `frame_ms` and `update(frame) -> bool`
/// (speech state after smoothing), so a neural VAD can replace the energy detector unchanged.
pub trait Vad: Send {
    fn frame_ms(&self) -> u32;
    fn update(&mut self, frame: &[f32]) -> bool;
}

/// RMS threshold with hysteresis: speech starts after `start_frames` loud frames in a row and
/// ends after `hang_frames` quiet ones.
///
/// shortcut: fixed threshold, no noise-floor tracking; use a neural VAD or an adaptive floor for noisy rooms.
pub struct EnergyVad {
    threshold_db: f64,
    frame_ms: u32,
    start_frames: u32,
    hang_frames: u32,
    speech: bool,
    run: u32, // consecutive frames that disagree with the current state
}

impl EnergyVad {
    pub fn new(threshold_db: f64, frame_ms: u32, start_frames: u32, hang_frames: u32) -> Self {
        Self {
            threshold_db,
            frame_ms,
            start_frames,
            hang_frames,
            speech: false,
            run: 0,
        }
    }
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::new(-50.0, 20, 2, 10)
    }
}

impl Vad for EnergyVad {
    fn frame_ms(&self) -> u32 {
        self.frame_ms
    }

    fn update(&mut self, frame: &[f32]) -> bool {
        let ms = frame
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            / frame.len().max(1) as f64;
        let loud = 20.0 * (ms.sqrt() + 1e-10).log10() > self.threshold_db;
        self.run = if loud != self.speech { self.run + 1 } else { 0 };
        if self.run
            >= if loud {
                self.start_frames
            } else {
                self.hang_frames
            }
        {
            self.speech = loud;
            self.run = 0;
        }
        self.speech
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(amp: f32) -> Vec<f32> {
        vec![amp; 320]
    }

    #[test]
    fn needs_two_loud_frames_to_start_and_ten_quiet_to_stop() {
        let mut v = EnergyVad::default();
        assert!(!v.update(&frame(0.1)));
        assert!(v.update(&frame(0.1)));
        for _ in 0..9 {
            assert!(v.update(&frame(0.0)));
        }
        assert!(!v.update(&frame(0.0)));
    }

    #[test]
    fn a_single_blip_is_ignored_and_resets_the_run() {
        let mut v = EnergyVad::default();
        assert!(!v.update(&frame(0.1)));
        assert!(!v.update(&frame(0.0)));
        assert!(!v.update(&frame(0.1)));
        assert!(v.update(&frame(0.1)));
    }
}
