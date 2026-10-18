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

/// Which voice activity detector a [`crate::Pipeline`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VadKind {
    /// Fixed threshold (`SessionConfig::vad_threshold_db`): the Python reference's behaviour.
    #[default]
    Energy,
    /// Tracks the noise floor; see [`AdaptiveVad`].
    Adaptive,
}

/// Tunables of [`AdaptiveVad`]; the defaults are the values chosen in docs/research/vad-noise.md.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveVadParams {
    /// Noise floor = this percentile of the last `window_frames` smoothed frame levels.
    pub floor_percentile: f32,
    pub window_frames: usize,
    /// The floor may rise at most this fast (dB per second); it falls freely.
    pub rise_db_per_s: f32,
    /// Frame levels are averaged (as power) over this many frames before comparison.
    pub smooth_frames: usize,
    /// A frame is loud when above floor + this and above `abs_min_db`.
    pub start_margin_db: f32,
    /// Speech continues while a frame is above floor + this.
    pub stay_margin_db: f32,
    pub abs_min_db: f32,
    /// Speech opens when at least `start_votes` of the last `start_window` frames are loud.
    pub start_window: usize,
    pub start_votes: usize,
    /// Quiet frames in a row that end speech.
    pub hang_frames: u32,
    /// Frames heard before the first decision (the floor needs a baseline).
    pub warmup_frames: usize,
    /// The first floor estimate is only trusted from a stationary stretch: the p90-p10 spread of the last `settle_frames`
    /// smoothed levels must be at most `settle_spread_db`. Silence and steady noise are within a few dB; speech modulates by
    /// much more within a second, so until a pause or a steady stretch the conservative prior floor stays and speech that
    /// starts the stream is never learned as the noise floor.
    pub settle_frames: usize,
    pub settle_spread_db: f32,
    /// A stationary stretch whose median level is above `settle_loud_db` is more likely a close talker than a noisy room, so it
    /// has to stay stationary for `settle_loud_frames` before it counts as noise.
    pub settle_loud_db: f32,
    pub settle_loud_frames: usize,
}

impl Default for AdaptiveVadParams {
    fn default() -> Self {
        Self {
            floor_percentile: 0.2,
            window_frames: 250,
            rise_db_per_s: 4.0,
            smooth_frames: 3,
            start_margin_db: 10.0,
            stay_margin_db: 5.0,
            abs_min_db: -55.0,
            start_window: 12,
            start_votes: 8,
            hang_frames: 10,
            warmup_frames: 25,
            settle_frames: 30,
            settle_spread_db: 6.0,
            settle_loud_db: -18.0,
            settle_loud_frames: 40,
        }
    }
}

/// Energy VAD whose threshold follows the room: speech = frame level above (tracked noise floor + margin).
///
/// - Floor: a low percentile of the recent frame levels (speech and non-speech alike, so a changing noise level is
///   followed), limited in how fast it may rise so a burst or a long vowel cannot drag it up; it falls freely.
/// - Start: enough loud frames among the last few (see [`AdaptiveVadParams`]), so a click or a cough shorter than
///   ~250 ms never opens a segment. The pipeline's preroll buffer returns the audio before the decision.
/// - Stay: a frame above the lower stay threshold continues speech; `hang_frames` quiet frames end it.
///
/// shortcut: energy only, so non-stationary noise (a second talker, music) still passes; the confidence check on the
/// decoded text is the second line of defence. Upgrade to a neural VAD if that proves insufficient.
pub struct AdaptiveVad {
    p: AdaptiveVadParams,
    frame_ms: u32,
    window: std::collections::VecDeque<f32>,
    power: std::collections::VecDeque<f64>,
    floor_db: f32,
    floor_known: bool,
    speech: bool,
    votes: std::collections::VecDeque<bool>,
    quiet_run: u32,
    heard: usize,
}

impl AdaptiveVad {
    pub fn new() -> Self {
        Self::with_params(AdaptiveVadParams::default())
    }

    pub fn with_params(p: AdaptiveVadParams) -> Self {
        Self {
            p,
            frame_ms: 20,
            window: Default::default(),
            power: Default::default(),
            floor_db: -60.0,
            floor_known: false,
            speech: false,
            votes: Default::default(),
            quiet_run: 0,
            heard: 0,
        }
    }

    /// Current noise-floor estimate in dBFS (for diagnostics).
    pub fn floor_db(&self) -> f32 {
        self.floor_db
    }

    fn track_floor(&mut self, db: f32) {
        // digital silence says nothing about the room; letting it in drags the floor to -200 dB and the next noise reads as speech
        if db < self.p.abs_min_db - 30.0 {
            return;
        }
        self.window.push_back(db);
        if self.window.len() > self.p.window_frames {
            self.window.pop_front();
        }
        if self.window.len() < self.p.warmup_frames {
            return; // keep the prior until enough has been heard
        }
        if !self.floor_known && !self.looks_stationary() {
            return; // speech-like start: keep the conservative prior instead of learning the speech as the noise floor
        }
        let mut v: Vec<f32> = self.window.iter().copied().collect();
        let k = ((v.len() - 1) as f32 * self.p.floor_percentile) as usize;
        let target = *v.select_nth_unstable_by(k, f32::total_cmp).1;
        let rise = self.p.rise_db_per_s * self.frame_ms as f32 / 1000.0;
        self.floor_db = if !self.floor_known || target < self.floor_db {
            target
        } else {
            target.min(self.floor_db + rise)
        };
        // the first estimate replaces the prior outright: no ramp-up false trigger at stream start
        self.floor_known = true;
    }
}

impl AdaptiveVad {
    /// Silence or steady noise rather than speech; see [`AdaptiveVadParams::settle_frames`].
    fn looks_stationary(&self) -> bool {
        let stats = |n: usize| {
            let mut v: Vec<f32> = self.window.iter().rev().take(n).copied().collect();
            v.sort_by(f32::total_cmp);
            let at = |q: f32| v[((v.len() - 1) as f32 * q).round() as usize];
            (at(0.9) - at(0.1), at(0.5))
        };
        let n = self.p.settle_frames.max(1);
        if self.window.len() < n {
            return false;
        }
        let (spread, median) = stats(n);
        if spread > self.p.settle_spread_db {
            return false;
        }
        if median <= self.p.settle_loud_db {
            return true;
        }
        let m = self.p.settle_loud_frames.max(n);
        self.window.len() >= m && stats(m).0 <= self.p.settle_spread_db
    }
}

impl Default for AdaptiveVad {
    fn default() -> Self {
        Self::new()
    }
}

impl Vad for AdaptiveVad {
    fn frame_ms(&self) -> u32 {
        self.frame_ms
    }

    fn update(&mut self, frame: &[f32]) -> bool {
        let ms = frame
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            / frame.len().max(1) as f64;
        if !ms.is_finite() {
            return self.speech; // NaN or infinite samples: no update
        }
        self.power.push_back(ms);
        if self.power.len() > self.p.smooth_frames.max(1) {
            self.power.pop_front();
        }
        let mean = self.power.iter().sum::<f64>() / self.power.len() as f64;
        let db = (10.0 * (mean + 1e-20).log10()) as f32;
        self.track_floor(db);
        self.heard += 1;
        let start_thr = (self.floor_db + self.p.start_margin_db).max(self.p.abs_min_db);
        let stay_thr = (self.floor_db + self.p.stay_margin_db).max(self.p.abs_min_db - 4.0);
        if self.speech {
            self.quiet_run = if db > stay_thr { 0 } else { self.quiet_run + 1 };
            if self.quiet_run >= self.p.hang_frames {
                self.speech = false;
                self.votes.clear();
            }
        } else {
            // votes cast against the unsettled prior floor are meaningless: start counting after the warm-up
            if self.heard > self.p.warmup_frames {
                self.votes.push_back(db > start_thr);
            }
            if self.votes.len() > self.p.start_window {
                self.votes.pop_front();
            }
            if self.heard >= self.p.warmup_frames
                && self.votes.iter().filter(|v| **v).count() >= self.p.start_votes
            {
                self.speech = true;
                self.quiet_run = 0;
            }
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

    fn level(db: f32) -> Vec<f32> {
        // a square wave of the requested RMS
        vec![10f32.powf(db / 20.0); 320]
    }

    fn feed(v: &mut AdaptiveVad, db: f32, n: usize) -> bool {
        (0..n).map(|_| v.update(&level(db))).last().unwrap()
    }

    #[test]
    fn adaptive_ignores_steady_noise_that_energy_vad_would_call_speech() {
        let mut e = EnergyVad::default();
        let mut a = AdaptiveVad::new();
        // -40 dBFS room noise is above the fixed -50 dB threshold
        assert!((0..100).map(|_| e.update(&level(-40.0))).last().unwrap());
        assert!(!feed(&mut a, -40.0, 400));
        assert!((a.floor_db() + 40.0).abs() < 1.0, "{}", a.floor_db());
        // speech 20 dB above the noise opens after 8 loud frames (160 ms) and closes ~200 ms (+ smoothing) after it ends
        assert!(!feed(&mut a, -20.0, 7));
        assert!(feed(&mut a, -20.0, 1));
        assert!(feed(&mut a, -40.0, 9));
        assert!(!feed(&mut a, -40.0, 3));
    }

    #[test]
    fn adaptive_ignores_clicks_shorter_than_the_minimum_speech_duration() {
        let mut a = AdaptiveVad::new();
        feed(&mut a, -60.0, 100);
        for _ in 0..5 {
            assert!(!feed(&mut a, -20.0, 4)); // 80 ms bursts (smoothing stretches them to 6 frames)
            feed(&mut a, -60.0, 10);
        }
    }

    #[test]
    fn digital_silence_does_not_drag_the_floor_so_later_noise_stays_closed() {
        let mut a = AdaptiveVad::new();
        feed(&mut a, -200.0, 50); // 1 s of zeros
        let open = (0..400).filter(|_| a.update(&level(-50.0))).count();
        assert!(open < 30, "noise held the gate open for {open} frames");
        assert!(a.floor_db() > -60.0, "{}", a.floor_db());
    }

    #[test]
    fn non_finite_samples_are_ignored() {
        let mut a = AdaptiveVad::new();
        feed(&mut a, -40.0, 400);
        let floor = a.floor_db();
        assert!(!a.update(&[f32::NAN; 320]));
        assert!(!a.update(&[f32::INFINITY; 320]));
        assert!(!feed(&mut a, -40.0, 50));
        assert_eq!(a.floor_db(), floor);
    }

    /// Syllable-like modulation: 8 frames at `hi` dB, 4 at `lo` dB, repeated.
    fn speechlike(v: &mut AdaptiveVad, hi: f32, lo: f32, frames: usize) -> Vec<bool> {
        (0..frames)
            .map(|i| v.update(&level(if i % 12 < 8 { hi } else { lo })))
            .collect()
    }

    #[test]
    fn speech_that_starts_the_stream_is_not_learned_as_the_noise_floor() {
        let mut a = AdaptiveVad::new();
        let s = speechlike(&mut a, -25.0, -50.0, 300); // 6 s of modulated speech, no silence heard yet
        let first = s.iter().position(|&x| x).expect("speech never detected");
        assert!(first <= 40, "opened only at frame {first}");
        assert!(
            s[first..].iter().filter(|&&x| x).count() > 250,
            "speech dropped out: {}",
            s[first..].iter().filter(|&&x| x).count()
        );
        assert!(
            a.floor_db() < -50.0,
            "floor was dragged up to {}",
            a.floor_db()
        );
    }

    #[test]
    fn without_the_settle_check_speech_at_the_start_is_lost() {
        // negative control: the previous behaviour (trust the first percentile) drags the floor up into the speech
        let mut a = AdaptiveVad::with_params(AdaptiveVadParams {
            settle_spread_db: f32::INFINITY,
            settle_loud_db: f32::INFINITY,
            ..Default::default()
        });
        let s = speechlike(&mut a, -25.0, -50.0, 300);
        let open = s.iter().filter(|&&x| x).count();
        assert!(
            open < 150,
            "the control no longer reproduces the bug ({open} of 300 frames open): {}",
            a.floor_db()
        );
    }

    #[test]
    fn a_pause_after_speech_settles_the_floor_to_the_real_noise() {
        let mut a = AdaptiveVad::new();
        speechlike(&mut a, -25.0, -50.0, 200);
        feed(&mut a, -62.0, 120); // a 2.4 s pause in a quiet room
        assert!((a.floor_db() + 62.0).abs() < 2.0, "{}", a.floor_db());
        assert!(!feed(&mut a, -62.0, 5));
    }

    #[test]
    fn floor_follows_a_rising_noise_level_slowly() {
        let mut a = AdaptiveVad::new();
        feed(&mut a, -60.0, 100);
        feed(&mut a, -45.0, 50); // 1 s later it has moved by at most ~4 dB
        assert!(a.floor_db() < -54.0, "{}", a.floor_db());
        feed(&mut a, -45.0, 800);
        assert!((a.floor_db() + 45.0).abs() < 1.0, "{}", a.floor_db());
    }
}
