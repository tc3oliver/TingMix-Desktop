# Background noise: adaptive VAD, confidence suppression and quiet-point cuts

Status: VERIFIED on synthetic speech + generated noise with the real TingMix-Large (MLX fp32, beam+LM) on one machine
(M4 Max); EXPERIMENTAL as a product default (no real-microphone or real-music recordings were used by the harness).
Everything below is off by default in the library; `SessionConfig::live_recommended()` turns it on.

## Problem

Observed 2026-10-18 (phone microphone through Continuity, real model): noise makes the app emit finals like "m" and "我"; a video
played through a speaker never reaches an endpoint, so segments end at the 25 s cap mid-sentence. Cause: `EnergyVad` is a
fixed -50 dBFS threshold (the Python reference's behaviour, kept as the compatibility default). Any bed above -50 dBFS
(a room, a fan, background music) is "speech" for ever, so the silence endpoint never fires, and noise-only stretches are
decoded and the model writes a lone BPE piece.

## What was built

- `AdaptiveVad` (`crates/tingmix-streaming/src/vad.rs`): noise floor = 20th percentile of the last 5 s of 3-frame-smoothed
  levels (first estimate immediate, rise limited to 4 dB/s, falls freely); a frame is loud above floor + 10 dB (and above
  -55 dBFS), speech opens when 8 of the last 12 frames are loud (so clicks and coughs under ~160 ms do nothing), stays
  open above floor + 5 dB, ends after 200 ms of quiet; nothing is decided during the first 0.5 s. Segment audio before the
  decision is recovered by the existing 0.8 s preroll. Parameters: `AdaptiveVadParams`. `EnergyVad` is unchanged.
- Confidence: `AsrBackend::transcribe_batch_scored` (default: no confidence, behaviour unchanged). MLX Large reports the
  mean max-posterior over frames whose best label is not blank (least confident 29.5 s chunk wins); it reports 0.0 when beam+LM wrote
  text although every frame's best label was blank, the signature of a hallucination.
- `Suppression` (`suppress.rs`, applied in `Decoder`, also to a final reused from the tail partial): a final is dropped if
  confidence < 0.5, or if it has at most 2 characters and confidence < 0.9; a partial is dropped only below 0.3 (partials
  are revisable and first words must stay fast). A dropped final closes its segment with empty text (the caption is
  retracted); nothing is emitted for a segment that never showed text. A backend without confidence is never suppressed.
- Quiet-point cut (`SessionConfig::max_cut_search_s`, buffered models only): when a segment reaches `max_segment_s`, cut at the quietest 60 ms
  of its last 3 s, close the final there and continue straight into the next segment from the same sample; no audio is
  dropped or duplicated (unit test).
- `tingmix-bench`: generated noise (white, pink, fan = rumble + 100/200 Hz hum, babble = six time-reversed TTS voices,
  music = chord bed with tremolo) at a given SNR against the speech RMS or an absolute dBFS, `--no-speech`, model-free VAD
  evaluation (`--vad-only`), segmentation of a recording (`--wav`), `--speed` for text-only replays, CER against the TTS
  prompt or the clean-audio oracle. No third-party recordings: speech is the repo's own TTS fixtures.

## Measurements

Model-free segmentation, 600 s per condition (`benchmarks/streaming/noise/vad-only-matrix.json`, 98 rows): with energy VAD every
noise bed at 30 dB SNR or louder (and white/pink/fan/babble at -53 dBFS and above) leaves the segment open until the 25 s cap
(19 merged segments per run; noise-only streams yield 2.47 false segments/min, i.e. permanently open; pink noise at 40 dB even gave 2.86/min). Adaptive VAD:
speech coverage 1.0 at 40, 30, 20 and 15 dB SNR for all four noise types, 0.85-0.94 at 10 dB (2-4 of 47 utterances missed), and
0.0-0.41 at 5 dB (28-47 missed); noise-only false segments: 0 in 60 min each of white, pink and babble, 6 in 60 min of the fan bed
(exactly one per 10-minute run: a slow-rumble artifact near stream start that I did not chase down); detection delay 0.21 s.

Real model, 120 s replays at 2x speed, speech = the 8 fixtures in random order, gaps 1-4 s (`mlx-suppress-off/`,
`mlx-suppress-on/`; stream CER against the clean-audio oracle texts (tests/fixtures/large/expected_texts.json); "miss" = utterances without a final;
fin95 = final latency P95 from speech end in stream seconds, indicative because the machine was shared). The off-runs
predate the confidence plumbing, so their JSON has no per-event confidence.

| Condition | energy VAD | adaptive VAD | adaptive + suppression |
|---|---|---|---|
| clean | 0.035 / miss 0 / fin95 0.88 | 0.040 / 0 / 0.94 | 0.040 / 0 / 0.92 |
| white, SNR 20 dB | 0.129 / miss 7 / 4.69 | 0.109 / 0 / 0.85 | 0.109 / 0 / 0.82 |
| pink, 20 dB | 0.094 / 7 / 4.68 | 0.077 / 0 / 0.84 | not run |
| fan, 20 dB | 0.036 / 7 / 4.69 | 0.030 / 0 / 1.00 | not run |
| babble, 20 dB | 0.096 / 7 / 4.70 | 0.041 / 0 / 0.84 | not run |
| white, 10 dB | 0.298 / 7 / 4.69 | 0.280 / 0 / 0.78 | 0.280 / 0 / 0.78 |
| music bed, -35 dBFS | 0.036 / 7 / 4.68 | 0.025 / 0 / 0.83 | 0.025 / 0 / 0.82 |

Noise only, 90 s, speed 2 (false segments per minute, characters written in false finals):

| Noise | energy | energy + suppression | adaptive (+ suppression) |
|---|---|---|---|
| white 20 dB | 2.80/min, 5 chars ("我" x5) | 2.80/min, 1 char | 0 / 0 |
| fan 20 dB | 2.80/min, 9 chars ("mm", "m") | 2.80/min, 0 | 0 / 0 |
| babble 20 dB | 2.80/min, 53 chars | 2.80/min, 0 | 0 / 0 |
| music -35 dBFS | 2.80/min, 7 chars ("m", "iss") | 0.56/min, 0 | 0 / 0 |

Confidence data behind the thresholds (suppression off, MLX fp32): false finals "m"/"mm" 0.31-0.46, babble text 0.32-0.58,
"我" in white noise 0.82-0.92; real speech finals 0.90-1.0 clean (min 0.70 in garbled 10 dB white noise); real short words
(synthetic TTS: 好 對 是 謝謝 好的 yes no hello) 0.90-1.0, "okay"/"ok" 0.63-0.97; a non-word ("嗯") decoded to "o"/"你" at 0.0-0.3. The
short-fragment bar of 0.9 therefore removes most hallucinated single characters but would also drop a real short word that the
model hears at under 0.9 in noise (one "ok" at 0.63 in the 10 dB pink run, 對 at exactly 0.90 passes); this is the chosen trade.

Suppression alone (energy VAD) leaves the first-partial latency unchanged because the short-fragment rule applies to finals
only; an earlier variant that also applied it to partials delayed first partial P95 from 0.63 s to 1.13 s and was dropped.

Quiet-point cut, continuous speech (4 fixtures back to back with 50-100 ms gaps, pink bed at -55 dBFS, adaptive VAD,
suppression on, 100 s; `benchmarks/streaming/noise/cut/`): stream CER against the prompts 0.152 with the hard cap, 0.092 with the cut. Only 5 utterances, one
seed: indicative. On a real 527 s zh-TW video with a music bed (a local file, not committed) energy VAD gave 22
segments of which 21 ended at the cap; with the cut, 23 segments with median 23.2 s and none cut at the cap; the adaptive
VAD found the pauses and gave 104 segments (median 2.3 s, longest 12 s), none at the cap.

## Recommendation

Switch the live-capture path to `SessionConfig::live_recommended()` (adaptive VAD + `Suppression::RECOMMENDED` + 3 s
quiet-point cut). It removes the permanently-open segment and the noise hallucinations in every tested bed, keeps clean
speech CER (0.035 vs 0.040, within noise) and first-partial latency, and shortens final latency P95 in noise from about
4.7 s to about 0.85 s. Keep `SessionConfig::default()` (EnergyVad, no suppression, hard cap) for file transcription and the
Python-parity goldens. The app wiring lives in `src-tauri` (decoder).

## Limits

- Energy-only VAD: a nearby second talker or music with speech-like dynamics still opens segments (the babble bed here is
  stationary-ish); the confidence check is the second line of defence. At 10 dB SNR 6-15 % of speech is missed (a few utterances) and at 5 dB most of it.
- Synthetic TTS speech and generated noise on one machine; thresholds should be re-checked with real microphones before
  they are treated as final. The MLX confidence is greedy-frame based while the text comes from beam+LM.
- The first 0.5 s after a stream starts makes no speech decision (audio is not lost: preroll covers it).
- A suppressed final retracts a caption the user already saw as a partial; the UI must treat an empty final as "remove".

Reproduce: `crates/tingmix-bench` with `--features mlx`; `benchmarks/streaming/noise/run_vad_matrix.py`,
`run_mlx_matrix.py` (see their headers); raw JSON in `benchmarks/streaming/noise/`.

## Speech that starts the stream (fix, 2026-10-18)

The first floor estimate used to be taken from the first half second. Audio that began with speech (`mix_08s.wav`, speech at sample 0)
made the floor equal to the speech, so the live preset lost the first ~2.4 s. The estimate is now trusted only from a stationary stretch
(`AdaptiveVadParams::settle_*`): the p90-p10 spread of the last 30 smoothed frames (0.6 s) must be at most 6 dB, and a stretch whose
median is above -18 dBFS (a close talker rather than a noisy room) must stay that stationary for 40 frames. Until then the conservative
prior floor (-60 dBFS) stays, so speech is never learned as noise. Natural speech modulates by far more than 6 dB within 0.6 s; the TTS fixtures
are stationary and loud, which is what the -18 dBFS clause is for. Tests: `speech_that_starts_the_stream_is_not_learned_as_the_noise_floor`,
a negative control that reproduces the old behaviour, and `live_preset_keeps_speech_that_starts_at_sample_zero` (replay of mix_08s; fails on the old code).
The model-free matrix re-run (`benchmarks/streaming/noise/vad-only-matrix-after-settle.json`) equals the committed one on all 98 rows (false segments, missed
utterances, coverage, merges, detection delay). Tuning that was tried and rejected: a spread-only rule at 50 frames added one 1-1.5 s false segment at the start of
every noisy recording (24 of 24 noisy runs), and a "must last 5 s if loud" rule delayed detection by up to 9 s under steady noise. Open: loud, modulated
noise (babble) with speech starting immediately cannot be told apart by energy; real-speech and real-room recordings are not covered by these synthetic checks.

## Non-stop compressed speech (fix, 2026-10-18)

A live test with a YouTube talk show played into the microphone skipped parts of sentences. Replaying the first 300 s of
that video through the live preset (`SessionConfig::live_recommended()`, MLX Large, no microphone) covered only 217 s with
segments; the other 83 s never reached the model. File mode covered all 300 s. The audio is broadcast-compressed speech over
a music bed with almost no pauses: frame levels p20 -28, median -22, p90 -15 dBFS. A floor at the 20th percentile of the last
5 s therefore sits inside the speech, the start threshold (floor + 10 dB) is reached by only the loudest quarter of frames, and
the stay threshold (floor + 5 dB) is near the median, so segments kept closing mid-sentence.

Change: `floor_percentile` 0.2 -> 0.1 and `stay_margin_db` 5 -> 3. Coverage of the same 300 s (`tingmix-bench --vad-only
--wav`, model-free): default 217 s / 63 segments; 0.1 alone 283 s; 0.1 + stay 3 296 s / 19 segments; 0.05 300 s. Slowing the
floor's rise only while speech is open was tried and rejected (239 s at 0.25 dB/s). With the real model the live preset now
writes continuous text over the 300 s, and no final was suppressed.

Model-free matrix (`vad-only-matrix-floor-p10.json` vs `vad-only-matrix-after-settle.json`, adaptive rows): speech in noise
improves everywhere (missed utterances at 10 dB SNR: white 2 -> 0, pink 2 -> 0, fan 4 -> 0; at 5 dB pink 47 -> 36, fan 28 -> 3,
babble 30 -> 8). White, pink and babble noise alone still open no segment. The cost is the steady fan bed: false segments
0.10 -> 0.79 per minute at every SNR, and one more merged segment per 10 min of fan + speech. Real model, suppression on
(`mlx-floor-p10/`): fan alone opened 0.56 false segments per minute whose finals were suppressed, but a partial "m" (confidence
0.34-0.36) showed for a few seconds before the retraction; clean and fan + speech CER 0.025 and 0.023; music bed alone 0.
The fan + speech run reports one missed utterance: en_02s followed 3.4 s later by zh_08s became one segment, and its text
starts that segment's final. The 0.05 percentile was rejected for fan noise alone (2.9 false segments per minute).

Tests: `non_stop_compressed_speech_stays_open` (fails on the old defaults); the settle negative control pins the old 0.2
percentile it was written for. Not checked: real rooms with fans or air conditioning, and how often the brief "m" partial
appears there. A neural VAD remains the real fix for speech that energy cannot separate from its background.

## Bridging short gaps between segments (2026-10-18)

Over the whole 1988 s of the same video the live preset left 8 gaps of 1.1-2.7 s between segments (1958 s covered). Each gap,
decoded on its own, held speech: quieter stretches (median -24 to -29 dBFS against -22 overall) that stay below the start
threshold, so the VAD does not reopen until the speech gets louder, and the 800 ms preroll does not reach back far enough.
Lowering the start margin or lengthening the hang closes the gaps but raises false segments on fan noise alone from 24 to 36-132
per 30 min (`hang_frames` 15/20, `start_margin_db` 8), so the VAD stays as it is.

Change: `SessionConfig::bridge_s` (live preset 4 s, default 0 = Python-identical). A segment that opens within 4 s of the previous
one's end starts at that end, so the audio between them is decoded with the next segment. The model window budget uses the larger
of preroll and bridge. Model-free coverage of the whole video: 1958 -> 1987 s. Real model, whole video: no gap above 1 s, no final
suppressed, 29385 -> 29743 characters; the gap speech now appears in the text. One segment (around 1155 s, which starts with a
2.3 s bridged stretch) came out shorter than before (22 vs 74 characters) while its neighbour gained text, so bridging can also
make an individual decode worse. Fixture replays with the bridge (`mlx-bridge/`, suppression on, adaptive): clean CER 0.031
(0.025 without the bridge in `mlx-floor-p10/`, 0.035 in the original table; a difference of a few characters), fan + speech
0.030, babble + speech 0.035, fan alone 0.56 false segments per minute with every final suppressed, music bed alone 0.
Test: `a_short_gap_between_segments_is_bridged`.
