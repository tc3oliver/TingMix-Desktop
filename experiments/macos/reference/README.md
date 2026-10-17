# TingMix-Large CPU fp32 reference (oracle)

Research harness (its own `uv` project; embed2asr and tingmix installed from git at the pinned commits, see
`docs/research/PINNED.md`). Status: VERIFIED for the 8 synthetic-TTS fixtures in `tests/fixtures/audio/`.

Regenerate (about 35 s on the M4 Max; weights are downloaded to `$HF_HOME`):

    cd experiments/macos/reference && uv sync && uv run python dump.py

Pipeline: `embed2asr.audio.load_audio` (ffmpeg or soundfile, 16 kHz mono f32) -> 29.5 s chunks (tails < 0.1 s dropped) ->
`AutoFeatureExtractor` (128-bin log-mel) -> EmbeddingGemma-2 `audio_tower` fp32 + TingMix-Large LoRA delta
(r16/alpha32, all 12 layers) -> `last_hidden_state` [T,1536] -> Reader (fp32) -> logits [T,5025] (blank id 0) ->
greedy CTC, and CTC prefix beam search + char 5-gram LM (alpha 0.35, beta 1.5, beam 16). Chunk texts are joined with
`""` (CJK only) or `" "` as in `Embed2ASR.transcribe`.

## Dump format: `out/<clip_id>.npz` (gitignored; `out/MANIFEST.sha256` lists sha256 of every file)

Absolute path: `<repo>/experiments/macos/reference/out/`

One group of arrays per 29.5 s chunk `k` (only `mix_29s`, 34.6 s, has two: `c0_*`, `c1_*`); T = valid frames:

| key | dtype / shape | meaning |
|---|---|---|
| `c{k}_mel` | f32 [1, F, 128] | feature-extractor output (input to the tower; F = 100/s of audio, unpadded) |
| `c{k}_mel_mask` | bool [1, F] | feature-extractor `input_features_mask` |
| `c{k}_features` | f32 [T, 1536] | tower `last_hidden_state` (before embed_audio), 25 fps, valid frames only |
| `c{k}_logits` | f32 [T, 5025] | Reader output (raw logits, not log-softmax) |
| `c{k}_greedy_ids` | i64 [N] | greedy CTC ids after collapse and blank removal |
| `c{k}_n_frames` | i64 scalar | T |

`out/results.json`: versions, command, LM settings and per-clip `ref` (TTS prompt text), `greedy`, `beam_lm` texts.
Model output is lowercase English, no punctuation, and the LM-unit placeholder behaviour of `embed2asr.lm` applies.

## Committed goldens (`tests/fixtures/large/`)

`{zh_02s,en_02s}_logits.npy` (f32 [T,5025]), `{zh_02s,en_02s}_greedy_ids.npy` (i64), `expected_texts.json` (all 8 clips,
greedy and beam+LM). Use these for decoder parity (logits in -> ids/text out) without running the encoder.

## Observed output (reference, not ground truth: the audio is synthetic TTS)

Short Mandarin and English clips are exact; code-switched and long clips contain recognition errors of the model itself
(e.g. `mix_04s` -> `...這個ai的設計`, `en_08s` -> `stays comm`). Beam+LM differs from greedy on `zh_16s` (`確保`) and
`mix_29s` (`然後`). Parity targets for ports are these dumps, not the TTS prompt text.

## Tolerances

Not yet fixed here; the MPS results in `benchmarks/apple/mps/` propose per-stage tolerances from measured error.
