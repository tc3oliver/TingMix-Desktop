# SenseVoice-Small as a second engine (2026-10-19)

Status: **EXPERIMENTAL**. The backend reproduces the reference implementation (sherpa-onnx). Its accuracy on real zh-TW speech
was measured on the 55 error-attribution clips, against references that are still provisional.

## What it is

SenseVoice-Small (Tongyi Lab, Alibaba; `FunAudioLLM/SenseVoiceSmall`) is a non-autoregressive encoder with a CTC head that
handles Mandarin, Cantonese, English, Japanese and Korean, with built-in language ID and inverse text normalisation
(punctuation, digits). The app uses the sherpa-onnx export
(`csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` at `2365bae`, `model.int8.onnx` 239 MB + `tokens.txt`).

It runs on the ONNX Runtime the app already bundles (1.31.0, CPU). No second runtime is needed: sherpa-onnx itself is not
linked. Its C++ library would bring a second copy of ONNX Runtime (1.28.2 in the 1.13.8 wheel) into the process.

Pipeline (`crates/tingmix-runtime-onnx/src/sensevoice.rs`), each step taken from sherpa-onnx 1.13.8:

1. Kaldi fbank: 25 ms / 10 ms with snip edges, DC removal, pre-emphasis 0.97, Hamming, 512-point power spectrum, 80 Kaldi-mel
   triangles from 20 Hz to Nyquist, `ln(max(x, f32::EPSILON))`. Samples are scaled to the int16 range, because the metadata
   says `normalize_samples = 0`.
2. LFR (`lfr.cc`, FunASR `apply_lfr`): stack 7 frames centred on every 6th frame. The first frame is repeated 3 times on the
   left, the last frame past the end, and the output has `ceil(n / 6)` frames. Older sherpa-onnx releases did not pad. Using
   that variant changed the text of 1 of 10 clips.
3. CMVN from the model metadata (`neg_mean`, `inv_stddev`).
4. Graph inputs `x [1,T,560]`, `x_length`, `language` (metadata `lang_auto`), `text_norm` (`with_itn`). Output `logits [1,T+4,V]`.
5. Greedy CTC (blank 0). Then drop the first four labels (language, emotion, event, ITN), which are query tokens, not text.
   Pieces are joined SentencePiece-style.
6. In the app only: OpenCC `s2tw` (`ferrous-opencc`, Apache-2.0), because SenseVoice writes Simplified characters.

Windows of up to 30 s are decoded on their own, as TingMix-Large does with 29.5 s windows.

## Parity (`experiments/sensevoice`, `crates/tingmix-runtime-onnx/tests/sensevoice.rs`)

Oracle: sherpa-onnx 1.13.8 `OfflineRecognizer.from_sense_voice` (language auto, ITN on), on 10 clips: 3 of the export's test
clips plus the 8 synthetic fixtures. The LFR features are compared with kaldi-native-fbank 1.22.3 under the same settings.

- Features: the largest absolute difference over all frames of all clips is 2.4e-4.
- Text: 9 clips match exactly. On zh_16s an enumeration comma (、) differs; its logit leads by 0.109. Running our pipeline on
  sherpa's own ONNX Runtime (1.28.2) gives the same result, so the cause is feature noise at the 1e-4 level, not the runtime.
  The test ignores punctuation and allows at most one such clip.
- Same result with the pyke ONNX Runtime build (dev) and the vendored 1.31.0 (release).
- Same result on native Windows 11 (AMD Ryzen 7 9700X, vendored ONNX Runtime 1.31.0): features within 7.7e-4, the same one
  comma on zh_16s.

## Accuracy on real zh-TW speech (`benchmarks/sensevoice/real-zh-tw-55clips.json`)

These are the same 55 clips (647 s, two videos), the same references and the same normalisation as
[error-attribution.md](error-attribution.md) section 2. The reference texts went through OpenCC s2t, as the TingMix output did,
so Simplified output costs nothing here.

| engine | MER | CER zh | WER en | S | D | I |
|---|---|---|---|---|---|---|
| SenseVoice-Small (int8, greedy) | **0.102** | 0.092 | 0.518 | 0.050 | 0.019 | 0.032 |
| TingMix-Large, the app's beam + LM | 0.212 | 0.203 | 0.600 | 0.094 | 0.111 | 0.006 |

- By video: product review 0.120 vs 0.273, finance show 0.085 vs 0.158.
- Against the product video's subtitles, an independent reference: 0.116 vs 0.277.
- SenseVoice deletes far less (D 0.019 vs 0.111) and inserts more (I 0.032 vs 0.006). Some of the insertions are fillers
  that the subtitles omit.
- Caveats: the references are Whisper large-v3 drafts until a human checks them, the sample is two videos, and no significance
  test was run. The subtitle check is independent of Whisper and points the same way.

Live path: the app's own live captioning path (`SessionConfig::live_recommended()`, VAD segments, 25 s segment cap) on the
11 regions, scored like error-attribution section 3:

| live path | MER | D | I |
|---|---|---|---|
| SenseVoice-Small, cap 25 s (today's preset) | **0.103** | 0.018 | 0.030 |
| TingMix-Large (MLX), cap 25 s | 0.244 | 0.150 | 0.007 |
| SenseVoice-Small, cap 6 s | 0.118 | 0.025 | 0.031 |
| TingMix-Large (MLX), cap 6 s | 0.209 | 0.088 | 0.014 |

SenseVoice loses almost nothing between clips and the live path (0.102 to 0.103), while TingMix-Large loses 3.2 points. The 6 s
cap that helps TingMix-Large makes SenseVoice worse, so the segment cap should stay where it is or become per engine. Against
the product subtitles the live path gives 0.111 against 0.290.

Speed: on an Apple M4 Max, 4 CPU threads, about 0.011× real time (647 s of audio decoded in 7.3 s). Load takes 0.3 to 0.8 s and
peak memory is 0.5 GB. TingMix-Large on MLX needs about 2 GB.

## Licence

The weights are under the FunASR Model Open Source License Agreement 1.1. It allows free use, copying, modification and sharing
with attribution, as long as the model names are kept. It disclaims responsibility ("for reference and learning purposes"),
and the publisher may revise the terms. The catalog entry therefore requires acceptance before download, as the TingMix
entries do. The terms are not an OSI licence; the owner should read them before publishing.

## Publishing (owner)

`crates/tingmix-runtime-onnx/fixtures/sensevoice-small-onnx-int8.manifest.json` is the unsigned catalog entry. Its files
download straight from the sherpa-onnx repository at the pinned revision, so nothing is re-hosted.

1. Set `min_app_version` to the first release that contains this backend. Older builds would otherwise list the entry and
   fail its test.
2. `cargo run -p tingmix-models --bin manifest-sign -- sign <release key> tingmix-release-1 <manifest> <signed>`
3. Append the signed entry to `catalog.json` in the model repository.

The entry is `os: any`, `arch: any`, so it also covers Windows x86_64, where the catalog has no TingMix package yet. Parity
was checked on Windows; the app's install, probe and live path were run on macOS only.

## Compared with Qwen3-ASR

See [qwen3-asr.md](qwen3-asr.md). On the same clips Qwen3-ASR-0.6B (the app's int4 build) scores 0.094 and 0.091 on the
live path, mostly from better English. The 1.7B model (MLX, `benchmarks/qwen3-asr/real-zh-tw-55clips-mlx.json`) adds almost
nothing over 0.6B and is 2 to 7 times slower.

## Open questions

- Human-verified references for the 55 clips (error-attribution section 8). They could move both engines' numbers.
- Hotwords and phrase biasing: SenseVoice has none in this export.
