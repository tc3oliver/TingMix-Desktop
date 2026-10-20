# Qwen3-ASR-0.6B as a third engine (2026-10-20)

Status: **EXPERIMENTAL**. The backend reproduces the export's reference implementation token for token on the fixtures, on
macOS and Windows. Accuracy on real zh-TW speech was measured on the 55 error-attribution clips, against references that are
still provisional.

## What it is

Qwen3-ASR (Qwen team, Alibaba; Apache-2.0) is an audio encoder in front of a Qwen3 language-model decoder. It covers 30
languages and 22 Chinese dialects and writes punctuation. The app uses the ONNX export by `andrewleech/qwen3-asr-onnx`
(`andrewleech/qwen3-asr-0.6b-onnx` at `4fc24a1`, int4 RTN decoders, FP32 encoder, FP16 embedding table, 2.0 GB). It runs on
the ONNX Runtime the app already bundles (1.31.0, CPU).

Pipeline (`crates/tingmix-runtime-onnx/src/qwen3.rs`), following the export repository's `src/{mel,prompt,inference}.py`:

1. Whisper log-mel: periodic Hann 400, reflect-padded centred STFT with hop 160, 128 Slaney mel filters (0 to 8 kHz),
   `log10`, floor at max − 8, `(x + 4) / 4`, last frame dropped.
2. Encoder `mel [1,128,T]` to `audio_features [1,N,1024]`, with N = 13 per 100 frames plus three stride-2 convolutions over
   the rest.
3. Chat prompt: system, user `<|audio_start|>` + N × `<|audio_pad|>` + `<|audio_end|>`, assistant. `decoder_init` gets the
   ids, positions, features and pad offset, and returns logits and the KV cache.
4. Greedy decoding through `decoder_step`: the next token's embedding row from `embed_tokens.bin`, and the KV cache handed
   back unchanged (`SessionOutputs::remove`, no copy). It stops at `<|im_end|>` or `<|endoftext|>`, or after 32 + 15 tokens
   per second of audio.
5. The answer is `language <name><asr_text><text>`; only the text is kept. Byte-level BPE is decoded from `vocab.json` with
   the GPT-2 byte table, so no tokenizer library is needed.
6. In the app only: OpenCC `s2tw`.

**One deviation from the reference: the first generated token is fixed to `language`.** The int4 decoder quantises its
activations (MatMulNBits accuracy level 4), so the first token is extremely sensitive to the input. On zh_02s the
reference's own margin between `language` and a wrong piece (`ology`) is 0.40 logits. Uniform mel noise of 1e-5 changes the
encoder output by 3e-6, yet it flips that token in 2 of 10 trials, and the window then comes out empty. Every answer starts
with `language`, so fixing that token loses nothing, and the model still chooses the language.

## Parity (`experiments/qwen3-asr/ref_onnx.py`, `crates/tingmix-runtime-onnx/tests/qwen3.rs`)

Oracle: the export repository at `4aee2ae` (its own mel, prompt and greedy loop) on the same int4 graphs, Python onnxruntime
1.23.2.

- Log-mel: largest difference 4.4e-5 on macOS and 6.9e-5 on Windows.
- Token ids: identical on all 7 fixtures of at most 30 s, on macOS (pyke and vendored 1.31.0) and on Windows 11 (Ryzen 7
  9700X, vendored 1.31.0). mix_29s is longer than one window and is not compared.

## Accuracy on real zh-TW speech (`benchmarks/qwen3-asr/real-zh-tw-55clips-onnx.json`)

These are the same 55 clips, references and normalisation as [error-attribution.md](error-attribution.md) and
[sensevoice.md](sensevoice.md).

| engine | clips MER | product vs subtitles | live path MER | live WER en | RTF (M4 Max, 4 threads) | peak memory |
|---|---|---|---|---|---|---|
| Qwen3-ASR-0.6B int4 (the app) | 0.094 | 0.108 | **0.091** | **0.224** | 0.15 | 3.4 GB |
| Qwen3-ASR-0.6B, FP32 decoders | 0.088 | 0.092 | — | — | 0.35 to 0.49 | 7.0 GB |
| Qwen3-ASR-0.6B, MLX fp16 (Python, reference only) | 0.084 | 0.092 | — | — | 0.02 to 0.03 (GPU) | — |
| SenseVoice-Small | 0.102 | 0.116 | 0.103 | 0.506 | 0.011 | 0.5 GB |
| TingMix-Large | 0.212 | 0.277 | 0.244 | 0.647 | — | about 2 GB |

- On the live path, against the product subtitles: Qwen3 0.098, SenseVoice 0.111, TingMix-Large 0.290.
- int4 costs about 0.6 points against the FP32 decoders, but it is 2 to 3 times faster and needs half the memory. FP32 on the
  CPU sits close to the live scheduler's 0.5 load limit, so the app ships int4.
- English words are where Qwen3 differs most: live WER 0.22 against 0.51 for SenseVoice. Code-switched speech is common in
  the material this app is for.
- Speed: 6.6× faster than real time on the CPU. See the next section for live latency.
- Caveats: provisional Whisper references (an autoregressive model may favour another one; the subtitle column does not
  depend on Whisper and shows the same order), two videos, no significance test.

## Live latency (`benchmarks/qwen3-asr/live-latency.json`)

Live captions re-decode the open segment for every partial. In continuous speech a segment grows to the 25 s cap, and
Qwen3 got slower with it: about 2 s per decode at 25 s on 4 threads, so captions lagged up to 2 s and a final came up to
3 s after the speech. Most of that was writing the whole answer again (61 tokens) and the prefill (1.1 s).

Two changes:

1. A window that extends the previous one (the next partial, or the final of the same segment) keeps the previous answer
   minus its last 10 tokens as a prefix and only generates the rest. Without a guard this sent two of 11 regions into
   repetition loops (MER 0.211), so an answer that writes more than the new audio can hold is thrown away and decoded from
   scratch.
2. `SessionConfig::live_recommended()` gives partials 0.8 of real time instead of 0.5 (owner decision: fresher captions
   over CPU use). 1.0 updated more often but each decode then waited longer.

Measured with `live-latency` (8 threads, M4 Max, recording fed in real time):

| | partials | mean lag | max lag | final delay (max) |
|---|---|---|---|---|
| before, product r0 (25 s of continuous speech) | 35 | 0.70 s | 1.96 s | 3.19 s |
| after, product r0 | 60 | 0.52 s | 1.18 s | 1.67 s |
| before, finance r5 | 36 | 0.68 s | 1.93 s | 1.19 s |
| after, finance r5 | 63 | 0.56 s | 1.16 s | 1.18 s |

Live-path accuracy stays where it was: MER 0.092 against 0.091, 0.099 against 0.098 on the subtitles. English WER moved
from 0.224 to 0.247, within the run-to-run spread of the few English words in this set. A shorter segment cap was the
other option, and cost more (8 s cap: MER 0.095, English WER 0.318).

What is left is the prefill: the decoder reads every audio token again on each partial (1.1 s at 25 s on 4 threads).
Reusing the audio part of the KV cache would remove it, but the Whisper mel is normalised over the whole window, so earlier
features change as audio arrives.

## Licence

The Qwen3-ASR weights are Apache-2.0, and so is the export repository. The ONNX files are a modified form of the weights
(graph export, int4 quantisation), which the catalog entry states in its licence notes. No acceptance step is required.

## Publishing

`crates/tingmix-runtime-onnx/fixtures/qwen3-asr-0.6b-onnx-int4.signed.json` is the catalog entry: `min_app_version` 0.2.0,
signed with `tingmix-release-1`. Its files download from the export repository at the pinned revision. It needs 8 GB of
RAM (peak 3.4 GB plus the app and the OS). First run offers it ahead of the other models where it fits
(`setup::first_run_rank`).

## Open questions

- Faster inference: the Mac's GPU through CoreML, or MLX directly (the MLX port reaches RTF 0.03), would make the FP32
  accuracy affordable.
- Language hints (`language Chinese` forced) and context biasing (the prompt's system turn accepts context text) are not
  used yet.
- Human-verified references (#5) may move all numbers.
