# MLX for TingMix-Large: Python parity, C++ port, Rust FFI

Status: **EXPERIMENTAL**. The graph is implemented in MLX (Python and C++) and matches a PyTorch CPU fp32 run on 8 synthetic
fixtures; a Rust-only path (Rust log-mel, MLX C++ tower+reader, Rust CTC/LM decoder behind `AsrBackend`) reproduces the oracle
texts on all 8 clips. It is not **VERIFIED**: no real-speech accuracy set was run (fixtures are synthetic TTS), and the shipped
build question (Metal toolchain, bundling) is open.

Pins: TingMix-Large `75071a1`, google/embeddinggemma-2 `914f7f8`, mlx-vlm `73d366e` (gemma4 `audio.py`), mlx 0.32.4,
torch 2.14.1 / transformers 5.19.0 / embed2asr `b582757` / tingmix `f44cbdb` for the oracle. Raw numbers:
`benchmarks/apple/mlx/` (`parity.json`, `bench-*.json`, `e2e.json`, rendered in `README.md`).

## Pipeline (experiments/macos/mlx)

| Step | File | What |
|---|---|---|
| oracle | `ref_torch.py` | embed2asr/tingmix on CPU fp32: mel, 1536-d `last_hidden_state`, reader logits, greedy and beam+LM text (alpha 0.35, beta 1.5, beam 16) |
| convert | `convert.py` | reads only `audio_tower.*` from the base file; merges `encoder_delta` LoRA (r16, alpha32, scale 2) in fp32 into 96 linear weights; Conv2d OIHW to OHWI, depthwise Conv1d (C,1,K) to (C,K,1); clip buffers (480 scalars) kept; reader conv (O,I,K) to (O,K,I); GRU repacked to MLX `Wx/Wh/b/bhn` (`b = b_ih + [b_hh_r, b_hh_z, 0]`, `bhn = b_hh_n`). Output is one fp32 safetensors, 1308 MB, 304,825,088 tower + 22,167,969 reader params (matches the 304,824,608 + 480 clip scalars and ~22.2M in the audit), no text/vision tower. sha256 in `out/*.sha256` (artifact itself is not in git) |
| model | `tingmix_mlx.py` | mlx-vlm `AudioEncoder` unchanged + a new Reader (conv blocks, 2-layer BiGRU, linear). fp16/bf16 are cast at load; clip bounds and `per_dim_scale` stay fp32 |
| parity | `parity.py`, `e2e.py` | features, logits, greedy and beam+LM text per dtype; e2e starts from the wav with the numpy mel port |
| bench | `bench.py` | wav to greedy text, one process per dtype |
| C++ | `crates/tingmix-runtime-mlx/cpp` | same graph on the MLX C++ API, C ABI in `tingmix_mlx.h` |

## Parity (vs PyTorch CPU fp32, 8 fixtures, 1.4 to 34.6 s clips, windowed to 29.99 s)

Worst case over fixtures; per-fixture rows in `benchmarks/apple/mlx/parity.md`.

| dtype | feature max-abs | feature cos min | logit max-abs | argmax-frame agree | greedy text == oracle | beam+LM text == oracle |
|---|---|---|---|---|---|---|
| fp32 | 6.0e-4 | 1.000000 | 6.2e-4 | 1.0000 | 8/8 | 8/8 |
| fp16 | 6.5e-2 | 0.999997 | 8.5e-2 | 1.0000 | 8/8 | 8/8 |
| bf16 | 4.6e-1 | 0.999837 | 2.9 (zh_16s) | 0.9980 | 8/8 | 8/8 |

Feature magnitudes are O(1) to O(10); bf16 max-abs outliers are sparse (cos stays above 0.9998). Mel front end: the mlx-vlm
numpy port differs from the HF extractor by at most 8.4e-4 absolute on log-mel; raw wav to greedy text matched the oracle on
24/24 (3 dtypes x 8 fixtures, `e2e.json`). The oracle's own errors on these fixtures (for example `comm` for `calm`, `pn`
for `PM`) are reproduced, which is the point of parity; this says nothing about WER on real speech.

Not covered: batch above 1 or padded batches (the encoder is not padding-invariant, see embed2asr notes), streaming, real
speech corpora, audio longer than the 29.5 s window.

## Latency, B=1, M4 Max 64 GB, macOS 27.0 (other agents ran GPU/CPU work concurrently, loadavg 7 to 8)

fp32, GPU, 30 timed runs after 3 warmups; full tables for all dtypes in `benchmarks/apple/mlx/README.md`.

| audio s | P50 ms | P95 ms | P99 ms | RTF P50 | mel / tower / reader+greedy ms |
|---|---|---|---|---|---|
| 2 | 20 | 25 | 35 | 0.010 | 1 / 13 / 7 |
| 4 | 29 | 30 | 30 | 0.007 | 1 / 15 / 13 |
| 8 | 48 | 53 | 55 | 0.006 | 2 / 22 / 25 |
| 16 | 85 | 89 | 97 | 0.005 | 3 / 34 / 49 |
| 29 | 154 | 178 | 179 | 0.005 | 5 / 54 / 99 |

Memory: process peak RSS 1.8 GiB (1.6 GiB right after load); MLX allocator peak 1.7 GiB fp32 at 29 s, 1.35 GiB for fp16/bf16.
For scale, the PyTorch CPU fp32 oracle encoder alone took 1.2 to 1.6 s per clip on this machine.

GPU use: the MLX default device was `gpu` and Metal was available. Control: the same graph on the MLX CPU stream took
639 to 660 ms at 29 s (4.2x slower) with 2.9 GiB RSS, so the GPU numbers are not a CPU path. Not measured: GPU utilization
counters, power and thermals (`powermetrics` needs sudo), streaming backlog, First Partial / Final P95 (needs the
session). CPU cores busy is about 0.9 (one dispatch thread).

Findings:
- The sequential GRU is now the largest block: 99 of 154 ms at 29 s (1500 dependent steps launched from the host). The tower
  is 54 ms. A fused GRU Metal kernel (`mx.fast.metal_kernel` or a C++ custom primitive) is the obvious next step; not done.
- fp16 and bf16 are slower than fp32 (P50 188 and 174 ms vs 154 ms at 29 s) and save memory only (1.35 vs 1.71 GiB): attention
  runs in fp32 and mixed-dtype promotion (fp32 clip bounds against half weights) adds casts. Use fp32 unless memory forces otherwise.
- Cold first run 182 ms vs 154 ms warm (no large JIT penalty); model load 0.09 s (lazy mmap).

## Can the same graph run from C++ behind a Rust FFI? Yes, proven for batch 1

`crates/tingmix-runtime-mlx` implements the whole tower and reader on the MLX C++ API (about 330 lines) and exposes five C
functions (`tm_load`, `tm_free`, `tm_vocab`, `tm_out_frames`, `tm_run`). Rust wraps them in `MlxLarge::load` / `logits`.

Measured through Rust (`cargo test -p tingmix-runtime-mlx`, with `MLX_PREFIX`, `TINGMIX_MLX_ARTIFACT`, `TINGMIX_MLX_REF`):
logits max-abs vs the PyTorch oracle 2.5e-5 (zh_02s), 4.2e-5 (en_02s), 1.3e-4 (mix_04s), 9.3e-5 (mix_29s). The standalone C++
driver (`experiments/macos/mlx/cpp/build.sh`) gives, for fp32 over all 8 fixtures, feature max-abs at most 4.7e-4, logit
max-abs at most 7.9e-4, argmax agreement 1.0, warm 29 s run 153 ms (same as Python, so Python dispatch is not the bottleneck).
fp16 and bf16 also run in C++ with the same error scale as Python. Every op used exists in the C++ API: `conv2d`,
`conv_general` with groups, `fast::rms_norm`, `fast::layer_norm`, `take`, `pad`, `softmax`, `erf`, `logaddexp`, `matmul`.
No custom kernel and no Python are needed. mlx-c was not evaluated; the C++ API plus a six-function C ABI was simpler.

Without `MLX_PREFIX` (or off aarch64 macOS) the crate builds a stub whose `load` returns `NotBuilt`, so the workspace still
builds and tests on other platforms and in CI.

### Gaps before this is a shippable backend (so it is **not** declared native-ready)

1. (closed in the section below) Rust log-mel front end and `AsrBackend` impl exist; they are not yet wired into the app's runtime selection.
2. Distribution. The prebuilt libmlx comes from the PyPI wheel: `libmlx.dylib` 21 MB, `libjaccl.dylib` 2 MB, `mlx.metallib`
   216 MB, which must sit next to the library or executable. A reproducible shipped build needs MLX built from source (MIT) in
   CI, pinned to a tag, and that needs the Xcode Metal toolchain. It is **missing on this machine** (`xcrun metal` fails:
   "missing Metal Toolchain"; fix is `xcodebuild -downloadComponent MetalToolchain`), so a source build was not attempted
   and the metallib was not trimmed. App size impact is about 240 MB unless the metallib is pruned to the used kernels.
3. Artifact: 1.3 GB fp32 safetensors loaded by MLX's own reader. Not yet signed/registered (model-registry task) and no
   smaller on-disk format (fp16 on disk was not produced; fp16 compute is slower anyway).
4. Single-window only (29.5 s); longer audio needs the chunking policy from `AsrBackend` callers. Batch 1 only.
5. Graph is rebuilt per call (no `mx.compile`); GRU is host-launched step by step (see findings).
6. Sandbox/notarization of a Metal-linked dylib inside a Tauri bundle, and behavior under App Sandbox, were not tested.
7. All numbers were taken on a loaded machine and on synthetic TTS fixtures; re-run on an idle machine and on real speech
   before using them as release gates.

## Native Rust path: wav to text (task #12)

Crates: `tingmix-frontend-gemma` (log-mel), `tingmix-runtime-mlx` (`MlxLargeBackend: AsrBackend`, `Decode::{Greedy, BeamLm}`),
decoding by `tingmix-decoder`. Chunking follows `embed2asr` `transcribe`: 29.5 s windows (472 000 samples), tails under
1 600 samples dropped, joined with `""` for CJK-only output and `" "` otherwise, each chunk encoded alone (batch 1).

**Mel parity** (`tingmix-frontend-gemma/tests/oracle.rs`, `TINGMIX_ORACLE_DIR=experiments/macos/reference/out`): all 9 chunks of
the 8 fixtures, frame count and valid-frame count equal the dump's `c{k}_mel` / `c{k}_mel_mask`, max-abs difference 4.8e-7 on
log-mel (gate 1e-5). The port mirrors transformers 5.19.0 `Gemma4AudioFeatureExtractor`: 160-sample left pad, right pad to a
multiple of 128, 321-sample unfold with the last sample dropped (no preemphasis), periodic Hann(320), FFT 512, magnitude, HTK
filters in Hz space (`mel_filter_bank`, no norm), `ln(mel + 1e-3)`, padded frames zeroed. FFT and mel run in f64 (numpy: f32).

**End to end** (`tingmix-runtime-mlx/tests/e2e.rs`): from the wav bytes through Rust only, greedy and beam+LM (0.35/1.5/16) text
equals `tests/fixtures/large/expected_texts.json` on 8/8 clips, 16/16 comparisons, fp32. Logits through the same Rust path were
already within 1.3e-4 of the dumps (`tests/parity.rs`, my own dumps) and the MLX fp32 Python graph within 2.1e-4 of the
reference task's dumps (`experiments/macos/mlx/cross_check.py`).

**Rust-side timings** (`cargo run --release -p tingmix-runtime-mlx --example bench`, raw `benchmarks/apple/mlx/rust-bench.json`),
fp32, M4 Max, 30 runs after 3 warmups, loadavg 10 to 16 from other agents (a first run at loadavg 7 to 17 gave up to 2x higher
and noisier values, which is why contention must be assumed in every number here):

| audio s | mel P50 ms | greedy e2e P50 / P95 / P99 ms | beam+LM e2e P50 / P95 / P99 ms | RTF P50 greedy / beam |
|---|---|---|---|---|
| 2 | 0 | 20 / 21 / 22 | 24 / 25 / 27 | 0.010 / 0.012 |
| 4 | 1 | 30 / 55 / 120 | 40 / 48 / 84 | 0.008 / 0.010 |
| 8 | 2 | 51 / 125 / 137 | 71 / 165 / 206 | 0.006 / 0.009 |
| 16 | 3 | 92 / 135 / 174 | 135 / 146 / 149 | 0.006 / 0.009 |
| 29 | 6 | 162 / 168 / 168 | 249 / 256 / 259 | 0.006 / 0.009 |

Rust adds nothing visible over the Python bench (154 ms at 29 s): mel is 6 ms and the beam+LM decode about 90 ms at 29 s.
Model load 0.07 to 0.14 s, character LM load 2.4 s (about 0.55 GiB). Peak RSS: 1.27 GiB after model load, 1.84 GiB with LM,
3.3 GiB in the bench process because it holds two backends (greedy and beam, two copies of the weights); one backend with LM
and a 29 s decode is the figure to budget (about 1.9 to 2 GiB).

**Buffered streaming** through `SessionWorker` (default `SessionConfig`: 480 ms hop, 600 ms endpoint silence, max load 0.5),
fixtures replayed in 100 ms chunks at real time followed by 1.5 s of silence, then flush; every decode re-runs the whole open
segment. Clips: zh_02s, en_08s, mix_08s, zh_16s (1.7, 5.1, 7.5, 20.4 s of audio):
- First partial arrived 0.50 to 0.51 s after the stream start for 7 of 8 runs (0.61 s once), i.e. the 480 ms scheduling hop plus
  one decode of 24 to 70 ms, not model latency.
- Partial staleness (arrival minus the partial's audio end): P50 6 to 10 ms, max 63 to 204 ms, except two beam+LM runs with
  P50 105 ms (decode landing between chunks).
- Per-update decode cost grows with the segment: P50 24 / 36 / 38 / 72 ms and P95 62 / 43 / 51 / 132 ms for greedy on the four
  clips (beam+LM up to 136 ms P50, 194 ms P95 at 20 s). No partial was coalesced; decode RTF stayed far below the 0.5 budget.
- Final arrived 0.68 to 0.91 s after the segment's audio end; that is the 600 ms endpoint silence plus a decode, so the
  model-side final latency is roughly 70 to 300 ms.
Not done: a 60-minute soak (task #14), power, and streaming of segments near the 25 s cap.

Remaining gaps are unchanged from the list above (Metal toolchain / source build and bundling, single window, batch 1,
no `mx.compile`, host-launched GRU), plus: no runtime-selection wiring and no Windows/other-platform equivalent.

## App integration: probe, worker, bundling (task #19)

**Probe** (`tingmix-runtime-mlx`, feature `probe`): `probe::MlxProbeTarget` implements `tingmix_models::probe::ProbeTarget`. It
spawns the `tingmix-mlx-worker` binary, which loads the model once and answers decode requests over stdin/stdout (request: u64
LE sample count + f32 LE samples; reply: `OK\t<text>` / `ERR\t<msg>`; first line `READY\t<mlx version>\t<device>`). A crash,
abort or hang in MLX/Metal kills the child, the parent returns an error (120 s timeout) and respawns on the next call.
`backend_version` and `device` come from the linked libmlx inside the child (`tm_runtime_info`), not from a guess.
Measured with `cargo test --release -p tingmix-runtime-mlx --features probe --test probe` (beam+LM, fp32, en_02s, 5 timed runs):
passed, device `gpu:Apple M4 Max`, backend_version `0.32.4`, `rtf_x1000` 15 (includes pipe transfer; the clip is 1.4 s),
first run 2.4 s (child start, model load and LM load). A separate run of the worker with LM on mix_08s: max RSS 1.85 GiB,
peak memory footprint 2.22 GiB, warm decode 64 ms.

**Bundling, tested**: a worker with its rpath rewritten to `@executable_path/../Frameworks` and `libmlx.dylib`, `libjaccl.dylib`,
`mlx.metallib` placed together in `Contents/Frameworks` decoded en_02s correctly from `/` as cwd. With `mlx.metallib` removed the
worker aborts (SIGTRAP) before `READY`; in-process that would take the app down, which is why decoding in production should use
the worker, or the in-process backend only after a successful probe. `build.rs` only sets the rpath on this crate's own
targets (the worker, tests, examples); a host binary linking the crate directly must add its own rpath. Ad-hoc `codesign -s -`
was applied after `install_name_tool`; Developer ID signing, hardened runtime and notarization of these three files were not tested.

**Stays EXPERIMENTAL**: parity and timings are on 8 synthetic TTS clips (no real-speech WER); libmlx is the PyPI wheel (no
source build; the Metal toolchain is not installed here); one machine (M4 Max) measured; fp32 only recommended; 29.5 s single
window, batch 1; the model file is not published and is installed by offline import of a locally converted artifact.
