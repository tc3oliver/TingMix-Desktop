# TingMix-Large on ONNX Runtime (Windows path)

Status words follow `CLAUDE.md`. Owner decision 2026-10-18: Windows transcribes with Large. Everything here is **EXPERIMENTAL**:
parity and speed are measured with the real TingMix-Large weights, but only on 8 synthetic TTS clips (no real-speech accuracy
set) and on one Windows PC. Weights: `google/embeddinggemma-2@914f7f8` audio tower plus `TingMix-Large@75071a1` (LoRA merged, fp32).

| Item | Status |
|---|---|
| mel to CTC logits ONNX graph with a dynamic frame axis matches the PyTorch fp32 oracle | VERIFIED on 8 clips / 9 chunks (rel-L2 <= 1.7e-6, greedy ids 100 %) |
| Rust backend `OnnxLarge` (Rust Gemma4 mel, ORT, Rust CTC/LM decode) reproduces the oracle texts | VERIFIED 8/8, greedy and beam+LM, on CPU (macOS arm64, Windows x64), DirectML and CUDA |
| Speed on the Windows PC (Ryzen 7 9700X, RTX 5070 Ti), CPU / DirectML / CUDA | EXPERIMENTAL (one machine, synthetic clips) |
| Compatibility probe (`OnnxProbeTarget`) and signed package manifest | VERIFIED passing on macOS and on Windows for all three providers |
| Real-speech accuracy, other CPUs/GPUs, fp16/int8 variants, bundled-installer size, signing | NOT DONE |

## The graph (`experiments/windows/large-onnx`)

`uv sync && uv run python export.py` writes `out/tingmix-large.onnx` (1 310 190 728 bytes, opset 17, torch 2.9.1 legacy exporter,
not in git; sha256 in `crates/tingmix-runtime-onnx/fixtures/*.json`). Input `mel [1, F, 128]` (log-mel of the **valid** frames, what
`Gemma4Frontend::extract` reports as `valid`), output `logits [1, T, 5025]`, `T = ceil(ceil(F/2)/2)`, CTC blank 0. `F` is dynamic, so
one file serves every window length up to 29.5 s (2949 frames) with no padding (the Reader's GRU is not length-aware, so padding
would change the output; docs/research/coreml.md measured 1.6 % of ids).

Rewrites of the HF modules, all exact (torch outputs compared before export; `check_torch.py`):

1. Blocked local attention without `Tensor.unfold`: two static slices of the chunk-padded, left-padded sequence; only the number of blocks depends on the length.
2. The 5-D blocked mask is built in the graph from the length (the Core ML study could use a trace-time constant; a dynamic axis cannot). Equal to the HF mask for all query rows that exist, checked for T = 1, 5, 12, 13, 24, 25, 188, 738. The sliding window is the 12 keys q-11..q (an off-by-one here gave 13 keys and wrong logits).
3. The attention matmuls are kept 4-D (heads and blocks merged into one batch axis). DirectML rejects the 5-D batched MatMul with `E_INVALIDARG` (80070057); the merge is exact (ONNX CPU parity unchanged).
4. Mask fill -3e4 (fp16-safe), `RMSNorm` on x/16 (bit-exact in fp32, needed by any fp16 variant), ReLU as `clamp(min=0)`, no `pack_padded_sequence` in the Reader.

Parity against `experiments/macos/reference/out` (CPU fp32 oracle): `parity.py` over all 9 chunks (clips 1.4 to 34.6 s; the 34.6 s clip is two
chunks) gives rel-L2 6.7e-7 to 1.7e-6, max abs 3.3e-5 to 8.8e-4, greedy ids equal on every chunk (`out/parity-cpu-mac.json`, not committed).

## Rust backend (`crates/tingmix-runtime-onnx`, `OnnxLarge`)

`OnnxLarge::load(model.onnx, units.json, Decode::{Greedy | BeamLm}, Provider::{Cpu | DirectMl | Cuda}, threads)`. Chunking, joining and
confidence follow `MlxLargeBackend` (29.5 s windows, tails under 0.1 s dropped, CJK joins with "", least-confident chunk decides).
A requested provider must initialise (no silent CPU fallback in the Rust API). `tests/large.rs` (env `TINGMIX_LARGE_ONNX`,
`TINGMIX_LARGE_DIR`) checks greedy and beam+LM text against `tests/fixtures/large/expected_texts.json`: **8/8 on both**, on macOS arm64
CPU and, with the Microsoft ORT 1.31.0 dll loaded by absolute path, on Windows for CPU, DirectML (wheel build 1.24.x) and CUDA (wheel build 1.31.0).
The ORT environment is now process-wide (ORT allows one; two backends in one process failed before).

## Windows measurements (Ryzen 7 9700X 8C/16T, 62 GB, RTX 5070 Ti 16 GB, Windows 11 26200)

Raw: `benchmarks/windows/large-onnx-rust-*.json` (Rust, wav in, 10 timed runs after 2 warm-ups, 8 threads) and
`benchmarks/windows/large-onnx-eps-*.json` (ORT Python wheels, model part only, with parity for every chunk). Clips: `mix_08s` (7.5 s of audio)
and the first 29.5 s window of `mix_29s`. "model" = Rust mel + `session.run`; "wav to text" adds the CTC decode.

| Provider (ORT build) | Window | model p50 / p95 ms | RTF (model) | wav to text greedy / beam+LM p50 ms | logits vs CPU oracle |
|---|---|---|---|---|---|
| CPU (Microsoft 1.31.0, load-dynamic) | 7.5 s | 237 / 246 | 0.032 | 235 / 268 | rel-L2 8.5e-7 |
| CPU | 29.5 s | 820 / 848 | 0.028 | 835 / 977 | |
| DirectML (wheel 1.24) | 7.5 s | 43 / 47 | 0.006 | 44 / 73 | rel-L2 1.5e-6 |
| DirectML | 29.5 s | 1152 / 1198 | 0.039 | 1181 / 1305 | |
| CUDA (wheel 1.31.0, pip cu13 + cuDNN) | 7.5 s | 31 / 39 | 0.004 | 32 / 64 | rel-L2 4.2e-4 (TF32), ids 100 % |
| CUDA | 29.5 s | 83 / 85 | 0.003 | 93 / 220 | |
| for scale: macOS M4 Max CPU (ORT 1.31.0) | 29.5 s | 293 | 0.010 | 308 / 393 | rel-L2 <= 1.7e-6 |

Observations that bound the conclusions:
- DirectML in the Python wheel took 337 ms (7.5 s) and 1190 ms (29.5 s) while the Rust run took 43 ms and 1152 ms; I did not find why
  the short window differs 8x (not tested: session options such as memory pattern / parallel execution, which DirectML documents as needing to be off).
  So the DirectML 7.5 s number is not stable evidence, and at 29.5 s it is slower than the CPU in both runs. Some nodes (shape ops, GRU) stay on the CPU EP, so strict mode (`disable_cpu_ep_fallback`) refuses the session for DirectML and CUDA.
- CUDA uses TF32 matmuls by default (rel-L2 4e-4 against the fp32 oracle; ids and texts unchanged).
- Both GPU paths need extra runtime files: CUDA about 1.6 GB of NVIDIA DLLs (cuBLAS 13, cuFFT, cuDNN 9, NVRTC, ...) plus 204 MB of ORT CUDA provider libraries, NVIDIA GPUs only. DirectML adds `DirectML.dll` (17.7 MB) and needs a DirectML build of `onnxruntime.dll` (the pinned Microsoft CPU build has no DML EP).
- First load of the model and LM: 3.6 to 6 s. Peak memory with two backends (greedy plus beam, i.e. two sessions) in one process on macOS: 4.2 GB; one backend with LM is about 2 to 2.5 GB.

### Default execution provider: CPU

Reasons from the numbers: it works on every machine with the pinned, redistributable Microsoft ORT 1.31.0 (no extra DLLs, offline), it is
exact (rel-L2 1e-6), and it is fast enough: a 29.5 s window decodes in 0.8 to 1.0 s on this PC (RTF 0.03). For live captions the re-decode
of the open segment costs 0.24 s at 8 s and about 0.6 s at 20 s; the session's hop stretches with measured cost, and the tail-partial policy
means a final adds no extra decode, so CPU is workable here (a slower CPU than a Ryzen 7 9700X is not measured). DirectML is **not**
recommended: slower than CPU at 29.5 s, partly on the CPU EP. CUDA is **opt-in**: 10x faster at 29.5 s, but only after the probe passes
(`onnx-cuda` has its own probe record) and only where the CUDA libraries exist; shipping them (1.6 GB) is a product decision, not a default.

## Probe and signed manifest

`OnnxProbeTarget` (feature `probe`) runs the graph in a child process, `tingmix-onnx-worker`, so a driver or runtime crash cannot take the
app down. Backend ids `onnx-cpu`, `onnx-directml`, `onnx-cuda` (equal to `Manifest::backend`) each get their own probe record; the version is
read from the loaded library (`runtime_version()`), the device is `cpu` / `gpu:directml` / `gpu:cuda`. Probe of `en_02s` ("good morning everyone",
beam+LM): passed on macOS CPU and on Windows for all three providers (rtf x1000 on the 1.4 s clip including pipe transfer: Windows CPU 61,
DirectML 28, CUDA 17; first run 5.3 to 6.4 s).

`crates/tingmix-runtime-onnx/fixtures/tingmix-large-onnx-fp32.signed.json` (signed with the `tingmix-dev-1` fixture key by `manifest-sign`; unsigned
source next to it): id `tingmix-large-onnx-fp32-r75071a1`, backend `onnx-cpu`, os/arch `any`, runtime `onnxruntime >= 1.31.0`, 6 files (the
`.onnx` imported offline, tokenizer/LM/meta from the pinned upstream revision), `modified: true` provenance (converter `experiments/windows/large-onnx/export.py`
at its commit), the same licence entries as the MLX package (embeddinggemma-2 acceptance), compat status EXPERIMENTAL. `tests/probe.rs` verifies the signature,
validates the manifest, checks tampering is rejected and, when the exported graph is present, that its size and sha256 equal the manifest's.
Re-exporting changes the sha256: re-sign the manifest and redo the `release` key signing (task "Model hosting").

## Notes for the app wiring (src-tauri was not touched)

1. A Large package is told apart by `manifest.family == "tingmix-large"`; build it with `OnnxLarge::load(graph, units.json, Decode::BeamLm(lm, BeamConfig::default()), Provider::Cpu, threads)` instead of `OnnxSmall`.
   The `.onnx` file is the manifest entry with component `model`; `units.json` and the LM are `tokenizer` and `lm` components as for MLX.
2. `tingmix_runtime_onnx::runtime_version()` gives `BackendReg.runtime_version` (the ORT library's own version string).
3. Probe: `OnnxProbeTarget::new(worker, Provider::Cpu, threads, true)`; ship `tingmix-onnx-worker(.exe)` next to the app executable
   (`OnnxProbeTarget::sibling_worker()`). In `load-dynamic` builds the child needs `TINGMIX_ORT_DYLIB` in its environment: set it in the app
   process before the first probe (the child inherits it) or the worker fails with "no ONNX Runtime library configured".
4. Memory: set the CPU Large requirement to 6 GiB (manifest); one preloaded backend with LM is about 2 to 2.5 GB, never preload two copies.
5. GPU providers only through their own backend ids after their probe passed; enabling them needs the cargo features `directml` / `cuda` on `tingmix-runtime-onnx` and an ORT library that contains the provider.

## Not done / open

- Real-speech accuracy and noise robustness of this graph (same weights as MLX, so expected equal; unmeasured). fp16/int8 variants (the RMS pre-scale is in place for fp16; not exported or measured).
- Why DirectML at 7.5 s differs between the Python wheel and the Rust run; any tuning of ORT CPU options (threads other than 8, `intra_op_spinning`); CPUs weaker than the 9700X.
- The `ort` DirectML/CUDA Rust path was only exercised with `load-dynamic` against wheel DLLs; the pyke prebuilt (`download-binaries`) EP packages were not tried.
- 1.3 GB package size and distribution (hosting, signing with the release key) are task "Model hosting".
