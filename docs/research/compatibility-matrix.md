# Runtime compatibility matrix

DRAFT for the lead to verify (task #11). Built only from committed documents and benchmark JSON; every number carries `file@commit`
(short hash of the last commit that touched the file, `git log -1 -- <file>` on 2026-10-18). Nothing was re-measured for this document.
Items that depend on uncommitted work or on tasks still running are marked `TODO(lead)`.

Status words (CLAUDE.md): **VERIFIED** = measured with the real TingMix model; **EXPERIMENTAL** = implemented or officially supported,
parity or speed on TingMix not (fully) established; **BLOCKED** = missing op, artifact or permission. "Not run" means nothing was
attempted. A VERIFIED/EXPERIMENTAL label here is never a claim about real-speech accuracy: all end-to-end parity uses the 8 synthetic TTS
fixtures (`tests/fixtures/audio`, `experiments/macos/reference/README.md@12d72f3`), and every Apple timing is one machine
(Mac Studio, Apple M4 Max 40-core GPU, 64 GB, macOS 27.0) that other agent jobs were using at the same time (loadavg 7 to 16;
`benchmarks/apple/mlx/README.md@36eb69e`, `docs/research/mlx-cpp-ffi.md@946a03d`, `experiments/macos/mps/README.md@0569a5d`).

Pins: TingMix-Large `75071a1`, google/embeddinggemma-2 `914f7f8`, tingmix `f44cbdb`, embed2asr `b582757` (`docs/research/PINNED.md`).

## 1. TingMix-Large

Oracle: PyTorch CPU fp32 (embed2asr + tingmix) on the 8 fixtures. Its own encoder cost on this machine was 1.2 to 1.6 s per clip
(`docs/research/mlx-cpp-ffi.md@946a03d`). It needs Python, so it is the numerical oracle only, not a runtime for the app.

| Runtime | Path / precision | Status | What was measured | Evidence | Not done / limits |
|---|---|---|---|---|---|
| PyTorch CPU fp32 | oracle | VERIFIED as oracle for the 8 fixtures | dumps of mel, features, logits, greedy and beam+LM text | `experiments/macos/reference/README.md@12d72f3`, `tests/fixtures/large/*` | synthetic TTS; no real-speech set |
| PyTorch MPS | fp32, fp16, bf16, `PYTORCH_ENABLE_MPS_FALLBACK=0` | EXPERIMENTAL (works, not shippable: Python/PyTorch) | no failing op in any stage; greedy and beam+LM text identical to the oracle on 8/8 clips in all three precisions. Feature/logit relative-RMS error vs CPU fp32 (worst clip): fp32 2.0e-6 / 1.4e-6, fp16 2.4e-3 / 9.2e-4, bf16 2.0e-2 / 4.7e-3. Tower+reader P50 at 29 s: 0.336 s (fp32), 0.335 s (fp16), 0.310 s (bf16); RTF P50 0.012 / 0.012 / 0.011 | `experiments/macos/mps/README.md@0569a5d`, `benchmarks/apple/mps/results.json@0569a5d` | batch 1; 15 timed runs, P99 is the max; CPU feature extraction and CPU Python decoder not on MPS; peak MPS allocation about 1.0 to 1.4 GB |
| MLX Python | fp32 (GPU) | EXPERIMENTAL | parity vs oracle (worst of 8): feature max-abs 6.0e-4, logit max-abs 6.2e-4, argmax agreement 1.0, greedy and beam+LM text 8/8. Latency P50 / P95 / P99: 2 s 20/25/35 ms, 8 s 48/53/55 ms, 29 s 154/178/179 ms (RTF P50 0.005 at 29 s); 29 s split 5 / 54 / 99 ms (mel / tower / reader+greedy); process peak RSS 1800 MiB | `benchmarks/apple/mlx/README.md@36eb69e`, `benchmarks/apple/mlx/parity.md@7574a39` | no real speech; no power or GPU counters; GRU is host-launched (99 of 154 ms at 29 s) |
| MLX Python | fp16 / bf16 (GPU) | EXPERIMENTAL; fp32 recommended | fp16 worst feature max-abs 6.5e-2, logit 8.5e-2, text 8/8; bf16 feature 4.6e-1, logit 2.9, argmax agreement 0.9980 (zh_16s), text 8/8. 29 s P50: 188 ms (fp16), 174 ms (bf16), slower than fp32 154 ms; MLX peak 1348 MiB vs 1712 MiB fp32 | same | memory-only benefit |
| MLX Python | fp32 on the MLX CPU stream (GPU-use control) | EXPERIMENTAL | 29 s P50 639 ms, RSS 2893 MiB (4.2x slower), so the GPU numbers are GPU numbers | `benchmarks/apple/mlx/README.md@36eb69e`, `docs/research/mlx-cpp-ffi.md@946a03d` | GPU utilisation counters not measured |
| MLX C++ via Rust FFI | fp32 | EXPERIMENTAL | Rust-driven logits max-abs vs oracle 2.5e-5 (zh_02s), 4.2e-5 (en_02s), 1.3e-4 (mix_04s), 9.3e-5 (mix_29s); standalone C++ driver: all 8 fixtures feature max-abs at most 4.7e-4, logit at most 7.9e-4, argmax agreement 1.0, warm 29 s run 153 ms; fp16/bf16 also run | `docs/research/mlx-cpp-ffi.md@946a03d` | batch 1, single 29.5 s window, graph rebuilt per call (no `mx.compile`) |
| Rust only (Rust log-mel + MLX C++ + Rust decoder) | fp32, greedy and beam+LM | EXPERIMENTAL | Rust log-mel vs reference dumps: max-abs 4.8e-7 (gate 1e-5) on all 9 chunks; wav to text equals the oracle on 8/8 clips for greedy and for beam+LM (16/16). Greedy e2e P50/P95/P99 at 29 s: 162/168/168 ms; beam+LM 249/256/259 ms; RTF P50 0.006 / 0.009; RSS 1.27 GiB after load, 1.84 GiB with LM (budget 1.9 to 2 GiB); LM load 2.4 s | `docs/research/mlx-cpp-ffi.md@946a03d` (raw `benchmarks/apple/mlx/rust-bench.json@0ffea4c`) | up to 2x noisier timings from contention; no real speech |
| MLX worker in the packaged app | fp32, beam+LM, worker process | EXPERIMENTAL | probe passed: device `gpu:Apple M4 Max`, MLX 0.32.4, `rtf_x1000` 15 (worker, 1.4 s clip); host test from the bundled `.app` with network denied: import with licence acceptance, activate, probe, selection (RTF 0.073), file transcription equal to the oracle beam+LM text, live session replayed from the wav gave 6 partials and 1 final with the same text; worker max RSS 1.85 GiB, peak footprint 2.22 GiB, warm decode 64 ms (mix_08s) | `docs/research/mlx-cpp-ffi.md@946a03d`, `docs/architecture/app.md@32e7e52` | GUI path, real microphone audio through MLX, real signing/notarization of the three MLX files, real-speech accuracy |
| Core ML, stage 1 Reader (T = 750, batch 1) | fp32 / fp16, trace | EXPERIMENTAL | converts. Median ms (CPU_ONLY / CPU_AND_GPU / CPU_AND_NE / ALL): fp32 116.5 / 114.2 / 117.7 / 112.3; fp16 111.8 / 116.8 / 116.1 / 111.9; output rel-L2 vs torch about 4e-7 (fp32), 2.2e-3 to 4.4e-3 (fp16). fp16 compute plan prefers the Neural Engine for 45 of 60 ops but latency is not better than CPU | `benchmarks/apple/coreml/stage1_reader.json@6bf96ac` | `torch.export` path: both fp32 and fp16 conversions failed (`ok: false`) |
| Core ML, stage 1b Reader dynamic length | fp16 | EXPERIMENTAL; a `RangeDim` model is wrong | `EnumeratedShapes` (188 and 750 frames) correct: 750 frames ALL 392.5 ms, CPU_ONLY 199.0 ms, 188 frames ALL 42.9 ms, CPU_ONLY 24.4 ms, greedy agreement 1.0. `RangeDim`: at 750 frames on ALL greedy agreement was 0.0013 (CPU_ONLY 1.0); at 188 frames on ALL prediction raised an error. Padding 188 frames into a 750 window changes greedy ids (agreement 0.984, max-abs 6.35) | `benchmarks/apple/coreml/stage1b_reader_dynamic.json@6bf96ac` | |
| Core ML, stage 2 sub-sampling conv (F = 2999) | fp32 / fp16 (trace values shown; export values are in the JSON) | EXPERIMENTAL | converts. fp32 CPU_AND_GPU 1.29 ms, CPU_ONLY 12.85 ms, rel-L2 about 4e-7; fp16 ALL 2.63 ms, CPU_AND_GPU 2.04 ms, CPU_AND_NE 4.00 ms, CPU_ONLY 7.84 ms, rel-L2 5.0e-4 (GPU) to 2.7e-3 (CPU_ONLY). All 12 fp16 ops are Neural-Engine-supported (compute plan), 2 placed on the NE in ALL | `benchmarks/apple/coreml/stage2_subsample.json@5a3865e` | placement is the compute-plan preference, not an execution counter |
| Core ML, stage 3 one Conformer block (T = 750, batch 1) | fp32 / fp16 (trace values shown; export values are in the JSON) | EXPERIMENTAL | converts. fp32 CPU_AND_GPU 4.26 ms, CPU_ONLY 19.0 ms, rel-L2 1.4e-7 to 3.3e-7; fp16 CPU_AND_GPU 5.5 ms, ALL 9.71 ms, CPU_AND_NE 15.3 ms, CPU_ONLY 22.1 ms, rel-L2 5.1e-4 (GPU), 6.4e-4 (ALL), 8.9e-4 (NE), 1.4e-3 (CPU_ONLY). fp16 plan: 44 of 170 ops on the NE, 123 on the GPU | `benchmarks/apple/coreml/stage3_block.json@5a3865e` | |
| Core ML, fp16 on the CPU path | one block, per-sub-block bisect | EXPERIMENTAL, a precision risk | error grows through the block on CPU_ONLY: after ff1 rel-L2 2.7e-3, after attention 2.3e-3, after the light conv 2.3e-2, after ff2 8.3e-2, after norm_out 0.54, while CPU_AND_GPU stays between 3.1e-4 and 8.0e-4 at every cut, 5.2e-4 after norm_out (reference absmax up to about 300) | `benchmarks/apple/coreml/bisect_stage3_fp16_cpu_cuts.json@5a3865e` | root cause not stated in the committed files |
| Core ML, whole 12-layer tower and Reader chain | | not in committed results | `benchmarks/apple/coreml/stage4_tower.json` and `experiments/macos/coreml/stage4_tower.py`, `stage5_encoder_reader.py` are uncommitted work of task #4 | | TODO(lead): add after #4 commits |
| Rust decoder (CTC greedy and beam+LM) | f32 log-probs | VERIFIED | identical to the Python reference on 303 decode, 5025 LM-unit, 12 greedy, 300+300 LM-probability, 60+60 beam comparisons; on the real oracle logits of the 8 fixtures: greedy and beam+LM text 8/8 identical. LM load 2.5 s and 579 MB RSS (Python reference 19.7 s, about 7.0 GB) | `docs/research/decoder-parity.md@c609cdc` | tie-breaking and libm can differ off macOS arm64 |
| Native Windows / Intel Mac | Large | not studied | no document or measurement exists for Large on Windows | | BLOCKED: no Windows machine; no study |

Apple-silicon coverage is a single chip generation (M4 Max). Other Apple chips, macOS versions below 27, and low-memory Macs
(Large needs about 1.9 to 2 GiB for MLX with LM, `docs/research/mlx-cpp-ffi.md@946a03d`) are **not measured**.

## 2. TingMix-Small

No checkpoint exists (`tc3oliver/TingMix-Small` returns HTTP 401; `docs/research/model-artifact-inventory.md@10026e0`). Every Small
number below is from seeded random-initialised weights of the production shape (d 512, hidden 640, 5 GRU layers, placeholder
vocabulary 4096) and proves the pipeline only (`docs/research/small-onnx.md@5fe2432`).

| Runtime | Path / precision | Status | What was measured | Evidence | Not done / limits |
|---|---|---|---|---|---|
| PyTorch (reference `Stream`) | random weights | VERIFIED for streaming == offline (random weights; no accuracy) | at production size, stream vs offline max log-prob difference 1.9e-6 for 15 chunkings; known defects: batch forward raises for clips under 992 samples, `finalize` drops up to 639 samples, `Stream.logp` memory growth | `docs/research/repository-audit.md@f80e59a`, `experiments/audit/results/small_stream_audit.json@10026e0` | no checkpoint |
| ONNX step graph | fp32, CPU | EXPERIMENTAL (pipeline-only weights) | export matches torch offline and chunked at 160/320/640 ms: production shape stream vs offline at most 5.7e-6, ONNX vs offline at most 1.5e-5, identical argmax; tiny model through Rust frontend + ONNX at most 3.5e-5 to 4.4e-5 | `docs/research/small-onnx.md@5fe2432`, `benchmarks/onnx-cpu/parity-python-prod-shape.json@5fe2432` | no real accuracy |
| Rust log-mel frontend (Small) | f32 | VERIFIED against the torch `LogMel` on 3 fixture wavs | max log-mel difference 3.2e-4 (en_02s 2.08e-4, mix_04s 1.26e-4, zh_08s 3.16e-4) | `docs/research/small-onnx.md@5fe2432` | |
| ONNX Runtime 1.31.0 CPU EP, macOS arm64 | fp32, 1 thread, 320 ms chunks | EXPERIMENTAL | RTF 0.0077, push P50 2.44 ms / P95 2.72 ms, peak RSS 136 MB (60 s of audio pushed as fast as possible); 160 ms: RTF 0.0103; 640 ms: 0.0065; 4 threads were worse (P95 13.0 ms: contention) | `docs/research/small-onnx.md@5fe2432`, `benchmarks/onnx-cpu/fp32-prod-shape.json@5fe2432` | Apple M4 Max, not the Windows reference PC; contention noise; random weights |
| ONNX Runtime CPU, 60-minute soak | fp32, 1 thread, 320 ms | EXPERIMENTAL (pipeline-only) | RTF 0.0080, P95 3.2 ms, 0 pushes slower than the chunk; peak RSS 375 MB including the 230 MB input buffer the benchmark holds (about 144 MB net vs 136 MB in the 60 s run) | `docs/research/small-onnx.md@5fe2432`, `benchmarks/onnx-cpu/soak-3600s-fp32-1t-320ms.jsonl@5fe2432` | growth over the hour not isolated further |
| ONNX INT8 | dynamic MatMul on a hand-unrolled GRU; static QDQ Conv | EXPERIMENTAL (mechanics only) | model size 62.4 MB to 16.0 MB (dynamic MatMul + static Conv, unrolled), P50 1 thread 4.52 ms to 1.63 ms, max abs log-prob drift 5.7e-3, argmax agreement 1.0 on random weights; ORT cannot dynamically quantise the ONNX `GRU` op | `docs/research/small-onnx.md@5fe2432`, `benchmarks/onnx-cpu/int8-python-ort-prod-shape.json@5fe2432` | accuracy regression unmeasurable without a checkpoint; unrolled graph is chunk-size specific |
| ONNX Runtime CPU, Windows x64 | | BLOCKED | no Windows run; the Windows target did not compile-check on this Mac (E0463, no Windows `core`); the CI `windows-latest` job has never run (no remote) | `docs/release/onnxruntime.md@55ae1ef` | needs a Windows machine or a CI run |
| Windows ML, DirectML, vendor EPs (MIGraphX, NvTensorRtRtx, OpenVINO, QNN, VitisAI, WebGPU) | | not run; documentation research only | decision recorded: CPU EP only in v1, EPs opt-in per device after a probe proves parity and speed; Windows ML catalog EPs need Windows 11 24H2 (build 26100) or newer; the Rust path to Windows ML is an open question | `docs/research/windows-inference.md@5fe2432` | `benchmarks/windows/` is empty |
| Core ML / MLX for Small | | not studied | | | |

## 3. Platform pieces that carry the runtimes

| Piece | Status | Evidence |
|---|---|---|
| Unsigned macOS arm64 `.app` and DMG (ORT bundled, no MLX) | built and launched with no Python on PATH; 64,820 KiB `.app`, 19,514,768-byte DMG; `otool -L` shows only system libraries | `docs/release/packaging.md@43fb351` |
| Signing, notarization, Gatekeeper | not done; no credentials; checklist only | `docs/release/signing-checklist.md@f80e59a` |
| Windows installer | not built | `docs/release/packaging.md@43fb351` |
| Microphone permission prompt and live capture from the `.app` | TODO(lead): user check, task #16 | |
| Streaming 60-minute soak for the app pipeline | TODO(lead): task #14 | |
| Dependency licences | generated inventory; MPL-2.0 crates flagged (symphonia, tauri's cssparser/selectors) | `docs/release/dependency-licenses.md@f8cf8c5` |
