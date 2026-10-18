# TingMix Desktop feasibility report

Status as of repository commit `2e912b5` (2026-10-18). Built only from committed documents and raw benchmark files; each number carries
`file@commit` (the last commit that touched the file). Nothing was re-measured for this report. Row-level detail for every runtime is in
[compatibility-matrix.md](compatibility-matrix.md); this report gives the decision view, the acceptance gates and the open items.

Status words. **VERIFIED**: measured with the real TingMix weights against the PyTorch CPU fp32 oracle (or on the real artifact), within the
scope named in the cell, which is almost always the 8 synthetic TTS fixtures. **EXPERIMENTAL**: implemented or officially supported, but speed,
real-speech accuracy or deployment is not (fully) established, or the evidence is synthetic or pipeline-only. **BLOCKED**: a hard blocker remains.
A backend carries one overall word; claims inside it carry their own.

## 1. Verdict

| Question | Answer | Status | Basis |
|---|---|---|---|
| TingMix-Large natively on Apple silicon without Python | Yes: MLX C++ behind a C ABI, Rust log-mel, Rust CTC/LM decoder. wav to text equals the oracle on 8/8 clips (greedy and beam+LM); 29 s in 162 ms (greedy) or 249 ms (beam+LM) P50 on an M4 Max. A real first run from the published catalog (1.42 GB download, verify, probe, transcribe) gave exactly the oracle text. | EXPERIMENTAL overall; parity and first run VERIFIED | `docs/research/mlx-cpp-ffi.md@7e6d484`, `docs/architecture/app.md@6098058` |
| TingMix-Large natively on Windows x64 | Yes: one mel-to-logits ONNX graph, Rust `OnnxLarge`, ONNX Runtime 1.31.0. CPU, DirectML and CUDA each reproduce the oracle text on 8/8 clips; the native Windows first run (catalog, 1.42 GB download, probe in a worker process, transcribe) gave exactly the oracle text. CPU is the default (29.5 s window in 0.82 s, RTF 0.028); CUDA is 10x faster but needs 1.6 GB of NVIDIA DLLs and is opt-in; DirectML is slower than the CPU at 29.5 s and is not recommended. | EXPERIMENTAL overall; parity and first run VERIFIED | `docs/research/large-onnx.md@b91dbf1`, `docs/release/windows-validation.md@090f734` |
| Is either path release-ready | No. Both bundles are unsigned (macOS ad hoc sealed, Windows NSIS unsigned); accuracy is measured on synthetic speech plus one real video (CER 0.32); no 60-minute run with the real Large model exists; one Mac and one Windows PC were measured. | EXPERIMENTAL | see sections 3 and 4 |
| PyTorch MPS | Every stage runs with fallback off in fp32, fp16 and bf16, text equal to the oracle on 8/8 clips; needs Python, so research and cross-check only. | EXPERIMENTAL | `experiments/macos/mps/README.md@81cb5d0` |
| Core ML | All five stages (subsample, block, tower, encoder+reader) convert, run and match the oracle, but are not faster than MLX (29.5 s: 158 to 163 ms vs 154 to 162 ms), need one fixed-length model per frame bucket, and the Neural Engine gave no benefit. Not the default. | EXPERIMENTAL | `docs/research/coreml.md@7e6d484` |
| TingMix-Small | The pipeline (frontend, streaming step graph, state carry, CTC across chunks, INT8 mechanics, 60-minute Windows soak) works with random weights; there is no checkpoint, so accuracy and the Small gates cannot be established. | BLOCKED (checkpoint unpublished); pipeline EXPERIMENTAL | `docs/research/small-onnx.md@21e9344`, `docs/release/windows-validation.md@090f734` |
| Noise robustness | Adaptive VAD + confidence suppression + quiet-point cut remove permanently open segments and noise hallucinations in every generated bed tested; recommended for live capture. | VERIFIED on synthetic speech + generated noise; EXPERIMENTAL as a default | `docs/research/vad-noise.md@b8b2f03` |
| Real-speech accuracy | One real 527 s zh-TW product-review video decoded through the app (MLX fp32): CER 0.322 with beam+LM, 0.316 greedy, hypotheses about 20 % shorter than the reference, brand names mostly wrong. No other real-speech set was used. | VERIFIED for that clip only | `docs/research/requests-to-training.md@7e6d484` (R14) |
| Dependencies and licences | `cargo deny check licenses sources bans advisories` all ok when run for this report (`zip` upgraded to 9.0.5, every crate `publish = false`); inventory of 417 crates, npm packages, ORT and MLX; MPL-2.0 only (symphonia, tauri's cssparser/selectors); no GPL/LGPL/AGPL. | VERIFIED (tool output) | `deny.toml`, `docs/release/dependency-licenses.md@2e912b5` |

## 2. What the audit established about the model

From `docs/research/repository-audit.md@7e6d484`, `docs/research/model-inference-graph.md@7e6d484`, `docs/research/model-artifact-inventory.md@7e6d484`:

- Audio tower 304,824,608 parameters (+480 clip scalars), 12 Conformer layers, 25 fps, 1536-d output; Reader 22,171,041 parameters; LoRA delta
  5,505,024 parameters (r16, alpha 32, 192 tensors, all 12 layers). The tower structure equals the real Google safetensors header (0 missing, 0 extra, 0 shape mismatches).
- The audio tower is one contiguous 609,650,176-byte range at the start of Google's 1.49 GB checkpoint: only bytes [0, 609,821,480) are needed.
- The tower is causal in attention (previous 11 frames) and in the depthwise conv; the Reader is bidirectional, which is why Large streams by re-decoding the open segment.
- The encoder is not padding-invariant, so parity uses batch 1 and the extractor's own padding.
- Details that took work to get right and are now covered by tests: flatten order (frequency, channel), periodic Hann, magnitude spectrum with HTK mel and `ln(x + 1e-3)`, LoRA merge `W + 2.0 B A`, the Reader's packed BiGRU.
- The language model is 698 MB of JSON (22.6M n-grams); the Rust decoder loads it in 2.5 s and 579 MB instead of 19.7 s and about 7 GB (`docs/research/decoder-parity.md@21eb60c`).

Conversion risks and operator inventory (consolidated). Ops used: Conv2d, LayerNorm, RMSNorm, SiLU, GLU, depthwise Conv1d, softplus, tanh, softmax, `unfold`,
`clamp` with tensor bounds, sin/cos table, packed BiGRU. Backend-specific hazards observed:

1. ORT's CPU LayerNormalization is numerically unsafe on clamped-silence log-mel frames; the Small graph writes LayerNorm out in two passes (`small-onnx.md@21e9344`).
2. Core ML: `RangeDim` Reader gives wrong output on the ALL compute unit; `torch.export` fails on the GRU; the blocked attention mask and relative positions become trace-time constants, so each frame-count bucket is its own model (`coreml.md@7e6d484`).
3. Core ML fp16 on the CPU path amplifies error through one Conformer block (rel-L2 0.54 after `norm_out`, versus 5.2e-4 on the GPU path) (`bisect_stage3_fp16_cpu_cuts.json@5a3865e`).
4. MLX fp16/bf16 are slower than fp32 and only save memory; the host-launched GRU is the largest block (99 of 154 ms at 29 s) (`benchmarks/apple/mlx/README.md@36eb69e`).
5. ONNX: DirectML rejects the 5-D batched MatMul (the attention matmuls are kept 4-D); a dynamic frame axis needs the blocked mask built in the graph; the Reader's GRU is not length-aware, so inputs are not padded (`large-onnx.md@b91dbf1`).
6. ORT cannot dynamically quantise the ONNX `GRU` operator (Small INT8 needs a MatMul-unrolled, chunk-size-specific graph) (`small-onnx.md@21e9344`).

LoRA merge and artifacts. Both Large paths merge the delta in fp32 into 96 linear weights and keep the 480 clip scalars. The MLX package is a 1.3 GB fp32
safetensors; the ONNX package is a 1.31 GB fp32 graph (a 655 MB fp16 variant exists but is not the default). Both are installed from signed manifests
(Ed25519, hash-verified, atomic activation, rollback, anti-rollback serial) fetched from a catalog on the owner's Hugging Face, or imported offline.

## 3. Acceptance targets (goals.md sections 7 and 13)

Targets are release gates, not observations. "Closest measurement" states what exists and its limits. No gate is declared passed for release: every
Large figure is synthetic speech on one machine with other load running, every Small figure is random weights. Timestamps are inference-side:
audio in, inference done, event received; UI render time is not included in any harness (`benchmarks/streaming/README.md@9b9b282`).

| Gate | Target Large / Small | Large, closest measurement | Small, closest measurement (pipeline-only) |
|---|---|---|---|
| First Partial P95 | <= 1.5 s / <= 1.0 s | MLX, 150 s real-time replay of 12 synthetic utterances: P95 0.140 s from onset (the first utterance, 2.4 s, includes the model load on first call and is excluded from the percentile) (`benchmarks/streaming/mlx-f32-150s-tail-live.json@21eb60c`). ONNX Large on Windows: not run through the session harness | Windows 60-minute soak, 281 utterances: P50/P95 0.057/0.073 s (`benchmarks/windows/soak-realtime-60min-onnx-pyke-ort.json@53025fd`) |
| Final latency P95 | <= 1.0 s / <= 0.8 s | MLX: after the endpoint P95 0.207 s (tail partial on) vs 0.384 s (off); from speech end P50 0.796 s / P95 0.807 s (on) vs 0.920 / 0.984 s (off) (same files and `mlx-f32-150s-tail-off.json@21eb60c`) | after endpoint P95 0.212 s; from speech end P95 0.812 s (same soak). The BRIEF gate does not say which of the two it means; the harness gates the after-endpoint value, and the from-speech-end value exceeds 0.8 s by 12 ms because it includes the 0.2 s VAD hang and 0.6 s silence endpoint |
| Streaming RTF | < 1.0 / <= 0.5 | MLX streaming RTF 0.236 (inference busy time / audio time); offline RTF 0.005 to 0.012 (`benchmarks/apple/mlx/README.md@36eb69e`). Windows CPU offline model RTF 0.028, CUDA 0.003 (`large-onnx.md@b91dbf1`) | 0.0079 (Windows soak), 0.043 (Mac 60 s replay, `benchmarks/streaming/onnx-small-60s.json@9b9b282`); target <= 0.5 on the reference PC, which is unspecified |
| Continuous operation >= 60 min | both | **Not run with the real Large model**: the real streaming runs are 150 s. A 60-minute run with the mock backend passed 9/9 soak assertions on the scheduler before a831f25 (`benchmarks/streaming/mock-60min-soak.json@5f633a0`) | 60 minutes of real-time audio, 281 utterances: all 9 assertions pass (pending finals max 1 and none left, no latency drift 0.796 s early and late, RSS max 115.5 MB with slope -14 MB/h, no stuck call, 0 errors) (`windows-validation.md@090f734`) |
| No accumulating backlog | both | no errors in the 150 s MLX runs; backlog samples are recorded in the JSON; long-run: not run | max 1 pending final in the 60-minute soak |
| Peak inference RSS | none / <= 1 GB | MLX streaming bench 3.2 to 3.8 GB; packaged worker max RSS 1.85 GiB (`mlx-cpp-ffi.md@7e6d484`); ONNX Large on Windows peak RAM 2.34 GB (CPU fp32), VRAM 2.32 GiB (CUDA fp32) (`large-onnx.md@b91dbf1`) | 115.5 MB max in the 60-minute soak on the Windows PC |
| Streaming accuracy regression | <= 5 % relative to the offline error rate (both) | not measured against an offline baseline on real speech. Stream CER vs the clean oracle on synthetic speech: 0.035 (energy VAD) / 0.040 (adaptive) (`vad-noise.md@b8b2f03`) | not measurable without a checkpoint |
| Parity stages 1 to 7 | explicit tolerances | stages 1 to 6 on the 8 synthetic clips, per runtime (mel 4.8e-7; features and logits per precision; greedy and beam+LM text 8/8 on MLX, ONNX CPU, DirectML, CUDA and Core ML fp32; MPS gates proposed 1e-3 / 2e-2 / 5e-2 relative RMS). Stage 7 (streaming finals vs offline) is covered only by the replay runs above | pipeline parity only |
| Real-speech CER/WER/MER | report | one real zh-TW video: CER 0.322 (beam+LM) | none |
| Power and thermals | investigate `powermetrics` | not measured (needs sudo) | not measured |
| User-visible subtitle P95 | separate from inference | not measured: no harness includes UI rendering | not measured |

## 4. Blockers and gaps

| ID | Item | Effect | Next step |
|---|---|---|---|
| B1 | No TingMix-Small checkpoint (Hub API 401) | Small accuracy, INT8 regression and all Small gates unmeasurable. **BLOCKED** | training session, R1 |
| B2 | Real-speech evidence is one clip (CER 0.32, hypotheses 20 % short, proper nouns wrong); the oracle texts themselves contain model errors | no accuracy claim for any backend; the weak point may be the model, not the runtimes | R12, R14 |
| B3 | Windows: one PC measured; installed-app live path, microphone, UI, SmartScreen, clean-VM install, offline installer, uninstall/upgrade not tested; Windows ARM64 not targeted | Windows is not release-validated | manual steps in `windows-validation.md@090f734` |
| B4 | Windows DirectML: the Python wheel and Rust runs differ 8x at 7.5 s (337 vs 43 ms) and the cause was not found; 29.5 s is slower than the CPU | DirectML stays unrecommended | not investigated |
| B5 | MLX comes from the PyPI wheel; the Metal toolchain is missing on the build Mac, so there is no source build and the 216 MB `mlx.metallib` is not trimmed (about 240 MB app size) | reproducible release build of the MLX backend | install the toolchain, build MLX from a pinned tag |
| B6 | No signing identity, notarization, Authenticode certificate or updater key; the Microsoft-signed ORT dylib and the MLX files need re-signing under hardened runtime | bundles are unsigned development artifacts | `docs/release/signing-checklist.md@7e6d484` |
| B7 | One Mac and one Windows PC, both shared during runs; no idle-machine rerun; weaker CPUs not measured | timings are indicative | rerun on idle machines |
| B8 | No 60-minute run with the real Large model; the real Large streaming runs are 150 s | the soak gate for Large is open | run `tingmix-bench --soak` with a Large backend |
| B9 | Reference PC for the Small gates is unspecified | RSS and RTF gates have no target hardware | owner decision |
| B10 | Gemma Prohibited Use Policy as a binding condition (E4) and re-hosting or re-packing Google weights (E5) are listed as open in the licence review, although the converted packages are published on the owner's Hugging Face | wording of the licence screen; whether the published packages need the notices recorded in the review | owner; `docs/research/licensing-review.md@7e6d484` |
| B11 | No committed record of a manual microphone-permission, UI or live-capture check on either OS | live path with a real microphone is unevidenced in the repository | record the check |

## 5. Requests to the training session (R1 to R14)

Full text in `docs/research/requests-to-training.md@7e6d484`. None has been answered.

| ID | Pri | Request | Status |
|---|---|---|---|
| R1 | P0 | TingMix-Small checkpoint ETA and release directory | open |
| R2 | P0 | Golden activations from the training environment | open |
| R3 | P1 | Machine-readable decoder block (alpha/beta/beam, LM hash) | open |
| R4 | P1 | Compact LM format | open (the Rust decoder reads the JSON today) |
| R5 | | Publish the CC0-only LM | withdrawn 2026-10-18 (owner decision) |
| R6 | P1 | Small streaming ONNX export contract | open |
| R7 | P1 | Front-end parameters in `small.json` | open |
| R8 | P2 | Small defect reports (short-input crash, finalize tail, log-prob retention, doc drift) | open |
| R9 | P1 | Reference numerics and tolerance statement | open |
| R10 | P2 | Question: causal or chunk-wise Reader | open |
| R11 | P2 | Licence hygiene in Hub repos and docs | open |
| R12 | P1 | Redistributable real-speech regression set | open |
| R13 | P1 | `finalize()` contract for native-streaming models | open |
| R14 | P1 | Expected accuracy on informal zh-TW consumer video; hot-word or domain-LM plans | open |

## 6. Owner decisions

| Date | Decision | Effect |
|---|---|---|
| 2026-10-18 | TingMix-Large commercial vs non-commercial use is out of scope: not pursued or escalated. | E1 to E3 kept as recorded facts and attribution requirements only; licence display, attribution and the no-mirror default unchanged (`licensing-review.md@7e6d484`); R5 withdrawn. |
| 2026-10-18 | Windows transcribes with Large (ONNX Runtime). | Large ONNX graph, `OnnxLarge`, probe and Windows packaging (`large-onnx.md@b91dbf1`). |
| 2026-10-18 | Repository `github.com/tc3oliver/TingMix-Desktop`, copyright holder tc3oliver, GitHub private vulnerability reporting as the contact; internal briefs rewritten as `docs/design/`; macOS `say` fixtures kept and documented. | Open-source clean-up (`README.md`, `NOTICE`, `tests/fixtures/audio/README.md`). |

Still open for the owner: E4 and E5 (B10) and the Small reference PC (B9).

## 7. Recommendation

1. Apple silicon: ship TingMix-Large through MLX (fp32) as the EXPERIMENTAL default; keep Core ML as a study (no speed or ANE benefit measured) and MPS as a cross-check. Before calling it release quality: a source-built, signed MLX (B5, B6), a real-speech evaluation (B2) and a 60-minute run with the real model (B8).
2. Windows: ship TingMix-Large with ONNX Runtime CPU (fp32, pinned Microsoft ORT 1.31.0) as the default; keep CUDA as an opt-in provider only if the 1.6 GB of NVIDIA libraries are accepted as a product decision, and keep DirectML off. fp16 is a candidate only after a real-speech WER comparison. Validate the installed app, the microphone path and a clean machine first (B3, B11).
3. Small: keep the ONNX Runtime CPU pipeline ready; no Small accuracy or acceptance claim until R1 delivers a checkpoint. INT8 of the recurrent layers needs the MatMul-unrolled graph or a different export (R6).
4. Accuracy: the one real video (CER 0.32) says the models, not the runtimes, are the limit on informal speech; treat R14 as the product-quality question for the training side, and do not market accuracy beyond the measured scope.
5. Do not describe any artifact as signed, notarized or accuracy-validated.
