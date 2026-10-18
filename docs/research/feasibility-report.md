# TingMix Desktop feasibility report

DRAFT, to be verified before publication. Date of drafting 2026-10-18, repository at `f80e59a`. Built only from committed documents
and benchmark files; each number carries `file@commit` (last commit that touched the file). Nothing was re-run for this report.
Detailed per-runtime rows are in `compatibility-matrix.md`; this report adds the decision view, the acceptance gates and the open items.
Placeholders `TODO` mark results that did not exist in committed form when this was drafted.

Status words (CONTRIBUTING.md): VERIFIED = measured with the real TingMix model; EXPERIMENTAL = implemented or officially supported, TingMix parity
or speed not (fully) established; BLOCKED = missing op, artifact or permission.

## 1. Verdict in one table

| Question | Answer | Status | Basis |
|---|---|---|---|
| Can TingMix-Large run natively on Apple Silicon without Python? | Yes, on one path: MLX C++ behind a C ABI called from Rust, Rust log-mel, Rust CTC/LM decoder. wav to text equals the CPU fp32 oracle on 8/8 synthetic clips (greedy and beam+LM). | EXPERIMENTAL | `docs/research/mlx-cpp-ffi.md@946a03d` |
| Is that path fast enough? | Offline: 29 s of audio in 162 ms (greedy) or 249 ms (beam+LM) P50 on an M4 Max, RTF 0.006 to 0.009. Buffered streaming first partial about 0.5 s, final 0.68 to 0.91 s after segment end (8 runs, synthetic, not P95). | EXPERIMENTAL | `docs/research/mlx-cpp-ffi.md@946a03d` |
| Is it shippable today? | No. Unsigned bundle only; MLX comes from the PyPI wheel (no source build: Metal toolchain missing on the build machine); no real-speech accuracy; one machine. | EXPERIMENTAL | `docs/research/mlx-cpp-ffi.md@946a03d`, `docs/release/packaging.md@43fb351` |
| PyTorch MPS | Works for every stage in fp32/fp16/bf16 with fallback off, but needs Python and PyTorch: research and cross-check only. | EXPERIMENTAL | `experiments/macos/mps/README.md@0569a5d` |
| Core ML | Reader, sub-sampling and one Conformer block convert and match; whole tower not in committed results; dynamic-length Reader via `RangeDim` produced wrong output; fp16 on the CPU path degrades. | EXPERIMENTAL (partial) | `benchmarks/apple/coreml/*.json@6bf96ac,5a3865e` |
| Can TingMix-Small run natively on CPU (ONNX Runtime + Rust)? | The pipeline works end to end with random weights: frontend, streaming step graph, state carry, CTC across chunks, INT8 mechanics. There is no checkpoint, so recognition accuracy and the acceptance gates cannot be established. | EXPERIMENTAL (pipeline) / BLOCKED (checkpoint) | `docs/research/small-onnx.md@5fe2432` |
| Windows | Not run. ONNX Runtime CPU EP is the planned baseline; Windows ML and vendor EPs are documentation research only. Large on Windows has not been studied. | BLOCKED (no Windows machine) | `docs/research/windows-inference.md@5fe2432`, `docs/release/onnxruntime.md@55ae1ef` |
| Licensing | Base encoder is Apache-2.0 and ungated; TingMix-Large source facts recorded; the commercial vs non-commercial question is not pursued by owner decision (2026-10-18). | recorded | `docs/research/licensing-review.md@f80e59a` |

## 2. What the audit established about the model (Phase 0)

All from `docs/research/repository-audit.md@f80e59a`, `docs/research/model-inference-graph.md@10026e0`, `docs/research/model-artifact-inventory.md@10026e0`:

- Audio tower 304,824,608 parameters (+480 clip scalars), 12 Conformer layers, 25 fps, 1536-d output; Reader 22,171,041 parameters; LoRA delta
  5,505,024 parameters (r16, alpha 32, 192 tensors, all 12 layers). Tower structure equals the real Google safetensors header (0 missing, 0 extra, 0 shape mismatches).
- The audio tower is one contiguous 609,650,176-byte range at the start of Google's 1.49 GB checkpoint: only bytes [0, 609,821,480) (header + tower) are needed.
- The tower is causal in attention (previous 11 frames) and in the depthwise conv; the Reader is bidirectional, which is why Large streams by re-decoding the open segment.
- The encoder is not padding-invariant, so parity must use batch 1 and the extractor's own padding.
- Rust frontend/decoder facts that took work to get right: flatten order (frequency, channel), periodic Hann, magnitude spectrum with HTK mel and `ln(x + 1e-3)`, LoRA merge `W + 2.0 B A`, the Reader's packed BiGRU.
- The LM is 698 MB of JSON (22.6M n-grams); the Rust decoder loads it in 2.5 s and 579 MB instead of 19.7 s and about 7 GB (`docs/research/decoder-parity.md@c609cdc`).

Conversion risks and operator inventory (consolidated): ops used are Conv2d, LayerNorm, RMSNorm, SiLU, GLU, depthwise Conv1d, softplus, tanh, softmax, `unfold`,
`clamp` with tensor bounds, sin/cos table, packed BiGRU (`model-inference-graph.md@10026e0`). Observed backend-specific hazards:

1. ONNX Runtime's CPU LayerNormalization is numerically unsafe on clamped-silence log-mel frames (variance via `E[x^2] - E[x]^2`); the exported Small graph writes LayerNorm out in two passes; other backends should be checked for silent frames (`small-onnx.md@5fe2432`).
2. Core ML `RangeDim` Reader gave greedy agreement 0.0013 at 750 frames on the ALL compute unit; `EnumeratedShapes` was correct but slower (`stage1b_reader_dynamic.json@6bf96ac`).
3. Core ML fp16 on the CPU path amplifies error through one Conformer block (rel-L2 0.54 after `norm_out`, versus 5.2e-4 on the GPU path) (`bisect_stage3_fp16_cpu_cuts.json@5a3865e`).
4. MLX fp16/bf16 are slower than fp32 and only save memory (29 s: 188 and 174 ms vs 154 ms); the host-launched GRU is the largest block (99 of 154 ms at 29 s) (`benchmarks/apple/mlx/README.md@36eb69e`, `mlx-cpp-ffi.md@946a03d`).
5. ORT cannot dynamically quantise the ONNX `GRU` operator, so INT8 of the recurrent weights needs a MatMul-unrolled graph that is chunk-size specific (`small-onnx.md@5fe2432`).

LoRA merge / artifact strategy: MLX conversion reads only `audio_tower.*`, merges the delta in fp32 into 96 linear weights and keeps the 480 clip scalars; the resulting
fp32 safetensors is 1.3 GB (`mlx-cpp-ffi.md@946a03d`). Whether a merged bf16/fp16 artifact keeps accuracy is covered only by the 8-clip parity (fp16 8/8 identical text, bf16 8/8). The Desktop
fetches the original files at pinned revisions; no mirror of Google or TingMix weights (`licensing-review.md@f80e59a`).

## 3. Acceptance targets (goals.md sections 7 and 13)

Targets are release gates, not observations. "Measured" is the closest committed measurement and is stated with its limits. No gate is marked passed here.

| Gate | Large target | Large: closest committed measurement | Small target | Small: closest committed measurement |
|---|---|---|---|---|
| First Partial P95 | <= 1.5 s | MLX buffered streaming via `SessionWorker`: first partial 0.50 to 0.51 s in 7 of 8 replays, 0.61 s once (hop 480 ms + one decode); 4 synthetic clips, replayed at real time, P95 not computed, no UI rendering (`mlx-cpp-ffi.md@946a03d`) | <= 1.0 s | not measured (needs session, endpointing, real checkpoint) (`small-onnx.md@5fe2432`) |
| Final latency P95 | <= 1.0 s | final 0.68 to 0.91 s after segment audio end (600 ms endpoint silence + decode), same 8 replays, P95 not computed (`mlx-cpp-ffi.md@946a03d`) | <= 0.8 s | not measured |
| Streaming RTF | < 1.0 | decode cost per update P50 24 / 36 / 38 / 72 ms on clips of 1.7 / 5.1 / 7.5 / 20.4 s (greedy), "far below the 0.5 budget"; offline RTF P50 0.005 to 0.012 (`mlx-cpp-ffi.md@946a03d`, `benchmarks/apple/mlx/README.md@36eb69e`) | <= 0.5 | RTF 0.0065 to 0.0103 (1 thread, fp32, 160 to 640 ms chunks) with random weights on an Apple M4 Max, not the reference PC (`small-onnx.md@5fe2432`) |
| Continuous operation >= 60 min | yes | TODO soak result | yes | 3600 s pipeline soak with random weights: RTF 0.0080, P95 3.2 ms, 0 pushes slower than the chunk (`benchmarks/onnx-cpu/soak-3600s-fp32-1t-320ms.jsonl@5fe2432`); not the app session, not a real model |
| No accumulating backlog | | no coalesced partial in the replays above (`mlx-cpp-ffi.md@946a03d`); long-run: TODO | required | 0 pushes slower than the chunk in the CPU runs (`small-onnx.md@5fe2432`) |
| Peak inference-process RSS | | MLX with LM about 1.9 to 2 GiB budget; worker max RSS 1.85 GiB, peak footprint 2.22 GiB (`mlx-cpp-ffi.md@946a03d`) | <= 1 GB on the reference PC | 136 MB peak (60 s run), 375 MB in the 3600 s run including a 230 MB benchmark buffer (about 144 MB net); reference PC unspecified and not measured (`small-onnx.md@5fe2432`) |
| Streaming accuracy regression | <= 5% relative to offline error rate | not measured: needs a real-speech set (R12); upstream reports final-vs-offline MER differences on its own streaming runs (`docs/streaming.md` in tingmix) but that is a different implementation | <= 5% | not measurable without a checkpoint |
| Parity gates (goals.md section 13 stages 1 to 7) | explicit tolerances before accepting a conversion | stages 1 to 6 on the 8 synthetic clips: mel 4.8e-7 (Rust), features/logits per precision, greedy and beam+LM text 8/8; stage 7 (streaming finals) covered only by the replays above (`mlx-cpp-ffi.md@946a03d`, `benchmarks/apple/mlx/parity.md@7574a39`); proposed relative-RMS gates 1e-3 (fp32), 2e-2 (fp16), 5e-2 (bf16) (`experiments/macos/mps/README.md@0569a5d`) | | pipeline parity only (`small-onnx.md@5fe2432`) |
| Power and thermals | investigate `powermetrics` | not measured (needs sudo) (`mlx-cpp-ffi.md@946a03d`) | | not measured |
| User-visible subtitle P95 (input to rendering) | separate from inference P95 | TODO: app-level timing | | TODO |

## 4. Blockers and gaps

| ID | Item | Effect | Owner / next step |
|---|---|---|---|
| B1 | No TingMix-Small checkpoint (Hub API 401) | Small accuracy, INT8 accuracy regression and every Small acceptance gate are unmeasurable | training session, R1 |
| B2 | No real-speech evaluation set; every parity check uses 8 synthetic TTS clips | no CER/WER/MER claim for any backend | training session, R12 |
| B3 | No Windows hardware or toolchain used; Windows target not compile-checked on the Mac; `windows-latest` CI untested (no remote) | Windows is unverified for ONNX CPU, installer, DLL loading, WebView2; Large on Windows unstudied | needs a Windows machine or the first CI run |
| B4 | Metal toolchain missing on the build machine (`xcrun metal` fails) | no MLX source build; the app uses the PyPI wheel's libmlx; the 216 MB `mlx.metallib` is not trimmed (about 240 MB app size) | install the toolchain, build MLX from a pinned tag in CI |
| B5 | No signing identity, notarization or updater key | bundle is ad hoc sealed only; hardened-runtime behaviour with Microsoft's ORT dylib and the MLX files untested | project owner, `docs/release/signing-checklist.md@f80e59a` |
| B6 | One Apple machine, shared during every run | timings indicative; no idle-machine rerun; other chips and macOS versions untested | rerun on an idle machine |
| B7 | Core ML whole tower and Reader chain not in committed results | no Core ML end-to-end conclusion | TODO |
| B8 | Reference PC for Small gates unspecified | RSS and RTF gates have no target hardware | owner decision |
| B9 | Gemma Prohibited Use Policy: whether it binds users (E4); re-hosting Google weights (E5) | wording of the licence screen; offline bundles | owner; both avoided by the current no-mirror default (`licensing-review.md@f80e59a`) |
| B10 | Open items from the packaging smoke test: install from the DMG into `/Applications`, Gatekeeper, clean-Mac run | | TODO |

## 5. Requests to the training session (R1 to R13)

Full text in `docs/research/requests-to-training.md@f80e59a`. None has been answered at drafting time.

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

## 6. Owner decisions

| Date | Decision | Effect |
|---|---|---|
| 2026-10-18 | TingMix-Large commercial vs non-commercial use is out of scope: not pursued or escalated. | E1 to E3 kept as recorded facts and attribution requirements only. Licence display, attribution and the "no mirror / no redistribution" default unchanged (`licensing-review.md@f80e59a`). R5 withdrawn. |

Still open for the owner: E4 (Prohibited Use Policy as a binding condition) and E5 (re-hosting or re-packing Google weights); B8 (reference PC for the Small gates).

## 7. Recommendation

1. Apple Silicon Large: continue on MLX C++ + Rust (fp32) as the only path with end-to-end parity; keep it behind the EXPERIMENTAL switch until a real-speech set (B2), an idle-machine rerun (B6) and a source-built, signed MLX (B4, B5) exist. Treat PyTorch MPS as a cross-check and Core ML as a study until a whole-tower result is committed (B7).
2. Small: keep the ONNX Runtime CPU pipeline ready and do not claim any Small accuracy or acceptance result until R1 delivers a checkpoint; INT8 for the recurrent layers needs the MatMul-unrolled graph or a different export (R6).
3. Windows: the first CI run and one physical machine are required before any Windows claim; ship CPU EP only in v1 per `windows-inference.md@5fe2432`.
4. Do not describe any artifact as signed, notarized, or accuracy-validated.

## 8. TODO

-: 60-minute app-level soak results (Large and, if available, Small) and user-visible subtitle P95.
- #16: result of the manual UI, microphone-permission and live-capture check from the `.app`.
- #4: Core ML whole-tower result once committed.
- Rebuild the release bundle from HEAD (`32e7e52` seals the bundle) and replace the codesign statement in `docs/release/packaging.md@43fb351`.
- Re-verify every number against the cited files before publishing this report.
