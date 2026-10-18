# TingMix Desktop: goals, requirements and acceptance gates

## Purpose

**TingMix Desktop** is a completely local speech recognition application for Windows and macOS. This document states what it is
meant to do, the requirements it is built against, and the gates a release has to pass. It is a statement of goals, not of
current status: results, measurements and open problems are recorded in `docs/research/`, and `README.md` states what works today.

## 1. Environment and Repositories

Reference development machine:

- Mac Studio M4 Max, 40-core GPU, 64GB unified memory
- macOS Apple Silicon

Target platforms:

- macOS arm64
- Windows x64

Repositories:

- Existing model project: https://github.com/tc3oliver/tingmix
- Research baseline: https://github.com/tc3oliver/embed2asr
- Released Large model: https://huggingface.co/tc3oliver/TingMix-Large
- New Desktop repository: `tc3oliver/tingmix-desktop`

The Desktop application lives in its own repository.

Model training happens in the repositories above, not here. If a model export, interface change, or additional checkpoint is
needed, it is written down as a request in `docs/research/requests-to-training.md` instead of changing the training
implementation.

## 2. Hard Requirements

1. Tauri 2 + React + TypeScript frontend.
2. Rust-native audio capture, streaming orchestration, VAD, transcription state management, model management, and inference.
3. macOS arm64 and Windows x64 support.
4. No Python, PyTorch, Conda, or external development environment required on end-user machines.
5. No external inference API, cloud speech processing, or mandatory localhost server.
6. All audio and transcripts remain local.
7. Real-time microphone transcription with Partial / Revision / Final events.
8. Audio-file transcription and TXT/SRT export.
9. Automatic hardware capability detection and runtime selection.
10. Automatic model download, integrity verification, version pinning, rollback, and offline loading.
11. Native macOS and Windows installers.
12. Training code and research-only dependencies must never be included in the distributed application.

Network access is permitted only for user-authorized application updates and model downloads.

Once the required models and runtimes are installed, inference must work without internet connectivity.

Support offline model import for computers that cannot download models directly.

## 3. Source Audit Requirements

Before relying on a model or repository, the actual repositories and model artifacts are inspected, and the exact commits and model
revisions are pinned (`docs/research/PINNED.md`). The audit is recorded in `docs/research/repository-audit.md`,
`docs/research/model-inference-graph.md` and `docs/research/model-artifact-inventory.md`, with exact source locations, tensor shapes,
model revisions, dependencies and unresolved issues.

### TingMix-Large

The audit covers the model, streaming and release code of `tc3oliver/tingmix` (`encoder.py`, `model.py`, `streaming/session.py`,
`core/events.py`, `core/transcript.py`, `tools/export_release.py`, `docs/benchmark.md`, `docs/licensing.md`) and of
`tc3oliver/embed2asr` (`encoder.py`, `reader.py`, `pipeline.py`, `lm.py`, `tokenizer.py`), and verifies the inference graph against
the selected released revision. Values quoted by the repositories are treated as initial information until verified:

- EmbeddingGemma 2 Audio Tower: approximately 304.8M parameters.
- 128-bin audio feature extraction.
- Conv subsampling and 12 Conformer layers.
- 25 fps, 1536-dimensional temporal features.
- Encoder LoRA, rank 16 and alpha 32 in the documented release.
- Approximately 22.2M-parameter bidirectional GRU-CTC Reader (not the older 9.66M Reader).
- CTC Prefix Beam Search with character 5-gram LM.
- Model-specific tokenizer and text normalization.

The audit checks that the Hugging Face artifacts contain everything necessary for reproducible inference, and identifies separately
downloaded Google base weights, license requirements, tokenizer files, LM files, and delta weights.

### TingMix-Small

Current implementation: causal Log-Mel, two strided Conv1D layers, unidirectional GRU, and CTC, with `Stream.accept()` and
`Stream.finalize()` retaining STFT, convolution, and GRU state. The implementation is verified rather than trusted from its
documentation, specifically:

- Causal Conv padding.
- Frame count and output shape.
- Chunk boundary behavior.
- STFT sample remainder.
- GRU hidden-state carry.
- CTC collapse across chunk boundaries.
- Variable-length inputs.
- Offline-versus-streaming numerical parity.

The reference tests use small randomly initialized networks; they do not establish real-model accuracy or deployment readiness.
Small's production checkpoint may be unavailable. Weights are never invented, and a mock model is never published as a functional ASR
release.

## 4. Official Documentation Verification

Verify the current official documentation before choosing dependencies or APIs.

Primary documentation:

- Tauri 2: https://v2.tauri.app/
- Rust: https://doc.rust-lang.org/
- PyTorch MPS: https://docs.pytorch.org/docs/stable/notes/mps.html
- Core ML Tools: https://apple.github.io/coremltools/
- Apple Core ML: https://developer.apple.com/documentation/coreml
- MLX: https://github.com/ml-explore/mlx
- MLX C++: https://ml-explore.github.io/mlx/build/html/dev/cpp.html
- ONNX Runtime: https://onnxruntime.ai/docs/
- Windows ML: https://learn.microsoft.com/windows/ai/
- Hugging Face Hub: https://huggingface.co/docs/huggingface_hub/
- Safetensors: https://huggingface.co/docs/safetensors/

Also inspect the existing MLX-VLM implementations:

- https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/audio.py
- https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/embedding_gemma2/embedding_gemma2.py

These contain a third-party MLX implementation of the relevant Audio Tower architecture.

Do not confuse the final EmbeddingGemma embedding output with TingMix's required 1536-dimensional temporal features.

For every technology decision, record:

- Documentation URL and version/date.
- Required API and minimum runtime version.
- Verified compatibility.
- Unsupported operations.
- Actual experiment results.
- Remaining risks.

Use three status categories:

- VERIFIED: measured successfully using the actual TingMix model.
- EXPERIMENTAL: implementation or official support exists, but TingMix parity or performance remains unverified.
- BLOCKED: required operations, artifacts, or permissions are unavailable.

Never report a successful conversion, GPU acceleration, or ANE execution without measured evidence.

## 5. Apple Silicon Research

The following experiments are run on an Apple-silicon Mac (reference machine: Mac Studio M4 Max).

### Experiment A — PyTorch MPS

Treat PyTorch as the reference implementation for conversion validation.

The current Embed2ASR default device selection uses CUDA or CPU, not MPS.

MPS selection is added in a separate research harness without changing the model training project.

Test:

1. Full Audio Tower forward.
2. Feature extraction and padding.
3. Conformer relative-position attention.
4. Five-dimensional blocked attention masks.
5. Reader bidirectional GRU.
6. CTC logits and final decoding.
7. FP32, FP16, and BF16 where supported.
8. Representative audio lengths: 2, 4, 8, 16, and 29 seconds.
9. Warmup latency, peak memory, RTF, and P95 latency.

Disable MPS CPU fallback during compatibility validation.

Explicitly record every failing operator and unsupported precision.

Do not silently accept CPU fallback as MPS success.

### Experiment B — MLX Audio Tower

Use the existing MLX-VLM Gemma4 Audio Encoder as the starting point.

Research and implement an isolated conversion pipeline:

1. Load the pinned Google Audio Tower.
2. Load the TingMix-Large LoRA delta.
3. Merge LoRA into base linear weights using FP32 arithmetic.
4. Map PyTorch tensors to MLX tensor layouts.
5. Handle Conv2D, Conv1D, ClippableLinear, Attention, and Projection weights.
6. Extract the 1536-dimensional temporal features.
7. Compare these features against PyTorch.
8. Port or implement the GRU-CTC Reader.
9. Reproduce logits and decoded text.
10. Measure full end-to-end latency and memory.

Do not include the unused text and vision towers in the final ASR artifact.

Investigate MLX C++ integration through a narrow Rust FFI boundary.

An MLX Python proof of concept is useful for parity research, but the production Desktop application must not require a Python runtime.

If MLX C++ deployment requires unsupported functionality, document the gap and stop short of declaring the native runtime ready.

### Experiment C — Core ML

Do not attempt an all-at-once conversion without isolated tests.

Convert incrementally:

1. Reader-only graph.
2. Conv subsampling.
3. Single Conformer block.
4. Full Audio Tower.
5. Complete Encoder + Reader.

Evaluate TorchScript and `torch.export` according to the current supported coremltools APIs.

Inspect:

- Relative-position attention.
- Dynamic slicing and reshaping.
- Local attention masks.
- Variable sequence lengths.
- Bidirectional GRU.
- Padding and masking.
- Numerical precision.
- Compute-unit placement.

Test CPU, GPU, and ANE placement separately where supported.

A Core ML conversion is not considered successful merely because an `.mlpackage` file was produced.

Require real execution, output parity, performance measurements, and compute-placement evidence.

If Core ML is slower or less accurate than MLX, keep it experimental and do not make it the Desktop default.

### Apple research record

- `docs/research/mlx-cpp-ffi.md`, `docs/research/coreml.md`, `experiments/macos/mps/README.md`
- `experiments/macos/mps/`, `experiments/macos/mlx/`, `experiments/macos/coreml/`
- `benchmarks/apple/`

Experimental Python scripts are never packaged into the production app.

## 6. Windows Native Inference

Use ONNX Runtime CPU as the baseline.

Investigate Windows ML as the hardware-acceleration integration layer, following the current Microsoft documentation.

Evaluate:

- NVIDIA Execution Providers.
- AMD-supported Execution Providers.
- Intel OpenVINO.
- DirectML compatibility where applicable.
- CPU Execution Provider.

Do not assume a provider supports a model merely because the device is manufactured by NVIDIA, AMD, or Intel.

Runtime selection must account for:

- Operating-system version.
- Processor architecture.
- Device vendor and supported generation.
- Driver availability.
- Execution Provider availability.
- Operator compatibility.
- Model precision.
- Runtime initialization success.
- Performance versus CPU baseline.

Do not hardcode GPU acceleration as universally faster.

Use ONNX Runtime CPU as the mandatory fallback for a compatible Small model.

A GPU Runtime is enabled only after the actual ONNX model passes correctness and performance checks.

Document Windows ML redistributable/runtime requirements and how to support fully offline installations.

### Windows record

- `docs/research/windows-inference.md`, `docs/release/windows-validation.md`
- `crates/tingmix-runtime-onnx/`
- `benchmarks/windows/`

Windows-specific performance claims require tests on physical Windows hardware.

Cross-compilation does not count as Windows Runtime validation.

## 7. Small ONNX Export and INT8

TingMix-Small is trained in the model repository; this project needs a release candidate from it. This project does not train models, and the inference export and validation tooling are developed independently of the weights.

Required ONNX interface:

Inputs:

- Audio features or frontend input.
- GRU hidden state.
- Any additional explicitly declared streaming state.

Outputs:

- CTC logits or log-probabilities.
- Updated GRU hidden state.
- Updated streaming state where applicable.

The ONNX graph must reproduce native streaming without reprocessing the entire utterance.

Preserve CTC decoder state across chunks, including the previous emitted token and blank/repeat handling.

Evaluate:

- FP32 baseline.
- Dynamic INT8 quantization for recurrent layers.
- Static QDQ quantization for suitable convolutional layers.
- CPU thread counts.
- Chunk sizes of 160, 320, and 640 ms.
- RSS and processing latency.
- Accuracy regression from quantization.

Frontend implementation must match the reference model exactly.

Do not substitute a generic Log-Mel configuration.

Test the complete frontend numerically, including windowing, FFT behavior, Mel filterbank, logarithm floor, and padding.

### Small acceptance targets

- CPU streaming RTF ≤ 0.5.
- First Partial P95 ≤ 1.0 second.
- Final latency P95 ≤ 800 ms.
- Continuous processing ≥ 60 minutes.
- Peak inference-process RSS ≤ 1GB on the target reference PC.
- No accumulating chunk-processing backlog.
- Streaming results within the approved accuracy regression threshold versus Small FP32 offline baseline.

Treat these as targets, not current achievements.

Use a real production candidate checkpoint for release acceptance.

## 8. Desktop Application Architecture

The application is separated as follows:

**React + TypeScript UI**

- Live subtitles.
- Microphone selection.
- Start / Stop.
- Model selection.
- Model download progress.
- Transcription history.
- Audio-file import.
- TXT / SRT export.
- Runtime status.
- Model/version management.
- Application settings.

**Tauri 2 Host**

- Application lifecycle.
- Window and system-tray integration.
- Native file dialogs.
- Permissions.
- Update integration.
- Typed commands and event transport.

**Rust Core**

- Audio capture.
- Resampling and PCM conversion.
- Ring buffer.
- VAD.
- Endpoint detection.
- Streaming session.
- Transcription event state.
- Model registry.
- Runtime selection.
- Model download and verification.
- File transcription jobs.
- Cancellation and error recovery.

**Native Inference Backend**

- ONNX Runtime CPU.
- Windows ML-compatible backend.
- Apple MLX backend, only after successful validation.
- Apple Core ML backend, only after successful validation.

Use explicit Rust traits for interchangeable inference backends.

Do not mix model-specific tensors into the shared audio and transcription event interfaces.

### Suggested repository layout

    tingmix-desktop/
    ├── README.md
    ├── LICENSE
    ├── Cargo.toml
    ├── package.json
    ├── src/
    │   ├── app/
    │   ├── components/
    │   ├── features/
    │   │   ├── live/
    │   │   ├── files/
    │   │   ├── models/
    │   │   └── settings/
    │   └── types/
    ├── src-tauri/
    │   ├── Cargo.toml
    │   ├── capabilities/
    │   ├── icons/
    │   ├── src/
    │   └── tauri.conf.json
    ├── crates/
    │   ├── tingmix-core/
    │   ├── tingmix-audio/
    │   ├── tingmix-streaming/
    │   ├── tingmix-decoder/
    │   ├── tingmix-models/
    │   ├── tingmix-runtime-onnx/
    │   ├── tingmix-runtime-mlx/
    │   └── tingmix-runtime-coreml/
    ├── experiments/
    │   └── macos/
    ├── benchmarks/
    │   ├── apple/
    │   └── windows/
    ├── tests/
    │   └── fixtures/
    ├── docs/
    │   ├── architecture/
    │   ├── research/
    │   └── release/
    └── .github/workflows/

Only create runtime crates when they have concrete implementations or experiments.

Avoid empty scaffolding and unnecessary abstraction.

## 9. Audio and Streaming Pipeline

Use Rust-native audio capture, preferably CPAL after verifying current platform support.

Architecture:

    Microphone
       |
       v
    Native Audio Callback
       |
       v
    Bounded PCM Ring Buffer
       |
       v
    Resampler -> 16 kHz Mono
       |
       v
    VAD / Endpoint Detection
       |
       v
    Streaming Scheduler
       |
       v
    Native Inference Worker
       |
       v
    CTC / LM Decoder
       |
       v
    Transcript Revision Manager
       |
       v
    Tauri Event Channel
       |
       v
    React Live Subtitles

Requirements:

- Audio callback must not block on model inference.
- Do not allocate large tensors or perform disk/network operations inside the audio callback.
- Use bounded channels with explicit overflow behavior.
- Preserve source timestamps and sample counts.
- Keep raw audio available until the segment is finalized.
- Coalesce stale Partial inference jobs instead of building an unbounded queue.
- Never silently discard unprocessed audio.
- Support microphone disconnect, sample-rate changes, sleep/resume, and inference failures.
- Properly terminate and clean up streams and model resources.

Start with the existing EnergyVAD behavior as the compatibility baseline.

If replacing it with a neural VAD, measure segmentation differences and final transcription accuracy before switching.

### Streaming Event Protocol

Preserve the existing TingMix protocol semantics:

    {
      "type": "partial",
      "segment_id": 12,
      "revision": 3,
      "text": "今天我們 review 這個 API",
      "start": 12.5,
      "end": 15.1,
      "stable_prefix_len": 4
    }

Rules:

- The same segment may receive multiple partial updates.
- Revision numbers increase monotonically.
- Each new partial replaces the old partial for that segment.
- A final segment is immutable.
- The UI must not append every partial to the transcript.
- Out-of-order and stale events must be rejected.
- Final events must remain recoverable after a UI reload or window recreation.
- Segment timestamps must remain consistent with the input audio timeline.

Use the existing Python event behavior as the protocol reference.

### Large Streaming

Large is an offline bidirectional model.

Use buffered re-decoding for live transcription:

- Start with the existing 480 ms partial scheduling behavior.
- Preserve preroll and endpointing.
- Re-decode the current open utterance.
- Replace partial hypotheses.
- Perform final decoding after endpoint detection.
- Never describe this as native stateful streaming.

The inference worker must be asynchronous relative to microphone capture.

### Small Streaming

Use the causal model's retained state.

- Process each audio chunk once.
- Preserve frontend, convolution, GRU, and decoder state.
- Do not rerun the entire growing segment.
- Finalize without losing buffered residual samples.
- Explicitly test repeated CTC tokens across chunk boundaries.

Both models expose the same public transcript event semantics.

## 10. Native CTC and LM Decoder

The Rust decoder is compatible with TingMix's released format.

Required components:

- Greedy CTC collapse.
- Tokenizer loading.
- SentencePiece-compatible decoding where needed.
- Prefix Beam Search.
- Character 5-gram LM.
- CJK versus English LM scoring behavior.
- Unicode handling.
- Configurable beam, alpha, and beta.

Match the reference decoder's behavior, not merely a generic CTC implementation.

Use the released decoder settings as defaults after verifying its model card and release artifacts.

Build golden fixtures from the Python reference:

- Identical logits.
- Identical vocabulary.
- Identical LM.
- Expected decoded token IDs.
- Expected output text.

Compare Rust output against the reference across Mandarin, English, and mixed-language samples.

Document any numerical or decoding difference.

Do not remove the LM simply to simplify Native deployment.

## 11. Model Management and Offline Security

The model registry uses an explicit artifact manifest.

Each release must identify:

- Model family.
- Model revision.
- Runtime backend.
- Operating system and architecture.
- Supported device capabilities.
- Required precision.
- Artifact URLs.
- SHA256 digests.
- Base model revision.
- Adapter revision.
- Tokenizer and LM revisions.
- Required runtime version.
- License and attribution metadata.
- Compatibility and benchmark metadata.

### Download workflow

1. Select a compatible model.
2. Display required download size and applicable license information.
3. Download to a temporary location.
4. Support safe resume and cancellation.
5. Verify all file hashes.
6. Verify a publisher-signed manifest using a public key embedded in the application.
7. Reject unknown, incomplete, or tampered artifacts.
8. Atomically activate the validated model.
9. Retain the previous working version for rollback.
10. Load future sessions entirely from the local model cache.

SHA256SUMS alone does not authenticate a download if the checksum file is also untrusted.

Investigate Ed25519-signed manifests and appropriate signing-key management.

Use immutable Hugging Face commit revisions rather than tracking `main` dynamically for production releases.

Do not execute downloaded model repository code.

Do not load arbitrary Pickle checkpoints in the Desktop application.

### Model licensing

Audit the Google base-model terms, TingMix model card, training-data restrictions, and dependency licenses.

Do not assume the TingMix Apache-2.0 source-code license automatically covers all model weights.

The current TingMix licensing notes identify CC BY-NC Parliament material and NTUML2021 licensing uncertainty. Escalate unresolved distribution or commercial-use restrictions.

Model installation must not bypass any required access restrictions or license acceptance.

An offline model bundle must still comply with all redistribution permissions.

## 12. Runtime Auto-Selection

Runtime selection is an explicit capability-based engine.

Selection priority:

1. Confirm OS and architecture compatibility.
2. Confirm the selected model's available runtime formats.
3. Verify installed runtime and provider availability.
4. Check hardware and driver compatibility.
5. Check memory requirements.
6. Run or consult a verified compatibility probe.
7. Compare representative performance against CPU.
8. Select the fastest compatible and validated runtime.
9. Persist the last known-good choice.
10. Fall back safely when device or driver state changes.

Do not select a backend using GPU vendor name alone.

Provide users with a runtime diagnostics panel showing:

- Active model and revision.
- Runtime name and version.
- CPU/GPU/ANE selection.
- Hardware capability summary.
- Model precision.
- Model memory usage.
- Current RTF.
- Reason for fallback.

Never label an execution as ANE accelerated unless profiling verifies its actual placement.

## 13. Benchmark and Acceptance

Use the reference TingMix PyTorch model as the numerical and recognition-quality oracle.

Build repeatable benchmark tooling that records:

- Device and operating-system version.
- Runtime and model revisions.
- Artifact SHA256.
- Precision.
- Audio duration.
- Warmup count.
- Inference latency.
- RTF.
- P50 / P95 / P99.
- First Partial P95.
- Final latency P95.
- Peak RSS.
- GPU or unified-memory usage.
- CPU utilization.
- Power consumption where measurable.
- Streaming backlog.
- CER / WER / MER.
- Exact-text match rate.
- Numeric differences in features and logits.

### Parity testing

Compare in stages:

1. Audio preprocessing.
2. Encoder temporal features.
3. Reader logits.
4. Greedy CTC.
5. Beam Search + LM.
6. Complete offline transcripts.
7. Streaming final transcripts.

Use fixed test fixtures, fixed model revisions, and identical scoring rules.

Do not demand bit-identical FP16/BF16 floating-point outputs when mathematically equivalent implementations use different kernels.

Set explicit numeric tolerances and recognition-quality regression gates before accepting a conversion.

### Streaming benchmarking

Replay recorded audio at actual real-time intervals.

Measure timestamps at the audio input, inference queue, inference completion, and subtitle rendering.

Do not use audio-clock simulation alone as evidence of end-to-end latency.

Test:

- Quiet rooms.
- Background noise.
- Mandarin.
- English.
- Intra-sentence code switching.
- Short utterances.
- Long meetings.
- Variable-length pauses.
- Rapid consecutive utterances.
- Microphone device changes.
- 60-minute continuous execution.

Distinguish inference P95 from user-visible subtitle P95.

### Initial targets

| Metric | Large | Small |
|---|---|---|
| First Partial P95 | ≤ 1.5 s | ≤ 1.0 s |
| Final latency P95 | ≤ 1.0 s | ≤ 0.8 s |
| Streaming RTF | < 1.0 | ≤ 0.5 |
| Continuous operation | ≥ 60 min | ≥ 60 min |
| Streaming accuracy regression | ≤ 5% relative to offline error rate | ≤ 5% relative to offline error rate |

Targets are release gates, not observed measurements.

### Power and thermal testing

On macOS, investigate system-supported power instrumentation such as `powermetrics`, while documenting any privilege requirements.

Separate:

- Idle power.
- Audio capture without inference.
- Small streaming.
- Large streaming.
- Sustained 30/60-minute operation.

Avoid claiming exact model power consumption if measurements only cover system-level power.

## 14. Desktop MVP Features

The MVP is a minimal but usable application.

### Live Transcription

- Select microphone.
- Start / Stop.
- Display current partial text.
- Update revisions in place.
- Commit final segments.
- Show elapsed time.
- Display recording/runtime status.
- Preserve transcript on stop.
- Export TXT and SRT.

### File Transcription

- Select or drag in an audio file.
- Decode common supported formats locally.
- Show processing progress.
- Display transcript.
- Export TXT and SRT.
- Allow cancellation.
- Handle corrupted or unsupported files without crashing.

Use an audited native decoding library, such as Symphonia where appropriate, instead of assuming FFmpeg can be freely bundled without license review.

### Model Setup

- Show available model families.
- Show compatibility information.
- Download verified artifacts.
- Display progress.
- Switch model.
- Show installed version.
- Remove unused versions.
- Offer rollback.

### Platform Integration

- macOS microphone privacy description and permission handling.
- Windows microphone privacy handling.
- Native file dialogs.
- System tray.
- Application shutdown cleanup.
- Persistent local configuration.

Use Tauri 2 capabilities with least-privilege permissions.

Do not expose arbitrary filesystem or shell execution capabilities to the frontend.

## 15. Packaging and Updates

### macOS

Produce an arm64 application bundle and DMG.

Verify:

- All native libraries are embedded correctly.
- No Python interpreter is required.
- No developer environment is required.
- Native inference works from the installed `.app`.
- Microphone permission is correctly requested.
- Application signing and notarization workflow.
- Gatekeeper-compatible distribution.

### Windows

Produce an x64 installer using the supported Tauri 2 bundling workflow.

Verify:

- ONNX Runtime libraries are included.
- Required Microsoft runtime dependencies are accounted for.
- CPU inference works on a clean system.
- No Python or CUDA Toolkit installation is required for baseline CPU inference.
- GPU backend requirements are clearly separated.
- Installer signing and update verification workflow.

### Updates

Use the Tauri 2 updater mechanism for application updates after verifying its current security and signing requirements.

Keep application updates separate from model updates.

An app update must not automatically replace a validated model with an incompatible checkpoint.

If signing credentials are not available, produce CI development artifacts and document the signing prerequisites. Do not claim production installers are signed.

## 16. Implementation Phases

Phases are sequential; independent research may run in parallel.

### Phase 0 — Audit and Feasibility

Tasks:

- Inspect source repositories.
- Pin model versions.
- Verify latest official documentation.
- Map complete inference graphs.
- Audit licenses and artifacts.
- Identify unsupported operations.

Deliverable: written feasibility report and explicit technical gates.

Exit criteria: no architecture-critical assumptions remain undocumented.

### Phase 1 — Rust Streaming Foundation

Tasks:

- Create the Desktop repository.
- Configure Tauri 2 + React + TypeScript.
- Implement native microphone capture.
- Implement ring buffer, VAD, endpointing, and session management.
- Implement typed transcript events.
- Build working live-caption UI with a deterministic Mock Inference Backend.

Exit criteria: the installed development app can capture audio and exercise Partial / Revision / Final without Python.

A mock backend is for infrastructure testing only, not a public ASR release.

### Phase 2 — Small ONNX CPU Integration

Tasks:

- Prepare ONNX export tooling.
- Define streaming-state tensor contracts.
- Implement Rust ONNX Runtime backend.
- Port the frontend and CTC decoder.
- Implement INT8 conversion experiments.
- Integrate real Small weights when delivered.
- Run parity and CPU benchmarks.

Exit criteria: actual Small inference works offline in the packaged application, with validated streaming and resource usage.

If weights are unavailable, finish the infrastructure and report the model-delivery dependency.

### Phase 3 — Apple Large Inference

Tasks:

- Verify PyTorch MPS.
- Convert the Audio Tower to MLX.
- Merge LoRA.
- Implement Reader and decoder parity.
- Benchmark full ASR.
- Investigate MLX C++ native integration.
- Complete an isolated Core ML conversion feasibility study.

Exit criteria: one completely native Large runtime passes model parity and performance validation.

If neither native backend passes, leave Large unavailable in the production model selector and document the blocker.

### Phase 4 — Model Management and Packaging

Tasks:

- Implement signed manifests.
- Implement verified downloads.
- Implement offline imports.
- Add hardware capability detection.
- Add runtime auto-selection.
- Configure native installers.
- Configure build and test CI.

Exit criteria: a clean system can install the App, install or import an authorized model, and run offline transcription.

### Phase 5 — Cross-Platform Release Validation

Tasks:

- Test macOS arm64.
- Test Windows x64.
- Validate microphone and file transcription.
- Validate model upgrades and rollback.
- Execute 60-minute streaming tests.
- Measure latency and resource usage.
- Review dependencies, permissions, and licenses.
- Produce a release candidate.

Exit criteria: all mandatory acceptance checks pass on actual target hardware.

Do not substitute simulated GPU capabilities or macOS cross-compilation for Windows hardware tests.

## 17. Required Testing and CI

At minimum the project has:

- Rust unit tests.
- Streaming-state tests.
- Frontend component tests.
- Event ordering tests.
- VAD and endpoint tests.
- Audio-resampling tests.
- ONNX state carry tests.
- Native decoder golden tests.
- Model manifest validation tests.
- Download corruption and rollback tests.
- Offline-mode tests.
- Packaging smoke tests.

Add GitHub Actions for:

- Rust formatting and Clippy.
- Rust unit tests.
- TypeScript type checking.
- Frontend tests.
- macOS build.
- Windows build.
- Dependency and security checks.

Heavy model benchmarks may run outside public CI, but benchmark results must contain reproducible commands, hardware metadata, and artifact hashes.

## 18. Engineering Principles

1. Research the actual model implementation before choosing inference libraries.
2. Verified model compatibility takes priority over theoretical accelerator support; MLX, Core ML, MPS, ONNX GPU and Windows ML
   compatibility are never assumed.
3. No Python in the distributed application.
4. Training code and checkpoints are not changed here; tokenizer, preprocessing, decoder and language-model behavior are not changed
   silently.
5. Production model revisions are pinned, never unpinned.
6. Published weights are not assumed to be legally redistributable.
7. Exact commands and raw benchmark results are preserved; failures and blocked work are reported explicitly.
8. Documentation is updated alongside implementation, and planned features are not described as implemented.
9. Windows runtime support is not claimed without execution on Windows hardware; ANE placement is not claimed without profiling
   evidence.
10. Private datasets, credentials, restricted weights and research checkpoints are never published or uploaded.
11. Small, independently testable commits; no unrelated refactoring of the TingMix and Embed2ASR repositories.

## 19. Final Deliverables

Deliverables:

**A. Technical Research**

- Repository audit.
- Model inference graph.
- Apple runtime comparison.
- Windows runtime comparison.
- Conversion risks and operator inventory.
- LoRA merge and artifact strategy.
- Small ONNX/INT8 strategy.
- Licensing and distribution review.

**B. Working Implementation**

- Tauri 2 desktop shell.
- Rust audio capture.
- Streaming engine.
- Partial / Final subtitle UI.
- Native model runtime abstraction.
- Implemented and tested inference backends.
- Model manager.
- File transcription.
- TXT/SRT export.

**C. Evidence**

- Parity reports.
- Benchmark results.
- Hardware/runtime compatibility matrix.
- Performance and memory measurements.
- Test results.
- Unsupported-feature list.

**D. Release Engineering**

- macOS build artifacts.
- Windows build artifacts.
- Installation instructions.
- Dependency/license inventory.
- Signing and release checklist.
- CI configuration.

## 20. Success criterion

A user can install TingMix Desktop on a supported Windows or macOS machine, obtain a compatible authorized model, disconnect from the internet, and transcribe microphone or audio-file speech entirely locally without installing Python or configuring a development environment.
