# TingMix Desktop

Local speech recognition for the desktop: live captions from the microphone and transcription of audio files, with
Partial / Revision / Final results and TXT / SRT export. All audio and text stay on your computer. The app is built with
Tauri 2 (Rust back end, React + TypeScript front end) and runs the recognizer natively, without Python.

It is the desktop front end for the [TingMix](https://github.com/tc3oliver/tingmix) models (Taiwanese Mandarin, English and
Mandarin-English code-switching), built on [Embed2ASR](https://github.com/tc3oliver/embed2asr) and Google's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) audio encoder.

> **Status: experimental, pre-release.** There are no signed or notarized builds yet, no published installers, and the speech
> models are not part of this repository. Until a model is installed the app runs in a clearly labelled **demo mode** that
> emits placeholder words, never recognized speech. See [Status](#status) and [Limitations](#limitations).

## Platforms

| Platform | State |
|---|---|
| macOS 13+ on Apple silicon (arm64) | builds and runs; TingMix-Large through MLX (EXPERIMENTAL) |
| Windows 11 x64 | builds, installs per user, starts; CPU inference through ONNX Runtime (EXPERIMENTAL, pipeline-tested with a placeholder network only) |
| Intel Mac, Linux, Windows on ARM | not targeted |

The user interface is available in English and Traditional Chinese and follows the operating-system language.

## What is in the repository

| Path | Contents |
|---|---|
| `src/`, `src-tauri/` | React UI and the Tauri host (capture, sessions, file jobs, settings, model manager, updater hook) |
| `crates/` | Rust libraries: audio capture and decoding, streaming session and events, log-mel front end, CTC and language-model decoder, signed model registry, ONNX Runtime and MLX back ends, benchmark harness |
| `tests/fixtures/` | small golden fixtures (synthetic speech, decoder and session cases) |
| `docs/` | architecture, release notes, and the research record (audit, parity, benchmarks) |
| `benchmarks/` | raw benchmark output with the commands that produced it |
| `experiments/`, `tools/` | Python research harnesses and export tools (each its own `uv` project; never part of the app) |

## Build from source

Clone the repository (`git clone https://github.com/tc3oliver/TingMix-Desktop.git`). Requirements: Rust 1.85 or newer (`rustup` or your package manager), Node 24 with pnpm 10, and the platform toolchain for
Tauri 2 ([prerequisites](https://v2.tauri.app/start/prerequisites/)): Xcode command-line tools on macOS; Visual Studio
Build Tools (MSVC) and the WebView2 runtime on Windows. Python is only needed for the research tools.

    pnpm install
    pnpm tauri dev                    # run the app in development (demo mode until a model is installed)

    pnpm typecheck && pnpm test       # front end
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace            # tests that need model files skip themselves when their inputs are absent

Release-style bundles load a pinned, checksum-verified ONNX Runtime at run time:

    tools/ort-fetch/fetch.sh macos-arm64      # or windows-x64
    pnpm bundle:macos                         # .app and .dmg, unsigned
    pnpm bundle:windows                       # NSIS installer, unsigned (run on Windows)

Details, results and what was and was not checked: [docs/release/packaging.md](docs/release/packaging.md),
[docs/release/onnxruntime.md](docs/release/onnxruntime.md), [docs/release/windows-validation.md](docs/release/windows-validation.md).
The MLX back end needs an MLX build and extra bundling steps: [docs/architecture/app.md](docs/architecture/app.md).

## Models

No model weights are stored in this repository, and none are bundled in the installers. The app installs a model from a
**signed package**: a manifest signed with an Ed25519 key whose public half is compiled into the app, listing every file with
its size, SHA-256 and licence. Packages can be downloaded by the app or imported offline from a folder or zip. Before
anything is installed the app shows each file's licence and asks you to accept it; unknown, incomplete or tampered files are
rejected, and only data formats are accepted (no pickles, no scripts).

- TingMix-Large needs Google's EmbeddingGemma 2 audio tower (Apache-2.0, downloaded from Google's Hugging Face repository at a
  pinned revision) plus the TingMix reader, LoRA adapter and language model. Their data and licence notes are in
  [docs/research/licensing-review.md](docs/research/licensing-review.md).
- TingMix-Small (a streaming CPU model) has no released checkpoint yet.

How trust, rollback and offline import work: [docs/architecture/model-management.md](docs/architecture/model-management.md).

## Status

Status words used throughout the repository: **VERIFIED** (measured with the real TingMix model), **EXPERIMENTAL**
(implemented or officially supported, TingMix parity or speed not fully measured), **BLOCKED** (missing operator, artifact or
permission).

- TingMix-Large on Apple silicon through MLX reproduces the reference recognizer's text on 8 synthetic clips (EXPERIMENTAL).
- The Rust CTC and language-model decoder and the Rust log-mel front end match the Python reference on the reference outputs.
- The Small streaming pipeline (front end, ONNX graph, state carry, INT8 mechanics) works end to end with placeholder weights;
  accuracy and acceptance targets cannot be measured without a trained checkpoint (BLOCKED).
- The research record, with the measured numbers and their limits, is in [docs/research/](docs/research/).

## Limitations

- No signed or notarized builds; macOS Gatekeeper and Windows SmartScreen will warn on development bundles.
- No real-speech accuracy evaluation of the desktop back ends yet; parity tests use synthetic speech.
- TingMix-Large re-decodes the open segment while streaming (the model is offline); latency is measured on one Mac and one
  Windows PC only.
- Models are not distributed from this repository; the training data of the published TingMix models carries its own licences.
- Updates are disabled until an updater key is configured.

## Contributing, security, conduct

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party licences of Rust and npm dependencies and bundled
binaries: [docs/release/dependency-licenses.md](docs/release/dependency-licenses.md). Model licences are separate and are
shown by the app before installation.
