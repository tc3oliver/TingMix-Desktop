# TingMix Desktop

Offline speech-to-text for macOS and Windows. It does live captions from the microphone and transcribes audio and video
files, in Taiwanese Mandarin, English, and the two mixed together. Nothing you say or transcribe leaves your computer.

它是一個離線的語音轉文字桌面程式，支援台灣華語、英文和中英夾雜，可以做即時字幕，也可以轉錄音訊或影片檔。聲音和文字都只留在你的電腦上。

You pick the speech model in Settings. Three are supported:

- [TingMix-Large](https://huggingface.co/tc3oliver/TingMix-Large): the audio encoder from Google's
  [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), a small CTC reader and a character language model.
- [SenseVoice-Small](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) (Alibaba): small and very fast.
- [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) (Alibaba): the most accurate of the three, best on mixed
  Chinese and English, and the slowest.

The app is written in Rust and TypeScript on [Tauri 2](https://v2.tauri.app/) and runs the models natively, with MLX on
Apple silicon and ONNX Runtime everywhere else. There is no Python in the app.

## Status

Early and unpolished. There are no prebuilt releases yet, so you have to build it yourself, and the builds are not code
signed. It works on the two machines it was developed on (an M4 Max Mac and a Ryzen 7 / RTX 5070 Ti PC); I haven't tried it
anywhere else.

On 11 minutes of real Taiwanese YouTube speech (a product review with music, and a finance show), live captions got this
share of characters and words wrong:

| model | error rate | speed (M4 Max CPU) | memory |
|---|---|---|---|
| Qwen3-ASR-0.6B | 9% | 0.15 s per second of audio | 3.4 GB |
| SenseVoice-Small | 10% | 0.01 s | 0.5 GB |
| TingMix-Large | 24% | | about 2 GB |

The reference transcripts haven't been checked by a person yet, so take these as rough. All three miss brand names and rare
proper nouns. Details: [docs/research/qwen3-asr.md](docs/research/qwen3-asr.md).

## Platforms

- macOS 13 or later on Apple silicon
- Windows 11, x64 (runs on the CPU; a GPU isn't needed)

## Models

No model is in this repository or in the installer. On first launch the app offers a download from the catalog at
[tc3oliver/TingMix-Desktop](https://huggingface.co/tc3oliver/TingMix-Desktop) on Hugging Face (TingMix-Large is about
1.4 GB). After that it works offline. SenseVoice and Qwen3-ASR aren't in the catalog yet; until they are, you can install
them from a folder.

Every package is described by a manifest that lists each file's size and SHA-256 and is signed with an Ed25519 key built
into the app. Anything that doesn't match is refused, and a package can never be replaced by an older one. You can also
install a package from a folder or a zip without a network connection. The details are in
[docs/architecture/model-management.md](docs/architecture/model-management.md).

Licences differ per model. TingMix-Large weights are Apache-2.0, but some of its training data is non-commercial (CC BY-NC)
or share-alike (CC BY-SA); the model card lists every source. SenseVoice uses the FunASR model licence, which you accept
before downloading. Qwen3-ASR is Apache-2.0.

## Building

You need Rust 1.85 or newer, Node 24 with pnpm 10, and the
[Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) for your platform (Xcode command line tools on macOS,
Visual Studio Build Tools and WebView2 on Windows).

```sh
git clone https://github.com/tc3oliver/TingMix-Desktop.git
cd TingMix-Desktop
pnpm install
tools/ort-fetch/fetch.sh macos-arm64     # or windows-x64: fetches the pinned ONNX Runtime
pnpm tauri dev
```

To build an installer, run `pnpm bundle:windows` on Windows. The macOS app with the MLX engine is built by
`tools/bundle-mlx.sh release`, which needs an MLX install; [docs/architecture/app.md](docs/architecture/app.md) explains
how.

Tests:

```sh
pnpm typecheck && pnpm test
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Tests that need the real model skip themselves unless you point them at one.

## Layout

```
src/            React UI
src-tauri/      Tauri host: audio capture, sessions, file jobs, settings, model setup
crates/         Rust libraries: audio, streaming and VAD, decoder, mel front end,
                model registry, MLX and ONNX Runtime engines, benchmarks
tests/fixtures/ small test inputs (synthetic speech, decoder cases)
experiments/    Python scripts used to port and check the model (not shipped)
tools/          export, packaging and fixture scripts
benchmarks/     raw benchmark results, with the command that produced each one
docs/           architecture notes, release notes and research write-ups
```

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Report security problems privately, as
described in [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Dependency licences are listed in
[docs/release/dependency-licenses.md](docs/release/dependency-licenses.md).
