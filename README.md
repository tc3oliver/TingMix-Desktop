# TingMix Desktop

Live captions and transcription for Taiwanese Mandarin, English and the two mixed together. It runs entirely on your
computer: no account, no cloud, and your audio never leaves the machine.

台灣華語、英文、中英夾雜都聽得懂的離線字幕工具。即時字幕、音訊和影片轉文字，全部在你的電腦上完成，不用帳號，不上雲端。

## What it does

- **Live captions** from your microphone, as you speak.
- **Transcribe files**: drop in an audio or video file and get the text.
- **Export** to TXT or SRT subtitles. Past sessions are kept in History.
- **Works offline** once a model is downloaded.

## Pick your model

Choose a speech model in Settings › Speech model. Each has its strengths:

| model | best for | errors* | memory |
|---|---|---|---|
| [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | accuracy, mixed Chinese and English | 9% | 3.4 GB |
| [SenseVoice-Small](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | speed, older or smaller machines | 10% | 0.5 GB |
| [TingMix-Large](https://huggingface.co/tc3oliver/TingMix-Large) | the original TingMix model | 24% | 2 GB |

\* Live captions on 11 minutes of real Taiwanese YouTube videos: a product review with background music, and a finance show.
The reference transcripts haven't been checked by a person yet, so treat the numbers as rough. Every model still misses
some brand names and rare names. [How this was measured](docs/research/qwen3-asr.md).

Today only TingMix-Large can be downloaded from inside the app. SenseVoice and Qwen3-ASR can be installed from a folder until
they're added to the download list.

## Requirements

- Mac with Apple silicon, macOS 13 or later
- Windows 11, x64. No GPU needed.

## Status

This is early software. There are no ready-made installers yet, so you build it yourself (below), and the builds aren't code
signed. It has been tested on one Mac and one Windows PC.

## Build it

You need Rust 1.85+, Node 24 with pnpm 10, and the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) for
your system.

```sh
git clone https://github.com/tc3oliver/TingMix-Desktop.git
cd TingMix-Desktop
pnpm install
tools/ort-fetch/fetch.sh macos-arm64     # on Windows: windows-x64
pnpm tauri dev
```

Installers: `pnpm bundle:windows` on Windows, `tools/bundle-mlx.sh release` on a Mac (needs MLX, see
[docs/architecture/app.md](docs/architecture/app.md)).

## Models and licences

Models aren't bundled. The app downloads them from
[Hugging Face](https://huggingface.co/tc3oliver/TingMix-Desktop) and checks every file against a signed list before using it.
Anything tampered with or out of date is refused. [Details](docs/architecture/model-management.md).

Each model has its own licence:

- **Qwen3-ASR**: Apache-2.0.
- **SenseVoice**: the FunASR model licence, shown for you to accept before download.
- **TingMix-Large**: Apache-2.0 weights, but part of the training data is non-commercial (CC BY-NC) or share-alike
  (CC BY-SA). Check the [model card](https://huggingface.co/tc3oliver/TingMix-Large) before commercial use.

## Under the hood

Rust and TypeScript on [Tauri 2](https://v2.tauri.app/). Models run natively, on MLX on Apple silicon and ONNX Runtime
elsewhere. There's no Python in the app.

```
src/            React UI
src-tauri/      app host: audio capture, sessions, files, settings, model setup
crates/         Rust libraries: audio, streaming, decoders, model registry, engines
experiments/    Python used to port and check models (not shipped)
tools/          export, packaging and fixture scripts
benchmarks/     raw benchmark results with the commands that produced them
docs/           architecture, release and research notes
```

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Report security problems privately, as
described in [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Dependency licences are listed in
[docs/release/dependency-licenses.md](docs/release/dependency-licenses.md).
