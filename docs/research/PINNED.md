# Pinned sources for all research in this repository

| Source | Revision | Date |
|---|---|---|
| github.com/tc3oliver/tingmix | `f44cbdbd2cd99598c99cf8d60da6cb41bb15d1db` | 2026-10-17 |
| github.com/tc3oliver/embed2asr | `b58275785501c084610b1570f1a446a31da5f933` | 2026-10-16 |
| huggingface.co/tc3oliver/TingMix-Large | `75071a1260fb21874f150df6ebd53716ac423f13` | 2026-10-17 |
| huggingface.co/google/embeddinggemma-2 (base encoder) | `914f7f89142e33e77833254d9c9b90c3cef7303b` | (pinned by tingmix/embed2asr source) |
| huggingface.co/tc3oliver/TingMix-Small | not published (HF API returns 401) | — |
| huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 (SenseVoice-Small ONNX export) | `2365baeacb507f821a0c8120fcee3d484dba7a07` | 2026-10-19 |
| sherpa-onnx (oracle, PyPI `sherpa-onnx`/`sherpa-onnx-core`) | `1.13.8` | 2026-10-19 |
| huggingface.co/Qwen/Qwen3-ASR-0.6B | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | 2026-10-19 |
| huggingface.co/Qwen/Qwen3-ASR-1.7B (accuracy probe only) | `7278e1e70fe206f11671096ffdd38061171dd6e5` | 2026-10-19 |
| huggingface.co/andrewleech/qwen3-asr-0.6b-onnx (ONNX export) | `4fc24a1402e74db89c4d2ef256875e71680128c4` | 2026-10-20 |
| github.com/andrewleech/qwen3-asr-onnx (export code, oracle) | `4aee2ae0c355c7bc33bd5c09befec9e9eabd4b50` | 2026-10-20 |

Read-only clones used for the audit: `~/Developer/scratch/tingmix-audit/{tingmix,embed2asr}` (disposable; re-clone at the
revisions above). Python research harnesses install these packages from git at the pinned commits, never from `main`.
