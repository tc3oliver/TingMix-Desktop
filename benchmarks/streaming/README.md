# Streaming latency and soak results

Harness: `crates/tingmix-bench` (real-time replay of `tests/fixtures/audio` through `SessionWorker`; see its `--help` text in
`src/main.rs`). Times are inference-side: audio in, inference done, event received. UI render time is not included. Every
JSON carries its command, environment (hardware, git sha, rustc, profile) and artifact sha256. Other processes ran on the
machine during all runs.

| File | What | Notes |
|---|---|---|
| `mock-60min-soak.json` | 60 min, mock backend with simulated cost (40 ms + 0.06 x audio per call), seed 7, scheduler before 0a68bc0 | 9/9 assertions pass: max 1 pending final, final latency early 1.395 s vs late 1.367 s, staleness 0.848 vs 0.824 s, RSS 27 to 34 MB (4.9 MB/h slope), longest call 1.57 s |
| `mock-2min.json` | 2 min, same simulated cost, scheduler before 0a68bc0 | final from speech end P50/P95 1.41/3.23 s |
| `mock-2min-tail-off.json` | same command after 0a68bc0 with `--tail-partial off` | P50/P95 1.38/2.85 s |
| `mock-2min-live.json` | same, live policy (`SessionWorker::spawn`, tail partial on) | P50/P95 0.81/2.29 s |
| `mlx-f32-150s-tail-off.json`, `mlx-f32-150s-tail-live.json` | real `MlxLargeBackend`, fp32 artifact, beam+LM, 150 s, 20 ms chunks | see below |
| `onnx-small-60s.json` | TingMix-Small step graph `tools/small-export/out/prod.onnx`, 60 s, native streaming | PIPELINE ONLY: seeded random weights (no Small checkpoint exists), so text is meaningless; compute RTF 0.043, final from speech end P50 0.79 s |

Mock runs characterise the harness, scheduler and queue only (the mock emits text even for silence, so its first-partial
numbers are optimistic). The 0.8 s floor of "final from speech end" is the VAD hang (0.2 s) plus `silence_ms` (0.6 s).

Real MLX Large (M4 Max, fp32 weights, beam+LM, 12 synthetic TTS utterances, first utterance excluded because the model is
loaded on the inference thread on its first call): first partial P95 0.14 s; final from speech end P50 0.92 s (tail partial
off) vs 0.80 s (live); final after endpoint P95 0.38 s vs 0.21 s; streaming RTF 0.24; no errors; peak RSS 3.2 to 3.8 GB.
Initial Large gates (first partial <= 1.5 s, final after endpoint <= 1.0 s, RTF < 1.0) are met on these clips; they are
synthetic speech on an idle-ish machine, not a release verdict.

Reproduce (MLX): `MLX_PREFIX=experiments/macos/mlx/.venv/lib/python3.12/site-packages/mlx cargo build -p tingmix-bench --release --features mlx`,
then `DYLD_LIBRARY_PATH=$MLX_PREFIX/lib target/release/tingmix-bench --backend mlx --artifact <fp32 safetensors> --units <units.json> --lm <charlm.json.gz> --dtype f32 --duration-s 150 --tail-partial live --out <file>`.
