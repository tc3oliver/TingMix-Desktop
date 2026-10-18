# TingMix-Small on ONNX Runtime (CPU): export, frontend, INT8, CPU benchmark

Status words follow `CLAUDE.md`. **Every number below comes from seeded random-init weights** (`SmallNet`, no checkpoint
exists: HF `tc3oliver/TingMix-Small` returns 401). They validate the pipeline (export, state carry, frontend, runtime,
quantization mechanics, throughput of a model of this shape). They say nothing about recognition accuracy, and the
vocabulary size (4096) is a placeholder. Release acceptance needs a real candidate checkpoint.

| Item | Status |
|---|---|
| Streaming step graph exports and matches torch (offline and chunked, 160/320/640 ms) | VERIFIED (random-init weights) |
| Rust log-mel frontend equals torch `LogMel` | VERIFIED (3 fixture wavs, max 3.2e-4 in log-mel) |
| Rust `ort` backend, `AsrBackend`/`AsrStream`, CTC repeat across chunk boundary, state carry | VERIFIED (random-init weights) |
| INT8 (dynamic MatMul, static QDQ Conv): size, latency, logit drift | EXPERIMENTAL (mechanics measured; accuracy regression unmeasurable without a checkpoint) |
| CPU RTF / latency / RSS on Apple M4 Max | EXPERIMENTAL (pipeline-only weights, shared machine, **not** the Windows reference PC) |
| Acceptance targets (RTF <= 0.5 on the reference PC, P95 first partial, final latency, RSS on reference PC) | NOT MEASURED (no checkpoint, no reference PC) |

Reference model: `tingmix@f44cbdb src/tingmix/models/small/model.py` (pinned in `PINNED.md`).

## Streaming step graph

Files: `tools/small-export/{common,export,parity,quantize,make_fixtures}.py` (uv project, torch 2.9.1, onnx 1.19.1,
onnxruntime 1.23.2, `tingmix` at the pinned commit). Converted models stay in `tools/small-export/out/` (git-ignored).

```
inputs : mel [1,K,80] raw log-mel (K multiple of 4, dynamic)   c1 [1,80,1]   c2 [1,d,1]   h [layers,1,hidden]
outputs: logp [1,K/4,V]                                        c1_out        c2_out        h_out
```

Deviation from the reference `Stream`, on purpose: the reference recomputes the convs on the previous 4 raw mel frames
as context. The graph instead carries one frame of state per conv (`c1` = last normalized mel frame, `c2` = last
conv1 output after GELU). Both are exact; this one recomputes nothing and, with zero initial state, reproduces the
offline left zero-padding with no "first chunk" flag. The offline pass is the same graph with zero state and all
frames (one graph, not two). The CTC state (previous raw id) is carried in Rust (`Collapser`).

Residual input: samples that do not complete a 4-frame block stay buffered and are dropped at the end of an
utterance, exactly like the offline reference (`T = frames - frames % 4`). `AsrStream::finalize` therefore has no
model state left to flush; the streaming session already feeds the residual audio before calling it
(`tingmix-streaming` deviation from the Python reference, see the session docs).

### Finding: ORT's LayerNormalization is unsafe on clamped silence

The first export used `nn.LayerNorm`. ORT's CPU LayerNormalization computes the variance as `E[x^2] - E[x]^2`.
A digitally silent log-mel frame is `ln(1e-6)` in all 80 bins (a near-constant frame with a large offset), where that
cancels catastrophically. The graph drifted by 0.05 in log-prob against a float64 reference on `zh_08s`
(torch float32 was 1e-6 away), independent of chunking and of ORT graph optimizations. The exported graph writes the
LayerNorm out two-pass (`mean`, `x - mean`, `mean((x-mean)^2)`), which brought ORT to 4e-5 from float64.
Any other backend (Core ML, MLX, DirectML) should be checked for the same behaviour on silent frames.

## Rust frontend (`crates/tingmix-runtime-onnx/src/frontend.rs`)

Identical to `LogMel`: STFT without centring (frame `t` = samples `[160t, 160t+512)`), periodic Hann 400 centred in a
512-point FFT (56 zeros each side), power spectrum, HTK mel triangles 20-7600 Hz (80 bands, area normalized by
`2/(hi-lo)`), `ln(max(x, 1e-6))`, sample remainder carried between pushes (`MelStream`). FFT: `rustfft` 6.4.1.

| wav | frames | max abs diff vs torch (log-mel) |
|---|---|---|
| en_02s | 138 | 2.08e-4 |
| mix_04s | 352 | 1.26e-4 |
| zh_08s | 857 | 3.16e-4 |

Goldens: `tests/fixtures/small/mel_*.npy`. The residual comes from float32 FFT/filter-weight differences near the
clamp; the end-to-end effect on logits is below (Rust frontend + ONNX vs torch).

## Parity (tiny model: d=32, hidden=48, 2 layers, V=40)

Rust (`cargo test -p tingmix-runtime-onnx -- --nocapture`; needs `tools/small-export/out/tiny.onnx`, built by
`make_fixtures.py`, not in git):

| check | max abs diff in log-prob |
|---|---|
| Rust frontend + ONNX vs torch offline, en_02s | 3.5e-5 |
| Rust frontend + ONNX vs torch offline, mix_04s | 4.4e-5 |
| chunked vs offline, 160 / 320 / 640 ms, both wavs | 9.5e-7 |
| chunk sizes 1, 777, 3999 samples vs offline | < 1e-4, identical CTC ids |

Also tested: a fresh state for the second half of the audio changes the output (state carry is real), CTC repeat
across chunk boundaries (any split of a raw id sequence equals one pass; a blank at the boundary keeps both tokens),
`transcribe_batch` equals streaming, and `StreamingSession` runs on the ONNX backend.

Python (`parity.py`, `benchmarks/onnx-cpu/parity-python-prod-shape.json`, prod-shape d=512, hidden=640, 5 layers):
reference `tingmix.Stream` vs offline <= 5.7e-6; ONNX step graph vs offline <= 1.5e-5; identical argmax ids, for
160/320/640 ms on en_02s, mix_04s, zh_08s.

## INT8 (prod-shape, random-init; `quantize.py`, onnxruntime 1.23.2, one 320 ms step, M4 Max)

ORT cannot dynamically quantize the ONNX `GRU` operator (5 layers hold about 12 of the 13 M parameters), so
`quantize_dynamic` on the exported graph only touches the output Linear. To quantize the recurrent weights the GRU is
written out as MatMuls and unrolled over a fixed number of steps (`StepNetUnrolled`, K=32 frames = 8 steps): the input
projection becomes one `[8,in]x[in,3H]` product per layer, the recurrent one a `[1,H]x[H,3H]` per step. That variant
measures INT8; it is chunk-size specific and not what ships.

| variant | size MB | p50 ms 1 thread | p50 ms 4 threads | max abs dlogp | mean abs dlogp | argmax agreement |
|---|---|---|---|---|---|---|
| fp32 (GRU op) | 62.4 | 4.52 | 1.44 | 0 | 0 | 1.0 |
| fp32 unrolled MatMul GRU | 62.6 | 4.75 | 1.57 | 1.9e-6 | 1.2e-7 | 1.0 |
| dynamic INT8 on GRU-op graph (only output Linear) | 54.5 | 4.28 | 1.31 | 4.7e-3 | 7.5e-4 | 1.0 |
| static QDQ INT8 on the 2 Convs (calibrated on fixture audio, per-channel) | 59.7 | 4.33 | 1.34 | 9.7e-4 | 1.5e-4 | 1.0 |
| dynamic INT8 MatMul on unrolled graph | 18.7 | 1.77 | 0.75 | 5.8e-3 | 8.6e-4 | 1.0 |
| static Conv QDQ + dynamic MatMul, unrolled | 16.0 | 1.63 | 0.68 | 5.7e-3 | 8.7e-4 | 1.0 |

Drift is against the FP32 graph on three fixture wavs with random weights; argmax agreement on random weights is not
an accuracy measure. Conclusion for the real model: the GRU weights dominate size and time, so quantizing the
recurrent stack (about 3.9x smaller, 2.8x faster on one thread here) is the only INT8 option that matters, and it needs
either a GRU decomposed to MatMuls or a runtime that quantizes `GRU`. Whether the drift is acceptable must be measured
with a real checkpoint on real speech (CER/WER vs FP32 offline); not done. Results: `benchmarks/onnx-cpu/int8-python-*.json`.

## CPU benchmark (Apple M4 Max, 16 logical CPUs, macOS 27, ONNX Runtime 1.31.0 prebuilt via `ort 2.0.0-rc.14`, CPU EP)

`cargo run --release -p tingmix-runtime-onnx --example bench_cpu -- <model.onnx> <out.json> "<label>" 60`
(each threads x chunk pair in a fresh process; 60 s of fixture speech pushed as fast as possible through frontend +
step graph + CTC; the first push is excluded from percentiles). Full JSON with command, hardware and model sha256:
`benchmarks/onnx-cpu/{fp32,static_conv,dyn_gruop}-prod-shape.json`. FP32, p50/p95 are per push of one chunk:

| threads | chunk ms | RTF | push p50 ms | push p95 ms | peak RSS MB |
|---|---|---|---|---|---|
| 1 | 160 | 0.0103 | 1.62 | 1.93 | 136 |
| 1 | 320 | 0.0077 | 2.44 | 2.72 | 136 |
| 1 | 640 | 0.0065 | 4.08 | 4.82 | 137 |
| 2 | 320 | 0.0066 | 2.06 | 2.36 | 136 |
| 4 | 320 | 0.0182 | 2.76 | 13.0 | 137 |
| 8 | 320 | 0.0096 | 2.78 | 5.05 | 137 |

Caveats, in order of importance: (1) this is random-init weights of the right shape on a 16-core Apple Silicon Mac, not
the Windows reference PC, so none of the acceptance targets is demonstrated; (2) the machine was shared with other
background jobs, so the p95/max and the 4/8-thread rows are contention noise (the 1-2 thread medians are the reliable
part); more threads did not help a batch-1 GRU-bound step; (3) one RTF of 0.01 means a step is far below the chunk
duration, "pushes slower than chunk" was 0 everywhere, i.e. no backlog.
A 3600 s soak (1 thread, 320 ms, `soak-3600s-fp32-1t-320ms.jsonl`): RTF 0.0080, p95 3.2 ms, 0 pushes slower than the
chunk. Its 375 MB peak RSS includes the 230 MB input buffer the benchmark itself holds; subtracting it leaves about
144 MB against 136 MB in the 60 s run, so growth over an hour is small but I did not isolate it further.
Time to first partial and final latency need the session, endpointing and a real model; not measured here.

## What is not done / next

- Real checkpoint: re-run `export.py`, `parity.py`, `quantize.py` and the benchmark; the token decoder
  (`units.json`, SentencePiece rule) is injected as a closure (`TokenDecoder`); wiring `tingmix_decoder::Vocab` in is
  the app's job.
- Verify `ort` EP behaviour on Windows (see `windows-inference.md`); ORT shipping for the app (`download-binaries`
  fetches a prebuilt library at build time; a release build must pin and bundle it).
- The dynamic-shape GRU-op graph is not quantizable for the recurrent weights; decide between an unrolled/Loop-based
  decomposition and asking the training repo for a different recurrent export.
