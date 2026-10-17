# TingMix-Large on PyTorch MPS (Experiment A)

Own `uv` project; reuses the loader in `../reference/common.py` and compares against the CPU fp32 oracle in
`../reference/out/` (run `../reference/dump.py` first). `PYTORCH_ENABLE_MPS_FALLBACK=0` is set before torch is imported,
so an unsupported op raises instead of silently running on the CPU. Raw output: `benchmarks/apple/mps/results.json`
(command, hardware, versions, artifact sha256, per-stage errors, every failure verbatim).

    cd experiments/macos/mps && uv sync && uv run python run.py --runs 15      # about 2 min

Status: EXPERIMENTAL as a runtime path (works, not a shipping candidate: needs Python/PyTorch); VERIFIED as a numeric
cross-check of the CPU oracle on the 8 synthetic fixtures. Torch 2.9.1, transformers 5.19.0, macOS 27.0, M4 Max 40-core GPU.

## Results (this run)

- Failing ops: none. Every stage ran on MPS with fallback off in fp32, fp16 and bf16: conv subsample, 5D blocked mask
  (identical to CPU), relative-position attention, 12 layers + output_proj, 2-layer BiGRU with `pack_padded_sequence`,
  CTC logits. Reader in fp16/bf16 also runs. `failures` in the JSON is empty.
- Not on MPS: the HF feature extractor (numpy, CPU, about 7 ms for 29 s), the greedy collapse and the Python beam+LM decoder
  (13 ms for 2 s up to 140 ms for 29 s, CPU).
- Text: greedy and beam+LM text is identical to the CPU oracle for all 8 clips in all three precisions.
- Numeric error vs CPU fp32 (worst clip, relative RMS = rms(diff)/rms(ref)):

| dtype | features | logits | logits max abs | proposed gate (rel RMS) |
|---|---|---|---|---|
| fp32 | 2.0e-6 | 1.4e-6 | 7.9e-4 | 1e-3 |
| fp16 | 2.4e-3 | 9.2e-4 | 0.42 | 2e-2 |
| bf16 | 2.0e-2 | 4.7e-3 | 1.57 | 5e-2 |

Gates are proposals sized at about 2-10x the measured worst case plus the requirement that greedy text equals the oracle;
raw logit max-abs is a poor gate (large logits move by O(1) in bf16 without changing the argmax).
Teacher-forced per-stage errors (zh_08s, mix_08s) are in the JSON under `stages`.

- Latency of tower+reader on MPS (greedy, 15 timed runs after 3 warmups, one clip at a time), RTF = total/duration where
  total adds CPU feature extraction and collapse:

| audio s | fp32 P50 / P95 s | fp16 P50 / P95 s | bf16 P50 / P95 s | RTF P50 (fp32/fp16/bf16) | first call fp32 s |
|---|---|---|---|---|---|
| 2 | 0.064 / 0.068 | 0.070 / 0.091 | 0.068 / 0.072 | .033 / .036 / .035 | 0.24 |
| 4 | 0.090 / 0.101 | 0.084 / 0.086 | 0.086 / 0.087 | .023 / .021 / .022 | 0.25 |
| 8 | 0.138 / 0.163 | 0.111 / 0.116 | 0.135 / 0.142 | .018 / .014 / .017 | 0.45 |
| 16 | 0.213 / 0.255 | 0.183 / 0.185 | 0.196 / 0.210 | .014 / .012 / .013 | 0.53 |
| 29 | 0.336 / 0.353 | 0.335 / 0.344 | 0.310 / 0.336 | .012 / .012 / .011 | 0.62 |

- Memory: peak MPS-allocated during timed runs about 1.0-1.4 GB (fp16/bf16 about 1.0 GB, fp32 about 1.3 GB). The
  `mps_driver_alloc_MB` (about 3.3 GB) and `rss_MB` values are process high-water marks that include the CPU copies and
  earlier dtypes in the same process, so they are an upper bound, not a per-dtype figure.

## Caveats

- Other agents ran CPU/GPU work concurrently (loadavg about 7 during the run), so timings are indicative; rerun on an
  idle machine before quoting them. P99 over 15 runs is the max; use more `--runs` for tail claims.
- Lengths are built by concatenating the fixture clips in manifest order and cropping (not real speech of that length).
- Batch size 1 only. fp16/bf16 gain little over fp32 here because attention logits and softmax are computed in fp32
  inside `Gemma4AudioAttention`, and short inputs are launch-overhead bound.
- The audio is synthetic TTS; recognition quality is not measured here (only equality with the CPU oracle).
