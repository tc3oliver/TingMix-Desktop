# Core ML incremental conversion study (TingMix-Large)

Status: **EXPERIMENTAL**. All five stages convert, run and match the PyTorch CPU fp32 oracle, but Core ML is not faster than
MLX and the Neural Engine (ANE) is slower than the GPU for this model, so Core ML is not the Desktop default.

Setup: Mac Studio M4 Max 64 GB, macOS 27, coremltools 9.0, torch 2.9.1, transformers 5.19.0, minimum deployment target macOS 15,
ML Program. Weights: `google/embeddinggemma-2@914f7f8` plus `TingMix-Large@75071a1` LoRA merged into the base linears in fp32
(`experiments/macos/coreml/common.py`). Oracle: PyTorch CPU fp32; the study's own oracle agrees with
`experiments/macos/reference` dumps (mix_08s logits max diff 6e-5; mel identical). Stage 5 `mix_29s` uses the reference chunk 0
(F=2950, T=738). Stages 1 to 4 used a 2999-frame mel of the same clip (T=750) and `mix_08s` (F=751, T=188).
Raw results (verbatim JSON, including error tracebacks): `benchmarks/apple/coreml/*.json`. Code: `experiments/macos/coreml/`
(`cd experiments/macos/coreml && uv sync && uv run python oracle.py && uv run python stage{1,1b,2,3,4,5}_*.py`).
Timings are median of 5 to 10 predictions after 3 warm-ups; another job used the GPU concurrently, so treat them as +-20 %.
Artifacts (`out/*.mlpackage`, 0.6 GB fp16 / 1.2 GB fp32) are not in git.

## Per-stage results

| Stage | Convert | Parity vs fp32 oracle | Latency (ms, median) | Notes |
|---|---|---|---|---|
| 1 Reader only, T=750 | TorchScript OK (fp32, fp16); **torch.export fails**: `NotImplementedError: Unsupported fx node gru, kind gru.input` | fp32 rel-L2 4e-7; fp16 2e-3 to 4e-3 (max abs 0.18 to 0.39), greedy ids 100 % | 112 to 134 on all four units | BiGRU lowers to 4 `while_loop`s (sequential); no unit helps. |
| 1b Reader dynamic shapes | `EnumeratedShapes{43,188,750}` OK; `RangeDim(16..760)` converts | enum: ids 100 %; **RangeDim on ALL: 0.1 % ids match (wrong output) at T=750 and an E5RT "Port descriptor cannot be changed" error at T=188**; CPU_ONLY correct | enum T=750: CPU 199, ALL 393 (slower than fixed) | Use fixed-length models. Padding a 188-frame input to 750 without `pack_padded_sequence` changes 1.6 % of ids (max logit diff 6.3). |
| 2 Subsample conv, F=2999 | TorchScript and export OK after rewrite | fp32 0 (max abs ~1e-5); fp16 rel-L2 5e-4 (GPU) to 2.7e-3 (CPU) | CPU 12.9 (fp32) / 7.8 (fp16), GPU 1.3 to 2.0, ALL 1.3 to 2.6 | Stock `nn.ReLU` after LayerNorm+permute gives wrong fp32 results on the CPU backend (rel-L2 1.2; `bisect_s2b.py`); `clamp(min=0)` fixes it. |
| 3 One Conformer block, T=750 | Stock HF attention fails both paths: TorchScript `TypeError ... int op ... self_attn` (int() of a traced size), export `Unsupported fx node unfold`. Passes with the fixed-T attention (`modules.FixedAttn`) | fp32 ~0; fp16 1.4e-3 (CPU), 5e-4 (GPU), 8.9e-4 (NE), 6.4e-4 (ALL) | fp32 CPU 19, GPU 4.3; fp16 CPU 12 to 22, GPU 5, ALL 9 to 10 | Without the RMSNorm prescale fp16 CPU/ANE error was **0.54** (`bisect_s3*.py`): activations reach ~300 so x^2 overflows fp16. |
| 4 Full tower mel to features, T=188 / T=750 | OK, TorchScript and export, 10 to 33 s | fp32 1.2e-6 to 1.7e-6; fp16 GPU 2.5e-3 / 1.9e-3, ALL 4.2e-3 / 3.8e-3, CPU_AND_NE 6.1e-3, CPU_ONLY 1.1e-2 / 1.0e-2. Reader greedy ids 100 % (99.7 % fp16 CPU_ONLY, T=750) | T=750 fp32: CPU 229, GPU 52, ALL 53; fp16: CPU 208, GPU 43, ALL 103, CPU_AND_NE 647 | See placement below. |
| 5 Encoder + Reader (mel to logits), F=751 / F=2950 | TorchScript OK (11 to 16 s) | fp32 rel-L2 5e-7, greedy ids 100 %, decoded text equal on both clips. fp16: GPU 1.4e-3 / 1.9e-3, ALL 2.4e-3 / 2.9e-3, CPU_AND_NE 2.0e-3 / 2.2e-3, CPU_ONLY 4.9e-3 / 5.0e-3; ids 100 %, text equal on all units | F=751 fp16: GPU 27, ALL 38, CPU 52, NE 354. F=2950: fp16 GPU 158, ALL 217, CPU 254, NE 1488; fp32 GPU/ALL 163, CPU 335 | First-load compile 1.6 to 14 s per unit. |

Tolerance used: fp32 rel-L2 <= 1e-5; fp16 rel-L2 <= 1e-2 on features and logits, plus 100 % greedy-id agreement and identical decoded text
(beam+LM text was not re-run on Core ML logits; the Rust decoder takes logits from any backend). Greedy text equality was checked on two
fixtures only (synthetic TTS), no real-speech accuracy.

## Placement evidence

`MLComputePlan` (static plan, per-op preferred device, top-level ops of `main` only, not a runtime trace):

| Model | ALL | CPU_AND_NE | CPU_AND_GPU |
|---|---|---|---|
| Stage 4 fp16 T=750 | 434 ANE, 1619 GPU | 782 ANE, 1175 CPU, 96 unplaced | 2053 GPU |
| Stage 4 fp32 | 1861 GPU | 1861 CPU | 1861 GPU |
| Stage 5 fp16 F=2950 | 480 ANE, 1618 GPU, 15 CPU | 2013 CPU, 100 unplaced | 2099 GPU, 14 CPU |

Reading: fp32 never uses the ANE. In fp16 the plan sends a minority of ops to the ANE, but wall-clock with `CPU_AND_NE` is 3 to 9 times
slower than GPU (stage 5 CPU_AND_NE F=2950: the plan shows only CPU placements, i.e. the ANE rejected the graph, and it runs at CPU speed
plus fallback overhead). So the ANE gives no benefit here. Per-op ANE execution was not traced: `powermetrics` needs sudo, which was not
used, and no Xcode performance report was generated. Statements about the ANE rest on the static plan plus latency only.

## Rewrites required (all numerically equivalent; torch outputs compared before conversion)

1. `Tensor.unfold` (export frontend unsupported) replaced by two static slices (stride equals chunk, context is 2 chunks).
2. Sequence length fixed per model: the blocked local-attention mask and relative position embeddings become trace-time constants; sizes are Python ints (TorchScript failed on `int()` of traced sizes).
3. Mask fill value -1e9 replaced by -3e4 (fp16-safe).
4. `RMSNorm` computed on `x/16` with `eps/256` (power of two: bit-exact in torch) to avoid fp16 overflow.
5. ReLU as `clamp(min=0)` in the subsample conv.
6. Reader: no `pack_padded_sequence`; requires exact-length input.

## Findings for the Desktop design

- Variable length: Core ML needs one model per frame-count bucket (RangeDim is unreliable on GPU/ANE, EnumeratedShapes is slower and cannot carry the T-dependent attention mask). A 29.5 s offline window plus bucketed shorter windows means several 0.6 GB models, or padding with a length-aware GRU (not available in Core ML).
- Versus MLX (`benchmarks/apple/mlx`): MLX fp32 does 29 s in 154 to 162 ms. Core ML best case for 29.5 s is 158 ms (fp16, CPU_AND_GPU) or 163 ms (fp32, ALL): equal within noise, not better, with a larger artifact set, an fp16 accuracy cost (still 100 % ids on 2 clips) and four graph rewrites. For 8 s: Core ML fp16 GPU 27 ms; MLX comparison not re-measured here.
- Recommendation: keep Core ML EXPERIMENTAL; MLX stays the Apple default (BRIEF section 5 rule: Core ML slower or less accurate than MLX is not the default).

## Not done / open

- No ANE runtime trace (needs sudo `powermetrics` or an Xcode performance report); no power/thermal data.
- No beam+LM text parity on Core ML logits; no streaming-bucket design; one machine; two fixtures for stage 5.
- `torch.export` path: converts for stages 2 to 4, fails for the GRU (stage 1/5 use TorchScript).
