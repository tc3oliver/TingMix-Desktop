# TingMix-Large inference graph

Audit date 2026-10-17. Sources are pinned in `PINNED.md`; abbreviations used for citations:

| Tag | Source | Revision |
|---|---|---|
| `TM:` | github.com/tc3oliver/tingmix | `f44cbdb` |
| `E2A:` | github.com/tc3oliver/embed2asr | `b582757` (src identical to tag `v0.1.0`, which tingmix pins) |
| `TF:` | `transformers==5.19.0`, `models/gemma4/{modeling,feature_extraction}_gemma4.py` | PyPI 5.19.0 |
| `EG2:` | huggingface.co/google/embeddinggemma-2 | `914f7f8` |
| `TML:` | huggingface.co/tc3oliver/TingMix-Large | `75071a1` |

Status legend. **VERIFIED** = measured on the real released artifact bytes (safetensors headers, hashes, configs).
**EXPERIMENTAL** = structure checked with randomly initialised weights built from the pinned config, or derived by reading
code; no TingMix numerics or speed were measured by this audit. **BLOCKED** = cannot be established from what is available.
Nothing in this document is a TingMix accuracy or speed claim. Raw outputs: `experiments/audit/results/`
(scripts and commands in `experiments/audit/`, Apple M4 Max, macOS 27.0, torch 2.9.1, transformers 5.19.0).

## 1. End-to-end pipeline

```
f32 mono 16 kHz  --(chunk <= 29.5 s)-->  Gemma4AudioFeatureExtractor  --> mel [1,F,128] + mask
  --> audio_tower (EG2, bf16 GPU / fp32 CPU) with TingMix LoRA delta --> features [1,T,1536], T ~ 25 fps
  --> Reader (fp32, bidirectional) --> logits [1,T,5025] --> CTC (greedy or prefix beam + char 5-gram LM) --> text
```

| Fact | Value | Source | Status |
|---|---|---|---|
| Encoder = `audio_tower` only, features = `last_hidden_state` (output of `output_proj`, 1536-d, before `embed_audio`) | | `E2A:src/embed2asr/encoder.py:1-3,49-50`, `TM:src/tingmix/models/large/encoder.py:3-4` | VERIFIED (key names in `EG2` header) |
| Audio tower parameters | 304,824,608 (+480 clip scalars) | `E2A:encoder.py:4`; module built from `EG2:config.json` audio_config has 304,824,608 params and 480 persistent buffers | VERIFIED (count) |
| Tower tensors in `EG2:model.safetensors` | 751 tensors, 304,825,088 elements, all BF16, 609,650,176 bytes | header, `results/header_embeddinggemma-2_model.safetensors.json` | VERIFIED |
| Whole `EG2` checkpoint | 1,376 tensors, 744,371,992 elements, 1,488,915,288 bytes (audio tower is 41 %) | same header | VERIFIED |
| Reader parameters | 22,171,041 (38 F32 tensors) | `TML:model.safetensors` header; hand count of the architecture below gives the same number | VERIFIED |
| LoRA delta | r 16, alpha 32, all 12 layers, 192 F32 tensors, 5,505,024 params | `TML:tingmix.json`, `TML:encoder_delta.safetensors` header | VERIFIED |
| Frame rate | 25 fps (40 ms per output frame) | `E2A:encoder.py:17`, shape trace below (30 s -> 750 frames) | VERIFIED (shape trace, random weights) |
| Window | 29.5 s per chunk (`TML:config.json` `chunk_seconds`); feature extractor truncates at 480,000 samples = 30 s | `E2A:pipeline.py:25,191`, `TF:feature_extraction_gemma4.py:232` | VERIFIED |
| The older 9.66M Reader is not used | Large reader is d 768, BiGRU 384 x 2, 4 conv blocks | `TML:config.json` reader block; header | VERIFIED |

## 2. Stage 0: audio preparation (Embed2ASR)

- Input: float32 mono 16 kHz, finite and non-empty (`E2A:audio.py:38-41`). Files go through ffmpeg `-ac 1 -ar 16000`
  (`E2A:audio.py:55-60`); ffmpeg mixes stereo as (L+R)/sqrt(2), i.e. +3 dB, while array input averages channels
  (`E2A:audio.py:4-6`). The Desktop decoder must pick one rule and record it; reference numbers use ffmpeg for files.
- Chunking: consecutive non-overlapping chunks of `int(29.5 * 16000) = 472,000` samples, tail chunks below 1,600 samples are
  dropped, no VAD, a word on a boundary can be split (`E2A:pipeline.py:26,189-193`). Chunk texts join with `""` if the joined
  text has no space, else `" "` (`E2A:pipeline.py:208-210`).
- Each input is encoded on its own by default. The encoder is **not padding-invariant** (`E2A:pipeline.py:185-188`: batching
  changes about 4 % of short clips by about one unit); `TM:docs/streaming.md` says the same about preroll/padding. Parity tests
  must therefore run batch size 1 with the extractor's own padding (below).

## 3. Stage 1: feature extractor (`Gemma4AudioFeatureExtractor`, `TF:feature_extraction_gemma4.py:49-294`)

Config from `EG2:preprocessor_config.json` (VERIFIED: file read at the pinned revision):
`feature_size 128, sampling_rate 16000, frame_length 320 (20 ms), hop_length 160 (10 ms), fft_length 512, fft_overdrive false,
min/max_frequency 0/8000, mel_floor 0.001, preemphasis 0.0, dither 0.0, input_scale_factor 1.0, per_bin_mean/stddev null,
padding_side right, padding_value 0`.

Algorithm (per clip, `_extract_spectrogram` `:169-224`, `__call__` `:226-294`):

1. Truncate to 480,000 samples; right-pad with zeros to a multiple of 128 samples (`pad_to_multiple_of=128`, `padding="longest"`,
   `:232-253`). Let `N'` be that length.
2. "Semicausal" left pad of `frame_length // 2 = 160` zeros (`:181`). `Lp = N' + 160`.
3. Frames of 321 samples at hop 160 are unfolded, the last sample is dropped (no preemphasis), so each frame is 320 samples
   `[i*160, i*160+320)` of the padded signal (`:185-198`). Frame count `F = (Lp - 321) // 160 + 1`.
4. Multiply by the **periodic** Hann window of 320 (`window_function(320)`, `:141`), `rfft` with n=512 (right zero pad), take the
   **magnitude** (not power) (`:200-205`).
5. `mel = magnitude @ mel_filters`, filters from `mel_filter_bank(257 bins, 128 mels, 0-8000 Hz, norm=None, mel_scale="htk")`
   (`:149-157`), shape `[257,128]`. The top filter falls between bins and may be all zero (code comment `:143-146`).
6. `log(mel + 0.001)` (`:208`), cast to float32 (`:286`).
7. Mask: frame i is valid only if sample `i*160 + 320` of the padded signal is real audio (`:219-223`); invalid frames are
   zeroed (`:289`).

Shapes measured with the real extractor config (random noise input, `results/gemma4_shapes.json`, EXPERIMENTAL because noise):

| Input | samples | `input_features` | mask true | tower output `[1,T,1536]` |
|---|---|---|---|---|
| 1.0 s | 16,000 | `[1,99,128]` | 99 | T = 25 |
| 3.0 s | 48,000 | `[1,299,128]` | 299 | T = 75 |
| 29.5 s | 472,000 | `[1,2950,128]` | 2949 | T = 738 |
| 30.0 s | 480,000 | `[1,2999,128]` | 2999 | T = 750 |
| 35.0 s (truncated) | 560,000 | `[1,2999,128]` | 2999 | T = 750 |

Output length `T = ceil(ceil(F/2)/2)` (checked for 202 input lengths from 800 to 480,000 samples, zero violations); the returned
length used by the reader is `attention_mask.sum()` (`E2A:encoder.py:50`) after the mask is sub-sampled twice with `[::2]`
(`TF:modeling_gemma4.py:388`), e.g. 2949 valid mel frames -> 738 valid output frames.
The valid length is `ceil(ceil(V/2)/2)` with `V` the number of valid mel frames (mask sum), not `F`: confirmed on the 9 real-model
chunks dumped by the CPU fp32 reference harness (`experiments/macos/reference/out/*.npz`, TingMix-Large + LoRA, VERIFIED), e.g.
`en_02s` has `F = 141`, `V = 140`, 35 valid frames; the 29.5 s chunk of `mix_29s` has `F = 2950`, `V = 2949`, 738 frames.

## 4. Stage 2: convolutional sub-sampling (`TF:modeling_gemma4.py:365-420`)

Input mel `[B,F,128]` -> `unsqueeze(1)` -> `[B,1,F,128]` (time x frequency).

| Op | Weights (EG2 header) | Output shape (F = 2950 trace) |
|---|---|---|
| `layer0.conv` Conv2d 1->128, k 3x3, stride 2, pad 1, no bias; `LayerNorm(128, eps 1e-6, no bias)` over channels; ReLU | `conv.weight [128,1,3,3]`, `norm.weight [128]` | `[1,128,1475,64]` |
| `layer1.conv` Conv2d 128->32, k 3x3, stride 2, pad 1, no bias; LayerNorm(32); ReLU | `conv.weight [32,128,3,3]`, `norm.weight [32]` | `[1,32,738,32]` |
| permute `(0,2,3,1)` -> `[B,T,32(freq),32(chan)]`, reshape `[B,T,1024]` | | `[1,738,1024]` |
| `input_proj_linear` Linear 1024->1024, no bias | `[1024,1024]` | `[1,738,1024]` |

Porting traps: the flatten order is **(frequency, channel)** with channel fastest, not (channel, frequency); the mask is multiplied
into the input of each conv layer (`:382`) and sub-sampled with `[::2]` after it (`:388`); LayerNorm is over the channel
axis after permuting channels last (`:385`); both convs use symmetric zero padding 1, so output frame j depends on mel frames up
to `4j+3` (measured below).

## 5. Stage 3: positional encoding and masks

- `Gemma4AudioRelPositionalEncoding` (`TF:modeling_gemma4.py:226-254`): `context_size = chunk(12) + left(13) - 1 + right(0) = 24`;
  positions `arange(context_size // 2, -1, -1) = [12, 11, ..., 0]` (13 positions); `inv_timescales = exp(-ln(10000)/511 * k)`,
  k = 0..511; embedding `[sin(pos*inv) , cos(pos*inv)]` concatenated -> `[13,1024]`. Not persisted in the checkpoint.
- Mask: `sliding_window_mask_function((left - 1, right)) = (12, 0)` (`:1875-1888`): query q may attend to key kv iff
  `0 <= q - kv < 12`, so each frame sees **itself and the 11 previous frames, no future** (right window 0 is never true).
  Combined with the padding mask, then `_convert_4d_mask_to_blocked_5d` (`:1915-1939`) gives a bool mask
  `[B,1,num_blocks,12,24]`, `num_blocks = ceil(T/12)` (63 for T = 750).

## 6. Stage 4: 12 Conformer layers (`TF:modeling_gemma4.py:533-584`)

Hidden size 1024, 8 heads, head_dim 128, FFN 4096, depthwise kernel 5, `hidden_act = silu`, `residual_weight = 0.5`,
`gradient_clipping = 1e10`, `rms_norm_eps = 1e-6`, `use_clipped_linears = true` (`EG2:config.json` audio_config, VERIFIED).
Per layer (Macaron order), `h` is `[B,T,1024]`:

```
h = FF1(h)                                   # FeedForward
r = h
h = norm_post_attn( clamp(Attn( norm_pre_attn( clamp(h) ) )) ) + r
h = LConv(h)
h = FF2(h)
h = norm_out( clamp(h) )
```

`clamp(x)` = clamp to +-min(1e10, finfo(dtype).max). With bf16/fp32 this is a no-op for practical values; **under fp16 it clamps
at +-65504**, which matters for a Core ML fp16 conversion (the `min(...)` is evaluated from the tensor dtype, `:440,523,557`).

- **RMSNorm** (`:205-223`): computed in float32, `x * (mean(x^2)+eps)^-0.5 * weight` -- the weight multiplies directly (no `1 + weight`).
- **FeedForward** (`:423-456`): `r=h; clamp; pre_layer_norm; ffw_layer_1 (1024->4096); silu; ffw_layer_2 (4096->1024); clamp;
  post_layer_norm; * 0.5; + r`.
- **ClippableLinear** (`:176-202`): `clamp(x, input_min, input_max)` -> `Linear (no bias)` -> `clamp(y, output_min, output_max)`.
  The four scalars are checkpoint buffers per ClippableLinear (480 total over 12 layers x 10 clippable linears x 4); weights
  `.../linear.weight`, scalars `.../{input,output}_{min,max}` (BF16, shape `[]`). Their values are inspected in
  `results/eg2_blob.json` (summary in `model-artifact-inventory.md`).
- **Attention** (`:257-362`), all of q/k/v/post are ClippableLinear 1024->1024; `relative_k_proj` Linear 1024->1024 no bias;
  `per_dim_scale [128]`:
  1. `q,k,v = proj(h).float().view(B,T,8,128)`.
  2. `q = q * (128^-0.5 / ln 2) * softplus(per_dim_scale)`; `k = k * (ln(1+e) / ln 2)`.
  3. `q` -> blocks `[B,nb,12,8,128]` (zero pad T to nb*12); `k,v` -> overlapping contexts `[B,nb,24,8,128]`: pad 12 frames on the
     left and `0 + 11` on the right, `unfold(size 24, step 12)` (`_extract_block_context` `:294-302`).
  4. `AC = q.permute(0,3,1,2,4) @ k.permute(0,3,1,4,2)` -> `[B,8,nb,12,24]`.
  5. `rel = relative_k_proj(pos[13,1024]).view(13,8,128)`; `BD = q_flat[B,8,nb*12,128] @ rel.permute(1,2,0)[8,128,13]`
     -> `[B,8,nb,12,13]`, then `_rel_shift` (pad last dim to 25, view `[..,12*25]`, keep `12*24`, view `[..,12,24]`, `:304-311`)
     -> `[B,8,nb,12,24]`. Entries that the shift fills with garbage lie outside the sliding-window mask.
  6. `w = AC + BD`; soft-cap: `tanh(w / 50) * 50` (`attention_logit_cap 50`); `masked_fill(~mask, -1e9)`; `softmax(-1)` in fp32.
  7. `out = w @ v.permute(0,3,1,2,4)` -> `[B,8,nb,12,128]` -> `[B,nb*12,1024]`, slice `[:T]`, `post` ClippableLinear.
- **LightConv1d** (`:492-530`): `r=h; pre_layer_norm; linear_start (1024->2048); GLU(dim -1) -> 1024; transpose;`
  depthwise `Conv1d(1024,1024,k=5,groups=1024,no bias)` with **causal left pad 4** (`Gemma4AudioCausalConv1d` `:459-489`,
  `left_pad = (5-1)*1 + 1 - 1 = 4`); `clamp; conv_norm (RMSNorm); silu; linear_end (1024->1024); + r`.

Tensor shapes for a 30 s window (T = 750, nb = 63), per layer:

| Tensor | Shape |
|---|---|
| layer input/output | `[1,750,1024]` |
| FFN hidden | `[1,750,4096]` |
| q blocks | `[1,63,12,8,128]` |
| k, v contexts | `[1,63,24,8,128]` |
| AC / BD (shifted) / probabilities | `[1,8,63,12,24]` |
| mask | `[1,1,63,12,24]` bool |
| rel-pos table / projected | `[13,1024]` / `[13,8,128]` |
| LConv after GLU | `[1,750,1024]`; after depthwise conv `[1,750,1024]` |

Per-layer weights (EG2 header, layer 0 shown; identical shapes for layers 1-11, VERIFIED by key/shape equality of the whole
tower against the module built from config: 0 missing, 0 extra, 0 shape mismatches):
`feed_forward{1,2}.ffw_layer_1.linear.weight [4096,1024]`, `ffw_layer_2.linear.weight [1024,4096]`, `pre/post_layer_norm.weight [1024]`;
`self_attn.{q,k,v}_proj.linear.weight [1024,1024]`, `post.linear.weight [1024,1024]`, `relative_k_proj.weight [1024,1024]`,
`per_dim_scale [128]`; `lconv1d.linear_start.linear.weight [2048,1024]`, `linear_end.linear.weight [1024,1024]`,
`depthwise_conv1d.weight [1024,1,5]`, `pre_layer_norm.weight`, `conv_norm.weight`; `norm_pre_attn/norm_post_attn/norm_out.weight [1024]`.

## 7. Stage 5: output projection

`output_proj`: Linear 1024->1536 **with bias** (`weight [1536,1024]`, `bias [1536]`). No final norm. Output `[B,T,1536]` is the
feature fed to the Reader; padded frames are zeroed in training (`TM:encoder.py:131-132`); at inference the reader masks them.
The text-space projection `embed_audio.embedding_projection` and the pooled 768-d EmbeddingGemma embedding are **not** used.

## 8. TingMix LoRA delta (`TM:encoder.py:44-70,146-165`, `TML:tingmix.json`)

- `y = W x + (alpha / r) * B (A x)` with `scale = 32 / 16 = 2.0`; `A [16, in]`, `B [out, 16]`, both **float32**, dropout 0 at
  inference, computed `(x.to(fp32) @ A.T) @ B.T`, scaled and cast to `x.dtype`, added to `base(x)` (`encoder.py:55-57`).
- It wraps the inner `nn.Linear` of every ClippableLinear whose attribute name is in `q_proj, k_proj, v_proj, post, ffw_layer_1,
  ffw_layer_2` in all 12 layers (`encoder.py:25,60-70`), i.e. 8 linears per layer (q,k,v,post + 2x FFN pairs) -> 192 tensors.
  Clipping stays outside the LoRA sum. Not adapted: `relative_k_proj`, LConv linears, sub-sampling, `output_proj`, norms.
- Key naming in `encoder_delta.safetensors`: `layers.{i}.feed_forward{1,2}.ffw_layer_{1,2}.linear.lora_{a,b}`,
  `layers.{i}.self_attn.{q_proj,k_proj,v_proj,post}.linear.lora_{a,b}`; shapes `lora_a [16,in]`, `lora_b [out,16]` (VERIFIED, header).
- Mathematically mergeable: `W' = W + 2.0 * B A` per target. The reference computes the bf16 base output and the fp32 LoRA term
  separately; a merged bf16 weight rounds differently. Parity thresholds for merged weights must be measured, not assumed
  (EXPERIMENTAL until a teammate measures it).
- `load_delta` rejects unknown tensor names and copies with dtype cast (`encoder.py:153-160`).

## 9. Reader (`E2A:reader.py:9-56`, config `TML:config.json`)

Input `x [B,T,1536]` float32 (`E2A:pipeline.py:203` calls `.float()`), `lengths [B]`; d 768, gru 384, blocks 4, dropout 0.

| Step | Op | Weights | Shape out |
|---|---|---|---|
| mask | `mask[b,t] = t < len_b` | | `[B,T,1]` |
| pre | `LayerNorm(1536, eps 1e-5)` -> `Linear(1536,768)` -> `GELU` (exact erf) | `pre.0.{weight,bias} [1536]`, `pre.1.weight [768,1536]`, `pre.1.bias [768]` | `[B,T,768]` |
| block x4 | `h = h + GELU(LayerNorm(768)(Conv1d(768,768,k5,pad 2)((h*mask)^T)^T))` | `blocks.i.conv.weight [768,768,5]`, `conv.bias`, `norm.{weight,bias} [768]` | `[B,T,768]` |
| GRU | 2-layer **bidirectional** GRU, input 768, hidden 384, over `pack_padded_sequence(h*mask, lengths)` | `rnn.weight_ih_l{0,1}{,_reverse} [1152,768]`, `weight_hh [1152,384]`, `bias_ih/hh [1152]` (gate order r,z,n as in PyTorch) | `[B,T,768]` (zero beyond length) |
| out | `Linear(768,5025)` | `out.weight [5025,768]`, `out.bias [5025]` | `[B,T,5025]` logits |

Parameter count by hand: 1,183,488 (pre) + 11,805,696 (4 blocks) + 5,317,632 (GRU) + 3,864,225 (out) = 22,171,041 = header total.

Porting traps: padding uses `pack_padded_sequence`, so the backward direction starts at each clip's true last frame -- run each
clip at its own length (batch size 1 does this naturally). `drop_gru = 0` in this release, so the single `nn.GRU(num_layers=2)`
path is used (`E2A:reader.py:36-38`). The conv uses zero padding at the sequence edges, after masking.

## 10. CTC and decoders

- Vocabulary: 5,025 units, `blank_id = 0` (`<blank>`), `units.json` is a JSON list of strings (`TML:units.json`, VERIFIED: 5,025
  entries, `[0] = "<blank>"`). Layout: blank, non-ASCII characters (CJK and Latin-extended), then 400-vocab SentencePiece pieces
  (`E2A:tokenizer.py:49`). `tokenizer_bpe.model` is only needed to **encode** text (training/LM building); decoding uses
  `units.json` plus the *presence* of the model (`E2A:tokenizer.py:65`: `... if self.sp else s`). Desktop needs the boolean from
  `config.json` `tokenizer.sentencepiece != null`, not a SentencePiece runtime.
- Greedy: `argmax` per frame over `logits[:len]`, `collapse_ctc` (merge repeats, drop blank, `E2A:ctc.py:4-11`), then
  `Tok.decode`: join units, replace `"▁"` by space, `" ".join(s.split())`, and remove spaces adjacent to non-ASCII characters
  (`_GAP` regex, `E2A:tokenizer.py:11,63-65`).
- LM beam search (`E2A:lm.py:120-185`): log-softmax in float32; per frame keep candidates with `logp > -14` (and blank) among the top
  `topk = 8`; prefix keys are unit-id tuples; score = `logP_ctc + alpha * sum logP_lm(scored chars) + beta * #scored chars`
  (+ `alpha * logP(EOS)` at the end); beam 16. Only single CJK units are scored; any other unit is LM-neutral, and a maximal run of
  neutral units enters the LM context as one placeholder `\x04` (`lm.py:25,113-117,129-141`); the output replaces `臺` with `台`
  (`lm.py:176`). Recommended parameters for this release: alpha 0.35, beta 1.5, beam 16 (`TML:tingmix.json` note, `TML:README.md`);
  library defaults are 0.25 / 1.5 / 16 (`E2A:pipeline.py:153-155`), so the Desktop manifest must carry the recommended values.
- LM file `lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz` (format `eg2-charlm-kn/1`): gzip JSON, order 5, discount D 0.75,
  22,618,443 n-gram entries (per order 4,595 / 561,669 / 3,193,287 / 7,541,286 / 11,317,606), 698,197,123 bytes uncompressed,
  113,932,911 bytes gzipped; loading it in Python peaks at about 6 GB RSS (`results/lm_stats.json`, VERIFIED). Probability
  `_p` is interpolated Kneser-Ney with lower order recursion `1/V` at the root, `V = |unigrams| + 1` (`lm.py:92-98`); this is what
  `tingmix-decoder` must reproduce.
- Text normalisation (`E2A:normalize.py`) is for training/scoring only; the model emits spoken-form text without digits or
  punctuation (`TML:README.md` limitations). Inverse text normalisation is a Desktop product decision, not part of the model.

## 11. Streaming behaviour of Large (what must be re-implemented in Rust)

TingMix-Large is **offline**: bidirectional reader, 29.5 s window (`TM:large/model.py:12-16`). "Streaming" is buffered
re-decoding of the open segment (`TM:streaming/session.py:1-7`): energy VAD (20 ms frames, -50 dBFS, 2 loud frames to start,
10 quiet frames hangover, `vad.py:13`), segment opens on speech with 800 ms preroll, closes after 600 ms non-speech or at
`min(25 s, 29.5 - 0.8 - 0.1) = 25 s`, re-decode every 480 ms of new audio (stretched when slow, `scheduler.py:16-18`),
stable prefix = common prefix of the last 2 partials, finals immutable (`transcript.py:26-33`). Reported numbers (`TM:docs/benchmark.md:25-28`):
per-update compute p50 55 ms on one MI300X, first partial p50 0.55 s, final 0.71 s after segment end -- GPU-server figures,
not Apple Silicon.

## 12. Structural findings relevant to the Apple/Windows backends

1. **The audio tower is causal in attention and in the LConv.** Measured by perturbing one mel frame (random weights, T = 400
   mel frames -> 100 output frames): perturbing mel frame 200, 201 or 203 changes output frames 50..99 and none below 50
   (`results/gemma4_shapes.json`, EXPERIMENTAL for TingMix weights but a structural property of the pinned module). Output frame
   j therefore depends on mel frames up to `4j+3` only (2 stride-2 convs with symmetric padding 1 = 3 frames = 30 ms lookahead),
   each attention layer looks back 11 frames, LConv looks back 4. A tower-level streaming cache (per-layer K/V of the last 11
   frames plus 4 LConv frames) is structurally possible; but the Reader is bidirectional and LoRA/Reader were trained on full
   windows, so streaming the tower alone only helps the compute split, not the Reader. Not implemented here.
2. The past receptive field of the 12 layers is at most `12 * (11 + 4) = 180` frames (7.2 s: 11 attention frames and 4 LConv
   frames per layer), so for the tower alone long chunks gain nothing over overlapping windows; this does not remove the
   need for a full-window Reader pass.
3. Blocked attention (`nb` blocks of 12) can be implemented as dense local attention with the same mask; the `_rel_shift`
   trick is only an implementation device. Any rewrite must reproduce `tanh` soft-cap, `-1e9` masking and fp32 softmax.
4. Ops used: Conv2d (3x3, s2), LayerNorm, ReLU, Linear, RMSNorm, SiLU, GLU, depthwise Conv1d, softplus, tanh, softmax, `unfold`,
   `clamp` with tensor bounds, sin/cos table. No data-dependent control flow once T is fixed; T is dynamic in the reference.
5. Approximate compute per 30 s window (hand-derived from the shapes above, **not measured**): tower 217 GMAC in the 12 layers
   (24.1 MMAC per frame per layer), 1.8 GMAC sub-sampling, 1.2 GMAC output projection, 0.56 GMAC attention matmuls; Reader
   22.1 MMAC per frame, 16.6 GMAC per window. About 0.44 TFLOP for the tower per 30 s of audio.
6. Reference numerics: published results were produced with bf16 on ROCm (MI300X, torch 2.10.0) and the encoder test notes
   "ROCm forward is not bit-deterministic (about 1 % run-to-run on noise input)" (`TM:tests/test_encoder_parity.py:38`,
   `E2A:docs/reproducibility.md:16`). Tolerances for fp32/CPU, MPS, MLX and Core ML parity should be set against that spread,
   not against zero.

## 13. Unresolved

- `embed2asr.encoder` loads the **whole** 1.49 GB checkpoint and discards the rest (`E2A:encoder.py:33-37`); the Desktop must read only the
  audio tower byte range (see `model-artifact-inventory.md`).
- No golden activations from the training side exist in either repository (only random-weight tests). Request R2 in
  `requests-to-training.md`.
- No conversion or accuracy result for TingMix-Large on Apple Silicon or Windows is claimed here; see teammates' reports.
