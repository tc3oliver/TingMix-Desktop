# Repository audit: tingmix, embed2asr, TingMix-Large, TingMix-Small

Audit date 2026-10-17 (Apple M4 Max, 64 GB, macOS 27.0). Pins: `PINNED.md` (tingmix `f44cbdb`, embed2asr `b582757`, TingMix-Large `75071a1`,
embeddinggemma-2 `914f7f8`). Citations use `TM:` / `E2A:` / `TML:` / `EG2:` / `TF:` as in `model-inference-graph.md`.
Status words: VERIFIED = measured on the real released artifact bytes or real model output; EXPERIMENTAL = random-weight or
read-from-code evidence only; BLOCKED = artifact missing. The audit did not measure any TingMix accuracy or speed itself.

Method: read every file named in BRIEF section 3 in full at the pinned commits (clones in `~/Developer/scratch/tingmix-audit/`), read the
`transformers==5.19.0` Gemma4 audio sources, read safetensors headers of the real artifacts by HTTP range, downloaded and hashed the released
files, ran the upstream Small tests, and wrote targeted Small checks (`experiments/audit/`, own `uv` project, raw JSON in
`experiments/audit/results/`).

## 1. Files inspected

| Repo | File | Read | Notes |
|---|---|---|---|
| TM | `src/tingmix/models/large/encoder.py` (165 lines) | all | tower load, LoRA, mask/pos helpers, delta loading |
| TM | `src/tingmix/models/large/model.py` (80) | all | adapter over Embed2ASR; applies `tingmix.json` delta |
| TM | `src/tingmix/models/base.py`, `streaming/{session,vad,endpoint,stability,scheduler,buffer}.py`, `core/{events,transcript,audio,registry}.py` | all | streaming contract |
| TM | `tools/export_release.py` (46) | all | release layout and `SHA256SUMS` |
| TM | `docs/{benchmark,licensing,model-card-large,streaming,small-design}.md` | all | |
| TM | `src/tingmix/models/small/{model,__init__}.py`, `tests/test_small.py`, `evaluation/cpu_rtf.py`, `training/small/train.py` | all (train.py: config, forward, export) | |
| TM | `tests/test_encoder_parity.py`, `pyproject.toml` | partial / all | slow tests need the real model; not run |
| E2A | `src/embed2asr/{encoder,reader,pipeline,lm,tokenizer,ctc,normalize,audio}.py` | all | `numnorm.py` (text-to-spoken numbers, scoring) not needed for inference |
| E2A | `docs/{limitations,model_card,reproducibility}.md`, `NOTICE.md`, `pyproject.toml` | relevant parts | |
| TML | `config.json`, `tingmix.json`, `units.json`, `README.md`, `SHA256SUMS`, all weights | all | `model-artifact-inventory.md` |
| EG2 | `config.json`, `preprocessor_config.json`, `README.md`, safetensors header and bytes `[0, 609,821,480)` | all | |
| TF | `models/gemma4/{modeling,feature_extraction}_gemma4.py`, `masking_utils.py`, `models/embedding_gemma2/modeling_embedding_gemma2.py` | audio paths | `model-inference-graph.md` |

## 2. Confirmation of the BRIEF's initial values (section 3 of the brief says to verify, not trust)

| BRIEF value | Result | Evidence | Status |
|---|---|---|---|
| Audio tower about 304.8M parameters | 304,824,608 parameters + 480 clip scalars | module built from pinned config; 751 tensors, 304,825,088 elements in the EG2 header | VERIFIED |
| 128-bin feature extraction | 128 HTK mel bins, 0-8000 Hz, 20 ms window, 10 ms hop, FFT 512, magnitude, `log(x+0.001)` | `EG2:preprocessor_config.json`; `TF:feature_extraction_gemma4.py` | VERIFIED |
| Conv subsampling and 12 Conformer layers | 2x Conv2d(3x3, stride 2) + Linear 1024, then 12 Macaron conformer layers (FF - chunked local attention - causal depthwise LConv - FF) | key/shape equality with the real header: 0 missing, 0 extra, 0 mismatched tensors | VERIFIED (structure) |
| 25 fps, 1536-d | T = ceil(ceil(V/2)/2) for V valid mel frames (30 s -> 750; 29.5 s chunk -> 738); 1536 from `output_proj` bias-ful Linear | shape trace + 9 real-model chunks from `experiments/macos/reference/out` | VERIFIED |
| Encoder LoRA r16 / alpha 32 | r 16, alpha 32, all 12 layers, 8 linears per layer, 192 tensors, 5,505,024 parameters | `TML:tingmix.json`, delta header | VERIFIED |
| About 22.2M-parameter bidirectional GRU-CTC Reader | 22,171,041 parameters: pre (LN+Linear 1536->768), 4 conv blocks (k5), 2-layer BiGRU 384, Linear 768->5025 | header + hand count | VERIFIED |
| CTC prefix beam search with character 5-gram LM | yes, alpha 0.35 / beta 1.5 / beam 16 recommended; LM order 5, D 0.75, 22.6M entries | `E2A:lm.py`, `TML:tingmix.json` note, `results/lm_stats.json` | VERIFIED |
| Model-specific tokenizer and normalisation | 5,025 units (blank + non-ASCII chars + 400 BPE pieces); decode uses `units.json` only; `norm()` is training/scoring only; output is spoken-form, no digits | `E2A:tokenizer.py`, `normalize.py`; `TML:README.md` | VERIFIED |
| Not the older 9.66M Reader | Large reader differs (d 768/gru 384/4 blocks vs Embed2ASR-v1 9,655,172 params) | `TML:config.json`; `E2A:docs/model_card.md` | VERIFIED |
| Small: causal log-mel, two strided Conv1D, unidirectional GRU, CTC | yes, 80 mel, 5-layer GRU 640 by default, 16,192,961 params with vocab 5025 | `TM:models/small/model.py` | VERIFIED (code); no checkpoint exists |

## 3. TingMix-Large findings

Severity: H = blocks or endangers the Desktop, M = needs a decision, L = note.

| ID | Sev | Finding | Evidence | Consequence / action |
|---|---|---|---|---|
| L1 | M | The reference loads the whole 1.49 GB EmbeddingGemma-2 checkpoint on CPU and discards everything but the audio tower. The tower is one contiguous 609,650,176-byte range at the start of the data section. | `E2A:encoder.py:33-37`; header analysis `model-artifact-inventory.md` | Desktop reads or downloads only bytes `[0, 609,821,480)`; saves 879 MB. |
| L2 | H | The encoder is **not padding-invariant**: batching or different padding changes output; streaming finals differ from offline for the same audio. | `E2A:pipeline.py:185-188`; `TM:docs/streaming.md` (final-vs-offline MER 4.6 %/0.3 % after tuning) | Parity tests must use batch size 1 and the extractor's own padding (to a multiple of 128 samples); streaming accuracy must be measured against the Desktop's own segmentation, not against offline numbers. |
| L3 | M | File transcription in the reference cuts audio into consecutive 29.5 s chunks with no overlap and no VAD; a word on a boundary can be split; tails under 1,600 samples are dropped. | `E2A:pipeline.py:26,189-193` | The Desktop's file mode can do better (VAD or overlap), but then it is a different algorithm from the one all published numbers use; label it EXPERIMENTAL until measured. |
| L4 | M | Large streaming is buffered re-decoding of the whole open segment every 480 ms (cost grows with segment length, up to 25 s); published latency (p50 55 ms/update, first partial 0.55 s) is from one MI300X GPU, not Apple Silicon or Windows CPU. | `TM:streaming/session.py:1-7`, `docs/benchmark.md:25-28` | Latency targets for Apple/Windows come from the backend benchmarks, not from these numbers. |
| L5 | M | The release recommends alpha 0.35 / beta 1.5 / beam 16 only in a free-text `note`; library defaults are 0.25 / 1.5 / 16. | `TML:tingmix.json`, `E2A:pipeline.py:153-155` | Decoder parameters must live in our signed manifest. Request R3. |
| L6 | M | `SHA256SUMS` omits the 113.9 MB LM file and is unsigned. | `model-artifact-inventory.md` section 1; `TM:tools/export_release.py:40` | Our manifest hashes all artifacts and is Ed25519-signed (task #8). |
| L7 | M | The LM is 698 MB of JSON (22.6M n-grams); parsing it in Python peaks at about 6 GB. | `results/lm_stats.json` | Needs a compact binary format built offline; the Rust decoder already streams the JSON (`tingmix-decoder`, flate2 + zlib-rs), but memory on user machines must be measured. Request R4. |
| L8 | M | Licensing of weights and LM is unresolved and internally contradictory. | `licensing-review.md` | E1-E3 escalated; default is "download from the original repos, non-commercial". |
| L9 | L | Two code paths unpickle research checkpoints (`torch.load(..., weights_only=False)`): `export_release.py:27`, `large/model.py:61`. The released safetensors files contain no pickle. | `TM:tools/export_release.py:27`, `TM:large/model.py:51,61` | Never port these; the Desktop reads safetensors only (BRIEF section 11). |
| L10 | L | `LargeModel.DEFAULT_REPO` still points at the older `tc3oliver/Embed2ASR-v1` (9.66M Reader). | `TM:large/model.py:9` | Do not use the default; always pass the pinned TingMix-Large path. |
| L11 | L | The Reader uses `pack_padded_sequence`, so the backward GRU starts at each clip's true end. | `E2A:reader.py:46-55` | Run each clip at its own length (batch size 1); any padded-batch port must replicate packing. |
| L12 | L | ffmpeg `-ac 1` mixes stereo as (L+R)/sqrt(2) (+3 dB) while array input averages channels; the output has no digits or punctuation. | `E2A:audio.py:4-6`, `TML:README.md` limitations | Desktop picks one downmix rule and documents it; inverse text normalisation is a product feature, not a model feature. |
| L13 | L | Events: `type` partial/final, `segment_id`, `revision` (final = last revision + 1), `text`, `start`, `end`, partial `stable_prefix_len`; finals are immutable and a closed segment rejects revisions. This is the shape the Desktop protocol already mirrors (`tingmix-core`). | `TM:core/events.py:9-35`, `core/transcript.py:8-38` | none |
| L14 | L | Published numbers come from bf16 on ROCm with about 1 % run-to-run noise on noise input; parity thresholds for Apple/Windows must be set from measured spread. | `TM:tests/test_encoder_parity.py:38`, `E2A:docs/reproducibility.md:16` | Reference harness owns this. |
| L15 | L | `docs/benchmark.md` still says "No weights are released yet" although the Hub release is dated 2026-10-17; `TM:docs/model-card-large.md` licence section is a to-do. | `benchmark.md:3,75` | Documentation drift; tracked in `licensing-review.md`. |

Also: `tingmix` pins `embed2asr` by tag `v0.1.0` (`TM:pyproject.toml:22`), not by the commit pinned in `PINNED.md`; `src/` is byte-identical between
`v0.1.0` and `b582757` (only `tools/bench_speed.py` and docs differ, `git diff --stat`), so the pins are equivalent for inference code.

## 4. TingMix-Small audit (BRIEF section 3 checklist)

Upstream tests: `tests/test_small.py`, 5 passed in 0.52 s (`results/upstream_test_small.txt`; torch 2.9.1 CPU). They use a 32/48/2-layer random network and
compare streaming with offline on 3 s of noise at chunk sizes 160, 777, 4000, plus a fake-model session test. They do **not** cover production-size dimensions,
other chunk boundaries, short inputs, tails, or memory. The targeted checks in `experiments/audit/small_stream_audit.py` (random weights, so
EXPERIMENTAL for everything that is about numbers and cannot say anything about accuracy) give:

| Check (BRIEF item) | Result | Detail |
|---|---|---|
| Causal Conv padding | PASS | `F.pad(x,(1,0))` before each stride-2 k3 conv (`model.py:66-67`); stream block re-computation with 4 mel frames of context and `h[1:]` (`model.py:99-107`) reproduces offline outputs exactly: 15 chunkings (1, 7, 159, 160, 161, 511, 512, 513, 639, 640, 641, 1600, 16000, random, whole) at d 512 / hidden 640 / 5 layers / vocab 5025: max abs difference 1.9e-6, argmax identical. |
| Frame count and output shape | PASS | `frames(n) = (n-512)//160+1` for n >= 512; output frames = `frames // 4`; first output frame needs **992 samples** (mel frames 0-3: `3*160+512`); 2,607 lengths (0-2,599, 4,000, 16,000, 16,001, 16,159, 16,160, 16,161, 33,333) agree between closed form and streaming. |
| Chunk boundary behaviour | PASS | see above; plus a deliberately broken variant (GRU state reset every chunk) differs by 0.055 in log-prob, so the test can detect a carry bug. |
| STFT sample remainder | PASS | `samples = samples[n*HOP:]` keeps exactly the unconsumed samples (`model.py:96-97`); equality holds for 1-sample chunks. `torch.stft` centres the 400-sample periodic Hann window inside the 512-sample frame (offset 56; measured), power spectrum `abs^2`, floor 1e-6, HTK mel scale (docstring says Slaney-style; `mel_filters` code is HTK with area normalisation). |
| GRU hidden-state carry | PASS | state `[5,1,640]` passed through `net.gru(h, state)` each call (`model.py:108`). |
| CTC collapse across chunk boundaries | PASS | `Stream.ids` keeps all argmax ids and the collapse is applied to the whole history, so repeats across boundaries merge and blank-separated repeats stay (423 chunk boundaries with equal ids on both sides in the test; ids identical to offline). With random weights blank behaviour is unrepresentative. |
| Variable-length inputs | PASS (batched) / **FAIL (short)** | padded batch of 2.0 s and 5.0 s equals the unbatched results (max diff 1.9e-6); an offline prefix equals the first frames of the longer run (diff 0). **But `SmallNet.forward` / `SmallModel.transcribe_batch` raises `RuntimeError` (conv kernel larger than padded input) for any batch whose longest clip is under 992 samples (62 ms): lengths 1-991, 991 lengths.** A mixed batch with one such clip survives. `Stream.accept` never raises (returns an empty hypothesis). |
| Offline-versus-streaming numerical parity | PASS | 1.9e-6 max log-prob difference at production dimensions; random weights only. |
| Finalize residual | **Gap** | `finalize()` only re-decodes the stored ids (`model.py:114-115`): no flush, no right padding. Audio after the last complete 4-mel-frame block is never decoded: up to **639 samples (39.9 ms)** at the tail, on top of the 62 ms start-up floor. Probably harmless for sentence-level output but it can clip a final short word; a Rust port should flush with zero padding or document the same behaviour. |
| Memory / long streams | **Gap** | `Stream.logp` keeps every frame's log-probs (5,025 floats) and `Stream.ids` every id: 1,809 MB per hour per stream at 25 fps (diagnostic field, not needed for decoding). The collapse re-runs over the full id list on every `accept` (O(n); 112 microseconds at 5 minutes of audio, negligible). Segments are bounded by the session endpointer, so impact is limited, but a port must not copy `logp`. |
| Inputs | note | `accept` expects float32 in [-1, 1]; int16 scaling is the caller's job (`TM:core/audio.py:21-26`). No mono/rate check. |
| Real-model accuracy, RTF | **BLOCKED** | No checkpoint (`TingMix-Small` Hub API 401; `TM:docs/small-design.md:1` "not trained yet"). `evaluation/cpu_rtf.py` needs one. |

Further Small observations: `docs/small-design.md` still names a conformer/Zipformer candidate with 20-80M parameters and int8 ONNX, but the code and trainer are a
16.2M-parameter GRU (default config), so documentation and implementation disagree (`small-design.md:21-26` vs `model.py:52-62`). The release format `small.json` stores
only the network config (`training/small/train.py:212` writes `net.cfg`): the front-end constants (sample rate, FFT 512, window 400, hop 160, 80 mels, mel range
20-7600 Hz, floor 1e-6) are hard-coded in code, so the format cannot detect a front-end change. The vocabulary is Large's 5,025-unit set (`train.py:134-135`,
`--vocab-from`). `cpu_rtf.py` measures only the time inside `accept()` (excludes capture, endpointing and finalisation).

## 5. Unresolved issues carried forward

| ID | Issue | Needed from | Status |
|---|---|---|---|
| U1 | No TingMix-Small checkpoint | training session (R1) | BLOCKED |
| U2 | No golden activations produced on the training side | training session (R2) | requested; the CPU fp32 oracle in `experiments/macos/reference` is the only reference |
| U3 | Decoder parameters not machine-readable; LM not hashed; LM too large for direct use | training session (R3, R4) | requested |
| U4 | Licence status of Parliament, NTUML2021, ASCEND-derived weights and LM | project owner | ESCALATED (`licensing-review.md` E1-E3) |
| U5 | Gemma Prohibited Use Policy: binding or not | project owner | ESCALATED (E4) |
| U6 | Small export interface (streaming ONNX), front-end parameters in `small.json` | training session (R6, R7) | requested |
| U7 | Rust dependency licence scan, Windows ML terms | CI task | open |
