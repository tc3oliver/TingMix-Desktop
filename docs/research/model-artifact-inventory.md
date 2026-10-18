# Model artifact inventory

Audit date 2026-10-17. Pins in `PINNED.md`. All sizes are bytes as reported by the Hugging Face tree API
(`GET /api/models/<repo>/tree/<rev>?recursive=true`, 2026-10-17) and re-measured on downloaded copies; sha256 values were computed
locally on the downloaded bytes and, for LFS files, equal the Hub LFS `oid`. Status words follow the legend in
`model-inference-graph.md` (VERIFIED = measured on the real released artifact bytes).

Local copies used (disposable, reproducible from the URLs below): `$MODEL_CACHE/tingmix-audit/large-75071a1/` and the
Hugging Face cache at `$HF_HOME` (`google/embeddinggemma-2 @ 914f7f8`). Scripts and raw JSON:
`experiments/audit/{st_headers,eg2_blob_check,lm_stats}.py`, `experiments/audit/results/`.

## 1. TingMix-Large `tc3oliver/TingMix-Large @ 75071a1260fb21874f150df6ebd53716ac423f13`

Hub metadata: `license: apache-2.0`, `gated: false`, last modified 2026-10-17T09:29:19Z, `base_model: google/embeddinggemma-2`
(`README.md` front matter). No `LICENSE`/`NOTICE` file in the repository tree.

| Path | Bytes | sha256 | In `SHA256SUMS`? | Role | Desktop |
|---|---|---|---|---|---|
| `model.safetensors` | 88,687,532 | `67a06c21d757fdfa0b347a84942097e7bf6315fc4b9d3437b58b008cc5b7e271` | yes | Reader, 38 F32 tensors, 22,171,041 params | download |
| `encoder_delta.safetensors` | 22,042,112 | `1b155cfeec560a07b975eea14eee515aa06d75074256989fc510129fab51d9cb` | yes | LoRA r16/a32, 192 F32 tensors, 5,505,024 params, all 12 layers | download; merge or apply at load |
| `config.json` | 660 | `eac5da8ce273202417906279698130b09c810111d75f82882ac323102ceb7431` | yes | format `embed2asr` v1, reader dims, vocab 5025, blank 0, 16 kHz, 25 fps, chunk 29.5 s, encoder id + revision | download, validate |
| `tingmix.json` | 412 | `5b818f0dcd7752539a449f49d834a684518e3feac006ceb929f8d6548d6beb6e` | yes | format `tingmix-large` v1, `encoder_delta` {mode lora, r 16, alpha 32, lora_layers null}, training step 12000, commit `c3522610`, recommended decoding note | download |
| `units.json` | 35,553 | `dbc1aba97abeeae49dc7556531a2f670d4dc5f39705b54ca28e84b8de9187756` | yes | 5,025 output units (JSON list; `[0] = "<blank>"`) | download |
| `tokenizer_bpe.model` | 245,401 | `959e7af76bb54482be709575e7ee3da49dce7f293f1e6e18cc99baaf2eeb0c7a` | yes | SentencePiece BPE model; only needed to encode text. Decoding needs only its presence flag | optional (ship only if text encoding is needed) |
| `lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz` | 113,932,911 | `c8251c39fd003df0527f074287acd5094f9da476c3bcd0a5eb86a5d6d9cc5a84` | **no** | char 5-gram LM, gzip JSON `eg2-charlm-kn/1`; 698,197,123 bytes uncompressed, 22,618,443 entries | download; convert to compact binary offline |
| `SHA256SUMS` | 496 | `04167074fb7ba3178e9c1f075f554b4cdf5bff0db3a9b0661b95e2054009e847` | n/a | unsigned checksum list | not trusted (see below) |
| `README.md` | 3,309 | `b70dfd7415c9719e45c1bb201fbc356fb8c17ddb8f1fe811c734317ed13557f1` | no | model card (license apache-2.0) | keep text for attribution screen |
| `.gitattributes` | 1,519 | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` | no | LFS rules | ignore |

Total of files the Desktop needs from this repo: 88,687,532 + 22,042,112 + 113,932,911 + 660 + 412 + 35,553 = **224,699,180 bytes**
(214 MiB), plus the base-encoder download below.

Findings:

1. `SHA256SUMS` does not list the 113.9 MB LM file, so the LM has no checksum in the repository (the Hub LFS oid matches the
   file we hashed, but only the manifest we sign can authenticate it). `tools/export_release.py:40` hashes only files present in
   the export directory; the LM is not covered by it (VERIFIED: the Hub tree lists the file, `SHA256SUMS` does not).
2. `SHA256SUMS` is unsigned and lives next to the data; goals.md section 11 requires a publisher-signed manifest (Ed25519) rather than this file.
3. `config.json` pins the base encoder by revision (`914f7f89...`), so the pairing is machine-readable (VERIFIED).
4. Reader tensors are F32 (`model.safetensors` metadata `{"format":"pt"}`); LoRA tensors are F32. No pickle is involved in
   these three safetensors files (header-only inspection, VERIFIED). The training export `.pt` is pickle
   (`TM:tools/export_release.py:27` uses `weights_only=False`; `TM:large/model.py:51` unpickles "trusted" files) and must never reach the app.
5. `tingmix.json` records `step 12000` and training commit `c3522610a4df4b6771da54e4202ed3ff15e07ae1` (an ancestor of the pinned
   `f44cbdb`, 2026-10-17 "fix(tools): raise container nofile limit"; VERIFIED with `git merge-base --is-ancestor`), so the weights
   were trained with code older than the audited tip. Training-code changes after that commit do not change the inference graph read
   here, but the audited `encoder.py` is the tip, not the training-time file.
6. Hub repo says nothing about a Small checkpoint: `tc3oliver/TingMix-Small` returns HTTP 401 for the API (same as at pinning time).

## 2. Google base encoder `google/embeddinggemma-2 @ 914f7f89142e33e77833254d9c9b90c3cef7303b`

Hub metadata (2026-10-17): `license: apache-2.0`, `gated: false` (downloadable without login or click-through), last modified
2026-10-06T15:29:23Z, `transformers_version 5.18.0.dev0`. No `LICENSE`/`NOTICE` file in the tree.

| Path | Bytes | sha256 | Needed by Desktop |
|---|---|---|---|
| `model.safetensors` | 1,488,915,288 | `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79` (= Hub LFS oid, recomputed on the downloaded file) | **audio tower prefix only**, see below |
| `config.json` | 4,455 | `b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3` | yes (audio_config), or embed the 25 values in our manifest |
| `preprocessor_config.json` | 511 | `ea2ae257e901064abdd98dceb19f2b0da06af600bed15e0f99f5c85c37ee9d78` | yes (feature extractor parameters) |
| `processor_config.json` | 1,788 | `168f6a08522f3ce5dea596d94d003af2fd691742d4f41fe1f9d8cce76bfbf69c` | no |
| `tokenizer.json` / `tokenizer.model` / `tokenizer_config.json` | 32,170,510 / 4,689,013 / 1,599 | LFS oids `4d777ef5...504828be4`, `e594c8a9...2ef03` / `17bd5d6e...004874` | no (text tokenizer) |
| `chat_template.jinja`, `config_sentence_transformers.json`, `modules.json`, `sentence_bert_config.json`, `1_Pooling/config.json`, `2_Normalize/config.json` | 1,016 / 1,565 / 413 / 747 / 90 / 97 | `4b852efc...`, `031e56a4...`, `3d02572a...`, `b1bcd9f2...`, `8759bdf7...`, `cdb09dfc...` | no |
| `README.md` | 21,730 | `b677a0ee2818c36f091b65fd3d9b4baea9f04d53b02b49cae916c74bf9682037` | no (quote license text for attribution) |

### Audio tower inside `model.safetensors` (VERIFIED on the downloaded 1.49 GB file)

- Header: 8-byte length + `171,296` bytes JSON, data section starts at byte offset `171,304`.
- All 751 `audio_tower.*` tensors (BF16, 304,825,088 elements) occupy one **contiguous** byte range at the **start** of the data
  section: data-relative `[0, 609,650,176)`, equal to the sum of their sizes; no other tensor overlaps it. Everything else
  (language model, vision tower, `embed_audio`) follows.
- Therefore **the first `171,304 + 609,650,176 = 609,821,480` bytes of the file are header + the entire audio tower.**
  One HTTP `Range: bytes=0-609821479` request fetches it (41 % of the file; the Hub honours range requests, used for the header reads here).
  sha256 of that prefix: `657170336d8fa1efa6bd7ef18bfd7a2ece6c6a52a449a33362de2f7e4c310ca2`; sha256 of the 171,304-byte header
  prefix alone: `75e406407f6c97cc0576e839499c190c5182b27f20d0e774636ef7bef0bf9ac8`.
- The prefix is not a valid standalone safetensors file (its header lists tensors beyond the end of the data). A loader must parse
  the header and read only `audio_tower.*` offsets, or the publisher must re-pack a tower-only safetensors file.
- Clipping scalars (480 BF16 values `input_min/input_max/output_min/output_max`): all finite, min -212.0, max 211.0, smallest non-zero
  magnitude 5.78 (`results/eg2_blob.json`). They are real bounds, not +-inf, so they must be applied (fp16 conversion included).

Download-design consequence (recommendation, not yet implemented): the manifest points the Desktop at the immutable Hub URL
`https://huggingface.co/google/embeddinggemma-2/resolve/914f7f89.../model.safetensors` with the byte range above and the prefix
sha256 above (computed and signed by us); the Desktop never needs Google weights re-hosted by us, and saves 879 MB per install.
Whether we may re-host or re-pack instead is a licensing question (see `licensing-review.md`).

## 3. What the Desktop must ship versus download

| Item | Ship in installer | Download on first use | Notes |
|---|---|---|---|
| App binaries, ONNX Runtime / MLX / Core ML runtime libraries | yes | no | no model weights in git or installer (CONTRIBUTING.md hard rule) |
| Apache-2.0 / third-party notices, Gemma Prohibited Use Policy link | yes | no | attribution screen text from the model cards |
| Signed manifest public key (Ed25519) | yes | no | key management is a model-registry task |
| Audio-tower bytes (609.8 MB BF16 prefix, or converted derivatives) | no | yes | hash in signed manifest |
| Reader + LoRA delta + units + config (111 MB) | no | yes | all three safetensors files are pickle-free |
| LM | no | yes, optional | 113.9 MB gz JSON as published; Python needs about 6 GB RAM to parse it, so a compact binary conversion (publisher-side tool) is required before it is a Desktop artifact. Greedy decoding works without it |
| Converted formats (MLX fp16/bf16, Core ML, ONNX) | no | yes | derived artifacts are produced and measured by the backend tasks; none exist in this inventory |

Disk footprint of the straightforward Large install (BF16 tower prefix as published + F32 reader/delta + LM gz): 609,821,480 +
88,687,532 + 22,042,112 + 113,932,911 + small files = about **835 MB** (before any conversion; a merged-LoRA or fp16 copy may double the tower on disk).

## 4. Other artifacts

| Artifact | Status |
|---|---|
| `tc3oliver/TingMix-Small` | **BLOCKED**: API returns 401 (not published). `TM:docs/small-design.md:1` says "interface only; not trained yet". Expected layout (`TM:models/small/__init__.py:4`): `small.json`, `model.safetensors`, `units.json`, `tokenizer_bpe.model`. With the default trainer config (d 512, hidden 640, 5 GRU layers, vocab 5025) the network has 16,192,961 parameters (instantiated from `TM:models/small/model.py:53-62`; no checkpoint exists, so no accuracy, size-on-disk or INT8 figure can be given). |
| `tc3oliver/Embed2ASR-v1 @ b40e1db5bf106bfe4481615591ce34a0c22c6231` | Not used. 9.66M-parameter Reader (goals.md: do not use for Large). Note `TM:large/model.py:9` still has it as `DEFAULT_REPO`. |
| Training exports (`*.pt`), training data, features | Not distributed; never Desktop inputs. |

## 5. Reproducibility check (is everything needed for inference on the Hub?)

For Large: **yes**, with the base encoder from Google. Reader weights, LoRA delta, units, config, decoder parameters (in `tingmix.json`
note and the model card), LM are all in `TML`; the feature extractor and tower architecture come from `EG2` config and the pinned
transformers code (`TF`, `transformers==5.19.0`; embedding_gemma2 appears first in 5.19.0 per `E2A:pyproject.toml:16-17`). Missing items:
(a) golden activations/logits from the training side, (b) a checksum for the LM, (c) a statement of the exact decoder default
(alpha 0.35) in a machine-readable file (only a free-text `note`), (d) licence files in both repos. See `requests-to-training.md`.
