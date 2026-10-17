# Rust decoder parity (`crates/tingmix-decoder`)

Reference: `embed2asr@b58275785501c084610b1570f1a446a31da5f933` (`ctc.py`, `tokenizer.py`, `lm.py`, `normalize.py`).
Release files: `tc3oliver/TingMix-Large@75071a1260fb21874f150df6ebd53716ac423f13` (`units.json` sha256 `dbc1aba9...7756`,
`lm/lmCN_cv_parl_ascend_ntuml.charlm.json.gz` sha256 `c8251c39fd003df0527f074287acd5094f9da476c3bcd0a5eb86a5d6d9cc5a84`,
113,932,911 bytes; kept outside git at `$MODEL_CACHE/tingmix-large-75071a1/`).

## Status

- Parity with the Python reference: **VERIFIED** on synthetic log-prob matrices (real 5025-unit vocabulary, real release LM).
- Real model output: **VERIFIED** on the CPU fp32 oracle logits of TingMix-Large (see "Real-logit check" below).
- The synthetic matrices are peaked, with confusers, repeats, blanks and flat frames; they exercise every branch of the
  decoder but are not real acoustic output.

## What is compared (`cargo test -p tingmix-decoder`)

| Check | Count | Result |
|---|---|---|
| `Vocab::decode` vs `Tok.decode` (random id sequences + `▁`/non-ASCII corners) | 303 | identical |
| `lm_units` for all 5025 units (incl. 臺->台, non-CJK = neutral) | 5025 | identical |
| Greedy ids + text (4 zh, 3 en, 4 mixed, 1 code-switch heavy) | 12 | identical |
| `CharLm::logp` bit-for-bit, small LM / release LM | 300 / 300 | identical |
| Beam ids + text, small LM, 12 cases x 5 configs | 60 | identical |
| Beam ids + text, release LM, 12 cases x 5 configs | 60 | identical |

Configs (alpha, beta, beam): (0.35, 1.5, 16) model card, (0, 0, 16) plain CTC beam, (0.6, 2.0, 8), (0.35, 1.5, 4), (1.0, 0.5, 32).
With the release LM the beam result differs from greedy in all 12 cases and differs across configs in all 12, so the LM
path is actually decisive in the fixtures. `topk=8` and the `-14` threshold are the reference constants.

Run the release-LM tests with `TINGMIX_LM_PATH=<lm file> cargo test -p tingmix-decoder --release` (they print `SKIPPED` and
pass without it). The small LM (`tests/fixtures/decoder/lm_small.charlm.json.gz`, 4.5 KB, built by the reference `CharLM.save`
from 24 sentences written for this test) lets CI run the full beam comparison without the 114 MB file.

Regenerate fixtures: `cd tools/golden-decoder && uv sync && uv run gen.py --release <dir with units.json + tokenizer_bpe.model> [--large-lm <file>]`.
The fixture matrices store float32 bit patterns (sparse rows: `floor` + overrides) so Rust decodes the exact array the reference decoded.

## Real-logit check (`tests/oracle.rs`)

Input: raw reader logits of the 8 synthetic-TTS clips from `experiments/macos/reference` (CPU fp32 oracle, `mix_29s` has
two chunks, joined with `""` or `" "` like `Embed2ASR.transcribe`). Rust applies its own f32 `log_softmax_rows`, then greedy
(argmax of the raw logits, as `pipeline.greedy`) and beam+LM (0.35 / 1.5 / 16, release LM), and compares the text to
`tests/fixtures/large/expected_texts.json`.

Result: 8 of 8 clips identical for greedy and for beam+LM (`TINGMIX_ORACLE_DIR=experiments/macos/reference/out`
`TINGMIX_LM_PATH=<lm> cargo test -p tingmix-decoder --release --test oracle`). Without the env vars only the two committed
clips (`zh_02s`, `en_02s`) run, greedy only. This includes the clips where beam+LM differs from greedy (`zh_16s`, `mix_29s`).

Divergence 8: Rust's `log_softmax` sums in a different order than torch, so log-probs may differ from the reference in
the last f32 ulp. Not observed to change any output here.

## LM load cost (Mac Studio M4 Max, release build)

| | Rust | Python reference |
|---|---|---|
| Load time | 2.5 s | 19.7 s |
| Memory after load | 579 MB RSS | ~7.0 GB peak RSS (dict of 22.6 M str keys) |
| Beam decode, 54-82 frames, beam 16 | 4-8 ms (first call; context cache cold) | not timed |

Debug builds (`cargo test` without `--release`) take about 115 s to load the LM because dependencies are unoptimised; use
`--release` for the large-LM tests, or set `[profile.dev.package."*"] opt-level = 3` in the workspace manifest.

n-gram counts: 4,595 / 561,669 / 3,193,287 / 7,541,286 / 11,317,606 for orders 1-5; D = 0.75; V = 4,596.
Layout: per order a sorted `Vec<u128>` of packed n-grams (21 bits per char, left aligned) with parallel counts, plus the
`tot`/`typ` history tables derived in one linear pass; lookups are binary searches.

## Divergences

1. **Candidate visiting order.** The reference takes the top-8 with `np.argpartition` and visits them in that arbitrary
   order; Rust sorts them by value, then by id. Scores do not depend on the visiting order (each candidate updates distinct
   accumulators), only the insertion order of equal-score prefixes, which decides ties in the stable sort. Exact score ties
   between different prefixes, or equal values straddling the top-8 boundary above -14, can therefore resolve differently.
   The generator asserts no fixture row has a boundary tie; no score tie occurred in the 120 comparisons.
2. **Platform libm.** `exp`, `log1p` and `ln` come from the system libm in both implementations. Bit-identical on macOS
   arm64 (this machine). On Windows/Linux a last-ulp difference is possible; it could only reorder near-ties.
3. **Input type.** Rust takes `f32` log-probs like the reference's `logits.float().log_softmax(-1)`; it does not apply
   log-softmax itself. Comparisons use the f32 value converted to f64, as `float(row[c])` does.
4. **Defaults.** `BeamConfig::default()` is the model-card setting 0.35 / 1.5 / 16 (the Python `beam_decode` function
   defaults to alpha 0.25; `tingmix-large` documents 0.35).
5. **Tokenizer.** Decoding needs only `units.json`; no SentencePiece runtime. `Vocab::from_units_json(path, sentencepiece)`
   takes the "release ships tokenizer_bpe.model" flag explicitly (the reference keys the join/gap logic on it, not on
   `is_bpe`). Encoding text to ids is not implemented (not part of decoding). `str.split()` whitespace is reproduced,
   including U+001C-U+001F.
6. **LM file.** Only `eg2-charlm-kn/1`, order <= 5, keys sorted and unique (as the reference writer emits); a file violating
   that is rejected instead of repaired. The Python `p` cache bound (4 M entries, then clear) is mirrored; it affects speed only.
7. **JSON floats.** `serde_json` needs the `float_roundtrip` feature; without it the fixture `logp` values parse one ulp off.
