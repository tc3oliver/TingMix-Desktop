# Licensing review for TingMix Desktop

Review date 2026-10-17. **This is an engineering audit of what the sources say, not legal advice.** Pins in `PINNED.md`.
Citations: `TM:` = tingmix `f44cbdb`, `E2A:` = embed2asr `b582757`, `EG2:` = `google/embeddinggemma-2 @ 914f7f8`,
`TML:` = `tc3oliver/TingMix-Large @ 75071a1`. URLs were fetched on 2026-10-17. Items marked **ESCALATE** need a human decision
before the corresponding distribution step; the Desktop must not ship or download the affected artifact by default until then.

## 1. Summary

| # | Question | Finding | Status |
|---|---|---|---|
| E1 | May TingMix-Large weights and the LM be used in a **commercial** or redistributed product? | The Hub card says Apache-2.0, but the maintainer's own review says the Parliament data is CC BY-NC 4.0 and that weights and LM are "likely non-commercial", and that the review is "to complete before any weight release". The weights were released on the Hub anyway. No completed review exists in the repositories. | **ESCALATE** |
| E2 | NTUML2021 training split: licence? | Not stated by the source. Two repo documents disagree on whether it was used for training (model card: "evaluation-only"; benchmark/licensing/Hub card: train split used). The released weights and LM include it. | **ESCALATE** |
| E3 | ASCEND (CC BY-SA 4.0) ShareAlike effect on weights/LM labelled Apache-2.0 | Maintainer's "conservative reading" is that ShareAlike may apply to derived weights/LM. Unresolved. | **ESCALATE** |
| E4 | Does the Gemma Prohibited Use Policy bind Desktop users? | Base model is Apache-2.0 and ungated on the Hub, but its README says "Deployments must adhere to the Gemma Prohibited Use Policy". Whether that is a licence condition is unclear. | **ESCALATE** (low cost: show the policy and require acknowledgement) |
| E5 | May we re-host or re-pack Google weights (tower-only file) or ship an offline bundle? | Apache-2.0 allows redistribution with notices (section 4 of the licence), subject to E4. No `LICENSE`/`NOTICE` file exists in the Hub repo, so the attribution text must come from the card. Not needed if the Desktop downloads from Google's repo directly. | Open, avoidable |
| E6 | People's Speech licence class of the subset used | TingMix lists "People's Speech clean (CC BY)", but the dataset card lists CC-BY and CC-BY-SA variants. Which configuration was used is not documented. | Open (low) |
| E7 | TingMix-Small | No checkpoint, no card, no licence (Hub API 401). | **BLOCKED** |

Recommended default until E1-E3 are answered: the Desktop is distributed **without model weights**; models are fetched by the
user from the original Hub repositories at pinned revisions after a licence screen (no mirrors, no offline bundles distributed
by us); the product is positioned as non-commercial. The offline-import path (BRIEF section 2) stays, importing files the user
obtained themselves.

## 2. Component table

| Component | Licence as published | Source (file:line / URL, date) | Obligations that reach the Desktop | Status |
|---|---|---|---|---|
| `google/embeddinggemma-2 @ 914f7f8` weights | Apache-2.0, `gated: false` | Hub API `https://huggingface.co/api/models/google/embeddinggemma-2` (cardData.license `apache-2.0`, gated false), 2026-10-17; `EG2:README.md:2,32` | Apache section 4: give recipients the licence, keep notices, mark changes if modified (conversion, LoRA merge, quantisation count as changes) | Clear for use; see E4/E5 |
| EmbeddingGemma 2 licence link | README links `https://ai.google.dev/gemma/docs/gemma_4_license`; that URL serves the page titled "Apache License 2.0" (full Apache text) with a banner "This documentation is deprecated. Please visit the new documentation at gemma.ai" | fetched 2026-10-17 | Pin the licence text we show to the Apache text; re-check the new location before release | Open (stale link) |
| Gemma Prohibited Use Policy | Use restrictions (illegal/harmful content, deception, etc.); page "Last modified: February 21, 2024" | `https://ai.google.dev/gemma/prohibited_use_policy`, fetched 2026-10-17; `EG2:README.md:282` "Deployments must adhere to the Gemma Prohibited Use Policy" | Display it and require acknowledgement at install; no technical impact | E4 |
| `tc3oliver/TingMix-Large @ 75071a1` (Reader + LoRA delta + units + LM) | Hub metadata `license: apache-2.0`, `gated: false`; card: "Apache-2.0 (model card)" | `TML:README.md:2`; `TM:docs/licensing.md:9` ("Proposed terms ... Apache-2.0", status "published") | See E1-E3; weights are "trained on" the datasets below | **ESCALATE** |
| TingMix source (`tingmix`), Embed2ASR source | Apache-2.0 | `TM:pyproject.toml:11`, `E2A:pyproject.toml:10`, `E2A:NOTICE.md` | Not shipped (Python); algorithms re-implemented in Rust, so cite and attribute | OK |
| `tc3oliver/Embed2ASR-v1` | Apache-2.0 | Hub API cardData.license, 2026-10-17 | Not used | n/a |
| Character 5-gram LM `lmCN_cv_parl_ascend_ntuml` | "shipped with the model" | `TM:docs/licensing.md:10`: built from CV + Parliament + ASCEND + NTUML-train transcripts | Count tables are derived from CC BY-NC and CC BY-SA transcripts (E1-E3); a CC0-only LM (`LM-O`, Common Voice zh-TW) is "built" but **not published** (`licensing.md:11`) | **ESCALATE**; request R5 |

## 3. Training-data licences (what the weights and LM were built from)

Source for the list: `TM:docs/licensing.md:9`, `TM:docs/model-card-large.md:20-23,37-42`, `TM:docs/benchmark.md:5-10`.
Dataset-card licences from `https://huggingface.co/api/datasets/<id>`, 2026-10-17.

| Data | TingMix claim | Dataset card (2026-10-17) | Notes |
|---|---|---|---|
| Common Voice zh-TW / en | CC0 | `fsicoli/common_voice_17_0`: cc0-1.0 (mirror; Mozilla's own card has no licence field) | Speakers must not be identified (licensing.md:18) |
| VoxPopuli | CC0 | `facebook/voxpopuli`: cc0-1.0, other | Audio notice "(c) European Union - Source: European Parliament" |
| LibriSpeech | CC BY | `openslr/librispeech_asr`: cc-by-4.0 | Attribution (Panayotov et al. 2015) |
| AMI | CC BY | `edinburghcstr/ami`: cc-by-4.0 | Attribution (Carletta et al.) |
| FLEURS | CC BY | `google/fleurs`: cc-by-4.0 | Attribution (Conneau et al. 2022) |
| People's Speech "clean" | CC BY | `MLCommons/peoples_speech`: cc-by-2.0/2.5/3.0/4.0 **and** cc-by-sa-3.0/4.0 | E6: confirm the configuration contains CC BY only |
| ASCEND | CC BY-SA 4.0 | `CAiRE/ASCEND`: cc-by-sa-4.0 | ShareAlike question E3 (`model-card-large.md:39` "conservative reading: ShareAlike for derived weights/LM") |
| Taiwan Parliament speech | CC BY-NC 4.0 | no dataset id given in the repositories | NonCommercial; `model-card-large.md:38`: "weights and the Parliament-derived LM are likely non-commercial". Provenance is not independently verified by this audit |
| NTUML2021 train split | "no licence stated" | no dataset id given | E2; `licensing.md:9,20` says the released model uses the train split; `model-card-large.md:33` and `Embed2ASR docs/limitations.md:28` call it evaluation-only |

Internal contradictions that a maintainer must reconcile (all VERIFIED by reading the pinned files):

1. `TM:docs/benchmark.md:75` says "No weights are released yet"; `benchmark.md:3` and the Hub show a release on 2026-10-17.
2. `TM:docs/model-card-large.md:37` titles the section "Licence review (to complete before any weight release)"; `licensing.md` says "Not legal advice. The maintainer's position:
   dataset licences govern redistribution of the data, which TingMix never does". That position is a maintainer opinion, not a review result;
   it does not address NonCommercial terms attaching to derived models.
3. NTUML2021 is described as evaluation-only (`model-card-large.md:33`) and as training data (`licensing.md:9`, `benchmark.md:7-8`).
4. `TM:docs/licensing.md:13-14` and `TML:README.md:15` say the encoder is "downloaded from Google under its own terms ... and the Gemma
   prohibited-use policy"; the Hub states Apache-2.0 and no gate. The statements are compatible only if E4 is read as "prohibited-use is a use restriction".
5. `TM:README.md:80`: "Model weights (when released) carry the terms of their base model and training data" -- which contradicts a blanket Apache-2.0 label on the Hub card.

## 4. What this means for the Desktop (decisions needed, defaults proposed)

1. **No model weights in git or in installers** (already a repo hard rule). Installers carry only licence notices and the manifest public key.
2. **Licence screen before download**, showing for each artifact: name, revision, size, licence id, link, and the Prohibited Use Policy; the user must
   accept; the acceptance and artifact hashes are recorded locally. For TingMix-Large the screen must show the unresolved data terms (NonCommercial
   note) until E1-E3 are closed.
3. **Never bypass access restrictions**: today both repos are public and ungated. If either becomes gated (HTTP 401/403), the downloader must stop and tell the
   user to accept terms on the Hub and import the file offline; it must not embed or ship credentials.
4. **Offline import** takes only files the user already has; the importer verifies hashes against the signed manifest and does not redistribute.
   An offline bundle produced by us is a redistribution and depends on E1-E5.
5. **Derived artifacts** (MLX/Core ML/ONNX conversions, merged LoRA, quantised copies) are modifications of Apache-2.0 Google weights and of TingMix weights. They must carry
   the upstream notices and a "modified" marker in the manifest, and are not published until E1-E5 are decided.
6. **Attribution screen** (in-app "About / Licences"): Google EmbeddingGemma 2 (Apache-2.0), TingMix and Embed2ASR (Apache-2.0 source, data terms as above), dataset attributions in section 3,
   and the third-party code below (use `cargo-about`/`cargo-deny` output at release time).

## 5. Code dependencies

Shipped or likely to be shipped (licences from crates.io API / GitHub API, fetched 2026-10-17; versions as declared in `Cargo.toml` files at the time of this audit):

| Dependency | Version | Licence | Used by | Note |
|---|---|---|---|---|
| cpal | 0.18.2 | Apache-2.0 | `tingmix-audio` | |
| ringbuf | 0.5.3 | MIT OR Apache-2.0 | `tingmix-audio` | |
| symphonia (features `all`) | 0.6.1 | **MPL-2.0** | `tingmix-audio` | file-level copyleft: fine to link; publish any modifications to Symphonia files; keep notices |
| thiserror, serde, serde_json | 2.0.21 / 1.0.229 / 1.0.154 | MIT OR Apache-2.0 | core, decoder | |
| flate2 (zlib-rs backend) | 1.1.10 / zlib-rs 0.6.8 | MIT OR Apache-2.0 / Zlib | `tingmix-decoder` | |
| hound | 3.5.1 | Apache-2.0 | tests only | |
| tauri | 2.12.4 | Apache-2.0 OR MIT | app (planned) | |
| ort (ONNX Runtime bindings) | 2.0.0-rc.14 | MIT OR Apache-2.0 | Small backend (planned) | release candidate version, pin deliberately |
| ONNX Runtime | repo licence MIT | `microsoft/onnxruntime` | Small / Windows | |
| MLX, mlx-c | MIT | `ml-explore/mlx`, `ml-explore/mlx-c` | macOS Large backend | |
| tokio / reqwest / ed25519-dalek / sha2 / safetensors / rubato / rustfft / ndarray | 1.53.3 / 0.13.5 / 3.0.0 / 0.11.1 / 0.8.0 / 5.0.1 / 6.4.1 / 0.17.2 | MIT / MIT OR Apache-2.0 / BSD-3-Clause / MIT OR Apache-2.0 / Apache-2.0 / MIT OR Apache-2.0 x3 | candidates | not yet in any manifest; confirm when added |

Research-only Python packages (installed under `experiments/**` and `tools/**`, never shipped; PyPI metadata in the audit venv): torch 2.9.1 BSD-3-Clause,
transformers 5.19.0 Apache-2.0, numpy 2.5.4 BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0, sentencepiece 0.2.2 Apache-2.0, safetensors 0.8.0 Apache,
huggingface_hub 2.2.0 Apache-2.0, scipy 1.18.1 BSD, embed2asr 0.1.0 and tingmix 0.0.1.dev0 Apache-2.0.

Open: no Rust licence scan has been run over the lockfile (no `cargo-deny` configuration exists yet); that belongs to the CI/packaging task and must cover transitive
dependencies before the first release. Windows ML / DirectML redistribution terms were not audited here.

## 6. Escalation list (for the project owner)

| ID | Decision needed | Blocks |
|---|---|---|
| E1 | Is the Parliament CC BY-NC 4.0 restriction inherited by weights and LM? Is the product non-commercial? | Any commercial use; mirrors; offline bundles |
| E2 | Was NTUML2021 train split lawfully usable, and is the Hub card correct? | Any distribution of TingMix-Large |
| E3 | ShareAlike (ASCEND) vs Apache-2.0 label | Re-labelling or redistribution of derived weights/LM |
| E4 | Is the Gemma Prohibited Use Policy a binding condition? | Wording of the licence screen |
| E5 | Re-hosting / re-packing Google weights | Offline bundles, tower-only artifact |
| E6 | People's Speech subset | Attribution text |
| R5 | Publish the CC0-only LM (`LM-O`) as a clean alternative | Commercially safe Large variant of the decoder |
