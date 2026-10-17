# TingMix Desktop — repo rules

Local-only speech recognition desktop app (Tauri 2 + React/TS + Rust). Pinned sources: `docs/research/PINNED.md`.

## Hard rules

- Never modify the training repos (`tc3oliver/tingmix`, `tc3oliver/embed2asr`). Needed changes become a written
  request in `docs/research/requests-to-training.md`.
- Never fabricate weights or results. Status words: VERIFIED (measured with the real TingMix model), EXPERIMENTAL
  (implemented / officially supported, TingMix parity or speed unmeasured), BLOCKED (missing op/artifact/permission).
- No Python in the shipped app. Python lives only under `experiments/` and `tools/`, each harness its own `uv` project
  (`uv sync`, pinned deps; tingmix/embed2asr installed from git at the pinned commits).
- Model weights, converted models and large dumps never enter git (see `.gitignore`). Large outputs go under
  `experiments/**/out/` or `$HF_HOME`. Small golden fixtures (< 1 MB each) may live in `tests/fixtures/`.
- Raw benchmark output is kept verbatim (JSON) next to the exact command, hardware and artifact sha256.
- No public network listeners, no cloud inference.

## Workspace

- Cargo workspace: root `Cargo.toml`, members `crates/*` (+ `src-tauri` once it exists). Shared deps go in
  `[workspace.dependencies]` only when two crates need them.
- Verify Rust: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`.

## Git

- Several agents commit here concurrently: stage only your own paths (`git add <paths>`), never `git add -A`;
  retry if `index.lock` exists. Small commits, imperative subject, body says what changed and why.
- No attribution trailers (no Co-Authored-By, no "Generated with"). No remote yet; never push.
