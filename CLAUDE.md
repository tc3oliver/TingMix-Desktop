# TingMix Desktop: guidance for contributors and coding agents

Local-only speech recognition desktop app (Tauri 2 + React/TypeScript + Rust). The human-facing version of these rules is
[CONTRIBUTING.md](CONTRIBUTING.md); pinned research sources are in `docs/research/PINNED.md`.

## Hard rules

- Never modify the training repositories (`tc3oliver/tingmix`, `tc3oliver/embed2asr`). Needed changes become a written
  request in `docs/research/requests-to-training.md`.
- Never fabricate weights or results. Status words: VERIFIED (measured with the real TingMix model), EXPERIMENTAL
  (implemented / officially supported, TingMix parity or speed unmeasured), BLOCKED (missing op/artifact/permission).
- No Python in the shipped app. Python lives only under `experiments/` and `tools/`, each harness its own `uv` project
  (`uv sync`, pinned deps; tingmix/embed2asr installed from git at the pinned commits).
- Model weights, converted models and large dumps never enter git (see `.gitignore`). Large outputs go under
  `experiments/**/out/` or `$HF_HOME`. Small golden fixtures (< 1 MB each) may live in `tests/fixtures/`.
- Raw benchmark output is kept verbatim (JSON) next to the exact command, hardware and artifact sha256.
- No public network listeners, no cloud inference. Never commit a credential or private key; never print one.
- Do not commit machine-specific paths (home directories, external volumes, ssh aliases, LAN addresses); use placeholders
  such as `<repo>`, `$HF_HOME` or environment variables.

## Workspace

- Cargo workspace: root `Cargo.toml`, members `crates/*` and `src-tauri`. Shared deps go in `[workspace.dependencies]` only
  when two crates need them.
- Verify Rust: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`.
  Verify the front end: `pnpm typecheck && pnpm test`. Dependencies: `cargo deny check`.

## Git

- Stage only your own paths (`git add <paths>`), never `git add -A`; several people or agents may work in one checkout.
  Small commits, imperative subject, a body that says what changed and why.
- No attribution trailers (no Co-Authored-By, no "Generated with").
