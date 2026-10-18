# Contributing to TingMix Desktop

Thanks for helping. This project is a local-only speech recognition app; changes are judged on correctness, privacy and
honest measurement. Please read this page and the [Code of Conduct](CODE_OF_CONDUCT.md) first.

## Ground rules

- **Privacy:** audio and transcripts never leave the machine. No cloud inference, no telemetry, no listening network
  sockets. Network use is limited to user-authorised model downloads and application updates.
- **No Python in the shipped app.** Python lives only under `experiments/` and `tools/`; each harness is its own
  [`uv`](https://docs.astral.sh/uv/) project with pinned dependencies (`uv sync`).
- **No model weights, converted models or large dumps in git** (see `.gitignore`). Small golden fixtures (under 1 MB each) may
  live in `tests/fixtures/`.
- **No secrets** in the repository, in command arguments, logs or fixtures. Signing keys are never committed.
- **Do not copy third-party text** (system prompts, transcripts or subtitles of other people's media, model output you did not
  generate) into the repository. Generate fixtures instead.
- **Report results honestly.** Use the status words: `VERIFIED` (measured with the real TingMix model), `EXPERIMENTAL`
  (implemented or officially supported, parity or speed unmeasured), `BLOCKED` (missing operator, artifact or permission).
  Never claim a successful conversion, GPU or Neural Engine use, signing or accuracy without a measurement. Keep raw benchmark
  output (JSON) next to the exact command, hardware and artifact SHA-256.
- Changes to the training repositories (`tc3oliver/tingmix`, `tc3oliver/embed2asr`) are not made here; write a request in
  `docs/research/requests-to-training.md` instead.

## Setup

See the build section of the [README](README.md). Before sending a change, run what applies:

    pnpm typecheck && pnpm test
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
    cargo deny check                   # licences, advisories, sources (install cargo-deny)
    python3 tools/licenses/inventory.py --check   # docs/release/dependency-licenses.md is current

Tests that need model files or large inputs skip themselves when their environment variables are unset; say in the pull
request which of them you ran.

## Pull requests

- One logical change per pull request, with tests for new logic and for every bug fixed (the failing case first).
- Match the surrounding code: naming, comment density and idiom. Comments say why, not what.
- Commit messages: imperative subject line, a body that says what changed and why it was wrong before. Do not add tool or
  assistant attribution trailers.
- Adding a dependency needs a reason in the pull request, its licence (it must pass `cargo deny`), and a regenerated
  `docs/release/dependency-licenses.md`. Prefer the standard library and existing dependencies.
- Changes to anything that downloads, verifies or activates a model, or that touches signing, need a security note in the
  description (see [SECURITY.md](SECURITY.md)).
- User-visible text goes through the translation files for both English and Traditional Chinese.

## Reporting bugs and proposing features

Use the issue templates. For a recognition problem include the app version, platform, backend and model shown in the
Diagnostics tab, and whether the audio can be shared (do not attach audio you do not have the right to share). Security
problems go through [SECURITY.md](SECURITY.md), not public issues.

## License

By contributing you agree that your contribution is licensed under the Apache License 2.0, the same as the project (section 5
of the license).
