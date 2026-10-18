# Contributing

Thanks for taking a look. A few things matter more than usual in this project.

**Privacy.** Audio and text stay on the user's machine. No telemetry, no cloud calls, and nothing that listens on a
network port. The only network traffic is the model download and app updates, and both happen only when the user starts
them.

**No Python in the app.** Python is fine under `experiments/` and `tools/`. Each folder there is its own
[uv](https://docs.astral.sh/uv/) project, so run `uv sync` inside it.

**Keep big files out of git.** That means model weights, converted models and large dumps. Test fixtures should stay under
about 1 MB each.

**Measure before you claim.** If a change makes something faster or more accurate, include the numbers, the command and the
hardware. Raw benchmark output goes under `benchmarks/` next to the command that produced it. Don't say something runs on
the GPU or the Neural Engine unless you measured it.

**Don't paste other people's content.** That includes transcripts or subtitles of someone else's media, and copied model
output. Generate test data instead.

## Before you open a pull request

```sh
pnpm typecheck && pnpm test
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo deny check
```

If you add a dependency, say why, make sure `cargo deny` passes, and regenerate the licence list with
`python3 tools/licenses/inventory.py`.

Keep each pull request to one change and add a test for anything new or fixed. Any UI text needs both the English and the
Traditional Chinese strings (`src/i18n/`). If you touch model download, verification or signing, explain the security
side in the description.

Commit messages: a short imperative subject, then a body saying what changed and why.

## Issues

Use the issue templates. For recognition problems, include the app version, your OS, and the details from Settings ›
About › Troubleshooting info. Only attach audio you're allowed to share.

By contributing you agree that your work is licensed under Apache-2.0, like the rest of the project.
