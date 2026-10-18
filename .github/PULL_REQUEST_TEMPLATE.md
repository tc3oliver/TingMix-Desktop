## What changed and why

<!-- What was wrong or missing before; what this changes. Link the issue if there is one. -->

## How it was checked

<!-- Commands you ran and their results. Tests that need model files skip themselves: say which you ran. -->

- [ ] `pnpm typecheck && pnpm test`
- [ ] `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
- [ ] `cargo deny check` and `python3 tools/licenses/inventory.py --check` (if dependencies changed)

## Checklist

- [ ] No model weights, converted models, large dumps, secrets or third-party text added
- [ ] New behaviour has tests; a bug fix has a test that failed before
- [ ] Results are labelled VERIFIED / EXPERIMENTAL / BLOCKED honestly; raw benchmark output is kept with its command
- [ ] User-visible text is translated (English and Traditional Chinese)
- [ ] Security-relevant change (model install, signing, updater, Tauri commands)? Added a note below.

<!-- Security note, if any -->
