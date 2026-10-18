## What and why

<!-- What was wrong or missing, and what this changes. Link the issue if there is one. -->

## Testing

<!-- What you ran and what happened. Tests that need the real model skip themselves; say if you ran any of them. -->

- [ ] `pnpm typecheck && pnpm test`
- [ ] `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
- [ ] `cargo deny check` (if dependencies changed)

<!-- If this touches model download, verification, signing, the updater or Tauri commands, add a short security note. -->
