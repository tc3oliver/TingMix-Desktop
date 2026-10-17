# Desktop app (`src-tauri/`, `src/`)

Status: EXPERIMENTAL. Built and unit-tested; the only transcription backend wired in is `MockBackend`, which the UI labels
**"Mock — not ASR"** (banner on every tab, backend name in the live footer, history entries and diagnostics). It emits
placeholder words, never recognized speech. Pinned versions: Tauri 2.12.4 / tauri-build 2.7.3 / tauri-plugin-dialog 2.8.3,
`@tauri-apps/api` 2.12.4, React 19.3.1, Vite 8.3.4, TypeScript 6.0.3, Vitest 5.0.5 (checked on crates.io / npm on 2026-10-18).

## Layout

| Path | Role |
|---|---|
| `src-tauri/src/live.rs` | Live session: capture thread -> `Resampler` -> `SessionWorker` (+ `MockBackend`) -> authoritative `Reducer` -> events |
| `src-tauri/src/files.rs` | File job: `decode_file_chunks` -> `StreamingSession`, cancellable, errors become `Failed` (also `catch_unwind` around the job) |
| `src-tauri/src/models.rs` | `tingmix-models` wrapper: overview, signed-manifest install with licence step, folder/zip import, activate, rollback, remove, verify |
| `src-tauri/src/store.rs` | Settings (`config dir/settings.json`) and history (`data dir/history/<id>.json`), atomic writes, validated ids |
| `src-tauri/src/commands.rs`, `lib.rs` | Typed commands, window/tray/drag-drop handling, shutdown cleanup |
| `src/features/{live,files,history,models,settings,diagnostics}` | React panels; `src/lib/api.ts` is the only place that calls `invoke`/`listen` |
| `src/features/live/reducer.ts` | TS twin of `tingmix_core::Reducer` (tests in `reducer.test.ts`) |
| `src/features/export/format.ts` | TXT and SRT formatting (tests in `srt.test.ts`) |

## Data flow

Host -> UI events: `tingmix://live` (`event` with `session_id`, or `status`), `tingmix://file` (`progress`, `event`, `done`,
`cancelled`, `failed`, tagged with `job_id`), `tingmix://models` (install/import progress), `tingmix://files-dropped`,
`tingmix://drag`. A partial replaces the open one of its segment by `revision`; a final is immutable. The host keeps the
authoritative reducer, so after a window reload the UI calls `live_snapshot` / `file_jobs` and merges the snapshot with the
same idempotent reducer (stale or duplicate events are ignored). The transcript stays after Stop (the last session is kept
in memory and, with "save history" on, on disk after every final).

## Least privilege

- `capabilities/default.json` grants `core:event:default` and one `allow-<command>` per app command (generated from the list
  in `build.rs`). There is no fs, shell, http or opener plugin; the dialog plugin is used only from Rust.
- The frontend never supplies a path: dialogs run in the host (`file_pick`, `models_import`, `models_begin_install`,
  `export_text`), dropped files are registered by the host and referred to by opaque id, history ids are validated
  (`[A-Za-z0-9_-]`, no separators). `export_text` only accepts `txt` or `srt` and content up to 64 MiB.
- CSP: `default-src 'self'`; the page loads nothing remote.
- macOS: `Info.plist` has `NSMicrophoneUsageDescription`; `Entitlements.plist` has `com.apple.security.device.audio-input` for
  hardened-runtime builds. Windows needs no manifest entry; the OS microphone privacy switch applies and a denied device
  surfaces as the host error shown under the Start button.
- Closing the window while recording hides it to the tray (tray menu: Show / Quit); otherwise closing quits. On exit the
  live session is stopped and its transcript persisted.

## Not done / limits

- No real backend: nothing here has measured recognition quality. Models can be installed, verified and activated, but no
  inference backend is registered, so the Models tab shows why none is eligible.
- File progress shows seconds of audio decoded, not a percentage: `tingmix-audio` does not expose the duration up front.
- SRT cues are not line-wrapped; one final segment is one cue.
- No updater yet (BRIEF section 15; task 10). The debug `.app` is unsigned; entitlements only take effect on a signed build.
- The UI was type-checked, unit-tested for the reducer and formatters, and built into the `.app`, but I had no screen or
  accessibility access in the build environment, so the layout has not been looked at.

## Verify

    pnpm install && pnpm typecheck && pnpm test
    cargo fmt -p tingmix-app --check && cargo clippy -p tingmix-app --all-targets -- -D warnings && cargo test -p tingmix-app
    cargo test -p tingmix-app -- --ignored mic_smoke --nocapture   # real microphone through the real pump
    pnpm tauri build --debug --bundles app                          # target/debug/bundle/macos/TingMix.app
