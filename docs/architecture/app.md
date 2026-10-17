# Desktop app (`src-tauri/`, `src/`)

Status: EXPERIMENTAL. Built and unit-tested. The default backend is `MockBackend`, labelled **"Mock — not ASR"** (banner on
every tab, backend name in the live footer, history entries and diagnostics); it emits placeholder words, never recognized
speech. Real backends are reachable only through runtime selection (next section). Pinned versions: Tauri 2.12.4 / tauri-build 2.7.3 / tauri-plugin-dialog 2.8.3,
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

## Runtime selection (`src-tauri/src/backends.rs`)

The next live session or file job uses a real backend only if all of this holds, otherwise the mock runs and the reason is
shown (Models tab, Diagnostics, banner):

1. an installed model is **active** (Models tab);
2. `tingmix_models::select::select_probed` accepts it: backend registered in this build, runtime present and new enough,
   RAM/CPU/chip requirements met, and a **fresh passing compatibility probe** (Models tab -> "Run compatibility probe": load,
   decode the shipped `zh_02s` fixture, compare with the oracle text, record RTF; stale after a runtime, OS, device or
   model change);
3. both real backends are EXPERIMENTAL and `select` only accepts VALIDATED ones, so they are registered as eligible only
   while **Settings -> Allow EXPERIMENTAL inference backends** is on (default off). The banner then reads
   `EXPERIMENTAL backend: <model> (<backend>) — EXPERIMENTAL`. Their true status is the `status_word` table in `backends.rs`.

A selected backend that fails to load makes Start / file import fail with the error; there is no silent fallback to the mock.

| Backend id | Crate | Status | Needs |
|---|---|---|---|
| `onnx-cpu` | `tingmix-runtime-onnx` (`load-dynamic`) | EXPERIMENTAL; no trained Small checkpoint exists, so no signed manifest for it yet | bundled ONNX Runtime 1.31.0; model component `.onnx` + `units.json` |
| `mlx` | `tingmix-runtime-mlx`, cargo feature `mlx` (off by default) | EXPERIMENTAL; slot only | Apple silicon; `model` (fp32 safetensors), `tokenizer/units.json`, `lm` components |

### MLX Large in the bundle (`tools/bundle-mlx.sh [debug|release]`)

Builds the app with `--features mlx` against `MLX_PREFIX` (default: the wheel in `experiments/macos/mlx/.venv`) and bundles
`Contents/MacOS/tingmix-mlx-worker` (a Tauri sidecar, rpath rewritten) plus `Contents/Frameworks/{libmlx.dylib, libjaccl.dylib,
mlx.metallib}`. Both executables carry only the rpath `@executable_path/../Frameworks`; `otool -L` shows `@rpath/libmlx.dylib`
and no absolute path (`TINGMIX_MLX_DEV_RPATH=1` adds the wheel path for `cargo test` builds only). Debug bundle: 327 MiB
(Frameworks 227 MiB, of which the metallib is 204 MiB, binaries 57 MiB, resources 44 MiB). Not signed with a real identity;
`codesign -s -` ad hoc only, hardened runtime and notarization untested.

The backend is built **on its own thread** (`Dyn::spawn`) and every call is forwarded to it: MLX streams are thread-local, and
a model loaded on one thread and decoded on another failed with `There is no Stream(gpu, 1) in current thread` (found by the
live part of the end-to-end test). Build errors therefore still surface at Start / file import.

Importing a package (folder or zip) now shows the licence step first: the signed manifest inside the package is verified, the
licences are listed, and `models_install` with the ticked ids records them and imports.

**End-to-end test** (`src-tauri/src/e2e.rs`, ignored unless `TINGMIX_E2E_MLX_PACKAGE` is set): offline import with licence
acceptance of `tingmix-large-mlx-fp32` -> activate -> selection refuses (experimental off; then no probe yet) -> probe in the
worker -> selection picks `mlx` (RTF 0.073) -> file transcription of `mix_08s.wav` equals the oracle beam+LM text -> live session
on the replayed wav gives 6 partials and 1 final with the same text. Run on 2026-10-18 from the bundled `.app`
(`Contents/MacOS/e2e-test`, the test binary copied next to the app binary so the bundled libraries are used) with
`sandbox-exec (deny network*)`, 131 s (the import re-hashes 1.4 GB). The package folder is the manifest plus the files it lists
(`$MODEL_CACHE/tingmix-large-mlx-package` here, assembled from `experiments/macos/mlx/out`, the pinned TingMix-Large
release files and `crates/tingmix-models/fixtures/tingmix-large-mlx-fp32.signed.json`; the manifest is signed with the dev key).

    tools/bundle-mlx.sh debug
    MLX_PREFIX=... TINGMIX_MLX_DEV_RPATH=1 cargo test -p tingmix-app --features mlx --no-run
    cp target/debug/deps/tingmix_app_lib-<hash> <app>/Contents/MacOS/e2e-test
    cd / && TINGMIX_E2E_MLX_PACKAGE=<package> TINGMIX_MLX_WORKER=<app>/Contents/MacOS/tingmix-mlx-worker \
      sandbox-exec -f nonet.sb <app>/Contents/MacOS/e2e-test --ignored mlx_import --nocapture

What this does not show: the GUI path (the test calls the same host functions the commands call, not the webview), real
microphone audio through MLX, behaviour with the sidecar after real signing, or anything about real-speech accuracy
(fixtures are synthetic TTS).

## ONNX Runtime bundle

The app depends on `tingmix-runtime-onnx` with `load-dynamic`, so nothing links or downloads ONNX Runtime at build time.
`tools/ort-fetch/fetch.sh` fetches the pinned, sha256-verified library into `vendor/onnxruntime/`; the bundle scripts add it
as Tauri resources through an extra config (kept out of `tauri.conf.json` so plain `cargo check`/`test` work without the
fetch): `pnpm bundle:macos` / `bundle:macos:debug` (`tauri.ort-macos.conf.json`) and `pnpm bundle:windows`
(`tauri.ort-windows.conf.json`; not built or run here, see `docs/release/onnxruntime.md`). At start-up the host resolves
`onnxruntime/<file>` with `BaseDirectory::Resource` and calls `set_runtime_library`; development runs use `TINGMIX_ORT_DYLIB`.
Re-signing the Microsoft-signed dylib with our identity belongs to the signing step (`docs/release/signing-checklist.md`).

## Updater

`tauri-plugin-updater` 2.14.0 is a dependency, but the plugin is registered **only when `plugins.updater.pubkey` is
non-empty in the build's config** (registering it without one makes plugin init fail). No key exists, so this build reports
"Updates are disabled" in Settings and Diagnostics. A release build passes a config with `plugins.updater` (`pubkey`,
HTTPS `endpoints`) and `bundle.createUpdaterArtifacts`; see section 3 of the signing checklist. The frontend never gets
updater plugin permissions: `updater_check` / `updater_install` are app commands that run the plugin in the host, and
install is refused while a live session runs. An update replaces the app bundle only; models live in the app data
directory (`models/`), are verified by their own signed manifests, and an app update never installs, switches or
re-activates one.

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
- Microphone permission (`src-tauri/src/mic.rs`, AVFoundation through `objc2-av-foundation`): `mic_permission_status` /
  `mic_permission_request` / `mic_open_settings`. A bundled macOS app only gets the system prompt when it asks; `live_start`
  now asks first when the status is undetermined and refuses to start capture when it is denied or restricted, with a message
  shown in the Live tab (plus an "Allow microphone access" button and, when denied, "Open System Settings" which opens
  `x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone` from the host). The status is re-read when the
  window regains focus. Windows has no per-app API (status `unsupported`; the OS privacy switch applies).
- Bundle signing: `bundle.macOS.signingIdentity` is `-` (ad hoc) with `hardenedRuntime` false, so the whole `.app` is sealed:
  `codesign -dv` shows `Identifier=app.tingmix.desktop`, `Info.plist entries=16`, entitlement `device.audio-input`. Hardened
  runtime must be turned on for Developer ID / notarization; then the bundled ONNX Runtime and MLX libraries (other teams'
  signatures) need re-signing with our identity or the library-validation entitlement (signing checklist).
- macOS: `Info.plist` has `NSMicrophoneUsageDescription`; `Entitlements.plist` has `com.apple.security.device.audio-input` for
  hardened-runtime builds. Windows needs no manifest entry; the OS microphone privacy switch applies and a denied device
  surfaces as the host error shown under the Start button.
- Closing the window while recording hides it to the tray (tray menu: Show / Quit); otherwise closing quits. On exit the
  live session is stopped and its transcript persisted.

## Not done / limits

- No backend has measured recognition quality in the app. The `onnx-cpu` path was exercised with the random-weight
  `tiny.onnx` graph (pipeline only, no network); the `mlx` slot is not compiled by default.
- File progress shows seconds of audio decoded, not a percentage: `tingmix-audio` does not expose the duration up front.
- SRT cues are not line-wrapped; one final segment is one cue.
- The updater is wired but disabled (no signing key); it has never fetched an update. The debug `.app` is unsigned;
  entitlements only take effect on a signed build.
- The UI was type-checked, unit-tested for the reducer and formatters, and built into the `.app`, but I had no screen or
  accessibility access in the build environment, so the layout has not been looked at.

## Verify

    pnpm install && pnpm typecheck && pnpm test
    cargo fmt -p tingmix-app --check && cargo clippy -p tingmix-app --all-targets -- -D warnings && cargo test -p tingmix-app
    cargo test -p tingmix-app -- --ignored mic_smoke --nocapture   # real microphone through the real pump
    tools/ort-fetch/fetch.sh macos-arm64 && pnpm bundle:macos:debug  # target/debug/bundle/macos/TingMix.app with the ORT dylib
    TINGMIX_ORT_DYLIB=<.app>/Contents/Resources/onnxruntime/libonnxruntime.1.31.0.dylib TINGMIX_TINY_ONNX=tools/small-export/out/tiny.onnx \
      cargo test -p tingmix-app -- --ignored onnx_bundle_smoke --nocapture   # bundled dylib + app code path, run it under a no-network sandbox
