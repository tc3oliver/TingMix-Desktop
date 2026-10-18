# Packaging results and procedure

Measured 2026-10-18 on the Mac Studio (Apple M4 Max, macOS 27.0), from a clean `git worktree` of commit `33e56e7` (tree without the
uncommitted work of other tasks), Homebrew Rust 1.99.0, pnpm 10.34.7. **The bundle is unsigned and not notarized.** No Windows build was made here.

## Procedure (macOS arm64)

    tools/ort-fetch/fetch.sh macos-arm64        # sha256-verified Microsoft ONNX Runtime 1.31.0 into vendor/ (git-ignored)
    pnpm install --frozen-lockfile
    pnpm bundle:macos                           # = tauri build --bundles app,dmg --config src-tauri/tauri.ort-macos.conf.json

Outputs: `target/release/bundle/macos/TingMix.app`, `target/release/bundle/dmg/TingMix_0.1.0_aarch64.dmg`. Cold build including all crates: 7 min 08 s
(`pnpm bundle:macos`, 285 s user CPU). The CI `package` job (tags and manual runs only) does the same on `macos-latest` and `windows-latest`.

## Results (commit 33e56e7)

| Item | Result |
|---|---|
| `TingMix.app` | 64,820 KiB on disk (63.29 MiB by Tauri); `Contents/MacOS/tingmix-app` 21,151,584 bytes; `Resources/onnxruntime/libonnxruntime.1.31.0.dylib` 44,830,288 bytes; ORT `LICENSE` 1,073 and `ThirdPartyNotices.txt` 338,088 bytes; `icon.icns`; `Info.plist` |
| DMG | `TingMix_0.1.0_aarch64.dmg` 19,514,768 bytes, sha256 `c8df5435fbbd2f6b83c416e194e531c67bd478185d39781e43863f710f67df51` (this build only; the bundle is not reproducible bit-for-bit across builds); `hdiutil verify`: checksum valid; mounts; contains `TingMix.app` and an `Applications` link; binary and ORT dylib identical to the ones in the `.app` |
| Bundled ORT | sha256 `2a50a63e7f41f8d4d551d3ee47248df3864f6540109fce8ad520ca3b51af4b70` equals the fetched file; still carries Microsoft's signature (TeamIdentifier `UBF8T346G9`) |
| `otool -L tingmix-app` | 20 libraries, all `/System/Library/Frameworks/*` or `/usr/lib/*` (AppKit, WebKit, CoreAudio, AudioToolbox, Security, SystemConfiguration, OSAKit, IOKit, CoreServices, libSystem, libobjc, libiconv, ...). No `/opt/homebrew`, `/usr/local`, venv, `/Users` or Python path, no `LC_RPATH`. ONNX Runtime is **not** linked (load-dynamic) |
| `otool -L libonnxruntime.1.31.0.dylib` | system frameworks (CoreML, Cocoa, Metal, IOSurface, CFNetwork, Network, ...), `/usr/lib/libc++`, `libz`, `libsqlite3`, and its own `@rpath/libonnxruntime.1.dylib` id with `@loader_path` rpath; no other paths |
| Launch with no Python | `env -i PATH=/bin HOME=<empty dir>`: `which python python3` finds nothing; the app stayed running, RSS about 111 MB after 7 s, one 960 x 700 on-screen window (`CGWindowListCopyWindowInfo`); `osascript -e 'tell application id "app.tingmix.desktop" to quit'` made it exit within 3 s. An earlier build of the same app without ORT (`target/release` of the main checkout, 16.85 MiB `.app`, 5.34 MiB DMG) behaved the same |
| Signature state | ad hoc, linker-signed only; no entitlements embedded (`codesign -d --entitlements` empty); `codesign --verify --deep --strict` fails ("code has no resources but signature indicates they must be present") because resources were added after the linker signature; Gatekeeper would reject it. This is expected for an unsigned development bundle and is exactly what the signing step has to fix |

## Not verified

- Inference from the installed `.app`: this smoke test only covers start, window and quit. Importing a model, probing, selecting a backend and transcribing from the packaged app is covered in `docs/architecture/app.md`.
- Microphone permission prompt and live capture from the `.app` (needs a person at the machine).
- Running from `/Applications` after copying out of the DMG, Gatekeeper/quarantine behaviour, notarization, stapling, hardened-runtime entitlements (library validation versus the Microsoft-signed ORT dylib is discussed in `onnxruntime.md`).
- Anything on a clean Mac without developer tools (this Mac has Xcode, Homebrew and mise; PATH was stripped but the file system was not).
- Windows: the NSIS installer, DLL loading, WebView2 and VC++ runtime handling, MSI. The `windows-latest` CI job is untested because the repository has no remote; the `unzip` fallback in the `package` job (7z) is a guess about the runner image.
- The DMG layout (background, icon positions) was not inspected visually.

## Windows installer configuration (applied in `src-tauri/tauri.conf.json`; not built on Windows)

Applied: `bundle.targets` is `["app", "dmg", "nsis"]` (no MSI/WiX), `bundle.windows.webviewInstallMode` is `embedBootstrapper` (the WebView2 bootstrapper is inside the installer, so an offline first install does not need a download; it only fetches the runtime if missing, so a machine with no network and no WebView2 still fails), `bundle.windows.nsis.installMode` is `currentUser` (no elevation; models and settings are per user). No signing keys are in the config. The file parses with the pinned Tauri 2.12.4 (a macOS build was run with it); the Windows values are unverified until a Windows build.

Fully offline installs: `pnpm bundle:windows:offline` adds `src-tauri/tauri.windows-offline.conf.json`
(`webviewInstallMode: offlineInstaller`), which embeds the complete WebView2 runtime installer so no download is ever needed.
Trade-off: the installer grows by roughly 120-130 MB (Microsoft's standalone x64 runtime; check the actual size of the first
build) versus a few MB for `embedBootstrapper`, so it is opt-in and `pnpm bundle:windows` keeps the embedded bootstrapper.
Windows 11 ships the WebView2 runtime with the OS, and up-to-date Windows 10 machines usually have it through Microsoft Edge
updates; on those the runtime installer is skipped either way, so the offline variant only matters for a Windows 10 machine
without WebView2 and without network. Only the config parse was checked (macOS build with both `--config` files); the size
figure and the install on a clean offline VM are unverified.

Original suggestions:

Existing: `src-tauri/tauri.ort-windows.conf.json` and `pnpm bundle:windows` (`--bundles nsis`). `tauri.conf.json` has `"targets": "all"`, which would
also request MSI on Windows; MSI needs WiX. Suggested changes, to be checked against `https://v2.tauri.app/reference/config/` for the pinned Tauri version before use
(none of them was built here):

1. Keep `bundle:windows` on `nsis` only until an MSI is required; make `targets` explicit (`["app", "dmg", "nsis"]`) so a bare `tauri build` does not depend on WiX.
2. Decide `bundle.windows.webviewInstallMode` explicitly (default downloads the WebView2 bootstrapper; offline machines need `embedBootstrapper` or `offlineInstaller`, which grows the installer).
3. Decide `bundle.windows.nsis.installMode` (`currentUser` avoids elevation, which suits a per-user model cache).
4. Do not put signing settings in the committed config (`certificateThumbprint`, `signCommand`); see `signing-checklist.md`.
5. After the first Windows build verify on a clean VM that `onnxruntime.dll` is loaded by absolute path from the install directory and not from System32.
