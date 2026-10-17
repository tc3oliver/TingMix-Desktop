# ONNX Runtime in release builds

Release builds use a pinned, sha256-verified **Microsoft ONNX Runtime 1.31.0** loaded at run time from the app
bundle. Development builds keep the default `ort` behaviour (a prebuilt ORT is downloaded at build time).

| | dev / CI default | release |
|---|---|---|
| cargo features of `tingmix-runtime-onnx` | `download-binaries` (default) | `--no-default-features --features load-dynamic` |
| ORT source | pyke build 1.31.0 fetched by `ort-sys` at build time (network) | `microsoft/onnxruntime` v1.31.0 release archive, fetched by `tools/ort-fetch/fetch.sh` |
| integrity | sha256 table inside `ort-sys` | archive sha256 in `tools/ort-fetch/MANIFEST.tsv` + per-file sha256 in `FILES.sha256`, both checked on every fetch |
| linked at build time | yes | no (`otool -L` shows no onnxruntime) |
| located at run time | linked | `set_runtime_library(abs path)`, else `TINGMIX_ORT_DYLIB`, else `ORT_DYLIB_PATH`; relative paths and a missing setting are errors, never a silent search |

`ort` is pinned `=2.0.0-rc.14` (needs ORT API >= 1.31 minor; a newer ORT logs a compatibility note, an older one is refused).

## Fetch

    tools/ort-fetch/fetch.sh [macos-arm64|windows-x64|all]     # needs network once; output in vendor/onnxruntime/ (git-ignored)

| platform | archive (GitHub release v1.31.0) | archive sha256 |
|---|---|---|
| macos-arm64 | `onnxruntime-osx-arm64-1.31.0.tgz` | `1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc` |
| windows-x64 | `onnxruntime-win-x64-1.31.0.zip` | `3958d8a44984160692c1b4ffd0de3cb48f8c39f9d34bb31e433db06e30b82195` |

The pinned hashes are the digests GitHub reports for the release assets, and the downloads matched them (2026-10-18).
That is integrity against a corrupted or swapped download, not an independent signature: the trust anchor is the
`microsoft/onnxruntime` release. The macOS dylib additionally carries Microsoft's code signature
(`codesign -dv`: TeamIdentifier `UBF8T346G9`). Windows ARM64 (`onnxruntime-win-arm64-1.31.0.zip`,
sha256 `ef6206fa...316d5`) is not in the manifest: add a line when an ARM64 build is a target.

## Bundle layout (Tauri resources)

Files to ship, relative to the repo root after `fetch.sh`, and their path inside the app's resource directory:

| platform | source | resource path |
|---|---|---|
| macOS arm64 | `vendor/onnxruntime/macos-arm64/lib/libonnxruntime.1.31.0.dylib` | `onnxruntime/libonnxruntime.1.31.0.dylib` |
| Windows x64 | `vendor/onnxruntime/windows-x64/lib/onnxruntime.dll` | `onnxruntime/onnxruntime.dll` |
| Windows x64 | `vendor/onnxruntime/windows-x64/lib/onnxruntime_providers_shared.dll` | `onnxruntime/onnxruntime_providers_shared.dll` |
| both | `vendor/onnxruntime/<platform>/LICENSE`, `ThirdPartyNotices.txt` | `onnxruntime/LICENSE`, `onnxruntime/ThirdPartyNotices.txt` |

At start-up resolve `onnxruntime/<file>` with `BaseDirectory::Resource`, and call
`tingmix_runtime_onnx::set_runtime_library(path)` before the first model load. Skip the symlinks
(`libonnxruntime.dylib`, `libonnxruntime.1.dylib`) and the `.dSYM`: the versioned file is loaded by absolute path.
The Windows DLL must be loaded by absolute path: Windows ships its own, older `onnxruntime.dll` in System32 for
Windows ML, and a bare name could resolve to that one. (Absolute-path loading on Windows is the design, not tested here.)

## Verified (macOS arm64, 2026-10-18)

- `cargo build --offline --release -p tingmix-runtime-onnx --no-default-features --features load-dynamic --example run_wav`
  builds without downloading; the binary has no onnxruntime link entry.
- `run_wav tiny.onnx mix_04s.wav <dylib>` under `sandbox-exec ... (deny network*)` (curl is refused with exit 6 in the
  same sandbox) loads Microsoft's 1.31.0 dylib and decodes: ids identical to the pyke build used in development.
- `cargo test -p tingmix-runtime-onnx --no-default-features --features load-dynamic` with `TINGMIX_ORT_DYLIB` set:
  11 tests pass against the Microsoft dylib.
- Missing/relative/unset library produces an `Error`, not a panic.

## Not verified

- **Windows target compile.** This Mac has Homebrew Rust with only `aarch64-apple-darwin` std; `cargo check --target
  x86_64-pc-windows-msvc` fails with E0463 (no `core`). Machine rules forbid rustup or a second toolchain manager, so
  it was not worked around. The CI `windows-latest` jobs are the Windows build check. No Windows runtime claim.
- Loading `onnxruntime.dll` on Windows, DLL search order, and the app's behaviour with the System32 copy present.
- **macOS signing/notarization.** The dylib is signed by Microsoft; a hardened-runtime app only loads libraries signed
  by its own team (or Apple) unless the entitlement `com.apple.security.cs.disable-library-validation` is set. Prefer
  re-signing the bundled dylib with our Developer ID in the bundling step over adding the entitlement. Not tested (no
  signing identity here); see `signing-checklist.md`.

## Licence

ONNX Runtime is MIT (Copyright Microsoft Corporation). Redistribution requires its `LICENSE` and
`ThirdPartyNotices.txt` (shipped next to the library, above) and an entry in `dependency-licenses.md`. The Microsoft
Windows archive contains CPU-only code; DirectML and other execution providers are separate downloads and are not bundled.

## Upgrading ORT

1. Bump the URL and `archive_sha256` lines in `MANIFEST.tsv` (take the digest from the GitHub release asset, confirm
   the download hashes to it), run `fetch.sh`, regenerate `FILES.sha256` from the extracted files.
2. Check the `ort` crate's supported ORT API minor; re-run the load-dynamic tests and the parity tests
   (`docs/research/small-onnx.md`).
