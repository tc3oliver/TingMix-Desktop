# Windows validation (physical PC)

Status words follow `CONTRIBUTING.md`. Everything below ran as **native Windows processes** (cargo, rustc, MSVC, PowerShell,
the installer, the app), started over `ssh <windows-test-host>` through WSL interop; nothing counts that ran in WSL/Linux. All model
numbers are **pipeline-only** (seeded random-init weights, no TingMix-Small checkpoint exists).

| Area | Status |
|---|---|
| Rust workspace builds and tests on Windows x64 (fmt, clippy, tests, whole workspace incl. the Tauri app crate, at 699e0ab) | VERIFIED except one open Windows-only test failure (finding 4) |
| Small ONNX backend with the pinned Microsoft ORT 1.31.0 win-x64 dll, loaded by absolute path | VERIFIED (pipeline-only weights) |
| NSIS installer builds, installs per user without elevation, bundles the pinned `onnxruntime.dll` | VERIFIED (unsigned) |
| Installed app starts, shows its window, responds and exits cleanly | VERIFIED (one launch, no interaction) |
| Installed app loads `onnxruntime.dll` from its install directory | NOT VERIFIED in the app: ORT is loaded lazily and no model is installed, so the dll was not loaded at start-up (see below) |
| CPU streaming performance of a Small-shaped model | EXPERIMENTAL (random weights, this PC only; no real checkpoint, so no acceptance gate is decided) |
| GPU execution providers | EXPERIMENTAL probe through ORT Python wheels, not through the Rust/`ort` path |
| Windows ML catalog EPs, microphone capture, UI, signing, SmartScreen, clean-VM install | NOT DONE |

## Machine and installed toolchain

Windows 11 Pro 10.0.26200, AMD Ryzen 7 9700X (8C/16T), 62 GB RAM, NVIDIA GeForce RTX 5070 Ti 16 GB (driver 610.88
reported by `nvidia-smi`, 32.0.16.1088 as Windows driver version), AMD iGPU, WebView2 runtime 144.0.3719.115. The PC was in
use (CPU load 46% at one benchmark start), so latency tails are noisy.

Installed with winget for this validation (versions as reported by winget/the tools):

| Tool | Version | Source |
|---|---|---|
| Rustup | 1.29.1 (`Rustlang.Rustup`; `--scope user` is rejected by that package, installs in the user profile anyway) | winget |
| Rust toolchain | stable-x86_64-pc-windows-msvc, rustc/cargo 1.99.1 (58cf8f9c3 2026-10-13) | rustup |
| Visual Studio Build Tools 2022 | 17.14.42 (installationVersion 17.14.37806.7), workload `Microsoft.VisualStudio.Workload.VCTools --includeRecommended`, MSVC 14.44.35207, Windows SDK 10.0.26100.0 | winget, no UAC prompt (the interop shell is already elevated) |
| pnpm | package 12.11.2; the repo's `packageManager` makes it run 10.34.7 | winget `pnpm.pnpm` |
| Already present | Git 2.51.2.windows.1, uv 0.10.8, Node v24.11.1, winget 1.29.380 | - |
| Fetched by `tauri build` | NSIS 3.13, nsis_tauri_utils 0.5.3 (hash-validated by Tauri) | github.com |

The repo was moved with `git bundle` (sha256 checked) into `C:\Users\User\src\tingmix-desktop` and cloned; the bundle's
`origin` was removed, nothing was pushed.

## Findings (Windows-only failures and what was done)

1. `crates/tingmix-runtime-mlx/examples/bench.rs` used `libc::getrusage`: `clippy`/`test --workspace` did not compile on Windows. Fixed (c8a377f).
2. `crates/tingmix-runtime-onnx/examples/bench_cpu.rs`, same cause. Fixed (ef0741d): peak working set via kernel32, PowerShell for hardware, hash and ORT version.
3. `src-tauri/src/mic.rs` `MicStatus` variants are only constructed on macOS, so `-D warnings` failed on Windows. Fixed upstream (2b3fcdd).
4. **Open (decoder owner):** `crates/tingmix-decoder/tests/golden.rs` `small_lm` calls `ps -o rss=` in `rss_mb()`; on Windows it fails with "program not found". Needs a `cfg(windows)` branch or no RSS assertion.
5. Chunked-vs-offline log-prob agreement of the Small step graph is 1.3e-4 on this x86 PC (9.5e-7 on Apple silicon): ORT's CPU kernels round per matrix shape. CTC ids stay identical. The test tolerance went from 1e-4 to 5e-4 (22e8c4b).
6. Packaging nit: the Windows installer contains `en.lproj\InfoPlist.strings` and `zh-Hant.lproj\InfoPlist.strings`, which are macOS resources (harmless, 297 bytes).
7. A workspace-wide test run unifies the `load-dynamic` feature that `src-tauri` requests, so the ONNX tests ran without a linked ORT and all 4 failed ("no ONNX Runtime library configured"). Same on macOS. Fixed in the tests (699e0ab: `TINGMIX_ORT_DYLIB`, else the `tools/ort-fetch` copy); CI and fresh checkouts must run `tools/ort-fetch/fetch.sh` before a workspace test run (`docs/release/onnxruntime.md`).
8. `cargo test --workspace` stops at the first failing test binary; use `--no-fail-fast` to see all results.

Native result at 699e0ab, whole workspace including `tingmix-app`: `cargo fmt --all --check` exit 0; `cargo clippy --workspace
--all-targets -- -D warnings` exit 0; `cargo test --workspace --no-fail-fast`: 34 test binaries ok, 1 failed (`tingmix-decoder`
`golden::small_lm`, item 4). Includes audio 8, core 6, streaming 12+5+7, runtime-onnx 6+1+4, models/registry 21+5+4+probe,
bench 3; the ONNX tests also pass with `--no-default-features --features load-dynamic` and `TINGMIX_ORT_DYLIB` set to the
pinned dll. (The MLX crate builds as its stub off Apple silicon.)

## ONNX Runtime from the install: absolute path, not System32

- `tools/ort-fetch/MANIFEST.tsv` archive re-downloaded on Windows: zip sha256 `3958d8a4...2195` matches; `onnxruntime.dll`
  sha256 `0c42dc4e...f4c` and `onnxruntime_providers_shared.dll` `e45d471e...723` equal `FILES.sha256`; Authenticode status
  Valid, signer Microsoft Corporation; version 1.31.0.20261008.6.402979a.
- `C:\Windows\System32\onnxruntime.dll` exists and is an older 1.17.260613 (Windows ML). The hijack risk is real.
- `run_wav.exe` (load-dynamic) with the absolute path: `(Get-Process).Modules` lists exactly one onnxruntime, the vendor one; the System32 copy is not loaded. No path configured: clean error, nothing loaded. Relative path: rejected.

## Small CPU benchmark (pipeline-only)

`benchmarks/windows/fp32-prod-shape-msft-ort-1.31.0.json` (command, hardware, model sha256 inside): prod-shaped random model
(62.4 MB), Microsoft ORT 1.31.0 CPU EP via load-dynamic, 60 s of fixture speech pushed as fast as possible through frontend +
step graph + CTC, per (threads x chunk) in its own process. RTF 0.0050 (8 threads, 640 ms) to 0.0200 (1 thread, 160 ms); push
p95 2.3 to 8.8 ms; zero pushes slower than their chunk; peak working set 108 to 110 MB. Single-thread 320 ms: push p50 4.3 ms,
p95 5.3 ms (the same graph on the M4 Max: 2.4 / 2.7 ms). 3600 s of audio pushed as fast as possible
(`soak-3600s-fp32-msft-ort-1.31.0.jsonl`): RTF 0.0142 (1 thread) and 0.0106 (4 threads), p95 5.8 ms, no push slower than its
chunk; peak working set 340 MB (it includes the 230 MB audio buffer the benchmark holds).

Real-time 60-minute soak (`soak-realtime-60min-onnx-pyke-ort.json`, `tingmix-bench --backend onnx --model prod.onnx --threads 4
--duration-s 3600 --soak`, 20 ms chunks, 281 synthetic utterances, tail-partial live policy): all 9 soak assertions pass. First
partial P50/P95 0.057/0.073 s from onset; final latency from speech end P50/P95 0.796/0.812 s (= VAD hang 0.2 s + silence_ms 0.6 s,
i.e. about 0 s after the endpoint because the final reuses the tail partial); streaming RTF 0.0079; max 1 pending final,
nothing left pending, no drift of final latency (early 0.796 s, late 0.796 s), RSS max 115.5 MB with slope -14 MB/h, no stuck
inference call, 0 backend errors. Caveats: random-init weights (the text means nothing), this run linked ort's own ORT 1.31.0 prebuilt
(default `download-binaries`), not the pinned Microsoft dll; the binary was built at 22e8c4b and the report's `git_sha`
(ffb0b91) is the checkout at the end of the run; a `cargo clippy/test` workspace build ran on the same PC around minute 5-10 of
the hour. Its `partial_staleness` metric (about 5.3 s, flat) is a property of the harness's definition for the Small model, not of drift.

## Execution-provider probe (`tools/ep-probe/probe.py`)

Same step graph (prod shape, 320 ms steps over a real log-mel fixture), each EP the ORT **Python wheel** offers, strict mode
(no CPU fallback) first. Raw: `benchmarks/windows/ep-probe-{cpu,directml,gpu-cuda}.json`.

| EP (wheel) | Init | Whole graph on the EP | Max abs d-logp vs CPU | argmax agreement | step p50 ms (CPU in the same run) |
|---|---|---|---|---|---|
| CPU (onnxruntime 1.31.0) | ok | yes | 0 | 1.0 | 2.1-3.0 (baseline) |
| DirectML (onnxruntime-directml 1.24.4) | ok | yes | 1.5e-5 | 1.0 | 0.89 (2.68) |
| CUDA (onnxruntime-gpu 1.31.0 with pip `nvidia-*-cu13` and cuDNN 9 wheels, no system CUDA) | ok | yes | 1.9e-4 | 1.0 | 1.56 (2.09) |
| TensorRT (same wheel) | silently CPU: `nvinfer_10.dll` missing | no | - | - | not applicable |
| Azure | listed, no GPU relevance | - | - | - | - |

What this does and does not show: DirectML and CUDA run this graph correctly and, at a 320 ms step, take about 1 ms; the CPU
EP already takes 2-4 ms, i.e. about 1% of real time, so there is no latency reason to enable a GPU for Small. Numbers come from
random weights and the Python wheels (DirectML wheel is 1.24.4, not the pinned 1.31.0; the pinned Rust path was not built with
the `directml`/`cuda` features). Power, thermals and contention with the user's GPU workload were not measured. The Windows ML
catalog EPs (NvTensorRtRtx, OpenVINO, MIGraphX, VitisAI) were **not** tried: they need the Windows App SDK runtime and a
download through Windows Update.

## Installer (NSIS) and installed app

- Build: `pnpm install --frozen-lockfile` then `pnpm bundle:windows` (= `tauri build --bundles nsis --config src-tauri/tauri.ort-windows.conf.json`), 1 m 51 s of Rust compile, built from 22e8c4b, output `target\release\bundle\nsis\TingMix_0.1.0_x64-setup.exe`, 10.54 MiB (11 052 493 bytes), sha256 `d8ea9b8fe5dff6260c43caeff4ad9461384f0bdb1da6a1f535c1762f8146c1ce`, **not signed** (Authenticode: NotSigned).
- Silent per-user install (`/S`, no elevation, exit 0) to `C:\Users\User\AppData\Local\TingMix`: `tingmix-app.exe` 19.2 MB, `uninstall.exe`, `onnxruntime\onnxruntime.dll` (sha256 equals the pinned `0c42dc4e...`), `onnxruntime_providers_shared.dll`, `LICENSE`, `ThirdPartyNotices.txt`; uninstall entry under HKCU (`TingMix 0.1.0`).

## Installed app: one launch

With the lead's and the user's approval, the installed `tingmix-app.exe` was started once from
`C:\Users\User\AppData\Local\TingMix`, left alone for 10 s (no mic, no UI interaction), inspected and closed:

- process running and `Responding: True`, window title `TingMix`;
- loaded modules matched against `onnx|DirectML`: **none**. The app preloads a backend only when a model is active, and no model
  is installed on this PC, so the ORT dll was not loaded. That the install-dir `onnxruntime.dll` is the pinned one is proven by
  its sha256 (above); that Rust loads exactly such a dll by absolute path and never the System32 copy is proven with `run_wav.exe`;
  the combination inside the app (model activated, ORT loaded from `...\TingMix\onnxruntime\`) still needs the manual steps below;
- `CloseMainWindow()` ended the process within 8 s; no `tingmix-app` process remained; the number of `msedgewebview2` processes
  (the user's other WebView2 apps) was unchanged (19 before and after);
- the run created `C:\Users\User\AppData\Roaming\app.tingmix.desktop` and `...\Local\app.tingmix.desktop` (app and WebView2 data).
  Uninstall: `C:\Users\User\AppData\Local\TingMix\uninstall.exe` (registry entry under HKCU); the data folders stay unless removed by hand.

## What is not done and exact manual steps for the user

Not done: microphone capture and the UI on Windows, Windows privacy prompt for the microphone, SmartScreen behaviour of an unsigned
installer, a clean-VM install (this PC already has the VC++ runtime, WebView2 and the dev toolchain), code signing, the offline
(`bundle:windows:offline`) installer, Windows ML catalog EPs, any real TingMix model, uninstall/upgrade paths.

Manual steps (about 10 minutes, with a model available):
1. Run `TingMix_0.1.0_x64-setup.exe` (build: `pnpm bundle:windows`, or the installer from this validation). Expect SmartScreen
   "unknown publisher" because the installer is unsigned; no elevation prompt should appear.
2. Start TingMix from the Start menu. In Models, import or install a model and activate it (the ORT dll is only loaded now).
3. While the app runs, in PowerShell: `(Get-Process tingmix-app).Modules | ? ModuleName -match onnx | select FileName` - it must show only
   `...\AppData\Local\TingMix\onnxruntime\onnxruntime.dll`, never `C:\Windows\System32\onnxruntime.dll`.
4. Live tab: choose the microphone, press Start. Windows Settings > Privacy & security > Microphone must allow desktop apps; speak and
   check that partials revise and a final appears; unplug/disable the microphone and check that the app reports it instead of hanging.
5. Quit, and check Task Manager shows no `tingmix-app.exe` left.
