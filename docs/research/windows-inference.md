# Windows inference for TingMix-Small: Windows ML, execution providers, offline install

Documentation research only. **Nothing here was run on Windows**; no Windows performance or compatibility claim is made.
Sources are the Microsoft Learn pages "What is Windows ML?", "Get started", "Windows ML execution providers",
"Accelerate AI models", "Install and deploy Windows ML", "Bring your own EPs" (page dates 2026-04 .. 2026-10, fetched
2026-10-18) and the ONNX Runtime execution-provider index. Re-read them before acting: the EP table and driver minimums
change.

## Decision for the Small model

1. **Mandatory baseline: ONNX Runtime CPU EP.** Small is a 13 M-parameter streaming GRU stack; on a CPU it is a
   batch-1 recurrent workload (see `small-onnx.md`: more threads did not help, a step is milliseconds). A GPU/NPU EP is
   unlikely to win on latency for it and adds start-up cost; "faster" must be shown, not assumed.
2. **Acceleration is opt-in per machine, after proof.** Enable an EP only if, on that device, (a) the real Small ONNX
   model compiles and runs on it, (b) log-probs match the CPU EP within tolerance, and (c) RTF/latency beat the CPU
   baseline. Otherwise keep CPU. Implemented as a startup probe in the runtime selector, never a vendor-based default.
3. **Do not infer support from the vendor.** Each EP below has a hardware/driver floor; operator coverage (GRU,
   Conv, LayerNorm-as-ops, MatMul, Gelu/Erf, LogSoftmax) must be tested per EP.

## Windows ML in one page

- Windows ML is the Windows-maintained copy of ONNX Runtime plus an `ExecutionProviderCatalog` API that downloads
  certified vendor EPs via Windows Update. Same ONNX APIs as ORT.
- Requirements: a Windows version supported by Windows App SDK, x64 or ARM64. CPU and DirectML work everywhere
  supported; **hardware-optimized EPs from the catalog need Windows 11 24H2 (build 26100) or newer**.
- Included with the runtime: CPU EP and DirectML (marked **legacy**). Everything else is downloaded on demand or
  supplied by the app.
- Runtime binaries (self-contained): `Microsoft.Windows.AI.MachineLearning.dll` (~1 MB), `onnxruntime.dll` (~20 MB),
  `DirectML.dll` (~20 MB), about 41 MB total.

### Catalog EPs (Windows ML 2.x and 1.8.x), with the stated requirements

| EP | Vendor | Stated requirement |
|---|---|---|
| MIGraphX | AMD | RDNA 3 or later GPU, AMD driver 25.10.13.09 or later |
| NvTensorRtRtx | NVIDIA | GeForce RTX 30xx or newer, driver >= 32.0.15.5585, CUDA 12.5 |
| OpenVINO | Intel | per Intel's "Windows ML support" guide (driver guidance there) |
| QNN | Qualcomm | Snapdragon X Elite / X Plus, Hexagon NPU driver >= 30.0.140.0 |
| VitisAI | AMD | Adrenalin 25.6.3 with NPU driver 32.00.0203.280 up to 25.9.1 with 32.00.0203.297 |
| WebGPU (experimental, 2.x only) | Microsoft | any DX12 GPU; needs preview NuGet `Microsoft.Windows.AI.MachineLearning` >= 2.4.66-preview |

Silicon mapping from "Accelerate AI models": NPU -> OpenVINO / QNN / VitisAI; GPU -> MIGraphX, NvTensorRtRtx,
OpenVINO, QNN, DirectML (legacy); CPU -> ORT CPU EP, OpenVINO. Microsoft itself recommends **pre-compiling and caching**
the model per EP (multi-second to multi-minute cold start otherwise).

## Mapping to this app (Rust + Tauri, `ort` crate)

- Today the Rust backend uses `ort 2.0.0-rc.14` with ONNX Runtime 1.31.0 prebuilt binaries (CPU EP). On Windows the
  same crate can link its own `onnxruntime.dll` (self-contained, our own pinned ORT) -- this is the lowest-risk path
  and keeps the offline story simple.
- Windows ML integration from Rust is not documented by Microsoft in the pages above: the supported languages for
  framework-dependent are C#, C++/WinRT and Python; **C/C++ is self-contained only (NuGet `Microsoft.Windows.AI.MachineLearning`
  with CMake config, targets `WindowsML::Api`, `WindowsML::OnnxRuntime`)**. A Rust app would have to bind the C API of that
  package (ORT-compatible C API plus the catalog API) or load its `onnxruntime.dll`. Open question, to be prototyped
  on real Windows hardware; do not promise it.
- EP libraries are registered through the ORT API (`RegisterExecutionProviderLibrary(name, dll)`), which is also how
  bring-your-own EPs are used. `ort` exposes `Environment::register_ep_library`; behaviour with the Windows ML
  `onnxruntime.dll` is untested.
- DirectML is "legacy" in Windows ML; do not build new acceleration on it without a measured reason.

## Offline / air-gapped installation

- **Self-contained deployment** (runtime inside the app, no external dependency) is the offline-friendly mode: ship the
  ORT/Windows ML DLLs next to the app; ~41 MB (about 20 MB without `DirectML.dll`, an unsupported but documented
  trim). No Windows Update, no Windows App SDK runtime install.
- **Framework-dependent** needs the Windows App SDK runtime installed on the machine (installer from Microsoft, or via
  Store/Windows servicing); it must be provided by the enterprise image or bundled installer for offline sites.
- **Catalog EPs are downloaded via Windows Update** -- unavailable offline or on managed devices that restrict it.
  Microsoft's documented alternative is **bring your own EPs**: ship the EP DLL package yourself (known packages:
  `NVIDIA/TensorRT-RTX-EP-ABI`, `Intel.ML.OnnxRuntime.EP.OpenVINO`, `Qualcomm.ML.OnnxRuntime.QNN`), register it
  explicitly. Cost: about 80 MB or more per EP, you own version pairing with the ORT inside Windows ML, and you must
  test registration, session creation and representative inference after each update; Microsoft does not guarantee
  forward compatibility of a bundled EP.
- The model and its optional pre-compiled EP cache are app data under our own model registry (hash-verified), so
  offline operation of the model itself is independent of Windows ML.
- Recommended release stance: ship CPU EP only in v1 (self-contained ORT, works offline everywhere), keep the EP probe
  behind a feature flag, and add bundled EPs per vendor only after measured wins on that hardware.

## What must be tested on physical Windows (nothing below is done)

1. Our ORT 1.31 CPU build on x64 and ARM64: parity with the macOS goldens, RTF, RSS, 60-minute soak.
2. Each candidate EP: model compiles, logits within tolerance of CPU, RTF vs CPU, cold start with/without a compiled
   cache, behaviour when the driver is below the stated minimum or missing.
3. Windows ML runtime from Rust (see open question above).
4. Offline machine: install, first run, no network, EP bundled vs absent.
5. Sleep/resume and audio-device changes with an active session.

`benchmarks/windows/` stays empty until those tests run on hardware; cross-compiling does not count.
