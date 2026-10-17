# Signing and release prerequisites

Status 2026-10-18: **none of these exist yet.** No signing identity, notarization credential, Authenticode certificate, updater key or
release manifest key is present in this repository or on the build machine, so CI builds are unsigned development artifacts and must
not be described as signed or notarized (BRIEF section 15). This file lists what has to be obtained and decided; it makes no claim that any
step works. Names of Tauri settings and environment variables were read from the Tauri 2 docs on 2026-10-18
(`https://v2.tauri.app/distribute/sign/macos/`, `.../sign/windows/`, `.../plugin/updater/`) and must be re-checked against the Tauri
version in `Cargo.lock` before first use.

Never put a credential in git, a workflow file, a command argument, shell history or a log. CI secrets go into GitHub encrypted secrets
(environment-scoped, with required reviewers for the release environment); local keys live in mode 600 files outside the repository.

## 1. macOS (arm64 `.app` and DMG)

| Prerequisite | Who / where | Notes |
|---|---|---|
| Apple Developer Program membership | account holder | needed for Developer ID |
| "Developer ID Application" certificate + private key | Keychain on a trusted Mac; export as `.p12` for CI | Tauri env: `APPLE_CERTIFICATE` (base64 p12), `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY` |
| Notarization credential | either Apple ID + app-specific password + team id (`APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID`) or an App Store Connect API key (`APPLE_API_ISSUER`, `APPLE_API_KEY`, `APPLE_API_KEY_PATH`) | prefer the API key (revocable, no personal password) |
| Hardened runtime + entitlements | `src-tauri/Entitlements.plist` currently grants only `com.apple.security.device.audio-input` | re-check once MLX/Metal and the native libraries are bundled (JIT or unsigned-memory entitlements must not be added unless proven necessary) |
| `NSMicrophoneUsageDescription` | `src-tauri/Info.plist` | must exist and be user-readable text |
| Native libraries inside the bundle | `libmlx.dylib`, `mlx.metallib`, ONNX Runtime dylib | every nested Mach-O must be signed with the same identity before notarization; verify with `codesign --verify --deep --strict` and `spctl -a -vv` on a clean Mac |

Open: deployment target (`minimumSystemVersion 13.0` in `tauri.conf.json`) versus the MLX minimum; DMG staple check; Gatekeeper run on a Mac that never had the developer tools.

## 2. Windows (x64 installer)

| Prerequisite | Notes |
|---|---|
| Code-signing option, decided by the owner | OV/EV Authenticode certificate (hardware token or HSM; CI cannot hold a plain exportable key for EV) or Azure Trusted Signing |
| Tauri configuration | `bundle.windows.certificateThumbprint` + `digestAlgorithm` + timestamp URL, or `bundle.windows.signCommand` for a remote signer; secrets via CI secrets, not `tauri.conf.json` |
| Timestamp server | RFC 3161 URL so signatures outlive the certificate |
| Microsoft runtimes | WebView2 (bootstrapper choice in the installer config) and the VC++ runtime needed by ONNX Runtime: decide embed versus download, then test on a clean VM |
| SmartScreen | reputation builds over time even with a valid signature; plan for the first-release warning |

## 3. Application updater (Tauri updater plugin)

| Prerequisite | Notes |
|---|---|
| Updater keypair | generated once with `tauri signer generate`; private key in `TAURI_SIGNING_PRIVATE_KEY` (+ `_PASSWORD`) for release CI only; public key goes into `plugins.updater.pubkey` |
| `bundle.createUpdaterArtifacts` | enable only on the release workflow |
| Update endpoint (`endpoints`) | HTTPS location of the update manifest; the updater is a user-authorised network use, kept separate from model downloads (BRIEF section 15); the app must not auto-activate a model on update |
| Key loss plan | a lost updater private key means existing installs cannot be updated; keep an offline backup and a documented rotation path |

The updater key is **independent** from the Apple/Microsoft identities and from the model manifest key below.

## 4. Model-manifest release key (replaces `tingmix-dev-1`)

Current state (`crates/tingmix-models/src/keys.rs`, `docs/architecture/model-management.md`): the only trusted key is the DEV key
`tingmix-dev-1`, whose private half sits in a developer home directory. Shipped fixtures are signed with it.

| Step | Notes |
|---|---|
| Generate the release key offline | `manifest-sign keygen <file>` on an offline machine or hardware-backed signer; the private key never touches CI or a laptop that browses |
| Embed the public key as a new id (for example `tingmix-release-1`) in `TRUSTED` and ship it in an app release; keep the dev key out of release builds | a build with `tingmix-dev-1` trusted must be refused by the release workflow (add a check) |
| Sign manifests only after the licence decisions E4-E5 (`docs/research/licensing-review.md`; E1-E3 are not pursued by owner decision) and the anti-rollback serial (registry hardening task) | |
| Rotation and compromise | add the new key, ship, drop the old one later; there is no online revocation list (`model-management.md`) |
| Backup | two offline copies, tested restore |

## 5. Release workflow prerequisites (not yet written)

- A separate `release.yml` triggered by a protected tag, running in a GitHub environment with required reviewers; it is the only workflow with signing secrets.
- Reproducible inputs: pinned action SHAs (the current `ci.yml` pins major tags only), `--locked` builds, `cargo deny` and the licence inventory green.
- Notices: attribution screen content from `docs/release/dependency-licenses.md` (MPL-2.0 notices for symphonia and tauri's `cssparser`/`selectors`) and model licence texts.
- Post-build checks on clean machines: signature/notarization status, microphone prompt, offline start with a pre-imported model, no Python on PATH.
- Smoke tests of the installers wait for the app (task #9).
