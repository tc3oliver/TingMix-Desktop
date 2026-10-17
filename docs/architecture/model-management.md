# Model management (crate `tingmix-models`)

Implements BRIEF sections 11 and 12. Status: EXPERIMENTAL (unit and integration tests only; no real download of the
1.7 GB TingMix-Large set has been run through it, and no UI uses it yet).

## Trust model

- A **signed manifest** is the only thing trusted. `SHA256SUMS` from a model repo is never trusted by itself.
- Envelope: `{ manifest, key_id, signature }`. The signature is Ed25519 over `serde_json::to_vec(&Manifest)` (compact JSON in
  struct field order; no maps, no floats). Parsing uses `deny_unknown_fields`, so unsigned data cannot ride along.
  Verification is `verify_strict` against the key named by `key_id` in `keys::TRUSTED` (compiled into the app).
  `VerifiedManifest` can only be built by `Verifier`, and `Registry::prepare` takes nothing else.
- Every file has `size` and `sha256`; the manifest binds the file list, URLs, licences and runtime requirements.
- Only data formats install: extensions `safetensors onnx json gz model txt md mlmodel bin npy`. No `.pt/.pkl/.py`, nothing
  is executed, unpickled or `trust_remote_code`'d. Paths must be relative, without `..`, `\`, `:`; duplicates are rejected
  case-insensitively.
- URLs must be `https://` and contain `/resolve/<40-hex commit>/` (immutable Hugging Face revision, never `main`).
  `Registry::allow_loopback_http(true)` (tests only) additionally allows `http://127.0.0.1:<port>/`.

### Signing key management

- Dev key: `~/.config/tingmix-desktop/dev-signing.key` (mode 600, directory 700, outside the repo, 64 hex chars).
  Its public half is embedded as `tingmix-dev-1` in `src/keys.rs`. It is a DEV key: replace by a release key held by the
  publisher (ideally on a hardware token / offline machine) before any public build.
- Tool (`cargo run -p tingmix-models --bin manifest-sign -- ...`): `keygen <file>` (create_new, mode 600, prints the public key
  only), `pubkey`, `sign <key> <key-id> <in.json> <out.json>` (refuses a key file that is not mode 600; the key is never an
  argument or output), `verify <signed.json>`.
- Rotation: add the new key to `TRUSTED`, ship an app update, sign new manifests with it, remove the old key in a later
  update. Compromise response: ship an update that drops the key. There is no online revocation list.
- Known gap: no anti-rollback. An attacker who can serve an old, validly signed manifest can make the app install an
  older version. Mitigation to add before release: a monotonically increasing `manifest_serial` plus a persisted
  high-water mark, and a signed expiry.

## Manifest (schema_version 1)

`id`, `family`, `revision`, `backend`, `os`, `arch` (`any` allowed), `precision`, `requirements` (`min_ram_bytes`,
`cpu_features`, `apple_chip_min_generation`), `runtime` (`name`, `min_version`), `base_model` / `adapter` / `tokenizer` /
`lm` (`id` + immutable `revision`), `files[]` (`path`, `component`, `url`, `size`, `sha256`), `licenses[]` (`id`, `name`,
`spdx`, `url`, `applies_to` components, `acceptance_required`, `attribution`, `notes`), `compat` (`status`, `rtf_x1000`,
`notes`), `created`, `min_app_version`. Every file component must be covered by a licence entry.

The first real manifest is `crates/tingmix-models/fixtures/tingmix-large-source.{manifest,signed}.json`: TingMix-Large
`75071a1` (reader, config, units, BPE model, encoder delta, tingmix.json, 114 MB char 5-gram LM) plus the
`google/embeddinggemma-2@914f7f8` base files (`model.safetensors` 1.49 GB, config and feature-extractor configs) under
`base/`. Total 1,713,866,623 bytes. Hashes for the TingMix files equal the repo's `SHA256SUMS` (asserted by a test); the
Google hashes are the LFS oid / measured sha256 of the pinned revision. Status `EXPERIMENTAL`: these are source-format
weights for converters and the PyTorch reference, no native backend consumes the version directly, and the
`source-safetensors` backend is not registered for selection.

Licences in that manifest: `tingmix-large-model` (Apache-2.0 declared; notes the CC BY-NC Parliament data and NTUML2021
train split without a stated licence, commercial use unresolved) and `google-embeddinggemma-2`. Acceptance is required for
both. For the Google base: on 2026-10-17 the Hugging Face API reported `gated=false` and licence tag `apache-2.0` (README:
Gemma 4 licence page), whereas the TingMix docs say users accept Google's terms and the Gemma prohibited-use policy.
Acceptance is therefore required conservatively; the licence audit (task 1) should settle it. The installer does not
carry a Hugging Face token and cannot download gated repos; if Google gates the repo later, the app must send the user to
the page to accept and import the files offline.

## On-disk layout and operations

```text
<root>/versions/<id>/...  manifest.signed.json   installed, verified (renamed in whole)
<root>/tmp/<id>/...  *.part                       staging, kept for resume
<root>/acceptances/<licence>.json                 user acceptance record (licence id, name, url, manifest, time)
<root>/active  previous  selection.json           one-line files replaced only by write-temp, fsync, rename
```

- **Install (network)**: `Registry::prepare` validates the manifest, checks licence acceptance and refuses an existing
  id; then `Downloader::install` fetches each file to `*.part`, resuming with `Range: bytes=N-` (a 200 reply restarts, a 416
  discards the partial, more bytes than the manifest size is an error), honours an `AtomicBool` cancel flag (partials are
  kept), hashes each finished file (a mismatch deletes the partial), and `Registry::commit` re-verifies the exact file set,
  sizes and hashes (extras and symlinks rejected) before the single directory rename into `versions/`. Nothing is visible
  as installed until that rename.
- **Install (offline)**: `import_dir` / `import_zip` take a folder or zip laid out like an installed version. Same
  verification; the zip is never extracted under its own entry names (no zip-slip), extra entries are rejected, reads are
  capped at the manifest size.
- **Activate**: validates the target (signature, file sizes), writes `previous` = old active, then `active` = new, each by
  atomic rename. A crash leaves the old pointer, or `previous == active`; stale `*.tmp` pointers are removed on open.
- **Rollback**: makes `previous` active again and keeps the replaced version as `previous` (so it can be undone); refuses if
  the previous version is damaged, leaving `active` untouched. `remove` refuses the active and previous versions.
- **Load**: `open_active` / `open_model` read only the local cache: re-verify the manifest signature and each file's
  existence and size. They never construct an HTTP client (`Downloader` is the only module that does). Full re-hashing
  (about 1.7 GB for Large) is `verify_full`, meant for install time and an explicit "verify" action, not every launch;
  a same-size bit flip is therefore only caught by `verify_full`.

## Runtime selection (`select`, `caps`)

`Capabilities::detect()`: OS, arch, CPU features (x86 avx2/avx512f/fma..., arm neon/fp16/dotprod), total RAM, Apple chip
generation (macOS `sysctl machdep.cpu.brand_string`) and a GPU vendor hint. The hint is display-only; a test checks that
changing it does not change the result. Windows GPU/driver detection is not implemented.

`select(installed, caps, backends, known_good)` applies, per installed manifest: OS/arch match, backend registered and
**VALIDATED** (EXPERIMENTAL and BLOCKED are never selected), runtime installed and at least `runtime.min_version`, CPU
features, Apple generation, RAM, and a passing compatibility probe (`BackendReg.probe`; a backend without a probe result
is rejected). Survivors are ranked by measured RTF (probe, else the manifest's), CPU path preferred on ties; the reason
string says whether the winner beat the best CPU path. Every rejection carries a human-readable reason for the
diagnostics panel. `save_known_good` / `load_known_good` persist the choice with a fingerprint of the machine; a valid
known-good choice on an unchanged machine and runtime version is kept, otherwise the reason says what changed
(hardware, runtime version, or the old choice's failure) and the engine re-selects.

Not implemented here: running the probe itself, the "ANE accelerated" label (needs profiling evidence; this crate never
emits it), the diagnostics UI, and the benchmark-vs-CPU job that fills `rtf_x1000`.

## Verification

`cargo test -p tingmix-models` (22 tests: tampered file, truncated file, tampered manifest, wrong key, unknown key and
fields, truncated download plus resume, ignored Range, corrupt partial, cancel, licence-before-request, interrupted
activation, rollback, offline load after the server is gone, zip-slip, shipped fixture signature and SHA256SUMS) and
`cargo clippy -p tingmix-models --all-targets -- -D warnings`. The test HTTP server binds 127.0.0.1 inside the tests only.
