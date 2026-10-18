# Security policy

## Supported versions

The project is pre-release. Only the latest commit on the default branch is supported; there are no released builds yet.

## Reporting a vulnerability

Please report privately, not in a public issue: use GitHub's private vulnerability reporting on this repository
(**Security > Report a vulnerability**, https://github.com/tc3oliver/TingMix-Desktop/security/advisories/new).

Include what you found, how to reproduce it, the affected commit or version, and the impact you expect. You can expect an
acknowledgement within a week. Please allow time for a fix before public disclosure; we will credit you if you wish.

## What is in scope

The application is designed so that audio and transcripts stay on the computer and only user-authorised downloads use the
network. Reports about any of the following are especially welcome:

- Model installation: bypassing signature, size or SHA-256 verification; path traversal or symlink issues in zip/folder import;
  installing a non-data file; rollback to an older signed manifest after the anti-rollback counter has advanced.
- Anything that sends audio, transcripts or identifiers off the machine, or opens a listening socket.
- The Tauri command surface: a webview or crafted file reaching a capability it should not have (file system, shell, arbitrary
  URLs), or content-security-policy bypasses.
- Memory-safety problems in the native code that load untrusted data (audio containers, model files, language-model files).
- Update mechanism: accepting an unsigned or tampered update.

Out of scope: vulnerabilities in third-party models or in upstream dependencies without a way to reach them from this
application (report those upstream), and findings that require a modified build or a compromised operating system.

## Keys and secrets

No private signing key is stored in this repository. The model-manifest public key compiled into development builds is a
development key and is replaced by a release key before any public build (see `docs/architecture/model-management.md` and
`docs/release/signing-checklist.md`). If you find a credential in the repository or its history, report it privately.
