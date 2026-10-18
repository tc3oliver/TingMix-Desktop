# Host message codes

Every user-facing message from the host (`src-tauri`) is a coded string `{"code","params","text"}` (`src-tauri/src/coded.rs`).
The UI translates `host.<code>` (`src/i18n/host.ts`) and falls back to a generic sentence for an unknown code; `text` is the
English fallback for logs and the diagnostics list. **`detail` parameters are raw technical text: show them only in the
Troubleshooting area, never in a sentence a normal user reads.** `mac` is a boolean: say "this Mac" when true, "this computer"
when false (the UI can also use its own platform check). A test (`coded::tests::every_code_in_the_sources_is_documented`)
fails when a code is used in `src-tauri/src` but missing here.

| Code | Params | Meaning |
|---|---|---|
| `mic.denied` | | microphone access refused in the OS |
| `mic.restricted` | `mac` | access restricted by a profile or parental controls |
| `mic.unanswered` | | the permission prompt was not answered |
| `mic.no_app_setting` | | this OS has no per-app microphone setting (open the OS privacy settings) |
| `live.busy` | | a live session is already running |
| `live.no_device` | | no input device |
| `live.device_not_found` | | the chosen device is gone |
| `live.device_error` | `detail` | the microphone could not be opened |
| `live.start_failed` | | the capture thread died while starting |
| `live.disconnected` | `detail` | the microphone was unplugged during a session |
| `live.rerouted` | | the OS switched the input device |
| `live.xrun` | | the audio device skipped a moment |
| `files.unknown` | | the chosen file or job no longer exists |
| `files.decode_failed` | `detail` | the audio file could not be decoded |
| `files.crashed` | | the decoder crashed on this file |
| `export.format` | `ext` | only TXT and SRT can be exported |
| `export.too_large` | | transcript too large to export |
| `export.write_failed` | `detail` | the export file could not be written |
| `update.disabled` | | this build has no update signing key |
| `update.stop_live` | | stop the live session before updating |
| `update.current` | | already up to date |
| `update.failed` | `detail` | update check or install failed |
| `backend.no_model` | | no model is active |
| `backend.unreadable` | `model`, `detail` | the active model cannot be opened |
| `backend.cant_use` | `model`, `why` (a coded message) | wraps the reason a model cannot be used |
| `backend.missing_engine` | `engine`, `mlx` | this build lacks the engine the model needs |
| `backend.untested` | `mac` | the model has no test result on this computer yet |
| `backend.test_failed` | `mac` | the test run failed |
| `backend.test_failed_detail` | `detail`, `mac` | the test run failed, with a reason |
| `backend.runtime_missing` | `engine`, `mac` | the runtime library is missing |
| `backend.runtime_old` | `engine` | the runtime is too old |
| `backend.ram` | `need`, `have` (GiB) | not enough memory |
| `backend.cpu` | `feature` | the processor lacks a required feature |
| `backend.chip` | | needs a newer Apple chip |
| `backend.platform` | `mac` | built for a different OS or CPU |
| `backend.unknown` | `detail` | a reason with no specific code |
| `backend.load_failed` | `model`, `detail` | the model could not be loaded into memory |
| `backend.worker_missing` | `detail` | the test helper program is missing |
| `models.import_only` | `file` | this package has no download address (add it from a folder or .zip) |
| `models.expired` | | the pending add request expired |
| `models.licence` | `licence` | a licence must be accepted first |
| `models.busy` | | another model operation is running |
| `models.cancelled` | | the operation was cancelled (downloaded bytes are kept) |
| `models.manifest_too_large` | | the package manifest is too large |
| `models.registry_error` | `detail` | the model registry could not be opened |
| `models.note` | | explanation line of the Models page |
| `models.download_failed` | `detail` | network failure during the download |
| `models.corrupt` | `detail` | a model file is damaged or incomplete |
| `models.rejected` | `detail` | the package signature or content was not accepted |
| `models.replay` | `detail` | an older version than the installed one |
| `models.already_installed` | `model` | |
| `models.not_installed` | `model` | |
| `models.no_previous` | | nothing to roll back to |
| `models.io_failed` | `detail` | a model file could not be read or written |
| `setup.required` | | a release build needs the model before it can transcribe |
| `setup.offline` | `detail` | the catalog could not be fetched or verified |
| `setup.no_platform` | | no model for this OS and CPU yet |
| `setup.no_engine` | | this build lacks the engine the available model needs |
| `system.failed` | `detail` | disk, dialog or OS failure |
| `system.url_not_allowed` | | `open_licence_url` was given an address that is not a licence page of an offered or installed model |

## Commands added for the product UX pass

- `open_licence_url(url)`: opens the system browser on a licence page. Allowed only for an `https://` URL that is exactly one of the
  `licenses[].url` of a manifest that is currently offered (`PendingInstall.licenses[].url`, including the first-run offer) or
  installed (`models_overview`); anything else fails with `system.url_not_allowed`. No opener plugin and no extra capability.
- `HistorySummary.first_line`: the first non-empty final text of the entry, cut to about 80 characters with an ellipsis (empty when
  the entry has no text). Use it to title live entries together with the start time.

## Fields that are not coded messages

- `LiveStatus.backend` is the engine label of the latest session and is **empty before any session has run**; `LiveStatus.demo` is
  true only when that session ran on the placeholder engine (debug builds). The UI must not infer "demo" from an empty label.
- Live history entries have an **empty `name`** (and `source == "live"`); the UI titles them from `started_at`. Entries saved by
  older builds as "Live session" are normalised to the empty name when read. File entries carry the file name.
