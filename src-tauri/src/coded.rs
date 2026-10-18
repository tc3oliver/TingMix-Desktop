//! User-facing host messages travel as `{"code": "...", "params": {...}, "text": "<English>"}` strings. The UI translates by
//! code and falls back to `text`; anything that is not such a payload is shown as is (technical details stay English).

use serde_json::{json, Value};

pub fn coded(code: &str, params: Value, english: impl Into<String>) -> String {
    json!({ "code": code, "params": params, "text": english.into() }).to_string()
}

/// The English text of a message that may be coded, for logs and the diagnostics list. Nested payloads in params are not
/// expanded: `text` already contains the full sentence.
pub fn english(s: &str) -> String {
    serde_json::from_str::<Value>(s)
        .ok()
        .and_then(|v| v.get("text").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_else(|| s.to_owned())
}

/// An unexpected failure on this computer (disk, dialog, OS call). The UI shows a plain sentence; `detail` is for the
/// Troubleshooting area only.
pub fn system(e: impl std::fmt::Display) -> String {
    coded(
        "system.failed",
        json!({ "detail": e.to_string() }),
        format!("Something went wrong on this computer: {e}"),
    )
}

/// A named failure with technical `detail`.
pub fn failure(code: &str, english: &str, e: impl std::fmt::Display) -> String {
    coded(
        code,
        json!({ "detail": e.to_string() }),
        format!("{english}: {e}"),
    )
}

/// "this Mac" on macOS, "this computer" elsewhere (English fallback text; the UI picks its own wording from `mac`).
pub fn computer() -> &'static str {
    if cfg!(target_os = "macos") {
        "this Mac"
    } else {
        "this computer"
    }
}

/// `tingmix-models` errors as coded messages. `Cancelled` stays the plain word the operation code checks for.
pub fn model_error(e: &tingmix_models::Error) -> String {
    use tingmix_models::Error as E;
    match e {
        E::Cancelled => "cancelled".into(),
        E::Download { reason, .. } => {
            failure("models.download_failed", "The download failed", reason)
        }
        E::HashMismatch { .. }
        | E::SizeMismatch { .. }
        | E::Missing(_)
        | E::Unlisted(_)
        | E::Archive(_) => failure(
            "models.corrupt",
            "The model files are damaged or incomplete",
            e,
        ),
        E::BadSignature(_) | E::UnknownKey(_) | E::Malformed(_) | E::Invalid(_) => {
            failure("models.rejected", "The model package was not accepted", e)
        }
        E::Replay { .. } => failure(
            "models.replay",
            "This is an older version than one already installed",
            e,
        ),
        E::LicenseNotAccepted(l) => coded(
            "models.licence",
            json!({ "licence": l }),
            format!("licence {l:?} must be accepted before installing"),
        ),
        E::ImportOnly(f) => coded(
            "models.import_only",
            json!({ "file": f }),
            format!(
                "{f} has no download URL: this version can only be imported from a folder or zip"
            ),
        ),
        E::AlreadyInstalled(id) => coded(
            "models.already_installed",
            json!({ "model": id }),
            format!("model {id:?} is already installed"),
        ),
        E::NotInstalled(id) => coded(
            "models.not_installed",
            json!({ "model": id }),
            format!("model {id:?} is not installed"),
        ),
        E::NoPrevious => coded(
            "models.no_previous",
            json!({}),
            "no previous version to roll back to",
        ),
        E::Io { .. } => failure(
            "models.io_failed",
            "A model file could not be read or written",
            e,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_round_trips_and_plain_text_passes_through() {
        let c = coded("mic.denied", json!({}), "Microphone access is denied.");
        let v: Value = serde_json::from_str(&c).unwrap();
        assert_eq!(
            (v["code"].as_str(), v["text"].as_str()),
            (Some("mic.denied"), Some("Microphone access is denied."))
        );
        assert_eq!(english(&c), "Microphone access is denied.");
        assert_eq!(english("io error at /x"), "io error at /x");
        assert_eq!(english("{not json"), "{not json");
        // technical detail rides along in params, never in the code
        let f = failure(
            "models.download_failed",
            "The download failed",
            "connection reset",
        );
        let v: Value = serde_json::from_str(&f).unwrap();
        assert_eq!(v["params"]["detail"], "connection reset");
        assert_eq!(english(&f), "The download failed: connection reset");
    }

    /// Walks the sources for string literals shaped like a message code and checks the contract document lists each one.
    #[test]
    fn every_code_in_the_sources_is_documented() {
        const NAMESPACES: [&str; 8] = [
            "mic", "live", "files", "export", "update", "backend", "models", "setup",
        ];
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let doc = std::fs::read_to_string(root.join("../docs/architecture/host-codes.md")).unwrap();
        let mut missing = std::collections::BTreeSet::new();
        let mut seen = 0;
        for entry in std::fs::read_dir(root.join("src")).unwrap().flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|x| x != "rs")
                || path.ends_with("coded.rs")
                || path.ends_with("e2e.rs")
            {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            for lit in text.split('"').skip(1).step_by(2) {
                let is_code = lit.split_once('.').is_some_and(|(ns, name)| {
                    (NAMESPACES.contains(&ns) || ns == "system")
                        && !name.is_empty()
                        && name.chars().all(|c| c.is_ascii_lowercase() || c == '_')
                });
                if is_code {
                    seen += 1;
                    if !doc.contains(&format!("`{lit}`")) {
                        missing.insert(lit.to_string());
                    }
                }
            }
        }
        assert!(
            seen > 30,
            "the scan found only {seen} codes; is the pattern still right?"
        );
        assert!(
            missing.is_empty(),
            "codes missing from docs/architecture/host-codes.md: {missing:?}"
        );
    }
}
