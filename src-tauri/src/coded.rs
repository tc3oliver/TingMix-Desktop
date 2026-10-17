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
    }
}
