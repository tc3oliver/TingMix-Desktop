//! Microphone permission. On macOS the system prompt appears only when an app explicitly asks (or opens an input device
//! while the status is undetermined); a bundled app that never asks gets silence. Elsewhere there is no per-app API.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
// the permission states are only produced by the macOS query
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub enum MicStatus {
    /// The user has not been asked yet.
    NotDetermined,
    Authorized,
    Denied,
    /// Blocked by a profile or parental controls; the user cannot change it in the app's name.
    Restricted,
    /// The OS has no per-app microphone permission API (the system privacy switch still applies).
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    Unsupported,
}

#[cfg(target_os = "macos")]
mod imp {
    use super::MicStatus;
    use block2::RcBlock;
    use objc2::runtime::Bool;
    use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaTypeAudio};
    use std::sync::mpsc::channel;
    use std::time::Duration;

    pub fn status() -> MicStatus {
        // SAFETY: plain class method; AVMediaTypeAudio is a valid media type constant.
        let s = unsafe {
            AVCaptureDevice::authorizationStatusForMediaType(
                AVMediaTypeAudio.expect("AVMediaTypeAudio"),
            )
        };
        match s {
            AVAuthorizationStatus::Authorized => MicStatus::Authorized,
            AVAuthorizationStatus::Denied => MicStatus::Denied,
            AVAuthorizationStatus::Restricted => MicStatus::Restricted,
            _ => MicStatus::NotDetermined,
        }
    }

    /// Shows the system prompt when the status is undetermined and waits for the answer (two minutes at most).
    pub fn request() -> MicStatus {
        if status() != MicStatus::NotDetermined {
            return status();
        }
        let (tx, rx) = channel();
        let handler = RcBlock::new(move |granted: Bool| {
            let _ = tx.send(granted.as_bool());
        });
        // SAFETY: the block is retained by the call until it runs; it only sends on a channel.
        unsafe {
            AVCaptureDevice::requestAccessForMediaType_completionHandler(
                AVMediaTypeAudio.expect("AVMediaTypeAudio"),
                &handler,
            )
        };
        let _ = rx.recv_timeout(Duration::from_secs(120));
        status()
    }

    pub fn open_settings() -> Result<(), String> {
        // fixed URL, no user input: the Microphone pane of Privacy & Security
        std::process::Command::new("/usr/bin/open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
            .status()
            .map_err(|e| e.to_string())
            .and_then(|s| {
                s.success()
                    .then_some(())
                    .ok_or_else(|| format!("open exited with {s}"))
            })
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::MicStatus;

    pub fn status() -> MicStatus {
        MicStatus::Unsupported
    }
    pub fn request() -> MicStatus {
        MicStatus::Unsupported
    }
    pub fn open_settings() -> Result<(), String> {
        Err(
            "there is no per-app microphone setting on this system; check the OS privacy settings"
                .into(),
        )
    }
}

pub use imp::{open_settings, request, status};

/// Err with a user-facing message when capture must not start; asks first when the user has not decided yet.
pub fn ensure_access() -> Result<(), String> {
    use crate::coded::coded;
    use serde_json::json;
    match request() {
        MicStatus::Authorized | MicStatus::Unsupported => Ok(()),
        MicStatus::Denied => Err(coded(
            "mic.denied",
            json!({}),
            "Microphone access is denied. Allow TingMix in System Settings > Privacy & Security > Microphone, then press Start again.",
        )),
        MicStatus::Restricted => Err(coded(
            "mic.restricted",
            json!({}),
            "Microphone access is restricted by a profile or parental controls on this Mac.",
        )),
        MicStatus::NotDetermined => Err(coded(
            "mic.unanswered",
            json!({}),
            "The microphone permission prompt was not answered. Press Start again and choose Allow.",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_is_readable_without_prompting() {
        // links AVFoundation and reads the status; any value is valid, it must not prompt or panic
        let s = status();
        assert!(matches!(
            s,
            MicStatus::NotDetermined
                | MicStatus::Authorized
                | MicStatus::Denied
                | MicStatus::Restricted
                | MicStatus::Unsupported
        ));
        println!("microphone status for this process: {s:?}");
    }
}
