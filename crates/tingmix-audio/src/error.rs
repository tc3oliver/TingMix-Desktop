#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("no input device available")]
    NoInputDevice,
    #[error("input device not found: {0}")]
    DeviceNotFound(String),
    #[error("microphone access denied by the operating system")]
    PermissionDenied,
    #[error("audio device error: {0}")]
    Device(String),
    #[error("cannot decode audio file: {0}")]
    Decode(String),
}

impl From<cpal::Error> for AudioError {
    fn from(e: cpal::Error) -> Self {
        match e.kind() {
            cpal::ErrorKind::PermissionDenied => AudioError::PermissionDenied,
            cpal::ErrorKind::DeviceNotAvailable => AudioError::NoInputDevice,
            _ => AudioError::Device(e.to_string()),
        }
    }
}
