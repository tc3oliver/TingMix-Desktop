use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("manifest is malformed: {0}")]
    Malformed(String),
    #[error("manifest rejected: {0}")]
    Invalid(String),
    #[error("manifest signature is not valid for any trusted key (key id {0:?})")]
    BadSignature(String),
    #[error("unknown signing key id {0:?}")]
    UnknownKey(String),
    #[error("{path}: sha256 mismatch (expected {expected}, got {actual})")]
    HashMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("{path}: size mismatch (expected {expected}, got {actual})")]
    SizeMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },
    #[error("{0}: file is missing")]
    Missing(String),
    #[error("{0}: file is not listed in the manifest")]
    Unlisted(String),
    #[error("license not accepted: {0}")]
    LicenseNotAccepted(String),
    #[error("model {0:?} is not installed")]
    NotInstalled(String),
    #[error("model {0:?} is already installed")]
    AlreadyInstalled(String),
    #[error("no previous version to roll back to")]
    NoPrevious,
    #[error("download of {url} failed: {reason}")]
    Download { url: String, reason: String },
    #[error("manifest {id:?} has serial {serial}, below the high-water mark {high_water} for lineage {lineage} (replay of an older release?)")]
    Replay {
        id: String,
        lineage: String,
        serial: u64,
        high_water: u64,
    },
    #[error("{0}: no download URL in the manifest (import-only file)")]
    ImportOnly(String),
    #[error("cancelled")]
    Cancelled,
    #[error("bad archive: {0}")]
    Archive(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn io(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> Error {
    let path = path.into();
    move |source| Error::Io { path, source }
}

impl Error {
    /// The stored manifest of an installed model is untrusted, unparsable or the wrong size: only these justify replacing
    /// an install with a verified download. Anything else (an I/O error, say) may be transient and must keep the install.
    pub fn replaceable(&self) -> bool {
        matches!(
            self,
            Error::UnknownKey(_)
                | Error::BadSignature(_)
                | Error::Malformed(_)
                | Error::SizeMismatch { .. }
        )
    }
}
