//! Model registry: signed manifests, verified download/import, atomic activation with rollback, and
//! capability-based runtime selection. Nothing here executes or unpickles model files.

pub mod caps;
pub mod catalog;
pub mod error;
pub mod fetch;
pub mod import;
pub mod keys;
pub mod manifest;
pub mod probe;
pub mod select;
pub mod store;

pub use error::{Error, Result};
pub use manifest::{Manifest, SignedManifest, VerifiedManifest, Verifier};
pub use store::{InstalledModel, Registry};
