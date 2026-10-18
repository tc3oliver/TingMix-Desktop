//! Model catalog: where the app learns which models exist. It is one signed manifest, or a JSON array of them, served from a
//! URL fixed at build time. Every entry is verified against the trusted keys exactly like an installed manifest; the catalog
//! itself carries no trust (a swapped or edited file simply fails verification).

use crate::error::{Error, Result};
use crate::fetch::Downloader;
use crate::manifest::{SignedManifest, VerifiedManifest, Verifier};

/// Upper bound for the catalog document.
pub const MAX_CATALOG_BYTES: u64 = 1 << 20;

#[derive(Debug)]
pub struct Catalog {
    /// Entries with a valid signature, newest serial first.
    pub manifests: Vec<VerifiedManifest>,
    /// Why entries were skipped (malformed, unknown key, bad signature); empty when every entry verified.
    pub rejected: Vec<String>,
}

impl Catalog {
    /// Entries built for this OS and architecture (`any` matches everything).
    pub fn for_platform(&self, os: &str, arch: &str) -> Vec<&VerifiedManifest> {
        self.manifests
            .iter()
            .filter(|m| {
                let m = m.manifest();
                (m.os == "any" || m.os == os) && (m.arch == "any" || m.arch == arch)
            })
            .collect()
    }
}

/// Parse and verify catalog bytes. Fails only when no entry verifies, with the first reason.
pub fn parse(bytes: &[u8], verifier: &Verifier) -> Result<Catalog> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| Error::Malformed(e.to_string()))?;
    let entries = match value {
        serde_json::Value::Array(a) => a,
        other => vec![other],
    };
    let (mut manifests, mut rejected) = (Vec::new(), Vec::new());
    for e in entries {
        let verified = serde_json::from_value::<SignedManifest>(e)
            .map_err(|e| Error::Malformed(e.to_string()))
            .and_then(|s| verifier.verify(s));
        match verified {
            Ok(m) => manifests.push(m),
            Err(e) => rejected.push(e.to_string()),
        }
    }
    if manifests.is_empty() {
        return Err(Error::Malformed(
            rejected
                .first()
                .cloned()
                .unwrap_or_else(|| "the catalog is empty".into()),
        ));
    }
    manifests.sort_by_key(|m| std::cmp::Reverse(m.manifest().serial));
    Ok(Catalog {
        manifests,
        rejected,
    })
}

/// Download the catalog from `url` and verify it.
pub fn fetch(url: &str, verifier: &Verifier, allow_loopback_http: bool) -> Result<Catalog> {
    let bytes = Downloader::new(allow_loopback_http).get_small(
        url,
        MAX_CATALOG_BYTES,
        allow_loopback_http,
    )?;
    parse(&bytes, verifier)
}
