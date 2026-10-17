//! Versioned model manifest and its Ed25519 signature envelope.
//!
//! Signed bytes = `serde_json::to_vec(&Manifest)`: compact JSON in struct field order, no maps and no floats, so the
//! bytes are a pure function of the parsed value. Parsing rejects unknown fields, so nothing unsigned can ride along.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

pub const SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub id: String,
    /// Immutable revision (Hugging Face commit sha), never a branch name.
    pub revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    /// Relative path inside the version directory (forward slashes, no `..`).
    pub path: String,
    /// Which part of the model this is (`reader`, `adapter`, `tokenizer`, `lm`, `base`, `meta`).
    pub component: String,
    pub url: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LicenseTerms {
    pub id: String,
    pub name: String,
    /// SPDX identifier when the terms are a standard licence.
    pub spdx: Option<String>,
    pub url: String,
    /// Components (`FileEntry::component`) these terms cover.
    pub applies_to: Vec<String>,
    /// The user must have accepted these terms (recorded locally) before the version can be installed.
    pub acceptance_required: bool,
    pub attribution: Vec<String>,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requirements {
    pub min_ram_bytes: u64,
    /// CPU feature names that must be present (`avx2`, `neon`, ...).
    pub cpu_features: Vec<String>,
    /// Minimum Apple Silicon generation (1 = M1, 4 = M4), when the artifact needs one.
    pub apple_chip_min_generation: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeReq {
    pub name: String,
    pub min_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Compat {
    /// Free-form validation status as recorded by the benchmark harness (VERIFIED / EXPERIMENTAL / BLOCKED ...).
    pub status: String,
    /// Real-time factor x1000 measured for this artifact, if any (integer to keep the signed bytes exact).
    pub rtf_x1000: Option<u32>,
    pub notes: String,
}

/// How a derived (converted, merged, quantised) artifact was made. Derived artifacts are modifications of the upstream
/// weights and must say so (Apache-2.0 section 4; docs/research/licensing-review.md section 4.5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub modified: bool,
    /// Immutable upstream revisions the artifact was derived from.
    pub sources: Vec<Source>,
    /// Converter script path in this repository and the commit it was run from.
    pub converter: String,
    pub converter_commit: String,
    /// What was changed (LoRA merge, layout transposes, dtype).
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    /// Anti-rollback counter, >= 1, strictly increasing across releases of one [`Manifest::lineage`].
    pub serial: u64,
    /// Unique, filesystem-safe version id (`[a-z0-9._-]`, at most 96 chars).
    pub id: String,
    pub family: String,
    pub revision: String,
    pub backend: String,
    pub os: String,
    pub arch: String,
    pub precision: String,
    pub requirements: Requirements,
    pub runtime: RuntimeReq,
    pub base_model: Option<Source>,
    pub adapter: Option<Source>,
    pub tokenizer: Option<Source>,
    pub lm: Option<Source>,
    pub files: Vec<FileEntry>,
    pub licenses: Vec<LicenseTerms>,
    pub compat: Compat,
    pub provenance: Option<Provenance>,
    pub created: String,
    pub min_app_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedManifest {
    pub manifest: Manifest,
    pub key_id: String,
    /// Hex Ed25519 signature over [`Manifest::canonical_bytes`].
    pub signature: String,
}

/// A manifest whose signature was checked against a trusted key. Only [`Verifier::verify`] builds one.
#[derive(Debug, Clone)]
pub struct VerifiedManifest(SignedManifest);

impl VerifiedManifest {
    pub fn manifest(&self) -> &Manifest {
        &self.0.manifest
    }
    pub fn signed(&self) -> &SignedManifest {
        &self.0
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 || !s.is_ascii() {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Dotted numeric version compare (`1.10.0` > `1.9`); non-numeric parts compare as 0.
pub fn version_at_least(have: &str, need: &str) -> bool {
    let parse = |v: &str| {
        v.split('.')
            .map(|p| p.parse::<u64>().unwrap_or(0))
            .collect::<Vec<_>>()
    };
    let (mut h, mut n) = (parse(have), parse(need));
    let len = h.len().max(n.len());
    h.resize(len, 0);
    n.resize(len, 0);
    h >= n
}

pub fn safe_rel_path(p: &str) -> bool {
    !p.is_empty()
        && p.len() <= 256
        && !p.starts_with('/')
        && !p.contains('\\')
        && !p.contains(':')
        && p.split('/')
            .all(|c| !c.is_empty() && c != "." && c != ".." && !c.chars().any(|ch| ch.is_control()))
}

/// Only data formats may be installed: nothing that can carry code (pickle, scripts, executables) is accepted.
const ALLOWED_EXT: &[&str] = &[
    "safetensors",
    "onnx",
    "json",
    "gz",
    "model",
    "txt",
    "md",
    "mlmodel",
    "bin",
    "npy",
];

impl Manifest {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("manifest serialises")
    }

    /// Releases that replace each other share a lineage; the serial only has to increase within one.
    pub fn lineage(&self) -> String {
        format!("{}.{}.{}", self.family, self.backend, self.precision)
    }

    pub fn download_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    pub fn licenses_requiring_acceptance(&self) -> impl Iterator<Item = &LicenseTerms> {
        self.licenses.iter().filter(|l| l.acceptance_required)
    }

    /// Structural checks. `allow_loopback_http` is for tests only: production URLs must be `https://` and pinned to a
    /// 40-hex commit (`/resolve/<sha>/`) so a branch cannot move under a signed hash.
    pub fn validate(&self, allow_loopback_http: bool) -> Result<()> {
        let bad = |m: String| Err(Error::Invalid(m));
        if self.schema_version != SCHEMA_VERSION {
            return bad(format!(
                "unsupported schema_version {}",
                self.schema_version
            ));
        }
        if self.serial == 0 {
            return bad("serial must be >= 1".into());
        }
        if self.id.is_empty()
            || self.id.len() > 96
            || !self
                .id
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b))
            || self.id.starts_with('.')
        {
            return bad(format!("bad id {:?}", self.id));
        }
        if self.files.is_empty() {
            return bad("no files".into());
        }
        let mut seen = std::collections::HashSet::new();
        for f in &self.files {
            if !safe_rel_path(&f.path) || f.path == crate::store::MANIFEST_FILE {
                return bad(format!("unsafe file path {:?}", f.path));
            }
            // Case-insensitive uniqueness: macOS and Windows file systems would alias these.
            if !seen.insert(f.path.to_lowercase()) {
                return bad(format!("duplicate path {:?}", f.path));
            }
            let ext = f.path.rsplit('.').next().unwrap_or("");
            if !ALLOWED_EXT.contains(&ext) {
                return bad(format!("file type not allowed: {:?}", f.path));
            }
            if !is_hex(&f.sha256, 64) {
                return bad(format!("{}: sha256 must be 64 lowercase hex chars", f.path));
            }
            if f.url.is_empty() {
                continue; // not published: installable by offline import only
            }
            let pinned = f.url.starts_with("https://")
                && f.url.split('/').any(|s| is_hex(s, 40))
                && f.url.contains("/resolve/");
            let loopback = allow_loopback_http && f.url.starts_with("http://127.0.0.1:");
            if !(pinned || loopback) {
                return bad(format!(
                    "{}: url must be https and pinned to a 40-hex commit under /resolve/",
                    f.path
                ));
            }
        }
        for l in &self.licenses {
            if let Some(c) = l
                .applies_to
                .iter()
                .find(|c| !self.files.iter().any(|f| &f.component == *c))
            {
                return bad(format!(
                    "license {} applies to unknown component {c:?}",
                    l.id
                ));
            }
        }
        for f in &self.files {
            if !self
                .licenses
                .iter()
                .any(|l| l.applies_to.contains(&f.component))
            {
                return bad(format!(
                    "{}: component {:?} is covered by no licence entry",
                    f.path, f.component
                ));
            }
        }
        Ok(())
    }
}

pub fn sign(manifest: &Manifest, key_id: &str, key: &SigningKey) -> SignedManifest {
    let sig: Signature = key.sign(&manifest.canonical_bytes());
    SignedManifest {
        manifest: manifest.clone(),
        key_id: key_id.to_string(),
        signature: hex(&sig.to_bytes()),
    }
}

/// Trusted publisher keys (`key_id`, public key). Production uses [`Verifier::embedded`].
pub struct Verifier {
    keys: Vec<(String, VerifyingKey)>,
}

impl Verifier {
    pub fn new(keys: &[(&str, [u8; 32])]) -> Result<Self> {
        let keys = keys
            .iter()
            .map(|(id, k)| {
                VerifyingKey::from_bytes(k)
                    .map(|v| (id.to_string(), v))
                    .map_err(|e| Error::Invalid(format!("bad public key {id}: {e}")))
            })
            .collect::<Result<_>>()?;
        Ok(Self { keys })
    }

    pub fn embedded() -> Self {
        Self::new(crate::keys::TRUSTED).expect("embedded keys are valid")
    }

    /// Parse (rejecting unknown fields) and verify strictly. Structural validation is separate: see `Registry`.
    pub fn parse(&self, bytes: &[u8]) -> Result<VerifiedManifest> {
        let signed: SignedManifest =
            serde_json::from_slice(bytes).map_err(|e| Error::Malformed(e.to_string()))?;
        self.verify(signed)
    }

    pub fn verify(&self, signed: SignedManifest) -> Result<VerifiedManifest> {
        let key = self
            .keys
            .iter()
            .find(|(id, _)| *id == signed.key_id)
            .ok_or_else(|| Error::UnknownKey(signed.key_id.clone()))?;
        let sig = unhex(&signed.signature)
            .and_then(|b| <[u8; 64]>::try_from(b).ok())
            .map(|b| Signature::from_bytes(&b))
            .ok_or_else(|| Error::BadSignature(signed.key_id.clone()))?;
        key.1
            .verify_strict(&signed.manifest.canonical_bytes(), &sig)
            .map_err(|_| Error::BadSignature(signed.key_id.clone()))?;
        Ok(VerifiedManifest(signed))
    }
}
