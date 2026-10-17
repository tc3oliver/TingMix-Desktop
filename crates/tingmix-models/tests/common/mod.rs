#![allow(dead_code)]
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use ed25519_dalek::SigningKey;
use tingmix_models::manifest::*;
use tingmix_models::{Registry, Verifier};

pub fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

pub fn verifier() -> Verifier {
    Verifier::new(&[("test", key(1).verifying_key().to_bytes())]).unwrap()
}

pub fn registry(dir: &std::path::Path) -> Registry {
    Registry::open(dir, verifier())
        .unwrap()
        .allow_loopback_http(true)
}

/// Deterministic pseudo-random bytes.
pub fn blob(n: usize, seed: u8) -> Vec<u8> {
    let mut x = seed as u32 + 1;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x >> 24) as u8
        })
        .collect()
}

pub fn contents(tag: u8) -> Vec<(&'static str, &'static str, Vec<u8>)> {
    vec![
        ("model.safetensors", "reader", blob(300_000, tag)),
        ("units.json", "tokenizer", b"[\"<blank>\",\"a\"]".to_vec()),
        ("lm/char.json.gz", "lm", blob(5_000, tag + 50)),
    ]
}

pub fn manifest(id: &str, base_url: &str, tag: u8) -> Manifest {
    let files = contents(tag)
        .into_iter()
        .map(|(p, c, b)| FileEntry {
            path: p.into(),
            component: c.into(),
            url: format!("{base_url}/{id}/{p}"),
            size: b.len() as u64,
            sha256: sha256_hex(&b),
        })
        .collect();
    Manifest {
        schema_version: 1,
        id: id.into(),
        family: "tingmix-large".into(),
        revision: "0".repeat(40),
        backend: "mlx".into(),
        os: "macos".into(),
        arch: "aarch64".into(),
        precision: "fp16".into(),
        requirements: Requirements {
            min_ram_bytes: 8 << 30,
            cpu_features: vec![],
            apple_chip_min_generation: Some(1),
        },
        runtime: RuntimeReq {
            name: "mlx".into(),
            min_version: "0.30".into(),
        },
        base_model: None,
        adapter: None,
        tokenizer: None,
        lm: None,
        files,
        licenses: vec![LicenseTerms {
            id: "model-terms".into(),
            name: "Test terms".into(),
            spdx: None,
            url: "https://example.invalid/terms".into(),
            applies_to: vec!["reader".into(), "tokenizer".into(), "lm".into()],
            acceptance_required: true,
            attribution: vec![],
            notes: String::new(),
        }],
        compat: Compat {
            status: "VERIFIED".into(),
            rtf_x1000: Some(12),
            notes: String::new(),
        },
        created: "2026-10-17".into(),
        min_app_version: "0.1.0".into(),
    }
}

pub fn signed(m: &Manifest) -> VerifiedManifest {
    verifier().verify(sign(m, "test", &key(1))).unwrap()
}

/// Write an importable folder (manifest + files), optionally corrupting nothing.
pub fn write_bundle(dir: &std::path::Path, m: &VerifiedManifest, tag: u8) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("manifest.signed.json"),
        serde_json::to_vec_pretty(m.signed()).unwrap(),
    )
    .unwrap();
    for (p, _, b) in contents(tag) {
        let f = dir.join(p);
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(f, b).unwrap();
    }
}

#[derive(Default)]
pub struct State {
    pub files: HashMap<String, Vec<u8>>,
    /// Cut the next response after this many body bytes (Content-Length still announces the full length).
    pub cut_next: Option<usize>,
    pub ignore_range: bool,
    pub requests: Vec<(String, Option<u64>)>,
}

pub struct Server {
    pub url: String,
    pub state: Arc<Mutex<State>>,
}

pub fn serve(ids: &[(&str, u8)]) -> Server {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://127.0.0.1:{}", l.local_addr().unwrap().port());
    let state = Arc::new(Mutex::new(State::default()));
    for (id, tag) in ids {
        for (p, _, b) in contents(*tag) {
            state.lock().unwrap().files.insert(format!("/{id}/{p}"), b);
        }
    }
    let st = state.clone();
    std::thread::spawn(move || {
        for conn in l.incoming() {
            let Ok(mut c) = conn else { continue };
            let mut r = BufReader::new(c.try_clone().unwrap());
            let mut line = String::new();
            if r.read_line(&mut line).is_err() {
                continue;
            }
            let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
            let mut range = None;
            loop {
                let mut h = String::new();
                if r.read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
                    break;
                }
                if let Some(v) = h.to_ascii_lowercase().strip_prefix("range: bytes=") {
                    range = v.trim().trim_end_matches('-').parse::<u64>().ok();
                }
            }
            let mut s = st.lock().unwrap();
            s.requests.push((path.clone(), range));
            let Some(body) = s.files.get(&path).cloned() else {
                let _ = c.write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                continue;
            };
            let start = if s.ignore_range {
                0
            } else {
                range.unwrap_or(0) as usize
            };
            let part = &body[start.min(body.len())..];
            let head = if start > 0 {
                format!("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len() - 1, body.len(), part.len())
            } else {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    part.len()
                )
            };
            let n = s
                .cut_next
                .take()
                .map(|k| k.min(part.len()))
                .unwrap_or(part.len());
            drop(s);
            let _ = c.write_all(head.as_bytes());
            let _ = c.write_all(&part[..n]);
        }
    });
    Server { url, state }
}
