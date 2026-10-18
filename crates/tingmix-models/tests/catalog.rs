mod common;
use common::*;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use tingmix_models::catalog::{self, MAX_CATALOG_BYTES};
use tingmix_models::manifest::sign;

/// One-shot server that answers every request with `body`.
fn serve_doc(body: Vec<u8>) -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!(
        "http://127.0.0.1:{}/catalog.json",
        l.local_addr().unwrap().port()
    );
    std::thread::spawn(move || {
        for conn in l.incoming() {
            let Ok(mut c) = conn else { continue };
            let mut r = BufReader::new(c.try_clone().unwrap());
            loop {
                let mut h = String::new();
                if r.read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
                    break;
                }
            }
            let _ = c.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            );
            let _ = c.write_all(&body);
        }
    });
    url
}

fn signed_json(id: &str, serial: u64, os: &str, arch: &str) -> serde_json::Value {
    let mut m = manifest(id, "http://127.0.0.1:1", 1);
    m.serial = serial;
    m.os = os.into();
    m.arch = arch.into();
    serde_json::to_value(sign(&m, "test", &key(1))).unwrap()
}

#[test]
fn single_signed_manifest_is_fetched_and_verified() {
    let url = serve_doc(serde_json::to_vec(&signed_json("m1", 1, "macos", "aarch64")).unwrap());
    let c = catalog::fetch(&url, &verifier(), true).unwrap();
    assert_eq!(c.manifests.len(), 1);
    assert_eq!(c.manifests[0].manifest().id, "m1");
    assert!(c.rejected.is_empty());
}

#[test]
fn array_is_filtered_by_platform_and_newest_serial_comes_first() {
    let doc = serde_json::json!([
        signed_json("old", 1, "macos", "aarch64"),
        signed_json("new", 3, "macos", "aarch64"),
        signed_json("win", 2, "windows", "x86_64"),
        signed_json("any", 1, "any", "any"),
    ]);
    let c = catalog::parse(&serde_json::to_vec(&doc).unwrap(), &verifier()).unwrap();
    let ids = |os: &str, arch: &str| {
        c.for_platform(os, arch)
            .iter()
            .map(|m| m.manifest().id.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(ids("macos", "aarch64"), ["new", "old", "any"]);
    assert_eq!(ids("windows", "x86_64"), ["win", "any"]);
    assert_eq!(ids("linux", "riscv64"), ["any"]);
}

#[test]
fn tampered_or_foreign_entries_are_skipped_and_an_all_bad_catalog_fails() {
    let mut tampered = signed_json("bad", 1, "macos", "aarch64");
    tampered["manifest"]["serial"] = serde_json::json!(9);
    let foreign = serde_json::to_value(sign(
        &manifest("foreign", "http://127.0.0.1:1", 1),
        "test",
        &key(7),
    ))
    .unwrap();
    let mixed = serde_json::json!([
        tampered.clone(),
        foreign.clone(),
        signed_json("good", 1, "macos", "aarch64")
    ]);
    let c = catalog::parse(&serde_json::to_vec(&mixed).unwrap(), &verifier()).unwrap();
    assert_eq!(c.manifests.len(), 1);
    assert_eq!(c.rejected.len(), 2);
    let all_bad = serde_json::json!([tampered, foreign]);
    assert!(catalog::parse(&serde_json::to_vec(&all_bad).unwrap(), &verifier()).is_err());
    assert!(catalog::parse(b"not json", &verifier()).is_err());
    assert!(catalog::parse(b"[]", &verifier()).is_err());
}

#[test]
fn urls_oversize_and_unreachable_servers_are_refused() {
    // plain http outside loopback, and loopback http unless allowed
    assert!(catalog::fetch("http://example.com/c.json", &verifier(), true).is_err());
    let url = serve_doc(b"[]".to_vec());
    assert!(catalog::fetch(&url, &verifier(), false).is_err());
    let big = serve_doc(vec![b' '; MAX_CATALOG_BYTES as usize + 10]);
    let e = catalog::fetch(&big, &verifier(), true)
        .unwrap_err()
        .to_string();
    assert!(e.contains("larger"), "{e}");
    // nothing listens on port 1
    assert!(catalog::fetch("http://127.0.0.1:1/c.json", &verifier(), true).is_err());
}
