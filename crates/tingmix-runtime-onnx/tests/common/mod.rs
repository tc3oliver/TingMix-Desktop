#![allow(dead_code)]
use std::path::{Path, PathBuf};

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Little-endian f32 C-order `.npy` (format 1.0/2.0): returns (shape, data).
pub fn read_npy(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&b[..6], b"\x93NUMPY");
    let (hlen, off) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[off..off + hlen]).unwrap();
    assert!(
        header.contains("'<f4'") && header.contains("False"),
        "{header}"
    );
    let shape: Vec<usize> = header[header.find("'shape': (").unwrap() + 10..]
        .split(')')
        .next()
        .unwrap()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data = b[off + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, data)
}

pub fn load_wav(name: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(root().join("tests/fixtures/audio").join(name)).unwrap();
    r.samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect()
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The seeded random-init tiny model written by `tools/small-export/make_fixtures.py` (not in git).
pub fn tiny_model() -> PathBuf {
    let p = root().join("tools/small-export/out/tiny.onnx");
    assert!(
        p.exists(),
        "missing {}: run `cd tools/small-export && uv sync && uv run python make_fixtures.py`",
        p.display()
    );
    p
}
