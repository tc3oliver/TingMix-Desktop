//! Mel parity against the reference oracle dumps (experiments/macos/reference/out, `c{k}_mel` / `c{k}_mel_mask`).
//! Set TINGMIX_ORACLE_DIR (relative paths resolve from the workspace root) and TINGMIX_FIXTURE_DIR is not needed:
//! the wav comes from tests/fixtures/audio via a minimal PCM16 reader. Skips when the dumps are absent.
use std::io::Read;
use std::path::{Path, PathBuf};

use tingmix_frontend_gemma::Gemma4Frontend;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn npy_f32(bytes: &[u8]) -> Vec<f32> {
    let len = if bytes[6] == 1 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize + 10
    } else {
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize + 12
    };
    bytes[len..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// RIFF PCM16 mono 16 kHz -> f32 (the fixtures are written that way).
fn wav(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    let mut p = 12;
    while &b[p..p + 4] != b"data" {
        p += 8 + u32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()) as usize;
    }
    let n = u32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()) as usize;
    b[p + 8..p + 8 + n]
        .chunks_exact(2)
        .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0)
        .collect()
}

#[test]
fn mel_matches_reference_dumps() {
    let Some(dir) = std::env::var_os("TINGMIX_ORACLE_DIR").map(|d| root().join(d)) else {
        eprintln!("skipped: set TINGMIX_ORACLE_DIR=experiments/macos/reference/out");
        return;
    };
    let fe = Gemma4Frontend::new();
    for id in [
        "zh_02s", "en_02s", "mix_04s", "zh_08s", "en_08s", "mix_08s", "zh_16s", "mix_29s",
    ] {
        let mut zip =
            zip::ZipArchive::new(std::fs::File::open(dir.join(format!("{id}.npz"))).unwrap())
                .unwrap();
        let mut get = |k: &str| {
            let mut v = Vec::new();
            zip.by_name(&format!("{k}.npy"))
                .unwrap()
                .read_to_end(&mut v)
                .unwrap();
            v
        };
        let audio = wav(&root().join(format!("tests/fixtures/audio/{id}.wav")));
        let (mut k, mut off) = (0, 0usize);
        while get_has(&mut zip_names(&dir, id), k) {
            let want = npy_f32(&get(&format!("c{k}_mel")));
            let mask = get(&format!("c{k}_mel_mask"));
            let n = off + 472_000;
            let mel = fe.extract(&audio[off..audio.len().min(n)]);
            let want_valid = mask[mask.len() - want.len() / 128..]
                .iter()
                .filter(|&&b| b != 0)
                .count();
            assert_eq!(mel.valid, want_valid, "{id} c{k} valid frames");
            assert_eq!(mel.frames * 128, want.len(), "{id} c{k} frames");
            let max = mel
                .data
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "{id} c{k}: frames {} valid {} max-abs {max:.2e}",
                mel.frames, mel.valid
            );
            assert!(max < 1e-5, "{id} c{k}: {max}");
            k += 1;
            off = n;
        }
    }
}

fn zip_names(dir: &Path, id: &str) -> Vec<String> {
    let z =
        zip::ZipArchive::new(std::fs::File::open(dir.join(format!("{id}.npz"))).unwrap()).unwrap();
    z.file_names().map(String::from).collect()
}

fn get_has(names: &mut [String], k: usize) -> bool {
    names.iter().any(|n| n == &format!("c{k}_mel.npy"))
}
