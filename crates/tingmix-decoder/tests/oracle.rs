//! Real TingMix-Large logits (CPU fp32 oracle, experiments/macos/reference) -> greedy and beam+LM text must equal
//! tests/fixtures/large/expected_texts.json.
//!
//! Always runs the two committed clips (greedy; beam too when TINGMIX_LM_PATH is set). With TINGMIX_ORACLE_DIR (path to
//! experiments/macos/reference/out, relative paths resolve from the workspace root) it runs all 8 clips.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use tingmix_decoder::{beam_decode, collapse_ctc, log_softmax_rows, BeamConfig, CharLm, Vocab};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[derive(Deserialize)]
struct Clip {
    greedy: String,
    beam_lm: String,
    n_chunks: usize,
}

#[derive(Deserialize)]
struct Expected {
    lm: Lm,
    clips: BTreeMap<String, Clip>,
}

#[derive(Deserialize)]
struct Lm {
    alpha: f64,
    beta: f64,
    beam: usize,
}

/// Minimal .npy reader for little-endian f32 / i64 C-order arrays: (shape, raw payload).
fn npy(bytes: &[u8]) -> (Vec<usize>, &[u8]) {
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let (len, start) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[start..start + len]).unwrap();
    assert!(header.contains("'fortran_order': False"), "{header}");
    let shape = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let shape = shape
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    (shape, &bytes[start + len..])
}

fn f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn join(parts: Vec<String>) -> String {
    if parts.concat().contains(' ') {
        parts.join(" ")
    } else {
        parts.concat()
    }
}

fn run(
    chunks: &[Vec<f32>],
    vocab: &Vocab,
    lm: Option<(&mut CharLm, BeamConfig)>,
) -> (String, Option<String>) {
    let v = vocab.len();
    let lps: Vec<Vec<f32>> = chunks.iter().map(|c| log_softmax_rows(c, v)).collect();
    // argmax of the raw logits like pipeline.greedy; first maximum wins
    let greedy = chunks
        .iter()
        .map(|c| {
            let am: Vec<u32> = c
                .chunks_exact(v)
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .fold(0, |b, (i, &x)| if x > r[b] { i } else { b })
                        as u32
                })
                .collect();
            vocab.decode(&collapse_ctc(&am, 0))
        })
        .collect();
    let beam = lm.map(|(lm, cfg)| {
        join(
            lps.iter()
                .map(|lp| beam_decode(lp, vocab, lm, &cfg).text)
                .collect(),
        )
    });
    (join(greedy), beam)
}

#[test]
fn oracle_logits_decode_to_expected_texts() {
    let exp: Expected = serde_json::from_slice(
        &std::fs::read(root().join("tests/fixtures/large/expected_texts.json")).unwrap(),
    )
    .unwrap();
    let vocab =
        Vocab::from_units_json(&root().join("tests/fixtures/decoder/units.json"), true).unwrap();
    let cfg = BeamConfig {
        alpha: exp.lm.alpha,
        beta: exp.lm.beta,
        beam: exp.lm.beam,
        ..Default::default()
    };
    let mut lm = std::env::var("TINGMIX_LM_PATH")
        .ok()
        .map(|p| CharLm::load(Path::new(&p)).unwrap());
    if lm.is_none() {
        println!("beam+LM part SKIPPED: set TINGMIX_LM_PATH to the TingMix-Large charlm.json.gz");
    }

    let mut inputs: BTreeMap<String, Vec<Vec<f32>>> = BTreeMap::new();
    for id in ["zh_02s", "en_02s"] {
        let bytes =
            std::fs::read(root().join(format!("tests/fixtures/large/{id}_logits.npy"))).unwrap();
        let (shape, raw) = npy(&bytes);
        assert_eq!(shape[1], vocab.len());
        inputs.insert(id.into(), vec![f32s(raw)]);
    }
    if let Ok(dir) = std::env::var("TINGMIX_ORACLE_DIR") {
        let dir = root().join(dir); // absolute paths replace the root
        for (id, clip) in &exp.clips {
            let mut z =
                zip::ZipArchive::new(std::fs::File::open(dir.join(format!("{id}.npz"))).unwrap())
                    .unwrap();
            let chunks = (0..clip.n_chunks)
                .map(|k| {
                    let mut buf = Vec::new();
                    z.by_name(&format!("c{k}_logits.npy"))
                        .unwrap()
                        .read_to_end(&mut buf)
                        .unwrap();
                    f32s(npy(&buf).1)
                })
                .collect();
            inputs.insert(id.clone(), chunks);
        }
    }

    for (id, chunks) in &inputs {
        let want = &exp.clips[id];
        assert_eq!(chunks.len(), want.n_chunks, "{id}");
        let (greedy, beam) = run(chunks, &vocab, lm.as_mut().map(|l| (l, cfg)));
        assert_eq!(greedy, want.greedy, "{id} greedy");
        if let Some(beam) = beam {
            assert_eq!(beam, want.beam_lm, "{id} beam+LM");
        }
        println!(
            "{id}: greedy{} ok",
            if lm.is_some() { " + beam+LM" } else { "" }
        );
    }
    println!("{} clips", inputs.len());
}
