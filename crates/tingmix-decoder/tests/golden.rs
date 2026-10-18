//! Exact comparison with the embed2asr reference. Fixtures: tests/fixtures/decoder (see tools/golden-decoder).
//! The release LM (114 MB) is not in git: set TINGMIX_LM_PATH to lmCN_cv_parl_ascend_ntuml.charlm.json.gz of
//! TingMix-Large@75071a1 to run the `large_lm` tests; without it they print a skip message and pass.

use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tingmix_decoder::{beam_decode, greedy_ids, BeamConfig, CharLm, Vocab};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/decoder")
}

fn read<T: for<'a> Deserialize<'a>>(name: &str) -> T {
    let p = dir().join(name);
    serde_json::from_slice(&std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .unwrap()
}

#[derive(Deserialize)]
struct Row {
    floor: u32,
    idx: Vec<usize>,
    val: Vec<u32>,
}

#[derive(Deserialize)]
struct Case {
    #[serde(rename = "T")]
    t: usize,
    rows: Vec<Row>,
    greedy_ids: Vec<u32>,
    greedy_text: String,
}

#[derive(Deserialize)]
struct DecodeCase {
    ids: Vec<u32>,
    text: String,
}

#[derive(Deserialize)]
struct Cases {
    vocab_size: usize,
    lm_units: Vec<Option<String>>,
    decode_cases: Vec<DecodeCase>,
    cases: std::collections::BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Beam {
    ids: Vec<u32>,
    text: String,
}

#[derive(Deserialize)]
struct Probe {
    prefix: String,
    w: String,
    logp: f64,
}

#[derive(Deserialize)]
struct Expected {
    configs: Vec<(f64, f64, usize)>,
    results: std::collections::BTreeMap<String, Vec<Beam>>,
    logp_probes: Vec<Probe>,
}

fn matrix(c: &Case, v: usize) -> Vec<f32> {
    let mut m = Vec::with_capacity(c.t * v);
    for r in &c.rows {
        let start = m.len();
        m.resize(start + v, f32::from_bits(r.floor));
        for (&i, &b) in r.idx.iter().zip(&r.val) {
            m[start + i] = f32::from_bits(b);
        }
    }
    m
}

fn vocab() -> Vocab {
    Vocab::from_units_json(&dir().join("units.json"), true).unwrap()
}

#[test]
fn vocab_decode_and_lm_units() {
    let cases: Cases = read("cases.json");
    let vocab = vocab();
    assert_eq!(vocab.len(), cases.vocab_size);
    assert_eq!(vocab.lm_units(), cases.lm_units);
    for (i, d) in cases.decode_cases.iter().enumerate() {
        assert_eq!(vocab.decode(&d.ids), d.text, "decode case {i}: {:?}", d.ids);
    }
}

#[test]
fn greedy_matches_reference() {
    let cases: Cases = read("cases.json");
    let vocab = vocab();
    for (name, c) in &cases.cases {
        let ids = greedy_ids(&matrix(c, cases.vocab_size), cases.vocab_size);
        assert_eq!(ids, c.greedy_ids, "{name}");
        assert_eq!(vocab.decode(&ids), c.greedy_text, "{name}");
    }
    println!("greedy: {} cases identical", cases.cases.len());
}

fn check_lm(lm_path: &Path, expected: &str) {
    let cases: Cases = read("cases.json");
    let exp: Expected = read(expected);
    let vocab = vocab();
    let t0 = Instant::now();
    let mut lm = CharLm::load(lm_path).unwrap();
    println!(
        "{}: loaded in {:.1}s, RSS {} MB",
        lm_path.display(),
        t0.elapsed().as_secs_f64(),
        rss_mb()
    );

    for (i, p) in exp.logp_probes.iter().enumerate() {
        let ctx: Vec<char> = p.prefix.chars().collect();
        let got = lm.logp(&ctx, p.w.chars().next().unwrap());
        assert_eq!(got, p.logp, "logp probe {i}: {:?} + {:?}", p.prefix, p.w);
    }
    let mut n = 0;
    for (name, c) in &cases.cases {
        let m = matrix(c, cases.vocab_size);
        for (k, &(alpha, beta, beam)) in exp.configs.iter().enumerate() {
            let cfg = BeamConfig {
                alpha,
                beta,
                beam,
                ..Default::default()
            };
            let t0 = Instant::now();
            let got = beam_decode(&m, &vocab, &mut lm, &cfg);
            let want = &exp.results[name][k];
            assert_eq!(
                (&got.ids, &got.text),
                (&want.ids, &want.text),
                "{name} config {k} {cfg:?}"
            );
            println!(
                "{name} cfg{k}: {} frames, {:.0} ms, ok",
                c.t,
                t0.elapsed().as_secs_f64() * 1e3
            );
            n += 1;
        }
    }
    println!(
        "{expected}: {} logp probes and {n} beam decodes identical",
        exp.logp_probes.len()
    );
}

/// Informational only (printed, never asserted): `ps` does not exist on Windows, where this reports 0.
fn rss_mb() -> u64 {
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
        .map_or(0, |kb| kb / 1024)
}

#[test]
fn small_lm() {
    check_lm(
        &dir().join("lm_small.charlm.json.gz"),
        "expected_small_lm.json",
    );
}

#[test]
fn large_lm() {
    let Ok(path) = std::env::var("TINGMIX_LM_PATH") else {
        println!(
            "SKIPPED large_lm: set TINGMIX_LM_PATH to the TingMix-Large charlm.json.gz to run it"
        );
        return;
    };
    check_lm(Path::new(&path), "expected_large_lm.json");
}
