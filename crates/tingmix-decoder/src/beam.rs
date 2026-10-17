use crate::lm::{CharLm, EOS, PH};
use crate::{Vocab, BLANK};
use std::collections::HashMap;

const NEG: f64 = -1e30;

/// Defaults are the TingMix-Large model card settings (the reference function itself defaults to alpha 0.25).
#[derive(Clone, Copy, Debug)]
pub struct BeamConfig {
    pub alpha: f64,
    pub beta: f64,
    pub beam: usize,
    pub topk: usize,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            alpha: 0.35,
            beta: 1.5,
            beam: 16,
            topk: 8,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct BeamOutput {
    pub ids: Vec<u32>,
    /// Text exactly as `beam_decode` returns it (臺 replaced by 台).
    pub text: String,
}

fn lse(a: f64, b: f64) -> f64 {
    if b <= NEG {
        a
    } else if a <= NEG {
        b
    } else {
        a.max(b) + (-(a - b).abs()).exp().ln_1p()
    }
}

/// LM state of a prefix: last chars of the LM context, sum of log P_lm, number of scored chars.
#[derive(Clone)]
struct State {
    ctx: Vec<char>,
    s: f64,
    n: usize,
}

struct Scorer<'a> {
    lm: &'a mut CharLm,
    units: Vec<Option<Vec<char>>>,
    st: HashMap<Vec<u32>, State>,
    keep: usize,
}

impl Scorer<'_> {
    fn state(&mut self, pf: &[u32]) -> State {
        if let Some(v) = self.st.get(pf) {
            return v.clone();
        }
        let State {
            mut ctx,
            mut s,
            mut n,
        } = self.state(&pf[..pf.len() - 1]);
        match &self.units[pf[pf.len() - 1] as usize] {
            None => {
                if ctx.last() != Some(&PH) {
                    s += self.lm.logp(&ctx, PH); // zh->en switch is an LM token; English content is never scored
                    ctx.push(PH);
                }
            }
            Some(u) => {
                for &ch in u {
                    s += self.lm.logp(&ctx, ch);
                    ctx.push(ch);
                }
                n += u.len();
            }
        }
        ctx.drain(..ctx.len().saturating_sub(self.keep));
        let v = State { ctx, s, n };
        self.st.insert(pf.to_vec(), v.clone());
        v
    }

    fn score(&mut self, pf: &[u32], pb: f64, pnb: f64, alpha: f64, beta: f64) -> f64 {
        let st = self.state(pf);
        lse(pb, pnb) + alpha * st.s + beta * st.n as f64
    }
}

/// `lm.beam_decode`: `lp` is row-major `[T, vocab.len()]` log-probabilities.
///
/// Difference to the reference, see docs/research/decoder-parity.md: the top-k candidates of a frame are visited in
/// descending value (ties: lower id first); the reference visits them in `np.argpartition` order, which only matters
/// for exact score ties between different prefixes.
pub fn beam_decode(lp: &[f32], vocab: &Vocab, lm: &mut CharLm, cfg: &BeamConfig) -> BeamOutput {
    let v = vocab.len();
    assert!(
        v > 0 && lp.len() % v == 0,
        "log-prob matrix is not [T, {v}]"
    );
    let mut sc = Scorer {
        lm,
        units: vocab
            .lm_units()
            .into_iter()
            .map(|u| u.map(|u| u.chars().collect()))
            .collect(),
        st: HashMap::from([(
            vec![],
            State {
                ctx: vec![],
                s: 0.0,
                n: 0,
            },
        )]),
        keep: 4,
    };
    sc.keep = sc.lm.order().saturating_sub(1).max(1);
    let (alpha, beta) = (cfg.alpha, cfg.beta);

    let mut beams: Vec<(Vec<u32>, f64, f64)> = vec![(vec![], 0.0, NEG)];
    for row in lp.chunks_exact(v) {
        let mut idx: Vec<usize> = (0..v).collect();
        if v > cfg.topk {
            idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            idx.truncate(cfg.topk);
        }
        let cand: Vec<usize> = idx
            .into_iter()
            .filter(|&c| row[c] > -14.0 || c == BLANK as usize)
            .collect();

        // insertion-ordered like a Python dict: the order is the tie-break of the stable sort below
        let mut new: Vec<(Vec<u32>, [f64; 2])> = Vec::new();
        let mut pos: HashMap<Vec<u32>, usize> = HashMap::new();
        let mut slot = |new: &mut Vec<(Vec<u32>, [f64; 2])>, k: Vec<u32>| -> usize {
            *pos.entry(k.clone()).or_insert_with(|| {
                new.push((k, [NEG, NEG]));
                new.len() - 1
            })
        };
        for (pf, pb, pnb) in &beams {
            let tot = lse(*pb, *pnb);
            for &c in &cand {
                let p = row[c] as f64;
                if c == BLANK as usize {
                    let i = slot(&mut new, pf.clone());
                    new[i].1[0] = lse(new[i].1[0], tot + p);
                    continue;
                }
                let mut ext = pf.clone();
                ext.push(c as u32);
                if pf.last() == Some(&(c as u32)) {
                    let i = slot(&mut new, ext);
                    new[i].1[1] = lse(new[i].1[1], pb + p);
                    let i = slot(&mut new, pf.clone());
                    new[i].1[1] = lse(new[i].1[1], pnb + p);
                } else {
                    let i = slot(&mut new, ext);
                    new[i].1[1] = lse(new[i].1[1], tot + p);
                }
            }
        }
        let mut scored: Vec<(f64, Vec<u32>, f64, f64)> = new
            .into_iter()
            .map(|(k, [pb, pnb])| (-sc.score(&k, pb, pnb, alpha, beta), k, pb, pnb))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("scores are finite")); // stable, like sorted()
        scored.truncate(cfg.beam);
        beams = scored
            .into_iter()
            .map(|(_, k, pb, pnb)| (k, pb, pnb))
            .collect();
    }

    let mut best: Option<(f64, &Vec<u32>)> = None;
    for (pf, pb, pnb) in &beams {
        let ctx = sc.state(pf).ctx;
        let f = sc.score(pf, *pb, *pnb, alpha, beta) + alpha * sc.lm.logp(&ctx, EOS);
        if best.is_none_or(|(b, _)| f > b) {
            best = Some((f, pf));
        }
    }
    let ids = best.expect("beam is never empty").1.clone();
    let text = vocab.decode(&ids).replace('臺', "台");
    BeamOutput { ids, text }
}
