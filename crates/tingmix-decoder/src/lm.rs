use crate::{err, Result};
use flate2::read::MultiGzDecoder;
use serde::de::{DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::collections::HashMap;
use std::fmt;
use std::io::BufReader;
use std::path::Path;

pub const BOS: char = '\x02';
pub const EOS: char = '\x03';
pub const PH: char = '\x04';

/// Longest supported order. An n-gram is packed into a u128 with 21 bits per char, first char in the high bits and
/// right-padded with zeros, so the numeric order of keys equals the code-point order of the strings.
const MAX_ORDER: usize = 5;
const BITS: u32 = 21;
const CHAR_MASK: u128 = (1 << BITS) - 1;
const ALL_SLOTS: u128 = (1 << (BITS as usize * MAX_ORDER)) - 1;

fn slot_shift(i: usize) -> u32 {
    BITS * (MAX_ORDER - 1 - i) as u32
}

/// Packs up to `MAX_ORDER` chars; `None` if a char is NUL (reserved as padding) or there are too many.
fn pack(chars: impl Iterator<Item = char>) -> Option<u128> {
    let mut k = 0u128;
    for (i, c) in chars.enumerate() {
        if c == '\0' || i >= MAX_ORDER {
            return None;
        }
        k |= (c as u128) << slot_shift(i);
    }
    Some(k)
}

/// n-gram counts of one order, sorted by packed key.
#[derive(Default)]
struct Counts {
    keys: Vec<u128>,
    counts: Vec<u32>,
}

/// Per history of length n-1: sum and number of distinct continuations over the order-n table (`tot` / `typ`).
#[derive(Default)]
struct Hist {
    keys: Vec<u128>,
    tot: Vec<u64>,
    typ: Vec<u32>,
}

/// `lm.CharLM`, interpolated Kneser-Ney with absolute discount `D`, loaded from `eg2-charlm-kn/1` gzip JSON.
pub struct CharLm {
    order: usize,
    d: f64,
    cnt: Vec<Counts>,
    hist: Vec<Hist>,
    v: f64,
    cache: HashMap<(u128, u32), f64>,
}

struct Tables(Vec<Counts>);
struct TableSeed(usize);
struct Table(Counts);

impl<'de> Deserialize<'de> for Tables {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Tables;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a list of n-gram count tables")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Tables, A::Error> {
                let mut out = Vec::new();
                while let Some(Table(t)) = a.next_element_seed(TableSeed(out.len() + 1))? {
                    out.push(t);
                }
                Ok(Tables(out))
            }
        }
        d.deserialize_seq(V)
    }
}

impl<'de> DeserializeSeed<'de> for TableSeed {
    type Value = Table;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Table, D::Error> {
        struct V(usize);
        impl<'de> Visitor<'de> for V {
            type Value = Table;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an n-gram -> count map")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Table, A::Error> {
                let mut t = Counts::default();
                while let Some((k, c)) = a.next_entry::<String, u32>()? {
                    if k.chars().count() != self.0 {
                        return Err(A::Error::custom(format!(
                            "order-{} table has key {k:?}",
                            self.0
                        )));
                    }
                    let key =
                        pack(k.chars()).ok_or_else(|| A::Error::custom("NUL char in n-gram"))?;
                    if t.keys.last().is_some_and(|&p| p >= key) {
                        // shortcut: the writer emits sorted keys; unsorted or duplicate files are rejected, not repaired
                        return Err(A::Error::custom("n-gram keys are not strictly increasing"));
                    }
                    t.keys.push(key);
                    t.counts.push(c);
                }
                Ok(Table(t))
            }
        }
        d.deserialize_map(V(self.0))
    }
}

#[derive(Deserialize)]
struct File {
    format: String,
    order: usize,
    #[serde(rename = "D")]
    d: f64,
    #[serde(rename = "BOS")]
    bos: String,
    #[serde(rename = "EOS")]
    eos: String,
    #[serde(rename = "PH")]
    ph: String,
    counts: Tables,
}

impl CharLm {
    /// Streams the gzip JSON; the 700 MB document is never held in memory.
    pub fn load(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).or_else(|e| err(format!("{}: {e}", path.display())))?;
        let gz = BufReader::with_capacity(
            1 << 20,
            MultiGzDecoder::new(BufReader::with_capacity(1 << 20, f)),
        );
        let file: File =
            serde_json::from_reader(gz).or_else(|e| err(format!("{}: {e}", path.display())))?;
        let want = (BOS.to_string(), EOS.to_string(), PH.to_string());
        if file.format != "eg2-charlm-kn/1" || (file.bos, file.eos, file.ph) != want {
            return err(format!(
                "{} is not an eg2-charlm-kn/1 language model",
                path.display()
            ));
        }
        Self::from_counts(file.order, file.d, file.counts.0)
    }

    fn from_counts(order: usize, d: f64, cnt: Vec<Counts>) -> Result<Self> {
        if order == 0 || order > MAX_ORDER || cnt.len() != order {
            return err(format!(
                "unsupported LM order {order} with {} tables",
                cnt.len()
            ));
        }
        // tot/typ come from groups of equal history; with sorted keys they are contiguous runs
        let hist = cnt
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let clear = !(CHAR_MASK << slot_shift(i));
                let mut h = Hist::default();
                for (&k, &c) in t.keys.iter().zip(&t.counts) {
                    let hk = k & clear;
                    if h.keys.last() != Some(&hk) {
                        h.keys.push(hk);
                        h.tot.push(0);
                        h.typ.push(0);
                    }
                    *h.tot.last_mut().unwrap() += c as u64;
                    *h.typ.last_mut().unwrap() += 1;
                }
                h
            })
            .collect();
        let v = (cnt[0].keys.len() + 1) as f64;
        Ok(Self {
            order,
            d,
            cnt,
            hist,
            v,
            cache: HashMap::new(),
        })
    }

    pub fn order(&self) -> usize {
        self.order
    }

    /// `CharLM._p` on a packed history of `len` chars.
    fn p_raw(&self, h: u128, len: usize, w: char) -> f64 {
        let lower = if len == 0 {
            1.0 / self.v
        } else {
            self.p_raw((h << BITS) & ALL_SLOTS, len - 1, w)
        };
        let hist = &self.hist[len];
        let Ok(i) = hist.keys.binary_search(&h) else {
            return lower;
        };
        let t = hist.tot[i] as f64;
        let key = h | (w as u128) << slot_shift(len);
        let c = self.cnt[len]
            .keys
            .binary_search(&key)
            .map_or(0, |j| self.cnt[len].counts[j]);
        (c as f64 - self.d).max(0.0) / t + self.d * hist.typ[i] as f64 / t * lower
    }

    /// `CharLM.logp(prefix, w)`; `ctx` is the prefix or any suffix of it at least `order - 1` chars long (the reference
    /// left-pads with BOS and keeps the last `order - 1` chars, so a longer prefix gives the same value).
    pub fn logp(&mut self, ctx: &[char], w: char) -> f64 {
        let n = self.order - 1;
        let tail = &ctx[ctx.len().saturating_sub(n)..];
        let h = pack(std::iter::repeat_n(BOS, n - tail.len()).chain(tail.iter().copied()))
            .expect("context chars are non-NUL");
        let key = (h, w as u32);
        if let Some(&p) = self.cache.get(&key) {
            return p.ln();
        }
        if self.cache.len() > 4_000_000 {
            self.cache.clear();
        }
        let p = self.p_raw(h, n, w);
        self.cache.insert(key, p);
        p.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_order_matches_string_order() {
        let a = pack("ab".chars()).unwrap();
        let b = pack("abc".chars()).unwrap();
        let c = pack("b".chars()).unwrap();
        assert!(a < b && b < c);
    }
}
