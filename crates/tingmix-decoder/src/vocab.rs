use crate::{err, Result};
use std::path::Path;

/// `tokenizer.Tok` restricted to decoding: units come from `units.json`, no SentencePiece runtime needed.
pub struct Vocab {
    itos: Vec<String>,
    /// The reference joins pieces and fixes spaces only when a SentencePiece model is attached.
    sentencepiece: bool,
}

/// `str.split()` whitespace: Rust's `White_Space` plus the ASCII separators U+001C..U+001F that Python also treats as space.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

impl Vocab {
    pub fn new(itos: Vec<String>, sentencepiece: bool) -> Self {
        Self {
            itos,
            sentencepiece,
        }
    }

    /// `units.json` is a JSON array of strings; `sentencepiece` is whether the release ships `tokenizer_bpe.model`.
    pub fn from_units_json(path: &Path, sentencepiece: bool) -> Result<Self> {
        let raw = std::fs::read(path).or_else(|e| err(format!("{}: {e}", path.display())))?;
        let itos: Vec<String> =
            serde_json::from_slice(&raw).or_else(|e| err(format!("{}: {e}", path.display())))?;
        Ok(Self::new(itos, sentencepiece))
    }

    pub fn len(&self) -> usize {
        self.itos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.itos.is_empty()
    }

    pub fn itos(&self) -> &[String] {
        &self.itos
    }

    /// `lm.is_bpe`
    pub fn is_bpe(&self) -> bool {
        self.itos.iter().any(|u| u.contains('▁'))
    }

    /// `Tok.decode`
    pub fn decode(&self, ids: &[u32]) -> String {
        let s: String = ids
            .iter()
            .map(|&i| self.itos[i as usize].as_str())
            .collect();
        if !self.sentencepiece {
            return s;
        }
        let spaced = s.replace('▁', " ");
        let words: Vec<&str> = spaced.split(py_space).filter(|w| !w.is_empty()).collect();
        let joined: Vec<char> = words.join(" ").chars().collect();
        // _GAP: a space next to a non-ASCII char carries no information
        let mut out = String::with_capacity(joined.len());
        for (i, &c) in joined.iter().enumerate() {
            let non_ascii = |c: Option<&char>| c.is_some_and(|c| !c.is_ascii());
            if c == ' '
                && (non_ascii(i.checked_sub(1).and_then(|j| joined.get(j)))
                    || non_ascii(joined.get(i + 1)))
            {
                continue;
            }
            out.push(c);
        }
        out
    }

    /// `lm.lm_units`: per unit the string the LM scores, `None` = LM-neutral.
    pub fn lm_units(&self) -> Vec<Option<String>> {
        if !self.is_bpe() {
            return self.itos.iter().cloned().map(Some).collect();
        }
        self.itos
            .iter()
            .map(|u| {
                let mut it = u.chars();
                match (it.next(), it.next()) {
                    (Some(c), None) if is_cjk(c) => {
                        Some(if c == '臺' { "台".into() } else { u.clone() })
                    }
                    _ => None,
                }
            })
            .collect()
    }
}

/// normalize.CJK (its U+3400-U+4DBF range lies inside the first one)
pub(crate) fn is_cjk(c: char) -> bool {
    matches!(c, '\u{2e80}'..='\u{9fff}' | '\u{f900}'..='\u{faff}' )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(units: &[&str]) -> Vocab {
        Vocab::new(units.iter().map(|s| s.to_string()).collect(), true)
    }

    #[test]
    fn decode_drops_spaces_next_to_cjk_only() {
        let t = v(&["<blank>", "▁hello", "你", "好", "▁world", "ç", "▁and"]);
        assert_eq!(t.decode(&[1, 2, 3, 4]), "hello你好world");
        assert_eq!(t.decode(&[1, 4]), "hello world");
        // the documented limitation: a space next to a non-ASCII letter is dropped too
        assert_eq!(t.decode(&[5, 6]), "çand");
    }
}
