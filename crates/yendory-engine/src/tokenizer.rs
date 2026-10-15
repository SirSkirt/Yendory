//! Lossless ASCII word/subword input tokenizer, not a knowledge or answer rule.
//! Stable token IDs: 0..127 are raw characters, 128.. are learned word units.
//! Unknown words always fall back to their characters.
use std::collections::BTreeMap;
pub const MAX_WORDS: usize = 256;
pub const BASE: usize = 128;
pub const TOKEN_CAPACITY: usize = BASE + MAX_WORDS;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tokenizer {
    pub words: Vec<String>,
}
fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'\''
}
impl Tokenizer {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn grow_from_corpus(&mut self, corpus: &str) -> usize {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for part in corpus.as_bytes().split(|b| !is_word(*b)) {
            if part.len() < 3 || part.len() > 24 || !part.is_ascii() {
                continue;
            }
            let token = String::from_utf8_lossy(part).to_string();
            if self.words.contains(&token) {
                continue;
            }
            *counts.entry(token).or_default() += 1;
        }
        let mut proposals: Vec<_> = counts.into_iter().filter(|(_, count)| *count >= 2).collect();
        proposals.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let previous = self.words.len();
        for (word, _) in proposals.into_iter().take(MAX_WORDS - self.words.len()) {
            self.words.push(word);
        }
        self.words.len() - previous
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u16>, String> {
        if !text.is_ascii() {
            return Err("The engine currently requires ASCII input".into());
        }
        let data = text.as_bytes();
        let mut result = Vec::with_capacity(data.len());
        let mut i = 0;
        while i < data.len() {
            let mut matched = None;
            if is_word(data[i]) && (i == 0 || !is_word(data[i - 1])) {
                // The final boundary matters: a partially typed word must
                // not be silently tokenized as a whole learned word.
                for (id, word) in self.words.iter().enumerate() {
                    let w = word.as_bytes();
                    if data[i..].starts_with(w)
                        && (i + w.len() == data.len() || !is_word(data[i + w.len()]))
                        && matched.map_or(true, |(_, len): (usize, usize)| w.len() > len)
                    {
                        matched = Some((id, w.len()));
                    }
                }
            }
            if let Some((id, len)) = matched {
                result.push((BASE + id) as u16);
                i += len;
            } else {
                result.push(data[i] as u16);
                i += 1;
            }
        }
        Ok(result)
    }
    pub fn decode(&self, tokens: &[u16]) -> Result<String, String> {
        let mut out = String::new();
        for &token in tokens {
            if (token as usize) < BASE {
                out.push(token as u8 as char);
            } else {
                let idx = token as usize - BASE;
                out.push_str(self.words.get(idx).ok_or("Unknown word token")?);
            }
        }
        Ok(out)
    }
    pub fn token_label(&self, token: u16) -> String {
        if (token as usize) < BASE {
            match token as u8 {
                b' ' => "␠".into(),
                b'\n' => "↵".into(),
                32..=126 => (token as u8 as char).to_string(),
                _ => format!("#{token}"),
            }
        } else {
            self.words.get(token as usize - BASE).cloned().unwrap_or_else(|| "?".into())
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.words.len() > MAX_WORDS {
            return Err("Tokenizer exceeded word capacity".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for word in &self.words {
            if word.len() < 3 || word.len() > 24 || !word.bytes().all(is_word) || !seen.insert(word) {
                return Err("Invalid or duplicate word token in checkpoint".into());
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lossless_unknown_word_and_punctuation() {
        let mut t = Tokenizer::new();
        t.grow_from_corpus("Mira opens a box. Mira opens the box.");
        for s in ["Mira opens a box.", "Tavik opens the blue box.", "Mira's cat!", "", "Mira opened a box"] {
            assert_eq!(t.decode(&t.encode(s).unwrap()).unwrap(), s);
        }
        assert!(t.encode("Mira opens a box.").unwrap().len() < "Mira opens a box.".len());
    }
    #[test]
    fn stable_ids_when_vocabulary_grows() {
        let mut t = Tokenizer::new();
        t.grow_from_corpus("Mira Mira opens opens");
        let original = t.encode("Mira").unwrap();
        t.grow_from_corpus("Tavik Tavik blue blue");
        assert_eq!(t.encode("Mira").unwrap(), original);
        assert_eq!(t.decode(&t.encode("Tavik Mira").unwrap()).unwrap(), "Tavik Mira");
    }
}
