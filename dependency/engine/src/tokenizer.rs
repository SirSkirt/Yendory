//! Lossless ASCII word/subword input tokenizer, not a knowledge or answer rule.
//! Stable token IDs: 0..127 are raw characters, 128.. are learned word units.
//! Unknown words always fall back to their characters.
use std::collections::{BTreeMap, HashMap};
/// A token id: 0..127 are ASCII characters, 128.. are learned word units. 32 bits, so a model
/// can be given a vocabulary far beyond 65,536 tokens.
pub type TokenId = u32;
/// Nova-2's word capacity (vocabulary 384 = 128 characters + 256 words). A model's own capacity
/// is `config.vocab - BASE` (`Bicameral::word_capacity`); this is the default and the limit
/// of the legacy `grow_from_corpus`.
pub const MAX_WORDS: usize = 256;
pub const BASE: usize = 128;
pub const TOKEN_CAPACITY: usize = BASE + MAX_WORDS;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tokenizer {
    /// Learned word units in id order (word `i` has token id `BASE + i`).
    words: Vec<String>,
    /// Word → index into `words`, kept in step with it (so encoding costs one lookup per word,
    /// however many word units there are).
    index: HashMap<String, u32>,
}
fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'\''
}
impl Tokenizer {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn from_words(words: Vec<String>) -> Self {
        let mut t = Self::default();
        t.set_words(words);
        t
    }
    /// The learned word units in id order.
    pub fn words(&self) -> &[String] {
        &self.words
    }
    /// Replaces the learned word units (vocabulary growth passes the old list plus new words,
    /// so existing ids never change).
    pub fn set_words(&mut self, words: Vec<String>) {
        self.index = words.iter().enumerate().map(|(i, w)| (w.clone(), i as u32)).collect();
        self.words = words;
    }
    pub fn push_word(&mut self, word: String) {
        self.index.insert(word.clone(), self.words.len() as u32);
        self.words.push(word);
    }
    /// Legacy frequency growth (minimum count 2, no name filter), up to `MAX_WORDS` words; the
    /// lesson and corpus pipelines use `learn::lessons::grow_vocabulary` instead.
    pub fn grow_from_corpus(&mut self, corpus: &str) -> usize {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for part in corpus.as_bytes().split(|b| !is_word(*b)) {
            if part.len() < 3 || part.len() > 24 || !part.is_ascii() {
                continue;
            }
            let token = String::from_utf8_lossy(part).to_string();
            if self.index.contains_key(&token) {
                continue;
            }
            *counts.entry(token).or_default() += 1;
        }
        let mut proposals: Vec<_> = counts.into_iter().filter(|(_, count)| *count >= 2).collect();
        proposals.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let previous = self.words.len();
        for (word, _) in proposals.into_iter().take(MAX_WORDS.saturating_sub(self.words.len())) {
            self.push_word(word);
        }
        self.words.len() - previous
    }
    /// Encodes ASCII text. A learned word unit is used only for a whole word: at the start of
    /// a run of word characters (letters, digits, apostrophes) whose entire run is that unit.
    /// Everything else is encoded character by character, so encoding is lossless.
    pub fn encode(&self, text: &str) -> Result<Vec<TokenId>, String> {
        if !text.is_ascii() {
            return Err("The engine currently requires ASCII input".into());
        }
        let data = text.as_bytes();
        let mut result = Vec::with_capacity(data.len());
        let mut i = 0;
        while i < data.len() {
            if is_word(data[i]) && (i == 0 || !is_word(data[i - 1])) {
                // The only unit that can match here is the whole run of word characters (a
                // unit must end at a word boundary, and units consist of word characters).
                let end = i + data[i..].iter().take_while(|&&b| is_word(b)).count();
                let run = &text[i..end];
                if let Some(&id) = self.index.get(run) {
                    result.push(BASE as TokenId + id);
                    i = end;
                    continue;
                }
            }
            result.push(data[i] as TokenId);
            i += 1;
        }
        Ok(result)
    }
    pub fn decode(&self, tokens: &[TokenId]) -> Result<String, String> {
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
    pub fn token_label(&self, token: TokenId) -> String {
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
    /// The original encoder: every word unit tried at every word start, longest whole-word
    /// match wins.
    fn encode_by_scanning(t: &Tokenizer, text: &str) -> Vec<TokenId> {
        let data = text.as_bytes();
        let (mut out, mut i) = (Vec::new(), 0);
        while i < data.len() {
            let mut matched = None;
            if is_word(data[i]) && (i == 0 || !is_word(data[i - 1])) {
                for (id, word) in t.words().iter().enumerate() {
                    let w = word.as_bytes();
                    if data[i..].starts_with(w)
                        && (i + w.len() == data.len() || !is_word(data[i + w.len()]))
                        && matched.map_or(true, |(_, len): (usize, usize)| w.len() > len)
                    {
                        matched = Some((id, w.len()));
                    }
                }
            }
            match matched {
                Some((id, len)) => {
                    out.push((BASE + id) as TokenId);
                    i += len;
                }
                None => {
                    out.push(data[i] as TokenId);
                    i += 1;
                }
            }
        }
        out
    }
    #[test]
    fn lookup_encoder_equals_scanning_encoder() {
        let words = ["the", "then", "box", "boxes", "don't", "abc123", "Mira", "a1b", "123"];
        let t = Tokenizer::from_words(words.iter().map(|w| w.to_string()).collect());
        let mut seed = 7u64;
        let alphabet = b"thenboxsdo'abc123Mir a1.\n!e";
        for _ in 0..3000 {
            let len = (seed % 40) as usize;
            let mut s = String::new();
            for _ in 0..len {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                s.push(alphabet[(seed >> 33) as usize % alphabet.len()] as char);
            }
            for text in [s.clone(), format!("{s} the box then boxes don't {s}")] {
                assert_eq!(t.encode(&text).unwrap(), encode_by_scanning(&t, &text), "{text:?}");
                assert_eq!(t.decode(&t.encode(&text).unwrap()).unwrap(), text);
            }
        }
    }
    #[test]
    fn ids_beyond_sixteen_bits() {
        let mut words: Vec<String> = (0..70_000).map(|i| format!("w{i:06}")).collect();
        words.push("zebra".into());
        let t = Tokenizer::from_words(words);
        let ids = t.encode("a zebra and w069999").unwrap();
        assert!(ids.contains(&(BASE as TokenId + 70_000)));
        assert!(ids.contains(&(BASE as TokenId + 69_999)));
        assert_eq!(t.decode(&ids).unwrap(), "a zebra and w069999");
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
