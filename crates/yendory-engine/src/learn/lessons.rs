//! Unified Teacher Lesson files -> training items. Exact port of `reference/lessons.py`.
//!
//! Blocks: `[Dialogue Lesson]`, `[Understanding Lesson]`, `[Multi-Turn Conversation Lesson]`,
//! `[Reasoning Lesson]`. Text before `[examples]` is explanatory prose (`T: ` lines, lighter
//! weight). About 10% of units in every block are held out by a stable hash of their text, so
//! the same question never lands in both training and test. Nothing here runs at inference;
//! answers are only training targets and evaluation keys.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Loss weight of question / user-turn text (context, lightly trained).
pub const PROMPT_WEIGHT: f32 = 0.15;
/// Loss weight of explanatory prose.
pub const PROSE_WEIGHT: f32 = 0.3;
/// Percentage of units held out.
pub const HELD_OUT_PERCENT: u32 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Prose,
    Dialogue,
    Understanding,
    Conversation,
    Reasoning,
}
impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Prose => "prose",
            Kind::Dialogue => "dialogue",
            Kind::Understanding => "understanding",
            Kind::Conversation => "conversation",
            Kind::Reasoning => "reasoning",
        }
    }
}

/// A piece of training text and the loss weight of the tokens it encodes to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub text: String,
    pub weight: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub kind: Kind,
    /// Source label (file name inside a course), for per-file records.
    pub source: String,
    /// True for held-out units: never trained on, used only to evaluate.
    pub held_out: bool,
    pub segments: Vec<Segment>,
    /// Dialogue / Understanding / Reasoning: the question shown after `U: `.
    pub prompt: Option<String>,
    /// The teacher's expected answer (evaluation key / training target only).
    pub answer: Option<String>,
    /// Reasoning: the contradicted answer, for the preference check.
    pub contradicted: Option<String>,
    /// Conversation: user turns and Nova's turns, in order.
    pub users: Vec<String>,
    pub replies: Vec<String>,
}
impl Item {
    fn new(kind: Kind, source: &str, held_out: bool, segments: Vec<Segment>) -> Self {
        Item {
            kind,
            source: source.into(),
            held_out,
            segments,
            prompt: None,
            answer: None,
            contradicted: None,
            users: Vec::new(),
            replies: Vec::new(),
        }
    }
}

/// `U: question\nN: ` (prompt weight) followed by `answer\n` (full weight).
pub fn qa(question: &str, answer: &str) -> Vec<Segment> {
    vec![
        Segment { text: format!("U: {question}\nN: "), weight: PROMPT_WEIGHT },
        Segment { text: format!("{answer}\n"), weight: 1.0 },
    ]
}

/// The prompt Nova sees for a question at evaluation time.
pub fn chat_prompt(question: &str) -> String {
    format!("U: {question}\nN: ")
}

/// Python `str.isspace` for ASCII.
fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c' | '\x1c'..='\x1f')
}
fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}
/// Python `str.splitlines` for ASCII text.
fn split_lines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if matches!(c, b'\n' | b'\r' | 0x0b | 0x0c | 0x1c..=0x1e) {
            out.push(&s[start..i]);
            if c == b'\r' && i + 1 < b.len() && b[i + 1] == b'\n' {
                i += 1;
            }
            start = i + 1;
        }
        i += 1;
    }
    if start < b.len() {
        out.push(&s[start..]);
    }
    out
}

/// IEEE CRC-32, as Python's `zlib.crc32`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// Stable held-out decision: lowercase, drop trailing `?.! `, collapse whitespace, CRC-32.
pub fn is_held_out(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    let trimmed = lower.trim_end_matches(['?', '.', '!', ' ']);
    let canon = trimmed.split(is_space).filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ");
    crc32(canon.as_bytes()) % 100 < HELD_OUT_PERCENT
}

fn block_kind(name: &str) -> Option<Kind> {
    match name {
        "Dialogue Lesson" => Some(Kind::Dialogue),
        "Understanding Lesson" => Some(Kind::Understanding),
        "Multi-Turn Conversation Lesson" => Some(Kind::Conversation),
        "Reasoning Lesson" => Some(Kind::Reasoning),
        _ => None,
    }
}

/// `[Name]\n ... ^[/Name]` blocks, scanned left to right without overlap.
fn blocks(text: &str) -> Vec<(&str, &str)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        let line_start = pos == 0 || text.as_bytes()[pos - 1] == b'\n';
        if line_start && text.as_bytes()[pos] == b'[' {
            if let Some(close) = text[pos + 1..].find("]\n") {
                let name = &text[pos + 1..pos + 1 + close];
                let body_start = pos + 1 + close + 2;
                if !name.is_empty() {
                    let end_tag = format!("[/{name}]");
                    let mut p = body_start;
                    let found = loop {
                        match text[p..].find(&end_tag) {
                            None => break None,
                            Some(off) => {
                                let at = p + off;
                                if at == 0 || text.as_bytes()[at - 1] == b'\n' {
                                    break Some(at);
                                }
                                p = at + 1;
                            }
                        }
                    };
                    if let Some(at) = found {
                        out.push((name, &text[body_start..at]));
                        pos = at + end_tag.len();
                        continue;
                    }
                }
            }
        }
        pos += 1;
    }
    out
}

/// The text after `label` up to the first occurrence of `next` (both must be present), with
/// surrounding whitespace removed; the lazy-group semantics of the reference regex.
fn reasoning_fields(line: &str) -> Option<[&str; 4]> {
    let rest = line.strip_prefix("Premise:")?;
    let q = rest.find("Question:")?;
    let premise = &rest[..q];
    let rest = &rest[q + "Question:".len()..];
    let s = rest.find("Supported:")?;
    let question = &rest[..s];
    let rest = &rest[s + "Supported:".len()..];
    let c = rest.find("Contradicted:")?;
    let supported = &rest[..c];
    let contradicted = &rest[c + "Contradicted:".len()..];
    Some([strip(premise), strip(question), strip(supported), strip(contradicted)])
}

/// Parses one lesson file. Non-ASCII text is rejected (the tokenizer is ASCII-only), and
/// `\r\n` / `\r` line endings are read as `\n`, like Python's text mode.
pub fn parse_lesson(text: &str, source: &str) -> Result<Vec<Item>, String> {
    if !text.is_ascii() {
        return Err(format!("{source}: lessons must be ASCII text"));
    }
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut items = Vec::new();
    for (name, body) in blocks(&text) {
        let Some(kind) = block_kind(name) else { continue };
        let (explanation, examples) = match body.find("[examples]\n") {
            Some(i) => {
                let rest = &body[i + "[examples]\n".len()..];
                (&body[..i], rest.split("[/examples]").next().unwrap_or(""))
            }
            None => (body, ""),
        };
        if !strip(explanation).is_empty() {
            items.push(Item::new(
                Kind::Prose,
                source,
                false,
                vec![Segment { text: format!("T: {}\n", strip(explanation)), weight: PROSE_WEIGHT }],
            ));
        }
        match kind {
            Kind::Dialogue | Kind::Understanding => {
                for line in split_lines(examples) {
                    let Some(i) = line.find("  ") else { continue };
                    let (q, a) = (strip(&line[..i]), strip(&line[i + 2..]));
                    let mut it = Item::new(kind, source, is_held_out(q), qa(q, a));
                    it.prompt = Some(q.into());
                    it.answer = Some(a.into());
                    items.push(it);
                }
            }
            Kind::Reasoning => {
                for line in split_lines(examples) {
                    let Some([premise, question, sup, con]) = reasoning_fields(strip(line)) else { continue };
                    let q = format!("{premise} {question}");
                    let mut it = Item::new(kind, source, is_held_out(&q), qa(&q, sup));
                    it.prompt = Some(q);
                    it.answer = Some(sup.into());
                    it.contradicted = Some(con.into());
                    items.push(it);
                }
            }
            Kind::Conversation => {
                let mut rest = examples;
                while let Some(open) = rest.find("[session]\n") {
                    let after = &rest[open + "[session]\n".len()..];
                    let Some(close) = after.find("[/session]") else { break };
                    let session = &after[..close];
                    rest = &after[close + "[/session]".len()..];
                    let mut it = Item::new(kind, source, is_held_out(session), Vec::new());
                    for line in split_lines(strip(session)) {
                        let Some(colon) = line.find(':') else { continue };
                        let (who, said) = (strip(&line[..colon]), strip(&line[colon + 1..]));
                        if who == "User" {
                            it.segments
                                .push(Segment { text: format!("U: {said}\nN: "), weight: PROMPT_WEIGHT });
                            it.users.push(said.into());
                        } else {
                            it.segments.push(Segment { text: format!("{said}\n"), weight: 1.0 });
                            it.replies.push(said.into());
                        }
                    }
                    items.push(it);
                }
            }
            Kind::Prose => unreachable!(),
        }
    }
    Ok(items)
}

/// A capitalized word that is rarely written in lowercase is treated as a name and kept
/// character-level, so the model learns to read and copy names instead of memorizing them.
fn is_name_like(word: &str, counts: &BTreeMap<String, usize>) -> bool {
    word.as_bytes()[0].is_ascii_uppercase()
        && (*counts.get(&word.to_ascii_lowercase()).unwrap_or(&0) as f64) < 0.2 * counts[word] as f64
}

/// Grows the learned word units from the TRAINING items only (held-out text never shapes the
/// vocabulary). Existing ids are kept; new words are appended by frequency, then byte order.
pub fn grow_vocabulary(
    existing: &[String],
    items: &[Item],
    capacity: usize,
    min_count: usize,
) -> Vec<String> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for it in items.iter().filter(|it| !it.held_out) {
        for seg in &it.segments {
            for w in seg.text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '\'')) {
                if (3..=24).contains(&w.len()) {
                    *counts.entry(w.to_string()).or_default() += 1;
                }
            }
        }
    }
    let have: BTreeSet<&str> = existing.iter().map(|s| s.as_str()).collect();
    let mut cand: Vec<(&String, usize)> = counts
        .iter()
        .filter(|(w, &c)| c >= min_count && !have.contains(w.as_str()) && !is_name_like(w, &counts))
        .map(|(w, &c)| (w, c))
        .collect();
    cand.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let mut out = existing.to_vec();
    out.extend(cand.into_iter().take(capacity.saturating_sub(existing.len())).map(|(w, _)| w.clone()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_zlib() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"hello"), 0x3610_A686);
        assert_eq!(crc32(b"The quick brown fox jumps over the lazy dog"), 0x414F_A339);
    }

    #[test]
    fn parses_all_block_types() {
        let text = "[Dialogue Lesson]\nGreetings start talks.\n[examples]\nhello  Hello!\nno separator\n\
                    [/examples]\n[/Dialogue Lesson]\n[Reasoning Lesson]\n[examples]\n\
                    Premise: Ada has a key. Question: Who has a key? Supported: Ada. Contradicted: Ben.\n\
                    [/examples]\n[/Reasoning Lesson]\n[Multi-Turn Conversation Lesson]\n[examples]\n\
                    [session]\nUser: hi\nNova: Hello!\n[/session]\n[/examples]\n[/Multi-Turn Conversation Lesson]\n";
        let items = parse_lesson(text, "t").unwrap();
        let kinds: Vec<Kind> = items.iter().map(|i| i.kind).collect();
        assert_eq!(kinds, [Kind::Prose, Kind::Dialogue, Kind::Reasoning, Kind::Conversation]);
        assert_eq!(items[1].answer.as_deref(), Some("Hello!"));
        assert_eq!(items[2].prompt.as_deref(), Some("Ada has a key. Who has a key?"));
        assert_eq!(items[2].contradicted.as_deref(), Some("Ben."));
        assert_eq!(items[3].users, ["hi"]);
        assert_eq!(items[3].replies, ["Hello!"]);
        assert!(parse_lesson("caf\u{e9}", "t").is_err());
    }
}
