//! Unified Teacher Lesson files -> training items. Exact port of `reference/lessons.py`.
//!
//! Blocks: `[Dialogue Lesson]`, `[Understanding Lesson]`, `[Multi-Turn Conversation Lesson]`
//! (sessions may include Nexus lines: `Environment:`, `Action:`, `Result:`),
//! `[Reasoning Lesson]`. Text before `[examples]` is explanatory prose (`T: ` lines, lighter
//! weight). About 10% of units in every block are held out by a stable hash of their text, so
//! the same question never lands in both training and test. Nothing here runs at inference;
//! answers are only training targets and evaluation keys.
//!
//! Two more blocks (written by the Lesson Designer, usable by hand):
//!
//! * `[Reading]`: text the model reads, trained at full weight like corpus reading and never
//!   held out. An optional first line `Mode: neural`, `Mode: memory` or `Mode: combined` says
//!   whether it trains the weights, is only studied into Study Mode memory, or both (default
//!   neural).
//! * `[Evaluation]`: the teacher's own held-out questions, one `question  answer` per line.
//!   When a file has one, these are its held-out units and nothing else is held out by hash.
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

/// What a `[Reading]` block is for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadingMode {
    /// Read into the weights (trained like corpus reading).
    #[default]
    Neural,
    /// Studied into Study Mode memory only; the weights do not change.
    Memory,
    /// Both.
    Combined,
}
impl ReadingMode {
    pub fn name(self) -> &'static str {
        match self {
            ReadingMode::Neural => "neural",
            ReadingMode::Memory => "memory",
            ReadingMode::Combined => "combined",
        }
    }
    pub fn parse(s: &str) -> Option<ReadingMode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "neural" => Some(ReadingMode::Neural),
            "memory" => Some(ReadingMode::Memory),
            "combined" => Some(ReadingMode::Combined),
            _ => None,
        }
    }
    /// The weights are trained on the text.
    pub fn trains(self) -> bool {
        self != ReadingMode::Memory
    }
    /// The text is studied into Study Mode memory.
    pub fn studies(self) -> bool {
        self != ReadingMode::Neural
    }
}

/// Reading text is split into units of about this many characters (at paragraph and sentence
/// boundaries), so a long passage is read in full across an epoch instead of one window of it.
pub const READING_CHUNK_CHARS: usize = 800;

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
    /// Course lessons and exams: how the answer is graded (accepted forms and rules). Without a
    /// key the answer must match exactly, as in the reference evaluation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<super::grading::AnswerKey>,
    /// Reading text (prose items from a `[Reading]` block): what it is for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reading: Option<ReadingMode>,
    /// Context the model sees before the question (Study Mode memories); part of the item's
    /// segments when trained, put before the prompt when evaluated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}
impl Item {
    /// True for a conversation with Nexus lines (an environment, actions, results).
    pub fn has_actions(&self) -> bool {
        self.segments.iter().any(|s| crate::nexus::has_markers(&s.text))
    }
    /// The model's lines in a conversation (replies and actions), each with everything before it.
    pub fn model_turns(&self) -> Vec<(String, &str)> {
        let mut history = String::new();
        let mut out = Vec::new();
        for seg in &self.segments {
            if seg.weight == 1.0 && !history.is_empty() {
                out.push((history.clone(), seg.text.trim_end_matches('\n')));
            }
            history.push_str(&seg.text);
        }
        out
    }
    /// True when training may use the item: not held out, and not reading meant only for memory.
    pub fn trainable(&self) -> bool {
        !self.held_out && self.reading.map_or(true, ReadingMode::trains)
    }
    /// How many outcomes evaluating this item gives: one per conversation turn, else one.
    pub fn outcome_count(&self) -> usize {
        match self.kind {
            Kind::Prose => 0,
            Kind::Conversation if self.has_actions() => self.model_turns().len(),
            Kind::Conversation => self.users.len().min(self.replies.len()),
            _ => usize::from(self.prompt.is_some() && self.answer.is_some()),
        }
    }
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
            key: None,
            reading: None,
            context: None,
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

/// The form a question is compared in: lowercase, trailing `?.! ` dropped, whitespace collapsed.
pub fn canonical(key: &str) -> String {
    let lower = key.to_ascii_lowercase();
    let trimmed = lower.trim_end_matches(['?', '.', '!', ' ']);
    trimmed.split(is_space).filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")
}

/// Stable held-out decision: CRC-32 of the canonical question.
pub fn is_held_out(key: &str) -> bool {
    crc32(canonical(key).as_bytes()) % 100 < HELD_OUT_PERCENT
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
pub(crate) fn blocks(text: &str) -> Vec<(&str, &str)> {
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
    // Nexus markers come only from `Environment:`, `Action:` and `Result:` lines, never from
    // the file's own characters.
    let text = text.replace(
        [crate::nexus::ENVIRONMENT, crate::nexus::ACTION, crate::nexus::RESULT, crate::memory::MEMORY],
        "",
    );
    let mut items = Vec::new();
    // Positions of the items from `[Evaluation]` blocks.
    let mut evaluation: BTreeSet<usize> = BTreeSet::new();
    for (name, body) in blocks(&text) {
        if name == "Reading" {
            let (mode, body) = reading_mode(body).map_err(|e| format!("{source}: {e}"))?;
            for chunk in reading_chunks(body) {
                let mut it = Item::new(
                    Kind::Prose,
                    source,
                    false,
                    vec![Segment { text: format!("{chunk}\n"), weight: 1.0 }],
                );
                it.reading = Some(mode);
                items.push(it);
            }
            continue;
        }
        if name == "Evaluation" {
            for line in split_lines(body) {
                let Some(i) = line.find("  ") else { continue };
                let (q, a) = (strip(&line[..i]), strip(&line[i + 2..]));
                if q.is_empty() || a.is_empty() {
                    continue;
                }
                let mut it = Item::new(Kind::Dialogue, source, true, qa(q, a));
                it.prompt = Some(q.into());
                it.answer = Some(a.into());
                evaluation.insert(items.len());
                items.push(it);
            }
            continue;
        }
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
                        } else if who == "Environment" {
                            // Written by the environment: context only, never trained on.
                            it.segments
                                .push(Segment { text: crate::nexus::environment_line(said), weight: 0. });
                        } else if who == "Result" {
                            it.segments.push(Segment { text: crate::nexus::result_line(said), weight: 0. });
                        } else if who == "Action" {
                            let line = crate::nexus::action_line(said);
                            it.replies.push(line.trim_end_matches('\n').into());
                            it.segments.push(Segment { text: line, weight: 1.0 });
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
    if !evaluation.is_empty() {
        // The teacher chose the held-out questions: everything else is taught, except a unit
        // asking an evaluation question (it would leak the answer).
        let eval_keys: BTreeSet<String> =
            evaluation.iter().filter_map(|&i| items[i].prompt.as_deref().map(canonical)).collect();
        let mut kept = Vec::with_capacity(items.len());
        for (i, mut it) in items.into_iter().enumerate() {
            if evaluation.contains(&i) {
                kept.push(it);
            } else if it.prompt.as_deref().map_or(true, |p| !eval_keys.contains(&canonical(p))) {
                it.held_out = false;
                kept.push(it);
            }
        }
        items = kept;
    }
    Ok(items)
}

/// The optional `Mode:` first line of a `[Reading]` block, and the text after it.
fn reading_mode(body: &str) -> Result<(ReadingMode, &str), String> {
    let trimmed = body.trim_start_matches(['\n', ' ', '\t']);
    if let Some(rest) = trimmed.strip_prefix("Mode:") {
        let (value, after) = rest.split_once('\n').unwrap_or((rest, ""));
        let mode = ReadingMode::parse(value).ok_or_else(|| {
            format!("[Reading] Mode must be neural, memory or combined, not '{}'", value.trim())
        })?;
        return Ok((mode, after));
    }
    Ok((ReadingMode::Neural, body))
}

/// Splits reading text into units of about [`READING_CHUNK_CHARS`]: whole paragraphs where
/// they fit, long paragraphs at sentence ends.
pub fn reading_chunks(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    for para in text.split("\n\n").map(strip).filter(|p| !p.is_empty()) {
        let para = para.split(is_space).filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ");
        if para.len() <= READING_CHUNK_CHARS {
            pieces.push(para);
            continue;
        }
        let mut current = String::new();
        for sentence in sentences(&para) {
            if !current.is_empty() && current.len() + 1 + sentence.len() > READING_CHUNK_CHARS {
                pieces.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(sentence);
        }
        if !current.is_empty() {
            pieces.push(current);
        }
    }
    // Short paragraphs are joined (a blank line between them) up to the chunk size.
    let mut out: Vec<String> = Vec::new();
    for p in pieces {
        match out.last_mut() {
            Some(last) if last.len() + 2 + p.len() <= READING_CHUNK_CHARS => {
                last.push_str("\n\n");
                last.push_str(&p);
            }
            _ => out.push(p),
        }
    }
    out
}

/// Sentences of a paragraph: split after `.`, `!` or `?` followed by a space (the punctuation
/// stays with its sentence).
pub fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let b = text.as_bytes();
    let mut start = 0;
    for i in 0..b.len() {
        if matches!(b[i], b'.' | b'!' | b'?') && (i + 1 == b.len() || b[i + 1] == b' ') {
            let s = text[start..=i].trim();
            if !s.is_empty() {
                out.push(s);
            }
            start = i + 1;
        }
    }
    let rest = text[start..].trim();
    if !rest.is_empty() {
        out.push(rest);
    }
    out
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
    let texts =
        items.iter().filter(|it| it.trainable()).flat_map(|it| it.segments.iter().map(|s| s.text.as_str()));
    grow_vocabulary_from_texts(existing, texts, capacity, min_count)
}

/// [`grow_vocabulary`] over plain training texts (used for corpora).
pub fn grow_vocabulary_from_texts<'a>(
    existing: &[String],
    texts: impl IntoIterator<Item = &'a str>,
    capacity: usize,
    min_count: usize,
) -> Vec<String> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for text in texts {
        for w in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '\'')) {
            if (3..=24).contains(&w.len()) {
                *counts.entry(w.to_string()).or_default() += 1;
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

    #[test]
    fn reading_and_evaluation_blocks() {
        let long = "Dogs are animals. ".repeat(60);
        let text = format!(
            "[Reading]\nMode: combined\nMax is a dog.\n\n{long}\n[/Reading]\n[Reading]\nPlain reading.\n[/Reading]\n\
             [Dialogue Lesson]\n[examples]\nIs Max a dog?  Yes.\nDo dogs bark?  Yes.\nDoes Max need water?  Yes.\n[/examples]\n[/Dialogue Lesson]\n\
             [Evaluation]\nDoes Max need water?  Yes.\nIs Max an animal?  Yes.\n[/Evaluation]\n"
        );
        let items = parse_lesson(&text, "t").unwrap();
        let reading: Vec<&Item> = items.iter().filter(|i| i.reading.is_some()).collect();
        assert!(reading.len() >= 3, "the long paragraph is split into units");
        assert!(reading.iter().all(|i| !i.held_out && i.segments[0].weight == 1.0 && i.kind == Kind::Prose));
        assert!(reading.iter().all(|i| i.segments[0].text.len() <= READING_CHUNK_CHARS + 1));
        assert_eq!(reading[0].reading, Some(ReadingMode::Combined));
        assert_eq!(reading.last().unwrap().reading, Some(ReadingMode::Neural));
        // The teacher's evaluation questions are the held-out units; nothing else is held out,
        // and the taught copy of an evaluation question is dropped.
        let held: Vec<&str> =
            items.iter().filter(|i| i.held_out).filter_map(|i| i.prompt.as_deref()).collect();
        assert_eq!(held, ["Does Max need water?", "Is Max an animal?"]);
        let taught: Vec<&str> =
            items.iter().filter(|i| !i.held_out).filter_map(|i| i.prompt.as_deref()).collect();
        assert_eq!(taught, ["Is Max a dog?", "Do dogs bark?"]);
        let memory_only = parse_lesson("[Reading]\nMode: memory\nText.\n[/Reading]\n", "t").unwrap();
        assert!(!memory_only[0].trainable());
        assert!(parse_lesson("[Reading]\nMode: dreaming\nText.\n[/Reading]\n", "t").is_err());
    }
}
