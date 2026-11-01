//! Lesson Designer: turns an imported `.txt` dataset into an editable lesson draft (blocks of
//! reading, instruction, examples, practice, extra practice and evaluation questions), checks
//! it for teaching problems, and writes it as a Unified Teacher Lesson file that the existing
//! Classroom and Course Builder teach.
//!
//! The organisation of an imported file is a proposal made from its layout (headings, `Q:` /
//! `A:` labels, question marks, worked-example labels); it is never assumed correct, every
//! block and item records where it came from, and the teacher edits the result. Nothing here
//! invents answers: a question without an answer in the file stays without one (and is not
//! trained) until the teacher writes it. Markdown is for the teacher; the lesson file gets
//! plain text (`markdown_to_plain`).
use super::corpus::normalize_ascii;
use super::grading::words;
use super::lessons::{blocks, canonical, is_held_out, parse_lesson, ReadingMode};
use super::session::EducationStage;
use crate::fingerprint::sha256_hex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    /// Text Nova reads (neural reading, Study Mode memory, or both).
    Reading,
    /// Instruction: the concept, principles and relationships, in the teacher's words.
    Lesson,
    /// Worked examples and demonstrations with their answers.
    Examples,
    /// Questions Nova answers herself (trained like examples in automatic teaching; asked
    /// interactively in a practice session).
    Practice,
    /// Reserve exercises used when a lesson needs more practice.
    ExtraPractice,
    /// The teacher's held-out questions: never trained, used to grade the lesson.
    Evaluation,
    /// Unified-format text kept as written (conversation sessions, reasoning lessons).
    Raw,
    /// Material the analysis could not place; not taught until the teacher moves it.
    Unsorted,
}
impl BlockKind {
    pub fn label(self) -> &'static str {
        match self {
            BlockKind::Reading => "Reading",
            BlockKind::Lesson => "Lesson",
            BlockKind::Examples => "Examples",
            BlockKind::Practice => "Practice",
            BlockKind::ExtraPractice => "Extra Practice",
            BlockKind::Evaluation => "Evaluation",
            BlockKind::Raw => "Unified text",
            BlockKind::Unsorted => "Unsorted",
        }
    }
    /// Blocks made of question / answer items (the others hold text).
    pub fn has_items(self) -> bool {
        matches!(
            self,
            BlockKind::Examples | BlockKind::Practice | BlockKind::ExtraPractice | BlockKind::Evaluation
        )
    }
}

/// Who wrote a piece of a draft.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// Taken from the imported file.
    #[default]
    Imported,
    /// Made by Yendory (for example Review Training material); needs the teacher's check.
    Generated,
    /// Written by the teacher.
    Teacher,
}

/// Where an item's answer came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerSource {
    /// Given in the imported file next to the question.
    Source,
    /// Made by Yendory: must be verified by the teacher before it is a training target.
    Generated,
    /// Written or corrected by the teacher.
    Teacher,
    /// No answer yet.
    #[default]
    Missing,
}

/// A line of the imported file (1-based).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceRef {
    pub file: String,
    pub line: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DraftItem {
    pub id: String,
    pub question: String,
    #[serde(default)]
    pub answer: String,
    #[serde(default)]
    pub answer_source: AnswerSource,
    /// The teacher confirmed the answer (required for generated answers and for answers the
    /// analysis paired with a question only by their position).
    #[serde(default)]
    pub verified: bool,
    #[serde(default)]
    pub origin: Origin,
    /// Changed by the teacher after import or generation.
    #[serde(default)]
    pub edited: bool,
    #[serde(default)]
    pub source: Option<SourceRef>,
    /// Why the analysis is unsure about this item (shown until verified or edited).
    #[serde(default)]
    pub flags: Vec<String>,
    /// Teacher note (never trained on).
    #[serde(default)]
    pub note: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DraftBlock {
    pub id: String,
    pub kind: BlockKind,
    #[serde(default)]
    pub title: String,
    /// Markdown (Reading, Lesson, Unsorted) or Unified lesson text (Raw).
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub items: Vec<DraftItem>,
    /// Reading blocks: neural reading, Study Mode memory, or both.
    #[serde(default)]
    pub mode: ReadingMode,
    /// Included when the lesson is taught automatically.
    #[serde(default = "yes")]
    pub use_in_teaching: bool,
    #[serde(default)]
    pub origin: Origin,
    #[serde(default)]
    pub edited: bool,
    /// First and last line of the imported file this block came from.
    #[serde(default)]
    pub lines: Option<(usize, usize)>,
}
fn yes() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LessonDraft {
    pub id: String,
    pub name: String,
    /// The imported file's name and fingerprint (empty for a lesson started from nothing).
    #[serde(default)]
    pub source_file: String,
    #[serde(default)]
    pub source_sha256: String,
    /// Education stage (Dense, Transformer, Co-op), or the whole brain.
    #[serde(default)]
    pub stage: Option<EducationStage>,
    /// The teacher marked the lesson finished (drafts may be incomplete).
    #[serde(default)]
    pub finished: bool,
    /// Teacher notes about the lesson (never trained on).
    #[serde(default)]
    pub notes: String,
    pub blocks: Vec<DraftBlock>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Blocks saving the lesson for teaching.
    Error,
    Warning,
    Info,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Issue {
    pub severity: Severity,
    pub block: Option<String>,
    pub item: Option<String>,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalyzeOptions {
    /// Divide a large dataset into several related lessons.
    pub split: bool,
}
impl Default for AnalyzeOptions {
    fn default() -> Self {
        AnalyzeOptions { split: true }
    }
}

/// A dataset larger than this (question items) is divided into lessons.
const SPLIT_ITEMS: usize = 40;
/// ... or larger than this (words of text).
const SPLIT_WORDS: usize = 3000;
/// No lesson gets more question items than this when divided.
const MAX_LESSON_ITEMS: usize = 60;

// ---------------------------------------------------------------------------------------------
// Analysis

struct Ids {
    block: usize,
    item: usize,
}
impl Ids {
    fn block(&mut self) -> String {
        self.block += 1;
        format!("b{}", self.block)
    }
    fn item(&mut self) -> String {
        self.item += 1;
        format!("i{}", self.item)
    }
}

#[derive(Debug)]
enum Element {
    Prose { text: String, first: usize, last: usize },
    Qa { q: String, a: Option<String>, worked: bool, by_layout: bool, line: usize },
    Noise { text: String, line: usize },
}

struct Section {
    title: String,
    level: usize,
    elements: Vec<Element>,
}

const QUESTION_LABELS: [&str; 7] = ["question:", "q:", "user:", "u:", "problem:", "exercise:", "prompt:"];
const ANSWER_LABELS: [&str; 7] = ["answer:", "a:", "nova:", "n:", "ans:", "response:", "output:"];
const WORKED_LABELS: [&str; 4] = ["solution:", "steps:", "working:", "explanation:"];

/// Strips a leading list number ("1.", "2)", "Q3.", "(4)").
fn strip_number(line: &str) -> &str {
    let t = line.trim_start();
    let t2 = t.strip_prefix(['Q', 'q']).unwrap_or(t);
    let digits = t2.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits > 0 && digits <= 4 {
        let rest = &t2[digits..];
        if let Some(r) = rest.strip_prefix(['.', ')', ':']) {
            return r.trim_start();
        }
    }
    if let Some(r) = t.strip_prefix('(') {
        let d = r.bytes().take_while(|b| b.is_ascii_digit()).count();
        if d > 0 && r[d..].starts_with(')') {
            return r[d + 1..].trim_start();
        }
    }
    t
}

/// `label` at the start of the line (any case), returning the rest.
fn labelled<'a>(line: &'a str, labels: &[&str]) -> Option<&'a str> {
    let lower = line.to_ascii_lowercase();
    labels.iter().find(|l| lower.starts_with(*l)).map(|l| line[l.len()..].trim())
}

fn heading(line: &str) -> Option<(usize, String)> {
    let t = line.trim();
    let hashes = t.bytes().take_while(|&b| b == b'#').count();
    if (1..=6).contains(&hashes) && t[hashes..].starts_with(' ') {
        return Some((hashes, t[hashes..].trim().trim_end_matches('#').trim().to_string()));
    }
    let lower = t.to_ascii_lowercase();
    for w in ["lesson ", "chapter ", "unit ", "section ", "topic "] {
        if let Some(rest) = lower.strip_prefix(w) {
            let id: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
            let after = rest[id.len()..].trim_start();
            if !id.is_empty() && (after.is_empty() || after.starts_with([':', '-', '.'])) && t.len() <= 80 {
                return Some((1, t.to_string()));
            }
        }
    }
    None
}

/// A question and its answer on one line: `question  answer` (two spaces, the Unified
/// format), a tab, or `question? -> answer` / `=> answer` / `Answer: answer`.
fn inline_pair(line: &str) -> Option<(String, String)> {
    let t = strip_number(line).trim_end();
    for sep in ["\t", " -> ", " => ", "  "] {
        if let Some(i) = t.find(sep) {
            let (q, a) = (t[..i].trim(), t[i + sep.len()..].trim());
            if !q.is_empty() && !a.is_empty() && (sep != "  " || !q.contains("  ")) {
                return Some((q.to_string(), a.to_string()));
            }
        }
    }
    let lower = t.to_ascii_lowercase();
    if let Some(i) = lower.find("? answer:") {
        let (q, a) = (t[..=i].trim(), t[i + "? answer:".len()..].trim());
        if !a.is_empty() {
            return Some((q.to_string(), a.to_string()));
        }
    }
    None
}

fn is_noise(line: &str) -> bool {
    let t = line.trim();
    // Markdown table rows (including the |---| separator) belong to the text around them.
    if t.starts_with('|') {
        return false;
    }
    t.len() >= 3 && t.chars().all(|c| matches!(c, '-' | '=' | '*' | '_' | '~' | '#' | ' '))
        || (!t.is_empty() && !t.chars().any(|c| c.is_ascii_alphabetic()))
        || t.to_ascii_lowercase().starts_with("page ") && t.len() < 12
}

fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Splits the file into sections (by headings) of prose, question / answer pairs and lines
/// that fit nowhere.
fn sections(text: &str) -> Vec<Section> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut out = vec![Section { title: String::new(), level: 0, elements: Vec::new() }];
    let mut para: Vec<&str> = Vec::new();
    let mut para_first = 0;
    let mut pending: Option<(String, usize, bool)> = None; // question, line, worked example
    let mut example_next = false;
    let flush_para = |out: &mut Vec<Section>, para: &mut Vec<&str>, first: usize, last: usize| {
        if !para.is_empty() {
            let text = para.join("\n");
            out.last_mut().unwrap().elements.push(Element::Prose { text, first, last });
            para.clear();
        }
    };
    let mut i = 0;
    while i < lines.len() {
        let raw = lines[i];
        let line = raw.trim();
        let n = i + 1;
        // An answer completing the pending question (possibly over several lines).
        if let Some((q, qline, worked)) = pending.take() {
            if let Some(a) = labelled(line, &ANSWER_LABELS).or_else(|| labelled(line, &WORKED_LABELS)) {
                let worked = worked || labelled(line, &WORKED_LABELS).is_some();
                let mut answer = a.to_string();
                while i + 1 < lines.len() {
                    let next = lines[i + 1].trim();
                    if next.is_empty()
                        || labelled(next, &QUESTION_LABELS).is_some()
                        || labelled(next, &ANSWER_LABELS).is_some()
                        || heading(next).is_some()
                        || next.ends_with('?')
                    {
                        break;
                    }
                    if let Some(w) = labelled(next, &WORKED_LABELS) {
                        answer = format!("{answer} {w}");
                    } else {
                        answer = format!("{answer} {next}");
                    }
                    i += 1;
                }
                out.last_mut().unwrap().elements.push(Element::Qa {
                    q,
                    a: (!answer.is_empty()).then_some(answer),
                    worked,
                    by_layout: false,
                    line: qline,
                });
                i += 1;
                continue;
            }
            if !line.is_empty() && labelled(line, &QUESTION_LABELS).is_none() && heading(line).is_none() {
                // The question continues on this line.
                pending = Some((format!("{q} {line}"), qline, worked));
                i += 1;
                continue;
            }
            out.last_mut().unwrap().elements.push(Element::Qa {
                q,
                a: None,
                worked,
                by_layout: false,
                line: qline,
            });
        }
        if line.is_empty() {
            flush_para(&mut out, &mut para, para_first, n.saturating_sub(1));
            i += 1;
            continue;
        }
        if let Some((level, title)) = heading(line) {
            flush_para(&mut out, &mut para, para_first, n - 1);
            out.push(Section { title, level, elements: Vec::new() });
            i += 1;
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("example") || lower.starts_with("worked example") || lower.starts_with("e.g.") {
            if let Some(colon) = line.find(':') {
                let rest = line[colon + 1..].trim();
                if rest.is_empty() {
                    example_next = true;
                    i += 1;
                    continue;
                }
                if let Some((q, a)) = inline_pair(rest) {
                    flush_para(&mut out, &mut para, para_first, n - 1);
                    out.last_mut().unwrap().elements.push(Element::Qa {
                        q,
                        a: Some(a),
                        worked: true,
                        by_layout: false,
                        line: n,
                    });
                    i += 1;
                    continue;
                }
            }
        }
        if let Some(q) = labelled(strip_number(line), &QUESTION_LABELS) {
            flush_para(&mut out, &mut para, para_first, n - 1);
            pending = Some((q.to_string(), n, std::mem::take(&mut example_next)));
            i += 1;
            continue;
        }
        if let Some((q, a)) = inline_pair(line) {
            flush_para(&mut out, &mut para, para_first, n - 1);
            out.last_mut().unwrap().elements.push(Element::Qa {
                q,
                a: Some(a),
                worked: std::mem::take(&mut example_next),
                by_layout: false,
                line: n,
            });
            i += 1;
            continue;
        }
        let bare = strip_number(line);
        if bare.ends_with('?') && word_count(bare) >= 2 {
            flush_para(&mut out, &mut para, para_first, n - 1);
            // The next line answers it when it is short, directly below, and not a question.
            let next = lines.get(i + 1).map(|l| l.trim()).unwrap_or("");
            let answers = !next.is_empty()
                && !next.ends_with('?')
                && heading(next).is_none()
                && labelled(next, &QUESTION_LABELS).is_none()
                && word_count(next) <= 15
                && !is_noise(next);
            if let Some(a) = labelled(next, &ANSWER_LABELS).or_else(|| labelled(next, &WORKED_LABELS)) {
                out.last_mut().unwrap().elements.push(Element::Qa {
                    q: bare.to_string(),
                    a: (!a.is_empty()).then(|| a.to_string()),
                    worked: std::mem::take(&mut example_next) || labelled(next, &WORKED_LABELS).is_some(),
                    by_layout: false,
                    line: n,
                });
                i += 2;
                continue;
            }
            out.last_mut().unwrap().elements.push(Element::Qa {
                q: bare.to_string(),
                a: answers.then(|| next.to_string()),
                worked: std::mem::take(&mut example_next),
                by_layout: answers,
                line: n,
            });
            i += if answers { 2 } else { 1 };
            continue;
        }
        if is_noise(line) {
            flush_para(&mut out, &mut para, para_first, n - 1);
            out.last_mut().unwrap().elements.push(Element::Noise { text: line.to_string(), line: n });
            i += 1;
            continue;
        }
        if para.is_empty() {
            para_first = n;
        }
        para.push(raw.trim_end());
        i += 1;
    }
    if let Some((q, qline, worked)) = pending.take() {
        out.last_mut().unwrap().elements.push(Element::Qa {
            q,
            a: None,
            worked,
            by_layout: false,
            line: qline,
        });
    }
    let last = lines.len();
    flush_para(&mut out, &mut para, para_first, last);
    out.retain(|s| !s.elements.is_empty() || !s.title.is_empty());
    out
}

fn item(ids: &mut Ids, file: &str, q: &str, a: Option<&str>, line: usize, flags: Vec<String>) -> DraftItem {
    DraftItem {
        id: ids.item(),
        question: q.trim().to_string(),
        answer: a.unwrap_or("").trim().to_string(),
        answer_source: if a.is_some_and(|a| !a.trim().is_empty()) {
            AnswerSource::Source
        } else {
            AnswerSource::Missing
        },
        verified: false,
        origin: Origin::Imported,
        edited: false,
        source: Some(SourceRef { file: file.into(), line }),
        flags,
        note: String::new(),
    }
}

fn block(ids: &mut Ids, kind: BlockKind, title: &str) -> DraftBlock {
    DraftBlock {
        id: ids.block(),
        kind,
        title: title.to_string(),
        text: String::new(),
        items: Vec::new(),
        mode: ReadingMode::Neural,
        use_in_teaching: kind != BlockKind::Unsorted,
        origin: Origin::Imported,
        edited: false,
        lines: None,
    }
}

fn widen(lines: &mut Option<(usize, usize)>, first: usize, last: usize) {
    *lines = Some(match *lines {
        Some((a, b)) => (a.min(first), b.max(last)),
        None => (first, last),
    });
}

/// The blocks of one section: its prose (instruction when the section has questions, reading
/// otherwise), its questions (worked ones as examples; of the others about a third become
/// examples when there are none, the rest practice, and a fifth of a long list extra practice),
/// and what fits nowhere.
fn section_blocks(ids: &mut Ids, file: &str, s: Section) -> Vec<DraftBlock> {
    let has_qa = s.elements.iter().any(|e| matches!(e, Element::Qa { .. }));
    let prose_kind = if has_qa { BlockKind::Lesson } else { BlockKind::Reading };
    let mut out: Vec<DraftBlock> = Vec::new();
    let mut qas: Vec<(DraftItem, bool)> = Vec::new();
    let mut noise = block(ids, BlockKind::Unsorted, &s.title);
    let push_text = |out: &mut Vec<DraftBlock>, ids: &mut Ids, text: &str, first: usize, last: usize| {
        match out.last_mut() {
            Some(b) if b.kind == prose_kind => {
                b.text.push_str("\n\n");
                b.text.push_str(text);
                widen(&mut b.lines, first, last);
            }
            _ => {
                let mut b = block(ids, prose_kind, &s.title);
                if s.level > 0 && !s.title.is_empty() {
                    b.text = format!("{} {}\n\n{text}", "#".repeat(s.level.min(3) + 1), s.title);
                } else {
                    b.text = text.to_string();
                }
                b.lines = Some((first, last));
                out.push(b);
            }
        }
    };
    for e in s.elements {
        match e {
            Element::Prose { text, first, last } => push_text(&mut out, ids, &text, first, last),
            Element::Qa { q, a, worked, by_layout, line } => {
                let mut flags = Vec::new();
                if by_layout {
                    flags.push(
                        "Answer paired with the question by its position (the line below); check it".into(),
                    );
                }
                if a.is_none() {
                    flags.push("No answer in the file".into());
                }
                qas.push((item(ids, file, &q, a.as_deref(), line, flags), worked));
            }
            Element::Noise { text, line } => {
                if !noise.text.is_empty() {
                    noise.text.push('\n');
                }
                noise.text.push_str(&text);
                widen(&mut noise.lines, line, line);
            }
        }
    }
    if !qas.is_empty() {
        let explicit_examples = qas.iter().any(|(_, w)| *w);
        let answered: Vec<usize> = (0..qas.len()).filter(|&i| !qas[i].0.answer.is_empty()).collect();
        let mut kind_of: Vec<BlockKind> =
            qas.iter().map(|(_, w)| if *w { BlockKind::Examples } else { BlockKind::Practice }).collect();
        if !explicit_examples && answered.len() >= 3 {
            for &i in answered.iter().take(answered.len().div_ceil(3).min(8)) {
                kind_of[i] = BlockKind::Examples;
            }
        }
        let practice: Vec<usize> = (0..qas.len())
            .filter(|&i| kind_of[i] == BlockKind::Practice && !qas[i].0.answer.is_empty())
            .collect();
        if practice.len() >= 10 {
            for &i in &practice[practice.len() - practice.len() / 5..] {
                kind_of[i] = BlockKind::ExtraPractice;
            }
        }
        for kind in [BlockKind::Examples, BlockKind::Practice, BlockKind::ExtraPractice] {
            let mut b = block(ids, kind, &s.title);
            for (i, (it, _)) in qas.iter().enumerate() {
                if kind_of[i] == kind {
                    let line = it.source.as_ref().map_or(0, |s| s.line);
                    widen(&mut b.lines, line, line);
                    b.items.push(it.clone());
                }
            }
            if !b.items.is_empty() {
                out.push(b);
            }
        }
    }
    if !noise.text.is_empty() {
        out.push(noise);
    }
    out
}

/// Imports a Unified Teacher Lesson file block by block: explanations become Lesson blocks,
/// dialogue and understanding examples Examples, `[Extra Practice]` Extra Practice,
/// `[Evaluation]` and `[Reading]` themselves, other blocks Unified text kept as written.
fn import_unified(ids: &mut Ids, file: &str, text: &str) -> Vec<DraftBlock> {
    let mut out = Vec::new();
    let line_of = |needle: &str| text.find(needle).map_or(0, |p| text[..p].matches('\n').count() + 1);
    let qa_lines = |body: &str| -> Vec<(String, String)> {
        body.lines()
            .filter_map(|l| l.find("  ").map(|i| (l[..i].trim().to_string(), l[i + 2..].trim().to_string())))
            .filter(|(q, a)| !q.is_empty() && !a.is_empty())
            .collect()
    };
    for (name, body) in blocks(text) {
        let first = line_of(body);
        match name {
            "Dialogue Lesson" | "Understanding Lesson" => {
                let (explanation, examples) = match body.find("[examples]\n") {
                    Some(i) => (&body[..i], body[i + 11..].split("[/examples]").next().unwrap_or("")),
                    None => (body, ""),
                };
                if !explanation.trim().is_empty() {
                    let mut b = block(ids, BlockKind::Lesson, "");
                    b.text = explanation.trim().to_string();
                    b.lines = Some((first, first + explanation.matches('\n').count()));
                    out.push(b);
                }
                let mut b = block(ids, BlockKind::Examples, "");
                for (q, a) in qa_lines(examples) {
                    let line = line_of(&format!("{q}  "));
                    b.items.push(item(ids, file, &q, Some(&a), line, Vec::new()));
                }
                if !b.items.is_empty() {
                    b.lines = Some((first, first + body.matches('\n').count()));
                    out.push(b);
                }
            }
            "Extra Practice" | "Evaluation" => {
                let kind =
                    if name == "Evaluation" { BlockKind::Evaluation } else { BlockKind::ExtraPractice };
                let mut b = block(ids, kind, "");
                for (q, a) in qa_lines(body) {
                    let line = line_of(&format!("{q}  "));
                    b.items.push(item(ids, file, &q, Some(&a), line, Vec::new()));
                }
                b.lines = Some((first, first + body.matches('\n').count()));
                out.push(b);
            }
            "Reading" => {
                let mut b = block(ids, BlockKind::Reading, "");
                let t = body.trim_start();
                let (mode, rest) = match t.strip_prefix("Mode:") {
                    Some(r) => {
                        let (v, after) = r.split_once('\n').unwrap_or((r, ""));
                        (ReadingMode::parse(v).unwrap_or_default(), after)
                    }
                    None => (ReadingMode::Neural, body),
                };
                b.mode = mode;
                b.text = rest.trim().to_string();
                b.lines = Some((first, first + body.matches('\n').count()));
                out.push(b);
            }
            _ => {
                let mut b = block(ids, BlockKind::Raw, name);
                b.text = format!("[{name}]\n{}[/{name}]", body);
                b.lines = Some((first, first + body.matches('\n').count()));
                out.push(b);
            }
        }
    }
    out
}

fn is_unified(text: &str) -> bool {
    blocks(text).iter().any(|(n, _)| {
        matches!(
            *n,
            "Dialogue Lesson"
                | "Understanding Lesson"
                | "Multi-Turn Conversation Lesson"
                | "Reasoning Lesson"
                | "Reading"
                | "Evaluation"
                | "Extra Practice"
        )
    })
}

fn draft_size(blocks: &[DraftBlock]) -> (usize, usize) {
    let items = blocks.iter().map(|b| b.items.len()).sum();
    let words = blocks.iter().map(|b| word_count(&b.text)).sum();
    (items, words)
}

fn new_draft(name: &str, file: &str, sha: &str, blocks: Vec<DraftBlock>) -> LessonDraft {
    LessonDraft {
        id: String::new(),
        name: name.to_string(),
        source_file: file.to_string(),
        source_sha256: sha.to_string(),
        stage: None,
        finished: false,
        notes: String::new(),
        blocks,
    }
}

/// Proposes lesson drafts for an imported text file: one, or (with `split`) several related
/// lessons for a large file, divided at its top-level headings or, without headings, by size.
/// Text is converted to ASCII first (the tokenizer's alphabet).
pub fn analyze(text: &str, file_name: &str, opts: AnalyzeOptions) -> Vec<LessonDraft> {
    let sha = sha256_hex(text.as_bytes());
    let (text, _) = normalize_ascii(text);
    let base = file_name.rsplit(['/', '\\']).next().unwrap_or(file_name);
    let base = base.strip_suffix(".txt").unwrap_or(base).to_string();
    let mut ids = Ids { block: 0, item: 0 };
    if is_unified(&text) {
        let blocks = import_unified(&mut ids, file_name, &text);
        return vec![new_draft(&base, file_name, &sha, blocks)];
    }
    let secs = sections(&text);
    let top = secs.iter().filter(|s| s.level > 0).map(|s| s.level).min().unwrap_or(0);
    // Group sections under each top-level heading (each group a possible lesson).
    let mut groups: Vec<(String, Vec<DraftBlock>)> = Vec::new();
    for s in secs {
        let starts = top > 0 && s.level == top;
        let title = s.title.clone();
        let blocks = section_blocks(&mut ids, file_name, s);
        if starts || groups.is_empty() {
            groups.push((title, blocks));
        } else {
            groups.last_mut().unwrap().1.extend(blocks);
        }
    }
    groups.retain(|(_, b)| !b.is_empty());
    let all: Vec<DraftBlock> = groups.iter().flat_map(|(_, b)| b.clone()).collect();
    let (items, words) = draft_size(&all);
    if !opts.split || (items <= SPLIT_ITEMS && words <= SPLIT_WORDS) {
        return vec![new_draft(&base, file_name, &sha, all)];
    }
    // One lesson per top-level section, and no lesson with too many questions.
    let mut lessons: Vec<(String, Vec<DraftBlock>)> = Vec::new();
    let source = if groups.len() > 1 { groups } else { vec![(String::new(), all)] };
    for (title, blocks) in source {
        let mut current: Vec<DraftBlock> = Vec::new();
        let mut count = 0;
        let mut part = 1;
        for mut b in blocks {
            while count + b.items.len() > MAX_LESSON_ITEMS {
                let room = MAX_LESSON_ITEMS - count;
                let mut head = b.clone();
                head.id = ids.block();
                head.items = b.items.drain(..room).collect();
                if !head.items.is_empty() {
                    current.push(head);
                }
                lessons.push((part_title(&title, part), std::mem::take(&mut current)));
                part += 1;
                count = 0;
            }
            count += b.items.len();
            current.push(b);
        }
        if !current.is_empty() {
            lessons.push((if part > 1 { part_title(&title, part) } else { title.clone() }, current));
        }
    }
    let n = lessons.len();
    lessons
        .into_iter()
        .enumerate()
        .map(|(i, (title, blocks))| {
            let name = if title.is_empty() {
                format!("{base} part {}", i + 1)
            } else {
                format!("{:02} {base} - {title}", i + 1)
            };
            let mut d = new_draft(if n == 1 { &base } else { &name }, file_name, &sha, blocks);
            d.notes = String::new();
            d
        })
        .collect()
}

fn part_title(title: &str, part: usize) -> String {
    if title.is_empty() {
        String::new()
    } else {
        format!("{title} ({part})")
    }
}

// ---------------------------------------------------------------------------------------------
// Markdown to plain text

/// Removes inline Markdown (emphasis, code, links, images, HTML tags) from one line.
fn inline_plain(line: &str) -> String {
    let mut s = String::with_capacity(line.len());
    let b: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        // ![alt](url) -> alt ; [text](url) -> text
        if (c == '[' || (c == '!' && b.get(i + 1) == Some(&'['))) && !(c == '[' && i > 0 && b[i - 1] == '\\')
        {
            let start = if c == '!' { i + 2 } else { i + 1 };
            if let Some(close) = (start..b.len()).find(|&j| b[j] == ']') {
                if b.get(close + 1) == Some(&'(') {
                    if let Some(end) = (close + 2..b.len()).find(|&j| b[j] == ')') {
                        s.extend(&b[start..close]);
                        i = end + 1;
                        continue;
                    }
                }
            }
        }
        if c == '<' {
            if let Some(end) = (i + 1..b.len()).find(|&j| b[j] == '>') {
                let inner: String = b[i + 1..end].iter().collect();
                let tagish =
                    inner.trim_start_matches('/').chars().next().is_some_and(|c| c.is_ascii_alphabetic())
                        && !inner.contains(' ')
                        || inner.starts_with('/')
                        || inner.contains('=');
                if tagish {
                    i = end + 1;
                    continue;
                }
            }
        }
        if c == '\\' && i + 1 < b.len() && "\\`*_{}[]()#+-.!|<>".contains(b[i + 1]) {
            s.push(b[i + 1]);
            i += 2;
            continue;
        }
        if c == '`' || c == '*' {
            i += 1;
            continue;
        }
        if c == '_' {
            // Emphasis underscores sit at a word edge; snake_case keeps its underscores.
            let before = i > 0 && b[i - 1].is_ascii_alphanumeric();
            let after = i + 1 < b.len() && b[i + 1].is_ascii_alphanumeric();
            if !(before && after) {
                i += 1;
                continue;
            }
        }
        s.push(c);
        i += 1;
    }
    s
}

/// Markdown as the plain text Nova is trained on: headings, emphasis, code fences, links,
/// images, quotes, rules and HTML removed (their text kept), bullets as "- ", table rows as
/// cells separated by "; ". Converted to ASCII.
pub fn markdown_to_plain(md: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut in_code = false;
    for line in md.replace("\r\n", "\n").split('\n') {
        let t = line.trim_end();
        let trimmed = t.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            out.push(t.to_string());
            continue;
        }
        if trimmed.len() >= 3 && trimmed.chars().all(|c| matches!(c, '-' | '*' | '_' | ' ')) {
            continue; // horizontal rule
        }
        let mut l = trimmed;
        let hashes = l.bytes().take_while(|&b| b == b'#').count();
        if (1..=6).contains(&hashes) && l[hashes..].starts_with(' ') {
            l = l[hashes..].trim().trim_end_matches('#').trim();
        }
        while let Some(r) = l.strip_prefix('>') {
            l = r.trim_start();
        }
        if l.starts_with('|') || (l.contains('|') && l.ends_with('|')) {
            let cells: Vec<String> = l
                .trim_matches('|')
                .split('|')
                .map(|c| inline_plain(c.trim()))
                .filter(|c| !c.is_empty())
                .collect();
            if cells.iter().all(|c| c.chars().all(|ch| matches!(ch, '-' | ':' | ' '))) {
                continue; // header separator
            }
            out.push(cells.join("; "));
            continue;
        }
        let bullet = ["- ", "* ", "+ "].iter().find_map(|p| l.strip_prefix(p));
        let text = match bullet {
            Some(rest) => format!("- {}", inline_plain(rest)),
            None => inline_plain(l),
        };
        out.push(text);
    }
    // Collapse runs of blank lines.
    let mut joined = String::new();
    let mut blank = 0;
    for l in out {
        if l.trim().is_empty() {
            blank += 1;
            continue;
        }
        if !joined.is_empty() {
            joined.push_str(if blank > 0 { "\n\n" } else { "\n" });
        }
        blank = 0;
        joined.push_str(&l);
    }
    normalize_ascii(&joined).0
}

/// One line of plain text for a question or an answer.
fn one_line(s: &str) -> String {
    markdown_to_plain(s).split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------------------------
// Validation

/// Words compared for near duplicates (lowercase, contractions expanded).
fn word_set(s: &str) -> BTreeSet<String> {
    words(s).into_iter().collect()
}

/// Share of words two texts have in common (Jaccard).
pub fn similarity(a: &str, b: &str) -> f32 {
    let (x, y) = (word_set(a), word_set(b));
    if x.is_empty() || y.is_empty() {
        return 0.;
    }
    x.intersection(&y).count() as f32 / x.union(&y).count() as f32
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Op(char),
    Eq,
    Other,
}

fn arith_tokens(text: &str) -> Vec<Tok> {
    let lower = text.to_ascii_lowercase();
    let mut out = Vec::new();
    let w: Vec<&str> = lower.split_whitespace().collect();
    let mut i = 0;
    while i < w.len() {
        let word = w[i];
        let two = w.get(i + 1).map(|n| format!("{word} {n}"));
        match (word, two.as_deref()) {
            (_, Some("multiplied by")) | (_, Some("divided by")) => {
                out.push(Tok::Op(if word == "multiplied" { '*' } else { '/' }));
                i += 2;
                continue;
            }
            ("plus", _) => out.push(Tok::Op('+')),
            ("minus", _) => out.push(Tok::Op('-')),
            ("times", _) => out.push(Tok::Op('*')),
            ("equals", _) => out.push(Tok::Eq),
            _ => {
                // Split a word like "3+4=7" or "(3" into numbers and symbols.
                let mut num = String::new();
                let flush = |num: &mut String, out: &mut Vec<Tok>| {
                    if !num.is_empty() {
                        match num.trim_end_matches('.').parse::<f64>() {
                            Ok(v) => out.push(Tok::Num(v)),
                            Err(_) => out.push(Tok::Other),
                        }
                        num.clear();
                    }
                };
                let chars: Vec<char> = word.chars().collect();
                for (k, &c) in chars.iter().enumerate() {
                    if c.is_ascii_digit()
                        || (c == '.' && !num.is_empty())
                        || (c == ','
                            && !num.is_empty()
                            && chars.get(k + 1).is_some_and(|d| d.is_ascii_digit()))
                    {
                        if c != ',' {
                            num.push(c);
                        }
                    } else {
                        flush(&mut num, &mut out);
                        match c {
                            '+' | '-' | '*' | '/' => out.push(Tok::Op(c)),
                            'x' if !out.is_empty()
                                && matches!(out.last(), Some(Tok::Num(_)))
                                && chars.len() == 1 =>
                            {
                                out.push(Tok::Op('*'))
                            }
                            '=' => out.push(Tok::Eq),
                            '?' | '!' | '.' | ',' | ':' | ';' | '(' | ')' => {}
                            _ => {
                                if !matches!(out.last(), Some(Tok::Other)) {
                                    out.push(Tok::Other);
                                }
                            }
                        }
                    }
                }
                flush(&mut num, &mut out);
            }
        }
        i += 1;
    }
    out
}

/// Evaluates `n op n op n ...` with * and / before + and -.
fn eval_chain(nums: &[f64], ops: &[char]) -> Option<f64> {
    let mut terms = vec![nums[0]];
    let mut adds = Vec::new();
    for (k, &op) in ops.iter().enumerate() {
        let n = nums[k + 1];
        match op {
            '*' => *terms.last_mut()? *= n,
            '/' => {
                if n == 0. {
                    return None;
                }
                *terms.last_mut()? /= n
            }
            _ => {
                adds.push(op);
                terms.push(n);
            }
        }
    }
    let mut v = terms[0];
    for (k, &op) in adds.iter().enumerate() {
        if op == '+' {
            v += terms[k + 1];
        } else {
            v -= terms[k + 1];
        }
    }
    Some(v)
}

/// Arithmetic written as `a op b ... = c` whose result is wrong: (expression, written, actual).
pub fn wrong_arithmetic(text: &str) -> Vec<(String, f64, f64)> {
    let t = arith_tokens(text);
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() {
        if let Tok::Num(first) = t[i] {
            let mut nums = vec![first];
            let mut ops = Vec::new();
            let mut j = i + 1;
            while j + 1 < t.len() {
                match (&t[j], &t[j + 1]) {
                    (Tok::Op(o), Tok::Num(n)) => {
                        ops.push(*o);
                        nums.push(*n);
                        j += 2;
                    }
                    _ => break,
                }
            }
            if !ops.is_empty() && j + 1 < t.len() && t[j] == Tok::Eq {
                if let (Tok::Num(written), Some(actual)) = (&t[j + 1], eval_chain(&nums, &ops)) {
                    if (written - actual).abs() > 1e-9 * actual.abs().max(1.) {
                        out.push((expression(&nums, &ops), *written, actual));
                    }
                    i = j + 2;
                    continue;
                }
            }
            i = j.max(i + 1);
            continue;
        }
        i += 1;
    }
    out
}

fn fmt_num(v: f64) -> String {
    if v.fract() == 0. && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn expression(nums: &[f64], ops: &[char]) -> String {
    let mut s = fmt_num(nums[0]);
    for (k, op) in ops.iter().enumerate() {
        s.push_str(&format!(" {op} {}", fmt_num(nums[k + 1])));
    }
    s
}

/// A question that is one sum (`What is 3 + 4?`, `seven plus...` excluded) and an answer whose
/// last number differs from its result: (expression, answer number, result).
pub fn suspicious_answer(question: &str, answer: &str) -> Option<(String, f64, f64)> {
    let qt = arith_tokens(question);
    if qt.contains(&Tok::Eq) {
        return None;
    }
    let nums: Vec<f64> =
        qt.iter().filter_map(|t| if let Tok::Num(n) = t { Some(*n) } else { None }).collect();
    let ops: Vec<char> = qt.iter().filter_map(|t| if let Tok::Op(o) = t { Some(*o) } else { None }).collect();
    if nums.len() < 2 || ops.len() != nums.len() - 1 {
        return None;
    }
    // The numbers and operators must alternate.
    let seq: Vec<&Tok> = qt.iter().filter(|t| !matches!(t, Tok::Other)).collect();
    let alternates = seq.iter().enumerate().all(|(k, t)| {
        if k % 2 == 0 {
            matches!(t, Tok::Num(_))
        } else {
            matches!(t, Tok::Op(_))
        }
    });
    if !alternates {
        return None;
    }
    let actual = eval_chain(&nums, &ops)?;
    let given =
        arith_tokens(answer).iter().rev().find_map(|t| if let Tok::Num(n) = t { Some(*n) } else { None })?;
    ((given - actual).abs() > 1e-9 * actual.abs().max(1.)).then(|| (expression(&nums, &ops), given, actual))
}

fn issue(severity: Severity, block: Option<&str>, item: Option<&str>, message: String) -> Issue {
    Issue { severity, block: block.map(String::from), item: item.map(String::from), message }
}

/// Teaching problems in a draft, for the teacher to review; nothing is changed.
pub fn validate(d: &LessonDraft) -> Vec<Issue> {
    let mut out = Vec::new();
    let used: Vec<&DraftBlock> = d.blocks.iter().filter(|b| b.use_in_teaching).collect();
    let evaluation: Vec<(&DraftBlock, &DraftItem)> = used
        .iter()
        .filter(|b| b.kind == BlockKind::Evaluation)
        .flat_map(|b| b.items.iter().map(move |i| (*b, i)))
        .collect();
    let taught_kinds = [BlockKind::Examples, BlockKind::Practice, BlockKind::ExtraPractice];
    let taught: Vec<(&DraftBlock, &DraftItem)> = used
        .iter()
        .filter(|b| taught_kinds.contains(&b.kind))
        .flat_map(|b| b.items.iter().map(move |i| (*b, i)))
        .collect();
    if d.name.trim().is_empty() {
        out.push(issue(Severity::Error, None, None, "The lesson needs a name".into()));
    }
    for b in &d.blocks {
        let bid = Some(b.id.as_str());
        if !b.use_in_teaching && b.kind != BlockKind::Unsorted {
            out.push(issue(
                Severity::Info,
                bid,
                None,
                format!("{} block not used in teaching", b.kind.label()),
            ));
        }
        if b.kind == BlockKind::Unsorted && b.use_in_teaching && !b.text.trim().is_empty() {
            out.push(issue(
                Severity::Warning,
                bid,
                None,
                "Unsorted material is never taught; move it into a block or delete it".into(),
            ));
        }
        if b.kind.has_items() && b.items.is_empty() {
            out.push(issue(Severity::Info, bid, None, format!("Empty {} block", b.kind.label())));
        }
        if !b.kind.has_items() && b.text.trim().is_empty() && b.kind != BlockKind::Unsorted {
            out.push(issue(Severity::Info, bid, None, format!("Empty {} block", b.kind.label())));
        }
        if b.kind == BlockKind::Raw {
            if let Err(e) = parse_lesson(&normalize_ascii(&b.text).0, "check") {
                out.push(issue(Severity::Error, bid, None, format!("Unified text cannot be read: {e}")));
            }
        }
        if b.kind == BlockKind::Reading && b.mode.studies() {
            out.push(issue(
                Severity::Info,
                bid,
                None,
                format!("Reading mode '{}' needs Study Mode on when taught (otherwise only neural reading happens)", b.mode.name()),
            ));
        }
        if !b.text.is_ascii() {
            out.push(issue(
                Severity::Info,
                bid,
                None,
                "Non-ASCII characters will be converted or dropped (Nova reads ASCII)".into(),
            ));
        }
        for (expr, written, actual) in wrong_arithmetic(&b.text) {
            out.push(issue(
                Severity::Warning,
                bid,
                None,
                format!("Suspicious arithmetic: {expr} = {} (it is {})", fmt_num(written), fmt_num(actual)),
            ));
        }
        for it in &b.items {
            let iid = Some(it.id.as_str());
            if it.question.trim().is_empty() {
                out.push(issue(Severity::Error, bid, iid, "An item has no question".into()));
                continue;
            }
            if it.answer.trim().is_empty() {
                let sev = if b.kind == BlockKind::Evaluation { Severity::Error } else { Severity::Warning };
                out.push(issue(
                    sev,
                    bid,
                    iid,
                    format!(
                        "No answer for \"{}\": not trained or graded until one is written",
                        short(&it.question)
                    ),
                ));
            }
            if it.answer_source == AnswerSource::Generated && !it.verified {
                out.push(issue(
                    Severity::Error,
                    bid,
                    iid,
                    format!(
                        "Generated answer for \"{}\" must be verified before it is a training target",
                        short(&it.question)
                    ),
                ));
            }
            if !it.verified && !it.edited {
                for f in &it.flags {
                    if f != "No answer in the file" {
                        out.push(issue(
                            Severity::Warning,
                            bid,
                            iid,
                            format!("\"{}\": {f}", short(&it.question)),
                        ));
                    }
                }
            }
            for text in [&it.question, &it.answer] {
                for (expr, written, actual) in wrong_arithmetic(text) {
                    out.push(issue(
                        Severity::Warning,
                        bid,
                        iid,
                        format!(
                            "Suspicious arithmetic: {expr} = {} (it is {})",
                            fmt_num(written),
                            fmt_num(actual)
                        ),
                    ));
                }
            }
            if let Some((expr, given, actual)) = suspicious_answer(&it.question, &it.answer) {
                out.push(issue(
                    Severity::Warning,
                    bid,
                    iid,
                    format!(
                        "Suspicious answer: {expr} is {}, the answer says {}",
                        fmt_num(actual),
                        fmt_num(given)
                    ),
                ));
            }
        }
    }
    // Duplicate questions.
    let mut seen: BTreeMap<String, (&DraftBlock, &DraftItem)> = BTreeMap::new();
    for (b, it) in evaluation.iter().chain(taught.iter()) {
        let key = canonical(&one_line(&it.question));
        if key.is_empty() {
            continue;
        }
        if let Some((pb, _)) = seen.get(&key) {
            let leak = (pb.kind == BlockKind::Evaluation) != (b.kind == BlockKind::Evaluation);
            out.push(issue(
                if leak { Severity::Error } else { Severity::Warning },
                Some(&b.id),
                Some(&it.id),
                if leak {
                    format!(
                        "\"{}\" is both an evaluation question and taught: it would leak the answer",
                        short(&it.question)
                    )
                } else {
                    format!("Duplicate question \"{}\" (also in {})", short(&it.question), pb.kind.label())
                },
            ));
        } else {
            seen.insert(key, (b, it));
        }
    }
    // Near duplicates of held-out questions in taught material and reading.
    let held: Vec<(&DraftBlock, &DraftItem)> = if evaluation.is_empty() {
        taught.iter().filter(|(_, it)| is_held_out(&one_line(&it.question))).copied().collect()
    } else {
        evaluation.clone()
    };
    for (_, e) in &held {
        let eq = one_line(&e.question);
        if word_count(&eq) < 4 {
            continue;
        }
        for (b, it) in &taught {
            if it.id == e.id || canonical(&one_line(&it.question)) == canonical(&eq) {
                continue;
            }
            if similarity(&it.question, &eq) >= 0.8 {
                out.push(issue(
                    Severity::Warning,
                    Some(&b.id),
                    Some(&it.id),
                    format!("\"{}\" is nearly the held-out question \"{}\"", short(&it.question), short(&eq)),
                ));
            }
        }
        let ce = canonical(&eq);
        for b in used.iter().filter(|b| matches!(b.kind, BlockKind::Reading | BlockKind::Lesson)) {
            if canonical(&markdown_to_plain(&b.text)).contains(&ce) {
                out.push(issue(
                    Severity::Warning,
                    Some(&b.id),
                    None,
                    format!(
                        "The held-out question \"{}\" appears in this {} text",
                        short(&eq),
                        b.kind.label()
                    ),
                ));
            }
        }
    }
    // Enough held-out questions to grade the lesson.
    let answered =
        |v: &[(&DraftBlock, &DraftItem)]| v.iter().filter(|(_, i)| !i.answer.trim().is_empty()).count();
    let eval_count = if evaluation.is_empty() {
        let n = taught
            .iter()
            .filter(|(b, i)| {
                b.kind != BlockKind::ExtraPractice
                    && !i.answer.trim().is_empty()
                    && is_held_out(&one_line(&i.question))
            })
            .count();
        if n == 0 && answered(&taught) > 0 {
            out.push(issue(
                Severity::Error,
                None,
                None,
                "No question is held out (about 1 in 10 is, by hash): add an Evaluation block so the lesson can be graded".into(),
            ));
        } else if n > 0 {
            out.push(issue(
                Severity::Info,
                None,
                None,
                format!("No Evaluation block: {n} question(s) will be held out by hash and never taught"),
            ));
        }
        n
    } else {
        answered(&evaluation)
    };
    if eval_count > 0 && eval_count < super::curriculum::FEW_QUESTIONS {
        out.push(issue(
            Severity::Warning,
            None,
            None,
            format!(
                "Only {eval_count} held-out question(s): grades will be coarse (aim for at least {})",
                super::curriculum::FEW_QUESTIONS
            ),
        ));
    }
    if answered(&taught) == 0
        && !used.iter().any(|b| b.kind == BlockKind::Reading || b.kind == BlockKind::Raw)
    {
        out.push(issue(
            Severity::Warning,
            None,
            None,
            "Nothing to teach: no answered examples or practice and no reading".into(),
        ));
    }
    out
}

fn short(s: &str) -> String {
    let s = s.trim();
    if s.len() > 60 {
        format!(
            "{}...",
            &s[..s.char_indices().take_while(|(i, _)| *i < 57).last().map_or(0, |(i, c)| i + c.len_utf8())]
        )
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------------------------
// Export

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Exported {
    /// The Unified Teacher Lesson file.
    pub text: String,
    /// Units taught (questions, conversations, explanations, reading).
    pub taught: usize,
    /// Held-out evaluation questions.
    pub held_out: usize,
    /// Reading units, and how many of them are studied into memory.
    pub reading: usize,
    pub reading_memory: usize,
    /// Extra practice questions (reserve).
    pub extra_practice: usize,
    /// Items left out, with the reason.
    pub skipped: Vec<String>,
}

/// Plain text that cannot close a block: a line starting with `[` gets a space before it.
fn safe_text(s: &str) -> String {
    s.lines()
        .map(|l| if l.trim_start().starts_with('[') { format!(" {l}") } else { l.to_string() })
        .collect::<Vec<_>>()
        .join("\n")
}

fn qa_line(it: &DraftItem, skipped: &mut Vec<String>) -> Option<String> {
    let q = one_line(&it.question);
    let a = one_line(&it.answer);
    if q.is_empty() {
        return None;
    }
    if a.is_empty() {
        skipped.push(format!("\"{}\": no answer", short(&q)));
        return None;
    }
    if it.answer_source == AnswerSource::Generated && !it.verified {
        skipped.push(format!("\"{}\": generated answer not verified", short(&q)));
        return None;
    }
    Some(format!("{q}  {a}"))
}

/// Writes the draft as a Unified Teacher Lesson file (only blocks used in teaching): each
/// Lesson block with the examples and practice after it becomes a `[Dialogue Lesson]` (the
/// instruction as its explanation), Reading `[Reading]`, Extra Practice `[Extra Practice]`,
/// Evaluation `[Evaluation]`, Unified text as written; Unsorted blocks are left out. The file
/// is read back with the lesson parser to count what will be taught.
pub fn export(d: &LessonDraft) -> Result<Exported, String> {
    let mut text = format!(
        "Lesson: {}\nMade with the Yendory Lesson Designer{}.\n\n",
        one_line(&d.name),
        if d.source_file.is_empty() { String::new() } else { format!(" from {}", one_line(&d.source_file)) }
    );
    let mut skipped = Vec::new();
    let mut extra = 0;
    // The open [Dialogue Lesson]: explanation and example lines.
    let mut explanation: Vec<String> = Vec::new();
    let mut examples: Vec<String> = Vec::new();
    let flush = |text: &mut String, explanation: &mut Vec<String>, examples: &mut Vec<String>| {
        if explanation.is_empty() && examples.is_empty() {
            return;
        }
        text.push_str("[Dialogue Lesson]\n");
        if !explanation.is_empty() {
            text.push_str(&safe_text(&explanation.join("\n\n")));
            text.push('\n');
        }
        if !examples.is_empty() {
            text.push_str("[examples]\n");
            for l in examples.iter() {
                text.push_str(l);
                text.push('\n');
            }
            text.push_str("[/examples]\n");
        }
        text.push_str("[/Dialogue Lesson]\n\n");
        explanation.clear();
        examples.clear();
    };
    for b in d.blocks.iter().filter(|b| b.use_in_teaching) {
        match b.kind {
            BlockKind::Lesson => {
                if !examples.is_empty() {
                    flush(&mut text, &mut explanation, &mut examples);
                }
                let plain = markdown_to_plain(&b.text);
                if !plain.trim().is_empty() {
                    explanation.push(plain);
                }
            }
            BlockKind::Examples | BlockKind::Practice => {
                examples.extend(b.items.iter().filter_map(|it| qa_line(it, &mut skipped)));
            }
            BlockKind::Reading => {
                flush(&mut text, &mut explanation, &mut examples);
                let plain = markdown_to_plain(&b.text);
                if plain.trim().is_empty() {
                    continue;
                }
                text.push_str(&format!(
                    "[Reading]\nMode: {}\n{}\n[/Reading]\n\n",
                    b.mode.name(),
                    safe_text(&plain)
                ));
            }
            BlockKind::ExtraPractice => {
                flush(&mut text, &mut explanation, &mut examples);
                let lines: Vec<String> = b.items.iter().filter_map(|it| qa_line(it, &mut skipped)).collect();
                if lines.is_empty() {
                    continue;
                }
                extra += lines.len();
                text.push_str("[Extra Practice]\n[Dialogue Lesson]\n[examples]\n");
                for l in lines {
                    text.push_str(&l);
                    text.push('\n');
                }
                text.push_str("[/examples]\n[/Dialogue Lesson]\n[/Extra Practice]\n\n");
            }
            BlockKind::Evaluation => {
                flush(&mut text, &mut explanation, &mut examples);
                let lines: Vec<String> = b.items.iter().filter_map(|it| qa_line(it, &mut skipped)).collect();
                if lines.is_empty() {
                    continue;
                }
                text.push_str("[Evaluation]\n");
                for l in lines {
                    text.push_str(&l);
                    text.push('\n');
                }
                text.push_str("[/Evaluation]\n\n");
            }
            BlockKind::Raw => {
                flush(&mut text, &mut explanation, &mut examples);
                text.push_str(normalize_ascii(b.text.trim()).0.as_str());
                text.push_str("\n\n");
            }
            BlockKind::Unsorted => {}
        }
    }
    flush(&mut text, &mut explanation, &mut examples);
    let items = parse_lesson(&text, &d.name)?;
    let reading: Vec<_> = items.iter().filter(|i| i.reading.is_some()).collect();
    Ok(Exported {
        taught: items.iter().filter(|i| !i.held_out && i.reading.map_or(true, ReadingMode::trains)).count(),
        held_out: items.iter().filter(|i| i.held_out).count(),
        reading: reading.len(),
        reading_memory: reading.iter().filter(|i| i.reading.is_some_and(ReadingMode::studies)).count(),
        extra_practice: extra,
        skipped,
        text,
    })
}

/// The file name a lesson is saved under in a course: its name with characters that are not
/// safe in file names replaced, the stage word added when the name lacks it.
pub fn file_name(d: &LessonDraft) -> String {
    let mut name: String = d
        .name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || " -_.()".contains(c) { c } else { '_' })
        .collect();
    name = name.trim().to_string();
    if name.is_empty() {
        name = "Lesson".into();
    }
    let lower = name.to_ascii_lowercase();
    let stage_word = match d.stage {
        Some(EducationStage::DenseKnowledge) if !lower.contains("dense") => " Dense",
        Some(EducationStage::TransformerLanguage) if !lower.contains("transformer") => " Transformer",
        Some(EducationStage::Cooperative) if !lower.contains("co-op") => " Co-op",
        _ => "",
    };
    format!("{name}{stage_word}.txt")
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATASET: &str = "# Animals\n\nDogs are animals. Animals need water to live.\nMax is a dog.\n\n\
        Q: Is Max a dog?\nA: Yes, Max is a dog.\n\nQ: Do animals need water?\nA: Yes.\n\n\
        What is a dog?\nAn animal.\n\nIs a cat a plant?\n\n\
        Example:\nQ: What is 3 + 4?\nSolution: 3 + 4 = 7.\n\n----\n\n\
        # The farm\n\nThe farm has cows and hens. The cows give milk.\n\n| animal | gives |\n|---|---|\n| cow | milk |\n";

    #[test]
    fn analysis_proposes_blocks_with_sources() {
        let drafts = analyze(DATASET, "animals.txt", AnalyzeOptions { split: false });
        assert_eq!(drafts.len(), 1);
        let d = &drafts[0];
        let kinds: Vec<BlockKind> = d.blocks.iter().map(|b| b.kind).collect();
        assert!(kinds.contains(&BlockKind::Lesson), "{kinds:?}");
        assert!(kinds.contains(&BlockKind::Examples));
        assert!(kinds.contains(&BlockKind::Practice));
        assert!(kinds.contains(&BlockKind::Reading), "a section without questions is reading");
        assert!(kinds.contains(&BlockKind::Unsorted), "the rule line fits nowhere");
        let items: Vec<&DraftItem> = d.blocks.iter().flat_map(|b| &b.items).collect();
        let max = items.iter().find(|i| i.question == "Is Max a dog?").unwrap();
        assert_eq!(max.answer, "Yes, Max is a dog.");
        assert_eq!(max.source.as_ref().unwrap().line, 6);
        // Paired by position: answered, but flagged for the teacher.
        let dog = items.iter().find(|i| i.question == "What is a dog?").unwrap();
        assert_eq!(dog.answer, "An animal.");
        assert!(!dog.flags.is_empty());
        // A question without an answer stays without one.
        let cat = items.iter().find(|i| i.question == "Is a cat a plant?").unwrap();
        assert_eq!(cat.answer_source, AnswerSource::Missing);
        assert!(cat.answer.is_empty());
        // The worked example is an example.
        let ex = d.blocks.iter().find(|b| b.kind == BlockKind::Examples).unwrap();
        assert!(ex.items.iter().any(|i| i.question == "What is 3 + 4?" && i.answer == "3 + 4 = 7."));
        // A table stays whole in its text block.
        let farm = d.blocks.iter().find(|b| b.kind == BlockKind::Reading).unwrap();
        assert!(farm.text.contains("| animal | gives |\n|---|---|\n| cow | milk |"), "{}", farm.text);
    }

    #[test]
    fn export_reads_back_and_keeps_evaluation_separate() {
        let mut d = analyze(DATASET, "animals.txt", AnalyzeOptions { split: false }).remove(0);
        d.blocks.push(DraftBlock {
            id: "e".into(),
            kind: BlockKind::Evaluation,
            title: String::new(),
            text: String::new(),
            items: vec![DraftItem {
                id: "e1".into(),
                question: "Does **Max** need water?".into(),
                answer: "Yes.".into(),
                answer_source: AnswerSource::Teacher,
                verified: true,
                origin: Origin::Teacher,
                edited: false,
                source: None,
                flags: vec![],
                note: "relationship".into(),
            }],
            mode: ReadingMode::Neural,
            use_in_teaching: true,
            origin: Origin::Teacher,
            edited: false,
            lines: None,
        });
        let ex = export(&d).unwrap();
        let items = parse_lesson(&ex.text, "x").unwrap();
        let held: Vec<_> = items.iter().filter(|i| i.held_out).collect();
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].prompt.as_deref(), Some("Does Max need water?"), "markdown removed");
        assert_eq!(ex.held_out, 1);
        assert!(ex.reading >= 1);
        assert!(ex.skipped.iter().any(|s| s.contains("cat")), "{:?}", ex.skipped);
        assert!(!ex.text.contains("----"), "unsorted material is not taught");
        // Reading is trained at full weight and never held out.
        let r = items.iter().find(|i| i.reading.is_some()).unwrap();
        assert!(!r.held_out && r.segments[0].weight == 1.0);
    }

    #[test]
    fn validation_flags_teaching_problems() {
        let mut d = analyze(DATASET, "animals.txt", AnalyzeOptions { split: false }).remove(0);
        let p = d.blocks.iter_mut().find(|b| b.kind == BlockKind::Practice).unwrap();
        p.items.push(DraftItem {
            id: "x".into(),
            question: "What is 6 + 7?".into(),
            answer: "12".into(),
            answer_source: AnswerSource::Generated,
            verified: false,
            origin: Origin::Generated,
            edited: false,
            source: None,
            flags: vec![],
            note: String::new(),
        });
        p.items.push(DraftItem { id: "y".into(), question: "Is Max a dog?".into(), ..p.items[0].clone() });
        let issues = validate(&d);
        let text = |s: &str| issues.iter().any(|i| i.message.contains(s));
        assert!(text("Suspicious answer: 6 + 7 is 13"), "{issues:#?}");
        assert!(text("must be verified"));
        assert!(text("Duplicate question"));
        assert!(text("No answer for \"Is a cat a plant?\""));
        assert!(text("by its position"));
        assert_eq!(wrong_arithmetic("so 3 + 4 = 8 and 2*3=6"), vec![("3 + 4".to_string(), 8., 7.)]);
        assert!(wrong_arithmetic("ten minus 4 equals 6, 10 - 4 = 6").is_empty());
    }

    #[test]
    fn markdown_becomes_plain_text() {
        let md = "## Colours\n\nThe **sky** is _blue_ and `grass` is green. See [this](http://x).\n\n\
                  - red\n* green\n\n| colour | thing |\n|---|---|\n| red | apple |\n\n```\ncode line\n```\n<script>alert(1)</script>";
        let plain = markdown_to_plain(md);
        assert_eq!(
            plain,
            "Colours\n\nThe sky is blue and grass is green. See this.\n\n- red\n- green\n\ncolour; thing\nred; apple\n\ncode line\nalert(1)"
        );
        assert_eq!(inline_plain("snake_case stays"), "snake_case stays");
    }

    #[test]
    fn large_datasets_are_divided() {
        let mut text = String::new();
        for s in 0..3 {
            text.push_str(&format!("# Topic {s}\n\nSome words about topic {s}.\n\n"));
            for i in 0..30 {
                text.push_str(&format!("Q: Question {s} {i}?\nA: Answer {i}.\n\n"));
            }
        }
        let drafts = analyze(&text, "big.txt", AnalyzeOptions::default());
        assert_eq!(drafts.len(), 3);
        assert!(drafts[0].name.contains("Topic 0"));
        assert_eq!(analyze(&text, "big.txt", AnalyzeOptions { split: false }).len(), 1);
    }

    #[test]
    fn unified_files_import_block_by_block() {
        let text = "[Dialogue Lesson]\nGreetings start talks.\n[examples]\nhello  Hello!\nbye  Goodbye!\n[/examples]\n[/Dialogue Lesson]\n\
                    [Reasoning Lesson]\n[examples]\nPremise: A. Question: B? Supported: C. Contradicted: D.\n[/examples]\n[/Reasoning Lesson]\n";
        let d = analyze(text, "u.txt", AnalyzeOptions::default()).remove(0);
        let kinds: Vec<BlockKind> = d.blocks.iter().map(|b| b.kind).collect();
        assert_eq!(kinds, [BlockKind::Lesson, BlockKind::Examples, BlockKind::Raw]);
        let ex = export(&d).unwrap();
        let before = parse_lesson(text, "u").unwrap();
        let after = parse_lesson(&ex.text, "u").unwrap();
        assert_eq!(before.len(), after.len());
    }
}
