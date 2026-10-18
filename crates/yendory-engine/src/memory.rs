//! Study Mode memory (training memory): statements Nova chose to keep from her reading, with
//! where they came from, and retrieval of the ones related to a question into her context.
//!
//! The memory is not part of the network and never answers for it. Retrieved memories are put
//! in front of a question as marked lines (`\x03statement\n`, like Nexus lines, never written
//! by outside text); the network has to learn, through ordinary training, whether and how to
//! use them. Nothing in the memory counts as learned: what Nova can answer without it is
//! measured separately from what she answers with it.
//!
//! What is kept: each sentence of a Reading block is a candidate; Nova's own prediction error
//! on it (mean loss per token, the network reading the sentence alone) ranks the candidates,
//! and she keeps the ones she cannot already predict: those above the passage's median and
//! above [`KNOWN_LOSS`], at most half of a passage, so passages are never copied whole. Those
//! two limits are fixed supporting rules; the ranking is the network's. As she learns, known
//! sentences fall below them and are no longer stored. The teacher's keep, edit and delete
//! decisions and reflection's usefulness measurements are recorded with each memory, as the
//! material a learned selection can later be trained on.
//!
//! Retrieval is an index (a supporting mechanism): memories sharing words with the question,
//! weighted by how rare the word is, plus one step to memories that share a word with those
//! (so "Max is a dog" and "Animals need water" bring "Dogs are animals" for "Does Max need
//! water?"). Questions held out for grading are never stored.
use crate::learn::grading::words;
use crate::learn::lessons::{canonical, sentences, Item, Kind, ReadingMode, Segment};
use crate::model::{Ablation, Bicameral};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Starts a memory line in the model's context: a remembered statement Study Mode retrieved.
/// Like the Nexus markers it never occurs in outside text (lesson files and user messages have
/// it removed), so only retrieval can put a memory in front of the model.
pub const MEMORY: char = '\x03';

/// Below this mean loss per token the network already predicts a sentence: nothing to keep.
pub const KNOWN_LOSS: f32 = 0.5;
/// Memories put in front of one question at most.
pub const RETRIEVE: usize = 4;
/// Retrieval records kept for inspection.
const LOG_LIMIT: usize = 400;

/// Function words left out of the index (they connect, they do not identify a concept).
const STOP: [&str; 64] = [
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "of", "to", "in",
    "on", "at", "by", "for", "with", "and", "or", "but", "if", "then", "so", "as", "it", "its", "this",
    "that", "these", "those", "there", "here", "what", "which", "who", "whom", "how", "why", "when", "where",
    "can", "could", "will", "would", "should", "has", "have", "had", "he", "she", "they", "we", "you", "i",
    "his", "her", "their", "our", "your", "my",
];
const NEGATION: [&str; 5] = ["not", "no", "never", "none", "nothing"];

/// The words a statement is indexed by: lowercase, function words and negations left out,
/// plural and third-person endings removed ("dogs" -> "dog", "needs" -> "need").
pub fn terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in words(text) {
        if STOP.contains(&w.as_str()) || NEGATION.contains(&w.as_str()) || w.len() < 2 {
            continue;
        }
        let t = stem(&w);
        if !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

fn stem(w: &str) -> String {
    if w.len() > 4 && w.ends_with("ies") {
        return format!("{}y", &w[..w.len() - 3]);
    }
    if w.len() > 4
        && (w.ends_with("ches") || w.ends_with("shes") || w.ends_with("sses") || w.ends_with("xes"))
    {
        return w[..w.len() - 2].to_string();
    }
    if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") && !w.ends_with("us") && !w.ends_with("is") {
        return w[..w.len() - 1].to_string();
    }
    w.to_string()
}

fn negated(text: &str) -> bool {
    words(text).iter().any(|w| NEGATION.contains(&w.as_str()))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemorySource {
    /// Lesson (file) name and the Reading text's fingerprint.
    pub lesson: String,
    #[serde(default)]
    pub sha256: String,
    /// Sentence number in the passage (from 1).
    #[serde(default)]
    pub sentence: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryOrigin {
    /// Chosen by Nova while studying.
    #[default]
    Studied,
    /// Written by the teacher.
    Teacher,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    pub id: u64,
    pub text: String,
    pub terms: Vec<String>,
    pub sources: Vec<MemorySource>,
    #[serde(default)]
    pub origin: MemoryOrigin,
    /// Changed by the teacher after it was stored.
    #[serde(default)]
    pub edited: bool,
    /// Nova's mean loss per token on the statement when it was stored.
    #[serde(default)]
    pub surprise: f32,
    /// The model's training step when it was stored.
    #[serde(default)]
    pub step: u64,
    #[serde(default)]
    pub created: u64,
    /// Times the same statement was studied again.
    #[serde(default)]
    pub seen: u32,
    /// Retrieved only when enabled (the teacher can switch a memory off without deleting it).
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(default)]
    pub retrieved: u32,
    #[serde(default)]
    pub last_retrieved: u64,
    /// Reflection: how much the memory lowered Nova's loss on related reading (nats per token;
    /// positive = it helped).
    #[serde(default)]
    pub usefulness: Option<f32>,
    /// The teacher checked the statement.
    #[serde(default)]
    pub verified: bool,
}
fn yes() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalRecord {
    pub time: u64,
    /// "training", "evaluation", "conversation", "reflection", "review".
    pub purpose: String,
    pub query: String,
    pub ids: Vec<u64>,
}

/// A teacher's decision about a memory: labelled examples of what is worth remembering.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub time: u64,
    pub id: u64,
    /// "kept", "edited", "deleted", "disabled", "enabled", "verified", "added".
    pub action: String,
    pub text: String,
    #[serde(default)]
    pub surprise: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryStore {
    pub entries: Vec<Memory>,
    #[serde(default)]
    pub next_id: u64,
    #[serde(default)]
    pub log: Vec<RetrievalRecord>,
    #[serde(default)]
    pub decisions: Vec<Decision>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    /// The two statements share a concept word.
    Shares,
    /// Most of their words are the same.
    Overlaps,
    /// Same words, but one is negated: they may contradict each other.
    Contradicts,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    pub a: u64,
    pub b: u64,
    pub kind: RelationKind,
    /// The shared words.
    pub terms: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StudyReport {
    pub lesson: String,
    pub candidates: usize,
    /// Stored as new memories: (id, text, surprise).
    pub stored: Vec<(u64, String, f32)>,
    /// Already in memory (studied again): ids.
    pub merged: Vec<u64>,
    /// Not kept because Nova already predicts them, or ranked below the passage's median.
    pub not_kept: Vec<(String, f32)>,
    /// Not kept because they match a question held out for grading.
    pub held_out: Vec<String>,
    /// New memories that may contradict an existing one: (new id, existing id).
    pub conflicts: Vec<(u64, u64)>,
}

fn jaccard(a: &[String], b: &[String]) -> f32 {
    let (x, y): (BTreeSet<&String>, BTreeSet<&String>) = (a.iter().collect(), b.iter().collect());
    if x.is_empty() || y.is_empty() {
        return 0.;
    }
    x.intersection(&y).count() as f32 / x.union(&y).count() as f32
}

/// Nova's mean loss per token on a statement read on its own (the network's surprise).
pub fn surprise(model: &Bicameral, text: &str) -> Result<f32, String> {
    let mut ids = model.tokenizer.encode(&format!("{text}\n"))?;
    ids.truncate(model.config.max_tokens + 1);
    if ids.len() < 2 {
        return Ok(0.);
    }
    let targets = ids[1..].to_vec();
    ids.pop();
    let mask = vec![1.; ids.len()];
    model.loss(&ids, &targets, &mask, Ablation::default())
}

/// Statements a passage offers: its sentences of 3 to 40 words that are not questions.
pub fn candidates(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split("\n\n") {
        let flat = para.split_whitespace().collect::<Vec<_>>().join(" ");
        for s in sentences(&flat) {
            let s = s.trim_start_matches("- ").trim();
            let n = s.split_whitespace().count();
            if (3..=40).contains(&n) && !s.ends_with('?') && s.chars().any(|c| c.is_ascii_alphabetic()) {
                out.push(s.to_string());
            }
        }
    }
    out
}

impl MemoryStore {
    pub fn get(&self, id: u64) -> Option<&Memory> {
        self.entries.iter().find(|m| m.id == id)
    }

    fn add(
        &mut self,
        text: &str,
        origin: MemoryOrigin,
        surprise: f32,
        step: u64,
        now: u64,
        source: Option<MemorySource>,
    ) -> u64 {
        self.next_id = self.next_id.max(self.entries.iter().map(|m| m.id).max().unwrap_or(0)) + 1;
        let id = self.next_id;
        self.entries.push(Memory {
            id,
            text: text.to_string(),
            terms: terms(text),
            sources: source.into_iter().collect(),
            origin,
            edited: false,
            surprise,
            step,
            created: now,
            seen: 1,
            enabled: true,
            retrieved: 0,
            last_retrieved: 0,
            usefulness: None,
            verified: origin == MemoryOrigin::Teacher,
        });
        id
    }

    /// An existing memory saying the same thing (same canonical text).
    fn same(&self, text: &str) -> Option<usize> {
        let c = canonical(text);
        self.entries.iter().position(|m| canonical(&m.text) == c)
    }

    /// Memories that may contradict `text`: mostly the same words, one of them negated.
    fn contradicting(&self, text: &str) -> Vec<u64> {
        let t = terms(text);
        let neg = negated(text);
        self.entries
            .iter()
            .filter(|m| negated(&m.text) != neg && jaccard(&m.terms, &t) >= 0.6)
            .map(|m| m.id)
            .collect()
    }

    /// Studies a passage: Nova ranks its sentences by her own surprise and keeps what she cannot
    /// already predict (see the module notes). `held_out` are questions and answers held out
    /// for grading; a sentence matching one is never stored.
    #[allow(clippy::too_many_arguments)]
    pub fn study(
        &mut self,
        model: &Bicameral,
        passage: &str,
        lesson: &str,
        sha256: &str,
        held_out: &[String],
        now: u64,
    ) -> Result<StudyReport, String> {
        let cands = candidates(passage);
        let mut report =
            StudyReport { lesson: lesson.to_string(), candidates: cands.len(), ..StudyReport::default() };
        let held: Vec<(String, Vec<String>)> = held_out.iter().map(|h| (canonical(h), terms(h))).collect();
        let mut scored = Vec::new();
        for (k, s) in cands.iter().enumerate() {
            let c = canonical(s);
            let t = terms(s);
            if held.iter().any(|(hc, ht)| {
                (hc.len() > 8 && c.contains(hc.as_str())) || (ht.len() >= 3 && jaccard(ht, &t) >= 0.8)
            }) {
                report.held_out.push(s.clone());
                continue;
            }
            scored.push((k, s.clone(), surprise(model, s)?));
        }
        let mut losses: Vec<f32> = scored.iter().map(|x| x.2).collect();
        losses.sort_by(|a, b| a.total_cmp(b));
        let median = losses.get(losses.len() / 2).copied().unwrap_or(0.);
        let limit = scored.len().div_ceil(2).max(1);
        let mut order: Vec<usize> = (0..scored.len()).collect();
        order.sort_by(|&a, &b| scored[b].2.total_cmp(&scored[a].2));
        let keep: BTreeSet<usize> = order
            .into_iter()
            .filter(|&i| scored[i].2 >= median && scored[i].2 > KNOWN_LOSS)
            .take(limit)
            .collect();
        for (i, (k, s, loss)) in scored.into_iter().enumerate() {
            let source =
                MemorySource { lesson: lesson.to_string(), sha256: sha256.to_string(), sentence: k + 1 };
            if let Some(j) = self.same(&s) {
                let m = &mut self.entries[j];
                m.seen += 1;
                if !m.sources.contains(&source) {
                    m.sources.push(source);
                }
                report.merged.push(m.id);
                continue;
            }
            if !keep.contains(&i) {
                report.not_kept.push((s, loss));
                continue;
            }
            let conflicts = self.contradicting(&s);
            let id = self.add(&s, MemoryOrigin::Studied, loss, model.step, now, Some(source));
            report.conflicts.extend(conflicts.into_iter().map(|c| (id, c)));
            report.stored.push((id, s, loss));
        }
        Ok(report)
    }

    /// A memory written by the teacher.
    pub fn add_teacher(&mut self, text: &str, step: u64, now: u64) -> u64 {
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let id = self.add(&text, MemoryOrigin::Teacher, 0., step, now, None);
        self.decisions.push(Decision { time: now, id, action: "added".into(), text, surprise: 0. });
        id
    }

    /// Changes, switches off or on, verifies or deletes a memory, recording the decision.
    pub fn decide(&mut self, id: u64, action: &str, text: Option<&str>, now: u64) -> Result<(), String> {
        let i = self.entries.iter().position(|m| m.id == id).ok_or("No such memory")?;
        let m = &mut self.entries[i];
        let surprise = m.surprise;
        match action {
            "edited" => {
                let t = text.ok_or("No text")?.split_whitespace().collect::<Vec<_>>().join(" ");
                if t.is_empty() {
                    return Err("A memory needs text".into());
                }
                m.text = t;
                m.terms = terms(&m.text);
                m.edited = true;
            }
            "disabled" => m.enabled = false,
            "enabled" => m.enabled = true,
            "verified" => m.verified = true,
            "kept" => {}
            "deleted" => {}
            _ => return Err(format!("Unknown action {action}")),
        }
        let text = m.text.clone();
        if action == "deleted" {
            self.entries.remove(i);
        }
        self.decisions.push(Decision { time: now, id, action: action.into(), text, surprise });
        Ok(())
    }

    /// How rare a word is among the memories (inverse document frequency).
    fn idf(&self) -> BTreeMap<&str, f32> {
        let n = self.entries.len() as f32;
        let mut df: BTreeMap<&str, f32> = BTreeMap::new();
        for m in self.entries.iter().filter(|m| m.enabled) {
            for t in &m.terms {
                *df.entry(t.as_str()).or_default() += 1.;
            }
        }
        df.into_iter().map(|(t, d)| (t, (1. + n / (1. + d)).ln())).collect()
    }

    /// The memories related to `query`, best first: shared words weighted by rarity, plus one
    /// step to memories sharing a word with those. At most `k`, all enabled.
    pub fn retrieve(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        let q = terms(query);
        let idf = self.idf();
        let live: Vec<&Memory> = self.entries.iter().filter(|m| m.enabled).collect();
        let direct: Vec<f32> = live
            .iter()
            .map(|m| {
                m.terms
                    .iter()
                    .filter(|t| q.contains(t))
                    .map(|t| idf.get(t.as_str()).copied().unwrap_or(0.))
                    .sum()
            })
            .collect();
        let mut score = direct.clone();
        for (i, m) in live.iter().enumerate() {
            if direct[i] > 0. {
                continue;
            }
            let hop = live
                .iter()
                .enumerate()
                .filter(|(j, h)| {
                    direct[*j] > 0. && h.terms.iter().any(|t| !q.contains(t) && m.terms.contains(t))
                })
                .map(|(j, _)| direct[j])
                .fold(0f32, f32::max);
            score[i] = 0.5 * hop;
        }
        let mut ranked: Vec<(u64, f32)> =
            live.iter().zip(&score).filter(|(_, &s)| s > 0.).map(|(m, &s)| (m.id, s)).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(k);
        ranked
    }

    /// The context lines for retrieved memories, keeping within `budget` characters.
    pub fn context(&self, ids: &[u64], budget: usize) -> String {
        let mut out = String::new();
        for id in ids {
            if let Some(m) = self.get(*id) {
                let line = format!("{MEMORY}{}\n", crate::nexus::sanitize(&m.text).replace('\n', " "));
                if out.len() + line.len() > budget {
                    break;
                }
                out.push_str(&line);
            }
        }
        out
    }

    /// Records a retrieval (for inspection) and counts it on the memories.
    pub fn record(&mut self, purpose: &str, query: &str, ids: &[u64], now: u64) {
        if ids.is_empty() {
            return;
        }
        for m in self.entries.iter_mut().filter(|m| ids.contains(&m.id)) {
            m.retrieved += 1;
            m.last_retrieved = now;
        }
        self.log.push(RetrievalRecord {
            time: now,
            purpose: purpose.into(),
            query: query.into(),
            ids: ids.to_vec(),
        });
        if self.log.len() > LOG_LIMIT {
            self.log.drain(..self.log.len() - LOG_LIMIT);
        }
    }

    /// How the memories relate: shared concept words, heavy overlaps, possible contradictions.
    pub fn relations(&self) -> Vec<Relation> {
        let mut out = Vec::new();
        for (i, a) in self.entries.iter().enumerate() {
            for b in &self.entries[i + 1..] {
                let shared: Vec<String> = a.terms.iter().filter(|t| b.terms.contains(t)).cloned().collect();
                if shared.is_empty() {
                    continue;
                }
                let j = jaccard(&a.terms, &b.terms);
                let kind = if negated(&a.text) != negated(&b.text) && j >= 0.6 {
                    RelationKind::Contradicts
                } else if j >= 0.8 {
                    RelationKind::Overlaps
                } else {
                    RelationKind::Shares
                };
                out.push(Relation { a: a.id, b: b.id, kind, terms: shared });
            }
        }
        out
    }
}

/// Reading text of a lesson to study: its `[Reading]` units in memory or combined mode.
pub fn reading_to_study(items: &[Item]) -> String {
    items
        .iter()
        .filter(|i| i.reading.is_some_and(ReadingMode::studies))
        .flat_map(|i| i.segments.iter().map(|s| s.text.trim().to_string()))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Questions and answers held out for grading in a lesson (never studied into memory).
pub fn held_out_texts(items: &[Item]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.held_out)
        .flat_map(|i| [i.prompt.clone(), i.answer.clone()])
        .flatten()
        .chain(
            items.iter().filter(|i| i.held_out && i.kind == Kind::Conversation).flat_map(|i| i.users.clone()),
        )
        .collect()
}

/// A copy of a question item with the memories retrieved for its question in front of it
/// (context, never a training target), if any are retrieved and it fits the context.
pub fn with_context(
    model: &Bicameral,
    store: &MemoryStore,
    item: &Item,
    k: usize,
) -> Option<(Item, Vec<u64>)> {
    let q = item.prompt.as_ref()?;
    let ids: Vec<u64> = store.retrieve(q, k).into_iter().map(|(id, _)| id).collect();
    if ids.is_empty() {
        return None;
    }
    let ctx = store.context(&ids, model.config.max_tokens / 2);
    if ctx.is_empty() {
        return None;
    }
    let total: String =
        std::iter::once(ctx.as_str()).chain(item.segments.iter().map(|s| s.text.as_str())).collect();
    if model.tokenizer.encode(&total).ok()?.len() > model.config.max_tokens {
        return None;
    }
    let mut it = item.clone();
    it.segments.insert(0, Segment { text: ctx.clone(), weight: 0. });
    it.context = Some(ctx);
    Some((it, ids))
}

/// Lesson items for training with Study Mode: the originals plus, for each taught question with
/// related memories, a copy with them in front, so the network learns to use (or ignore) them.
/// Held-out items are unchanged: grading stays without memory.
pub fn training_items(
    model: &Bicameral,
    store: &mut MemoryStore,
    items: &[Item],
    now: u64,
) -> (Vec<Item>, usize) {
    let mut out = items.to_vec();
    let mut added = 0;
    for it in items.iter().filter(|i| i.trainable() && i.prompt.is_some()) {
        if let Some((copy, ids)) = with_context(model, store, it, RETRIEVE) {
            store.record("training", it.prompt.as_deref().unwrap_or(""), &ids, now);
            out.push(copy);
            added += 1;
        }
    }
    (out, added)
}

/// The lesson's held-out questions with retrieved memories in front (for the "with memory"
/// score; never trained).
pub fn evaluation_items(model: &Bicameral, store: &mut MemoryStore, items: &[Item], now: u64) -> Vec<Item> {
    items
        .iter()
        .filter(|i| i.held_out && i.prompt.is_some())
        .map(|it| match with_context(model, store, it, RETRIEVE) {
            Some((copy, ids)) => {
                store.record("evaluation", it.prompt.as_deref().unwrap_or(""), &ids, now);
                copy
            }
            None => it.clone(),
        })
        .collect()
}

/// Study Mode during Classroom and course teaching: studies each lesson's memory-mode reading
/// before it is taught, adds memory copies of its questions to training, and measures its
/// held-out questions with memory after teaching.
pub struct Studying {
    pub store: MemoryStore,
    pub now: u64,
    studied: BTreeSet<String>,
}

impl Studying {
    pub fn new(store: MemoryStore, now: u64) -> Self {
        Studying { store, now, studied: BTreeSet::new() }
    }
}

/// Hooks into lesson teaching (see [`Studying`]).
pub trait StudyHook {
    /// Before a lesson is taught: the items to teach, and what was studied (first time only).
    fn before_lesson(
        &mut self,
        model: &Bicameral,
        name: &str,
        items: &[Item],
    ) -> Result<(Vec<Item>, Option<StudyReport>, usize), String>;
    /// After: held-out questions answered right with memory, of how many.
    fn after_lesson(&mut self, model: &Bicameral, items: &[Item]) -> Option<(usize, usize)>;
}

impl StudyHook for Studying {
    fn before_lesson(
        &mut self,
        model: &Bicameral,
        name: &str,
        items: &[Item],
    ) -> Result<(Vec<Item>, Option<StudyReport>, usize), String> {
        let mut report = None;
        if self.studied.insert(name.to_string()) {
            let text = reading_to_study(items);
            if !text.trim().is_empty() {
                let sha = crate::fingerprint::sha256_hex(text.as_bytes());
                report =
                    Some(self.store.study(model, &text, name, &sha, &held_out_texts(items), self.now)?);
            }
        }
        let (items, added) = training_items(model, &mut self.store, items, self.now);
        Ok((items, report, added))
    }
    fn after_lesson(&mut self, model: &Bicameral, items: &[Item]) -> Option<(usize, usize)> {
        let eval = evaluation_items(model, &mut self.store, items, self.now);
        let refs: Vec<&Item> = eval.iter().collect();
        let o = crate::learn::eval::evaluate(model, &refs, Ablation::default()).ok()?.overall();
        Some((o.correct, o.total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(texts: &[&str]) -> MemoryStore {
        let mut s = MemoryStore::default();
        for t in texts {
            s.add_teacher(t, 0, 0);
        }
        s
    }

    #[test]
    fn retrieval_follows_shared_concepts() {
        let s = store(&[
            "Dogs are animals.",
            "Animals need water.",
            "Max is a dog.",
            "Cats like milk.",
            "The sky is blue.",
        ]);
        let got: Vec<String> = s
            .retrieve("Does Max need water?", 4)
            .iter()
            .map(|(id, _)| s.get(*id).unwrap().text.clone())
            .collect();
        assert_eq!(got.len(), 3, "{got:?}");
        assert!(got.contains(&"Max is a dog.".to_string()));
        assert!(got.contains(&"Animals need water.".to_string()));
        assert!(got.contains(&"Dogs are animals.".to_string()), "one step through 'dog' / 'animal'");
        assert!(s.retrieve("What colour is grass?", 4).is_empty());
        let ctx = s.context(&s.retrieve("Max", 1).iter().map(|x| x.0).collect::<Vec<_>>(), 100);
        assert_eq!(ctx, "\x03Max is a dog.\n");
    }

    #[test]
    fn relations_and_decisions() {
        let mut s = store(&["Dogs are animals.", "Dogs are not animals.", "Max is a dog."]);
        let rel = s.relations();
        assert!(rel.iter().any(|r| r.kind == RelationKind::Contradicts && r.a == 1 && r.b == 2));
        assert!(rel.iter().any(|r| r.kind == RelationKind::Shares && r.terms == ["dog"]));
        s.decide(2, "disabled", None, 5).unwrap();
        assert!(
            s.retrieve("Are dogs animals?", 4).iter().all(|(id, _)| *id != 2),
            "a disabled memory is kept but not retrieved"
        );
        s.decide(3, "edited", Some("Max is a brown dog."), 6).unwrap();
        assert!(s.get(3).unwrap().edited && s.get(3).unwrap().terms.contains(&"brown".to_string()));
        s.decide(1, "deleted", None, 7).unwrap();
        assert!(s.get(1).is_none());
        assert_eq!(
            s.decisions.iter().map(|d| d.action.as_str()).collect::<Vec<_>>(),
            ["added", "added", "added", "disabled", "edited", "deleted"]
        );
        assert_eq!(terms("The dogs' bowls and boxes"), ["dog", "bowl", "box"]);
    }
}
