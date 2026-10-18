//! Rewards: an automatic grader, reward ledgers and reward-guided training emphasis.
//!
//! **Stage 1 (measurement).** Every answer Nova gives to a lesson question can be graded 0-100%
//! ([`grade_reply`]), or recorded as *uncertain* when no rule can judge it reliably; uncertain
//! answers never earn anything. Graded answers go into the [`RewardLedger`]:
//!
//! * **Progress** (within a question's reward cycle): one-time credit for new milestones only:
//!   a better grade, or (spelling lessons, experimental) a longer correct prefix of the answer.
//!   Answers are compared ignoring case and punctuation, so "HALLWAY!!!" after "Hallway" earns
//!   nothing new. Reworded copies of a question share one cycle (same answer and the same
//!   concept words), each cycle has a capped budget, and partial credit stops after a number of
//!   attempts. A completed cycle never reopens by itself.
//! * **Retention** (across checkpoints): a question answered right at a later checkpoint, after
//!   more training, earns a shrinking bonus (by default 0.6, then 0.2, 0.1, ...).
//! * **Generalization**: a question Nova was never trained on (a lesson's held-out questions,
//!   Review Training's evaluation questions) answered right earns the largest credit.
//! * **Relevance** (diagnostic only): words of the question's own source text that Nova uses in
//!   a wrong answer. Recorded to show what she recognises; never used by training.
//!
//! Answers given with Study Mode memories in front are recorded separately from answers by the
//! network alone, with a smaller budget. Correctness is always kept apart from credit: a correct
//! answer whose progress is already used up is still correct. Every decision is written to the
//! audit log with its reason, including credit withheld and why.
//!
//! **Protected examinations** (a course's final exam, the Exams page) never enter the ledger:
//! their questions are refused by [`Rewards::protect`], and nothing in this module reads them.
//!
//! **Stage 2 (reward-guided supervised training).** When enabled, a teaching session quizzes
//! Nova on the lesson's *training* units, and the ledger turns her grades and history into an
//! emphasis for each unit ([`RewardLedger::emphasis`]): knowledge gaps, repeated failures,
//! forgotten answers, weak concepts and difficult answers get more practice; correct and
//! retained answers get less, never none. The emphasis sets how many times each unit is trained
//! in an epoch, within the same total number of units as ordinary training. Targets never change:
//! they are the lesson's verified answers and the teacher's corrections, never Nova's own wrong
//! answers. Held-out and unseen questions never influence emphasis, so they stay independent.
use super::eval::{evaluate, Report};
use super::grading::{normalize, words, AnswerKey, GradeMode};
use super::lessons::{canonical, Item, Kind};
use super::Rng;
use crate::memory::{terms, MEMORY};
use crate::model::{Ablation, Bicameral};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Highest grade a partly right answer can get from answer-key rules.
pub const PARTIAL_MAX: f32 = 60.;
/// Audit entries kept (oldest dropped first).
pub const AUDIT_LIMIT: usize = 20_000;
/// Checkpoint results kept per question.
const CHECKPOINT_LIMIT: usize = 64;
/// Distinct answers remembered per reward cycle.
const ANSWER_LIMIT: usize = 200;
/// Most training copies of one unit in an epoch.
pub const MAX_COPIES: usize = 4;

const NEGATIONS: [&str; 6] = ["not", "no", "never", "none", "nothing", "nor"];

/// How much each piece of history adds to (or takes from) a unit's training emphasis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EmphasisWeights {
    /// Added at a 0% grade (scaled down as the grade rises).
    pub gap: f32,
    /// Added for repeated failures (full at four in a row).
    pub failures: f32,
    /// Added when an answer given at an earlier checkpoint was lost.
    pub forgotten: f32,
    /// Added for partial progress that is not complete yet.
    pub progress: f32,
    /// Added for a weak concept (other questions of the same pattern answered wrong).
    pub concept: f32,
    /// Added when new variations of the concept are answered poorly at first sight.
    pub variation: f32,
    /// Added for long answers (difficulty), when not yet right.
    pub difficulty: f32,
    /// A correct answer's emphasis is multiplied by this.
    pub correct_factor: f32,
    /// And by this once per checkpoint it was retained (up to four).
    pub retained_factor: f32,
    /// After this many failures in a row without any progress, a unit is "stuck"...
    pub stuck_after: u32,
    /// ...and its emphasis is capped here (it needs the teacher, not endless repetition).
    pub stuck_cap: f32,
}
impl Default for EmphasisWeights {
    fn default() -> Self {
        EmphasisWeights {
            gap: 1.0,
            failures: 0.5,
            forgotten: 0.75,
            progress: 0.25,
            concept: 0.5,
            variation: 0.5,
            difficulty: 0.25,
            correct_factor: 0.6,
            retained_factor: 0.6,
            stuck_after: 8,
            stuck_cap: 1.5,
        }
    }
}

/// Reward settings. Every value is a configurable default, not a fixed rule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardConfig {
    /// Stage 1: grade answers and keep the ledgers.
    pub record: bool,
    /// Stage 2: training emphasis from the ledger (needs `record`).
    pub weighted_training: bool,
    /// Experimental: letter-prefix progress, only in lessons marked `Grading: spelling`.
    pub letter_prefix: bool,
    /// Progress budget of one reward cycle (network alone).
    pub progress_budget: f32,
    /// Spelling: credit for the first correct letter...
    pub first_letter: f32,
    /// ...and for each further correct letter of the prefix.
    pub per_letter: f32,
    /// Partial milestones together earn at most this share of the budget; the rest needs the
    /// complete answer.
    pub partial_cap: f32,
    /// Partial credit stops after this many attempts in a cycle (the complete answer still
    /// earns the remaining budget).
    pub discovery_attempts: u32,
    /// Relevance (diagnostic): per recognised word, and the cap per cycle.
    pub relevance_word: f32,
    pub relevance_cap: f32,
    /// Recorded for a certainly wrong (0%) answer, once per distinct answer in a cycle. 0 by
    /// default; a negative value records a small penalty. Never applied to uncertain grades.
    pub incorrect_reward: f32,
    /// First correct answer of a trained question at a checkpoint.
    pub retention_first: f32,
    /// Second correct checkpoint; it halves each time after that...
    pub retention_repeat: f32,
    /// ...down to this.
    pub retention_floor: f32,
    /// First correct answer to a question Nova was never trained on.
    pub generalization: f32,
    /// Answers with Study Mode memories in front earn this share of the network-alone credit.
    pub memory_share: f32,
    /// Training units Nova is quizzed on at a session start (and, with Stage 2, each epoch).
    pub quiz_limit: usize,
    pub weights: EmphasisWeights,
    /// Emphasis limits: the minimum keeps every unit in training.
    pub min_emphasis: f32,
    pub max_emphasis: f32,
}
impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            record: true,
            weighted_training: false,
            letter_prefix: false,
            progress_budget: 1.0,
            first_letter: 0.10,
            per_letter: 0.09,
            partial_cap: 0.5,
            discovery_attempts: 20,
            relevance_word: 0.01,
            relevance_cap: 0.05,
            incorrect_reward: 0.,
            retention_first: 0.6,
            retention_repeat: 0.2,
            retention_floor: 0.02,
            generalization: 1.0,
            memory_share: 0.5,
            quiz_limit: 64,
            weights: EmphasisWeights::default(),
            min_emphasis: 0.25,
            max_emphasis: 3.0,
        }
    }
}

// ----------------------------------------------------------------------------------------
// Grader
// ----------------------------------------------------------------------------------------

/// How a grade was decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    /// The lesson's answer key (accepted forms, keywords, rules).
    AnswerKey,
    /// Exactly the expected answer.
    Exact,
    /// The expected answer apart from case, punctuation and spacing.
    Normalized,
    /// Partly meets the answer key's required words.
    KeyRules,
    /// Contains a short expected answer with other words.
    ContainsAnswer,
    /// Spelling lesson: correct letters of the answer's prefix.
    Spelling,
    /// Conversation: share of turns right.
    Turns,
    /// The teacher's grade.
    Teacher,
    /// Certainly wrong (no answer, the contradicted answer, the opposite meaning, nothing in
    /// common with the answer).
    Mismatch,
    /// No rule can judge it: recorded, never credited.
    Uncertain,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grade {
    /// 0-100, or None when uncertain.
    pub percent: Option<f32>,
    pub correct: bool,
    pub method: Method,
    pub reason: String,
}
impl Grade {
    fn certain(percent: f32, correct: bool, method: Method, reason: impl Into<String>) -> Grade {
        Grade { percent: Some(percent.clamp(0., 100.)), correct, method, reason: reason.into() }
    }
    pub fn uncertain(reason: impl Into<String>) -> Grade {
        Grade { percent: None, correct: false, method: Method::Uncertain, reason: reason.into() }
    }
    /// The teacher's grade (0-100) with their note.
    pub fn teacher(percent: f32, note: &str) -> Grade {
        let p = percent.clamp(0., 100.);
        let reason =
            if note.trim().is_empty() { "teacher's grade".to_string() } else { note.trim().to_string() };
        Grade::certain(p, p >= 100., Method::Teacher, reason)
    }
    /// Conversation turns: `correct` of `total` right.
    pub fn turns(correct: usize, total: usize) -> Grade {
        if total == 0 {
            return Grade::uncertain("no turns to grade");
        }
        Grade::certain(
            100. * correct as f32 / total as f32,
            correct == total,
            Method::Turns,
            format!("{correct} of {total} turns right"),
        )
    }
}

fn negative(w: &[String]) -> bool {
    w.iter().any(|x| NEGATIONS.contains(&x.as_str()))
}

/// Characters two normalised strings share at the start.
fn common_prefix(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// Content words of a text: its terms, plus numbers (which `terms` leaves out when short).
fn content(text: &str) -> Vec<String> {
    let mut t = terms(text);
    for w in words(text) {
        if w.chars().all(|c| c.is_ascii_digit()) && !t.contains(&w) {
            t.push(w);
        }
    }
    t
}

/// A reply compared with accepted sentences when no key rule decides: certainly wrong when it
/// says the opposite or has nothing in common, partly right when it contains a short answer,
/// otherwise uncertain.
fn compare_sentences(accepted: &[String], reply: &str) -> Grade {
    let rw = words(reply);
    let rt = content(reply);
    let mut best = 0f32;
    let mut contains_short = false;
    for a in accepted {
        let at = content(a);
        if at.is_empty() {
            continue;
        }
        let shared = at.iter().filter(|t| rt.contains(t)).count();
        if negative(&words(a)) != negative(&rw) && shared * 2 >= at.len() {
            return Grade::certain(0., false, Method::Mismatch, "says the opposite of the expected answer");
        }
        let share = shared as f32 / at.len() as f32;
        best = best.max(share);
        if at.len() <= 3 && shared == at.len() && rw.len() <= words(a).len() + 4 {
            contains_short = true;
        }
    }
    if best == 0. {
        return Grade::certain(
            0.,
            false,
            Method::Mismatch,
            "shares no content word with the expected answer",
        );
    }
    if contains_short {
        return Grade::certain(
            75.,
            false,
            Method::ContainsAnswer,
            "contains the expected answer with other words",
        );
    }
    Grade::uncertain("partly matches the expected answer; an answer key or the teacher must judge it")
}

/// Grades one reply 0-100% (or uncertain). `spelling` turns on letter-prefix grading (only for
/// lessons marked as spelling, with the experiment enabled). Reasoning and sentence answers are
/// judged by meaning rules (answer keys, negation, required words), never by shared letters.
pub fn grade_reply(
    expected: &str,
    key: Option<&AnswerKey>,
    contradicted: Option<&str>,
    reply: &str,
    spelling: bool,
) -> Grade {
    let r = normalize(reply);
    if r.is_empty() {
        return Grade::certain(0., false, Method::Mismatch, "no answer");
    }
    if let Some(bad) = contradicted {
        if normalize(bad) == r {
            return Grade::certain(0., false, Method::Mismatch, "gave the contradicted answer");
        }
    }
    if let Some(k) = key {
        if k.is_correct(reply) {
            return Grade::certain(100., true, Method::AnswerKey, "accepted by the answer key");
        }
        if spelling {
            return spelling_grade(k.target(), &r);
        }
        let w = words(reply);
        return match k.mode {
            GradeMode::Keyword => Grade::certain(
                0.,
                false,
                Method::AnswerKey,
                "not accepted by the answer key (no accepted keyword, or a conflicting one)",
            ),
            GradeMode::Sentence => {
                if let Some(n) = k.never.iter().find(|n| w.contains(n)) {
                    return Grade::certain(
                        0.,
                        false,
                        Method::KeyRules,
                        format!("contains '{n}', which the key forbids"),
                    );
                }
                if !k.must.is_empty() {
                    let have = k.must.iter().filter(|m| w.contains(m)).count();
                    return Grade::certain(
                        PARTIAL_MAX * have as f32 / k.must.len() as f32,
                        false,
                        Method::KeyRules,
                        format!("{have} of {} required words", k.must.len()),
                    );
                }
                compare_sentences(&k.accepted, reply)
            }
        };
    }
    if reply.trim() == expected.trim() {
        return Grade::certain(100., true, Method::Exact, "exactly the expected answer");
    }
    if r == normalize(expected) {
        return Grade::certain(
            100.,
            true,
            Method::Normalized,
            "the expected answer apart from case and punctuation",
        );
    }
    if spelling {
        return spelling_grade(expected, &r);
    }
    compare_sentences(&[expected.to_string()], reply)
}

fn spelling_grade(expected: &str, r: &str) -> Grade {
    let e = normalize(expected);
    let n = e.chars().count().max(1);
    let l = common_prefix(r, &e);
    Grade::certain(
        100. * l as f32 / n as f32,
        false,
        Method::Spelling,
        format!("first {l} of {n} letters right"),
    )
}

/// Words of a wrong answer, sorted by where they come from (diagnostic only).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Relevance {
    /// In the question's own source text (recognised), not the answer.
    pub recognized: Vec<String>,
    /// In neither the source nor the answer.
    pub invented: Vec<String>,
}
pub fn relevance(reply: &str, source: &str, expected: &str) -> Relevance {
    let src = terms(source);
    let exp = terms(expected);
    let mut r = Relevance::default();
    for t in terms(reply) {
        if exp.contains(&t) {
            continue;
        }
        if src.contains(&t) {
            r.recognized.push(t);
        } else {
            r.invented.push(t);
        }
    }
    r
}

// ----------------------------------------------------------------------------------------
// Identities
// ----------------------------------------------------------------------------------------

/// The question of an item (conversations: the user turns).
pub fn question_text(it: &Item) -> String {
    match it.kind {
        Kind::Conversation => it.users.join(" / "),
        _ => it.prompt.clone().unwrap_or_default(),
    }
}
/// The answer an item is trained on (conversations: Nova's turns).
pub fn expected_text(it: &Item) -> String {
    match it.kind {
        Kind::Conversation => it.replies.join(" / "),
        _ => it.answer.clone().unwrap_or_default(),
    }
}
fn short_hash(s: &str) -> String {
    crate::fingerprint::sha256_hex(s.as_bytes())[..16].to_string()
}
/// A question's identity: its lesson and its canonical text.
pub fn question_id(it: &Item) -> String {
    short_hash(&format!("{}\n{}", it.source, canonical(&question_text(it))))
}
/// The reward cycle: the expected answer and the question's concept words, so a reworded copy
/// (in any lesson) shares the cycle and cannot earn the same milestones again.
pub fn cycle_key(it: &Item) -> String {
    // "Where's" and "where is" are the same question.
    let mut t = terms(&question_text(it).replace("'s", " is"));
    t.sort();
    short_hash(&format!("{}|{}", normalize(&expected_text(it)), t.join(" ")))
}
/// The question's pattern: its last sentence with names and numbers replaced by `_`, e.g.
/// "dialogue: where is _" for "Where is Daniel?". Questions sharing a pattern are one concept.
pub fn template(it: &Item) -> String {
    let q = question_text(it);
    let last = q.split(['.', '?', '!']).map(str::trim).rfind(|s| !s.is_empty()).unwrap_or("");
    let mut out = Vec::new();
    for (i, tok) in last.split_whitespace().enumerate() {
        let w: String = tok.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '\'').collect();
        if w.is_empty() {
            continue;
        }
        let slot = w.chars().all(|c| c.is_ascii_digit())
            || (i > 0 && w.starts_with(|c: char| c.is_ascii_uppercase()));
        out.push(if slot { "_".to_string() } else { w.to_ascii_lowercase() });
    }
    format!("{}: {}", it.kind.name(), out.join(" "))
}
/// The text a question was asked about: the question and any non-memory context.
pub fn source_text(it: &Item) -> String {
    let ctx: Vec<&str> =
        it.context.as_deref().unwrap_or("").lines().filter(|l| !l.starts_with(MEMORY)).collect();
    format!("{} {}", ctx.join(" "), question_text(it))
}
/// True when Study Mode memories were in front of the question.
pub fn with_memory(it: &Item) -> bool {
    it.context.as_deref().is_some_and(|c| c.contains(MEMORY))
}

// ----------------------------------------------------------------------------------------
// Ledger
// ----------------------------------------------------------------------------------------

/// Where an answer came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// A taught unit, asked at a saved checkpoint (session start, after an attempt).
    Training,
    /// A taught unit, asked during training (between epochs; not a checkpoint).
    Session,
    /// A lesson's held-out question (never trained).
    HeldOut,
    /// A Review Training evaluation question (never trained).
    Review,
    /// Interactive practice, graded by the teacher.
    Practice,
}
impl Origin {
    /// Questions Nova was never trained on: generalization, not progress.
    pub fn unseen(self) -> bool {
        matches!(self, Origin::HeldOut | Origin::Review)
    }
}

/// Milestones of one track (network alone, or with memories) in a cycle.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Track {
    /// Highest milestone value credited so far.
    pub best: f32,
    /// The milestones credited, in order ("grade 60%", "prefix 3/7", "complete").
    pub milestones: Vec<String>,
    pub complete: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cycle {
    pub neural: Track,
    pub memory: Track,
    /// Recognised words already credited (diagnostic relevance).
    pub rewarded_words: BTreeSet<String>,
    pub relevance: f32,
    /// Normalised answers already given (case and punctuation ignored).
    pub answers: BTreeSet<String>,
    /// Attempts in the cycle (partial credit stops after the limit).
    pub attempts: u32,
    /// Wrong answers that already carried the incorrect-answer value.
    pub penalized: BTreeSet<String>,
}

/// One question at one checkpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckpointResult {
    pub model: String,
    pub step: u64,
    pub correct: bool,
    pub with_memory: bool,
    pub bonus: f32,
    /// first correct, generalized, retained, relearned, lost, unchanged checkpoint, ...
    pub event: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuestionRecord {
    pub id: String,
    pub lesson: String,
    pub question: String,
    pub expected: String,
    pub kind: Kind,
    pub template: String,
    pub cycle: String,
    /// Asked as a never-trained question (held-out, review evaluation).
    #[serde(default)]
    pub unseen: bool,
    /// Asked as a taught unit.
    #[serde(default)]
    pub trained: bool,
    /// Grade the first time it was asked as a taught unit (a new variation, before practice).
    #[serde(default)]
    pub first_percent: Option<f32>,
    /// Latest grade, network alone.
    #[serde(default)]
    pub last_percent: Option<f32>,
    /// Latest grade with memories in front.
    #[serde(default)]
    pub last_memory_percent: Option<f32>,
    /// Certain wrong answers in a row (network alone).
    #[serde(default)]
    pub failures: u32,
    #[serde(default)]
    pub attempts: u32,
    /// Ledger sequence of the latest answer (quiz rotation).
    #[serde(default)]
    pub last_seq: u64,
    #[serde(default)]
    pub checkpoints: Vec<CheckpointResult>,
    /// Retention bonuses given after the first.
    #[serde(default)]
    pub repeats: u32,
    #[serde(default)]
    pub retention: f32,
    #[serde(default)]
    pub generalization: f32,
    /// Lost since an earlier checkpoint and not answered right since.
    #[serde(default)]
    pub forgotten: bool,
    /// Teacher notes.
    #[serde(default)]
    pub notes: Vec<String>,
}
impl QuestionRecord {
    /// Checkpoints in a row, at the end, where the network alone answered right.
    pub fn retained_streak(&self) -> usize {
        self.checkpoints.iter().rev().filter(|c| !c.with_memory).take_while(|c| c.correct).count()
    }
}

/// One decision of the ledger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    pub seq: u64,
    /// Unix seconds.
    pub time: u64,
    pub model: String,
    pub step: u64,
    pub lesson: String,
    pub question_id: String,
    pub question: String,
    pub expected: String,
    pub origin: Origin,
    pub with_memory: bool,
    pub checkpoint: bool,
    pub reply: String,
    pub normalized: String,
    pub grade: Grade,
    /// New progress credit (0 when nothing new).
    pub progress: f32,
    pub milestone: Option<String>,
    /// Milestones of this track already credited before this answer.
    pub previous: Vec<String>,
    pub retention: f32,
    pub generalization: f32,
    /// Diagnostic only, never used by training.
    pub relevance: f32,
    pub recognized: Vec<String>,
    pub invented: Vec<String>,
    /// The incorrect-answer value, when applied.
    pub accuracy: f32,
    /// Credit withheld, and why.
    pub withheld: Vec<String>,
    /// The checkpoint event (first correct, retained, lost, ...).
    pub event: Option<String>,
    pub note: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Totals {
    pub progress: f32,
    pub retention: f32,
    pub generalization: f32,
    /// Diagnostic only.
    pub relevance: f32,
    pub accuracy: f32,
    pub answers: usize,
    pub uncertain: usize,
}

/// One teaching session as the ledger saw it (for the hacking checks and the history view).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionSummary {
    pub seq: u64,
    pub time: u64,
    pub model: String,
    pub lesson: String,
    pub step_start: u64,
    pub step_end: u64,
    pub learned: bool,
    pub weighted: bool,
    /// Credit earned during the session.
    pub progress: f32,
    pub retention: f32,
    pub generalization: f32,
    pub relevance: f32,
    /// Taught units asked at the start: share right (network alone).
    pub quiz_rate: Option<f32>,
    /// Held-out questions right, at the start and the end.
    pub heldout_before: f32,
    pub heldout_after: f32,
    pub heldout_loss_before: f32,
    pub heldout_loss_after: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RewardLedger {
    pub questions: BTreeMap<String, QuestionRecord>,
    pub cycles: BTreeMap<String, Cycle>,
    pub sessions: Vec<SessionSummary>,
    pub audit: VecDeque<AuditEntry>,
    pub totals: Totals,
    pub seq: u64,
    /// Answers refused because their question is in a protected examination (counted only).
    #[serde(default)]
    pub protected_refused: usize,
}

/// What Nova answered.
#[derive(Clone, Debug)]
pub enum Answer<'a> {
    Reply(&'a str),
    /// A conversation: turns right of total, and the replies for the log.
    Turns {
        correct: usize,
        total: usize,
        replies: String,
    },
}

/// One answer to record.
#[derive(Clone, Debug)]
pub struct Evidence<'a> {
    pub item: &'a Item,
    pub answer: Answer<'a>,
    pub origin: Origin,
    pub step: u64,
    /// The model is a saved checkpoint (counts for retention and generalization).
    pub checkpoint: bool,
    /// The teacher's grade (0-100) and note: replaces the automatic grade.
    pub teacher: Option<(f32, &'a str)>,
}

/// The ledger with its settings, the model it belongs to and the protected questions.
#[derive(Clone, Debug, Default)]
pub struct Rewards {
    pub config: RewardConfig,
    pub ledger: RewardLedger,
    /// The model's identity (checkpoints are recorded per model).
    pub model: String,
    /// Canonical protected-exam questions: refused, never recorded.
    pub protected: BTreeSet<String>,
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Keys a protected question is refused by.
fn protected_keys(it: &Item) -> Vec<String> {
    let mut k = vec![canonical(&question_text(it))];
    if let Some(p) = &it.prompt {
        k.push(canonical(p));
    }
    k
}

impl Rewards {
    pub fn new(config: RewardConfig, ledger: RewardLedger, model: &str) -> Self {
        Rewards { config, ledger, model: model.to_string(), protected: BTreeSet::new() }
    }

    /// Marks examination questions as protected: answers to them are refused and nothing about
    /// them is recorded or credited.
    pub fn protect<'a>(&mut self, items: impl IntoIterator<Item = &'a Item>) {
        for it in items {
            self.protected.extend(protected_keys(it));
        }
    }
    pub fn is_protected(&self, it: &Item) -> bool {
        protected_keys(it).iter().any(|k| self.protected.contains(k))
    }

    /// Stage 2 is on.
    pub fn weighted(&self) -> bool {
        self.config.record && self.config.weighted_training
    }

    /// Grades and records one answer. Returns the audit entry (None when refused as protected or
    /// recording is off).
    pub fn record(&mut self, ev: Evidence) -> Option<AuditEntry> {
        if !self.config.record {
            return None;
        }
        let it = ev.item;
        if self.is_protected(it) {
            self.ledger.protected_refused += 1;
            return None;
        }
        let cfg = &self.config;
        let l = &mut self.ledger;
        let mem = with_memory(it);
        let expected = expected_text(it);
        let spelling = cfg.letter_prefix && it.spelling;
        let (reply, grade) = match (&ev.answer, ev.teacher) {
            (Answer::Reply(r), Some((p, note))) => (r.to_string(), Grade::teacher(p, note)),
            (Answer::Turns { replies, .. }, Some((p, note))) => (replies.clone(), Grade::teacher(p, note)),
            (Answer::Reply(r), None) => (
                r.to_string(),
                grade_reply(&expected, it.key.as_ref(), it.contradicted.as_deref(), r, spelling),
            ),
            (Answer::Turns { correct, total, replies }, None) => {
                (replies.clone(), Grade::turns(*correct, *total))
            }
        };
        let normalized = normalize(&reply);
        let id = question_id(it);
        let ckey = cycle_key(it);
        l.seq += 1;
        let seq = l.seq;
        let q = l.questions.entry(id.clone()).or_insert_with(|| QuestionRecord {
            id: id.clone(),
            lesson: it.source.clone(),
            question: question_text(it),
            expected: expected.clone(),
            kind: it.kind,
            template: template(it),
            cycle: ckey.clone(),
            unseen: false,
            trained: false,
            first_percent: None,
            last_percent: None,
            last_memory_percent: None,
            failures: 0,
            attempts: 0,
            last_seq: 0,
            checkpoints: Vec::new(),
            repeats: 0,
            retention: 0.,
            generalization: 0.,
            forgotten: false,
            notes: Vec::new(),
        });
        let c = l.cycles.entry(ckey).or_default();
        q.attempts += 1;
        q.last_seq = seq;
        if ev.origin.unseen() {
            q.unseen = true;
        } else {
            q.trained = true;
        }
        let mut withheld = Vec::new();
        let repeat = !normalized.is_empty() && c.answers.contains(&normalized);

        // Progress: taught questions only, one-time milestones per track.
        let (mut progress, mut milestone, mut previous) = (0f32, None, Vec::new());
        let credited = ev.checkpoint || ev.origin == Origin::Practice;
        if grade.percent.is_none() {
            withheld.push(format!("uncertain grade: {}", grade.reason));
        } else if ev.origin.unseen() {
            withheld.push("never-trained question: counts for generalization, not progress".into());
        } else if !credited {
            withheld.push(
                "asked during training (not a saved checkpoint): guides practice, earns no credit".into(),
            );
        } else {
            c.attempts += 1;
            let budget = cfg.progress_budget * if mem { cfg.memory_share } else { 1. };
            let track = if mem { &mut c.memory } else { &mut c.neural };
            previous = track.milestones.clone();
            let (value, label) = if grade.correct {
                (budget, "complete".to_string())
            } else if grade.method == Method::Spelling {
                let n = normalize(&expected);
                let len = common_prefix(&normalized, &n);
                let v = if len == 0 { 0. } else { cfg.first_letter + cfg.per_letter * (len - 1) as f32 };
                (
                    v.min(cfg.partial_cap * cfg.progress_budget) * budget / cfg.progress_budget.max(1e-6),
                    format!("prefix {len}/{}", n.chars().count()),
                )
            } else {
                let p = grade.percent.unwrap_or(0.);
                ((budget * p / 100.).min(cfg.partial_cap * budget), format!("grade {p:.0}%"))
            };
            let limit = !grade.correct && c.attempts > cfg.discovery_attempts;
            if track.complete {
                withheld.push("cycle already complete: progress used up (still graded for accuracy)".into());
            } else if value <= track.best + 1e-6 {
                withheld.push(if repeat {
                    "same answer as before (case and punctuation ignored): no new milestone".to_string()
                } else {
                    "no new milestone beyond those already credited".to_string()
                });
            } else if limit {
                withheld.push(format!("partial-credit attempt limit ({}) reached", cfg.discovery_attempts));
            } else {
                progress = value - track.best;
                track.best = value;
                track.complete = grade.correct;
                track.milestones.push(label.clone());
                milestone = Some(label);
            }
        }

        // Relevance (diagnostic): recognised source words in a wrong answer, once per cycle.
        let rel =
            if grade.correct { Relevance::default() } else { relevance(&reply, &source_text(it), &expected) };
        let mut relevance_credit = 0f32;
        for w in rel.recognized.iter().filter(|_| credited) {
            if c.relevance + relevance_credit + cfg.relevance_word > cfg.relevance_cap + 1e-6 {
                break;
            }
            if c.rewarded_words.insert(w.clone()) {
                relevance_credit += cfg.relevance_word;
            }
        }
        c.relevance += relevance_credit;

        // The incorrect-answer value: certain 0% answers, once per distinct answer.
        let mut accuracy = 0f32;
        if credited
            && cfg.incorrect_reward != 0.
            && grade.percent == Some(0.)
            && c.penalized.insert(normalized.clone())
        {
            accuracy = cfg.incorrect_reward;
        }
        if !normalized.is_empty() && c.answers.len() < ANSWER_LIMIT {
            c.answers.insert(normalized.clone());
        }

        // Retention and generalization: saved checkpoints only, once per checkpoint.
        let (mut retention, mut generalization, mut event) = (0f32, 0f32, None);
        if ev.checkpoint && grade.percent.is_some() {
            let same = q
                .checkpoints
                .iter()
                .any(|x| x.model == self.model && x.step == ev.step && x.with_memory == mem);
            if same {
                withheld.push("already counted at this checkpoint".into());
            } else {
                let last = q.checkpoints.iter().rfind(|x| x.with_memory == mem).cloned();
                let ever = q.checkpoints.iter().any(|x| x.with_memory == mem && x.correct);
                let share = if mem { cfg.memory_share } else { 1. };
                let mut bonus = 0f32;
                let e = if grade.correct {
                    if !ever {
                        if ev.origin.unseen() {
                            bonus = cfg.generalization * share;
                            generalization = bonus;
                            "generalized: first correct answer to a never-trained question"
                        } else {
                            bonus = cfg.retention_first * share;
                            retention = bonus;
                            "first correct answer at a checkpoint"
                        }
                    } else if last.as_ref().is_some_and(|x| x.step < ev.step) {
                        q.repeats += 1;
                        let b = (cfg.retention_repeat * 0.5f32.powi(q.repeats as i32 - 1))
                            .max(cfg.retention_floor);
                        bonus = b * share;
                        retention = bonus;
                        if last.as_ref().is_some_and(|x| !x.correct) {
                            "relearned after being lost"
                        } else {
                            "retained at a later checkpoint"
                        }
                    } else {
                        withheld.push("no training since the last checkpoint: no retention bonus".into());
                        "unchanged checkpoint"
                    }
                } else if last.as_ref().is_some_and(|x| x.correct) {
                    if !mem {
                        q.forgotten = true;
                    }
                    "lost: answered right at the previous checkpoint"
                } else {
                    "not yet right"
                };
                q.retention += retention;
                q.generalization += generalization;
                q.checkpoints.push(CheckpointResult {
                    model: self.model.clone(),
                    step: ev.step,
                    correct: grade.correct,
                    with_memory: mem,
                    bonus,
                    event: e.to_string(),
                });
                if q.checkpoints.len() > CHECKPOINT_LIMIT {
                    q.checkpoints.remove(0);
                }
                event = Some(e.to_string());
            }
        }

        // History used by training emphasis (taught questions; never unseen ones).
        if let Some(p) = grade.percent {
            if mem {
                q.last_memory_percent = Some(p);
            } else {
                q.last_percent = Some(p);
                if !ev.origin.unseen() && q.first_percent.is_none() {
                    q.first_percent = Some(p);
                }
                if grade.correct {
                    q.failures = 0;
                    q.forgotten = false;
                } else if p == 0. {
                    q.failures += 1;
                }
            }
        }
        let note = ev.teacher.map(|(_, n)| n.trim().to_string()).filter(|n| !n.is_empty());
        if let Some(n) = &note {
            if !q.notes.contains(n) {
                q.notes.push(n.clone());
                if q.notes.len() > 20 {
                    q.notes.remove(0);
                }
            }
        }

        let t = &mut l.totals;
        t.progress += progress;
        t.retention += retention;
        t.generalization += generalization;
        t.relevance += relevance_credit;
        t.accuracy += accuracy;
        t.answers += 1;
        t.uncertain += usize::from(grade.percent.is_none());
        let entry = AuditEntry {
            seq,
            time: now(),
            model: self.model.clone(),
            step: ev.step,
            lesson: it.source.clone(),
            question_id: id,
            question: q.question.clone(),
            expected,
            origin: ev.origin,
            with_memory: mem,
            checkpoint: ev.checkpoint,
            reply,
            normalized,
            grade,
            progress,
            milestone,
            previous,
            retention,
            generalization,
            relevance: relevance_credit,
            recognized: rel.recognized,
            invented: rel.invented,
            accuracy,
            withheld,
            event,
            note,
        };
        l.audit.push_back(entry.clone());
        while l.audit.len() > AUDIT_LIMIT {
            l.audit.pop_front();
        }
        Some(entry)
    }

    /// Records evaluated outcomes of `items` (the report of `evaluate` on exactly these items).
    pub fn record_report(
        &mut self,
        items: &[&Item],
        report: &Report,
        origin: Origin,
        step: u64,
        checkpoint: bool,
    ) {
        let mut o = report.outcomes.iter();
        for it in items {
            let n = it.outcome_count();
            let outs: Vec<_> = (0..n).filter_map(|_| o.next()).collect();
            if outs.is_empty() {
                continue;
            }
            let answer = if it.kind == Kind::Conversation {
                Answer::Turns {
                    correct: outs.iter().filter(|x| x.correct).count(),
                    total: outs.len(),
                    replies: outs.iter().map(|x| x.reply.as_str()).collect::<Vec<_>>().join(" / "),
                }
            } else {
                Answer::Reply(&outs[0].reply)
            };
            self.record(Evidence { item: it, answer, origin, step, checkpoint, teacher: None });
        }
    }

    /// Asks Nova up to `quiz_limit` of the given taught units (those not asked for longest
    /// first) and records her answers. Returns (asked, right).
    pub fn quiz(
        &mut self,
        model: &Bicameral,
        items: &[&Item],
        origin: Origin,
        checkpoint: bool,
    ) -> Result<(usize, usize), String> {
        if !self.config.record || self.config.quiz_limit == 0 {
            return Ok((0, 0));
        }
        let mut gradeable: Vec<&Item> =
            items.iter().copied().filter(|it| it.outcome_count() > 0 && !self.is_protected(it)).collect();
        let mut seen = BTreeSet::new();
        gradeable.retain(|it| seen.insert((question_id(it), with_memory(it))));
        let last = |it: &Item| self.ledger.questions.get(&question_id(it)).map_or(0, |q| q.last_seq);
        gradeable.sort_by_key(|it| last(it));
        gradeable.truncate(self.config.quiz_limit);
        if gradeable.is_empty() {
            return Ok((0, 0));
        }
        let report = evaluate(model, &gradeable, Ablation::default())?;
        let before = self.ledger.seq;
        self.record_report(&gradeable, &report, origin, model.step, checkpoint);
        let right = self
            .ledger
            .audit
            .iter()
            .rev()
            .take_while(|e| e.seq > before)
            .filter(|e| e.grade.correct && !e.with_memory)
            .count();
        Ok((gradeable.len(), right))
    }
}

// ----------------------------------------------------------------------------------------
// Stage 2: emphasis
// ----------------------------------------------------------------------------------------

/// A taught unit's training emphasis and why.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Emphasis {
    pub value: f32,
    pub reasons: Vec<String>,
}

/// Concept statistics of taught questions (by pattern): latest and first-sight grades.
#[derive(Clone, Debug, Default)]
pub struct Concepts {
    by_template: BTreeMap<String, (Vec<f32>, Vec<f32>)>,
}
impl Concepts {
    /// From the ledger records of the given taught units.
    pub fn of(ledger: &RewardLedger, items: &[&Item]) -> Concepts {
        let mut c = Concepts::default();
        let mut seen = BTreeSet::new();
        for it in items {
            let id = question_id(it);
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Some(q) = ledger.questions.get(&id) {
                let e = c.by_template.entry(q.template.clone()).or_default();
                if let Some(p) = q.last_percent {
                    e.0.push(p / 100.);
                }
                if let Some(p) = q.first_percent {
                    e.1.push(p / 100.);
                }
            }
        }
        c
    }
    /// From every taught question in the ledger.
    pub fn all(ledger: &RewardLedger) -> Concepts {
        let mut c = Concepts::default();
        for q in ledger.questions.values().filter(|q| q.trained) {
            let e = c.by_template.entry(q.template.clone()).or_default();
            if let Some(p) = q.last_percent {
                e.0.push(p / 100.);
            }
            if let Some(p) = q.first_percent {
                e.1.push(p / 100.);
            }
        }
        c
    }
    fn mean(v: &[f32]) -> Option<f32> {
        (v.len() >= 3).then(|| v.iter().sum::<f32>() / v.len() as f32)
    }
    /// (latest accuracy, first-sight accuracy) of a pattern, with at least three questions.
    pub fn get(&self, template: &str) -> (Option<f32>, Option<f32>) {
        self.by_template.get(template).map_or((None, None), |(a, b)| (Self::mean(a), Self::mean(b)))
    }
}

impl RewardLedger {
    /// The training emphasis of a taught unit (1.0 = ordinary training).
    pub fn emphasis(&self, cfg: &RewardConfig, it: &Item, concepts: &Concepts) -> Emphasis {
        if it.outcome_count() == 0 {
            return Emphasis {
                value: 1.0,
                reasons: vec!["reading or explanation: not graded, ordinary training".into()],
            };
        }
        self.emphasis_of(cfg, &question_id(it), concepts)
    }

    /// The emphasis of a recorded question, by its id.
    pub fn emphasis_of(&self, cfg: &RewardConfig, id: &str, concepts: &Concepts) -> Emphasis {
        let w = &cfg.weights;
        let Some(q) = self.questions.get(id) else {
            return Emphasis { value: 1.0, reasons: vec!["not asked yet: ordinary training".into()] };
        };
        let mut e = 1f32;
        let mut reasons = Vec::new();
        let g = q.last_percent.map(|p| p / 100.);
        match g {
            None => reasons.push("no certain grade yet: ordinary training".into()),
            Some(g) if g < 1. => {
                e += w.gap * (1. - g);
                reasons.push(if g == 0. {
                    "answered 0%: a knowledge gap".to_string()
                } else {
                    format!("answered {:.0}%: partly known", g * 100.)
                });
            }
            _ => {}
        }
        if q.failures >= 2 {
            e += w.failures * q.failures.min(4) as f32 / 4.;
            reasons.push(format!("wrong {} times in a row", q.failures));
        }
        if q.forgotten {
            e += w.forgotten;
            reasons.push("forgotten since an earlier checkpoint".into());
        }
        let cycle = self.cycles.get(&q.cycle);
        let best = cycle.map_or(0., |c| c.neural.best);
        if g.is_some_and(|g| g < 1.) && best > 0. && !cycle.is_some_and(|c| c.neural.complete) {
            e += w.progress;
            reasons.push("partial progress made: close to the answer".into());
        }
        let (acc, first) = concepts.get(&q.template);
        if let Some(a) = acc.filter(|a| *a < 1.) {
            e += w.concept * (1. - a);
            reasons.push(format!("concept \"{}\" answered {:.0}% overall", q.template, a * 100.));
        }
        if let Some(f) = first.filter(|f| *f < 1.) {
            e += w.variation * (1. - f);
            reasons.push(format!("new variations of this concept answered {:.0}% at first sight", f * 100.));
        }
        if g.is_some_and(|g| g < 1.) {
            let d = (terms(&q.expected).len() as f32 / 8.).min(1.);
            if d >= 0.25 {
                e += w.difficulty * d;
                reasons.push("long or complex answer".into());
            }
        }
        if g == Some(1.) {
            e *= w.correct_factor;
            reasons.push("answered correctly: less repetition".into());
            let k = q.retained_streak().min(4);
            if k > 0 {
                e *= w.retained_factor.powi(k as i32);
                reasons.push(format!("retained across {k} checkpoint(s)"));
            }
        }
        if q.last_memory_percent == Some(100.) && g.is_some_and(|g| g < 1.) {
            reasons.push("right only with memories in front: not yet in her weights".into());
        }
        if q.failures >= w.stuck_after && best == 0. && e > w.stuck_cap {
            e = w.stuck_cap;
            reasons.push(format!("stuck after {} tries without progress: needs the teacher", q.failures));
        }
        if e < cfg.min_emphasis {
            reasons.push("kept at the minimum: never excluded from training".into());
        }
        Emphasis { value: e.clamp(cfg.min_emphasis, cfg.max_emphasis), reasons }
    }
}

/// Copies of each unit in an epoch: emphasis spread over the same total number of units as
/// ordinary training (stochastic rounding from `rng`), at most [`MAX_COPIES`] each.
pub fn plan_copies(emphasis: &[f32], rng: &mut Rng) -> Vec<usize> {
    let n = emphasis.len();
    let sum: f32 = emphasis.iter().sum();
    if n == 0 || sum <= 0. {
        return vec![1; n];
    }
    emphasis
        .iter()
        .map(|&e| {
            let c = e as f64 * n as f64 / sum as f64;
            let whole = c.floor();
            let extra = usize::from(rng.unit() < c - whole);
            (whole as usize + extra).min(MAX_COPIES)
        })
        .collect()
}

/// A unit in the emphasis report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmphasisRow {
    pub question: String,
    pub lesson: String,
    pub value: f32,
    pub copies: usize,
    pub reasons: Vec<String>,
}

/// What Stage 2 did in one epoch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmphasisPlan {
    pub units: usize,
    /// Training copies scheduled (about the same as `units`).
    pub copies: usize,
    pub quizzed: usize,
    pub quiz_right: usize,
    /// Units trained more / less than once, and kept at the minimum.
    pub raised: usize,
    pub lowered: usize,
    pub at_minimum: usize,
    /// The units with the most and the least emphasis.
    pub top: Vec<EmphasisRow>,
    pub low: Vec<EmphasisRow>,
}

/// Emphasis of every unit, the copies of each, and the report.
pub fn plan_epoch(
    rewards: &Rewards,
    train: &[&Item],
    rng: &mut Rng,
    quizzed: (usize, usize),
) -> (Vec<usize>, EmphasisPlan) {
    let concepts = Concepts::of(&rewards.ledger, train);
    let em: Vec<Emphasis> =
        train.iter().map(|it| rewards.ledger.emphasis(&rewards.config, it, &concepts)).collect();
    let values: Vec<f32> = em.iter().map(|e| e.value).collect();
    let copies = plan_copies(&values, rng);
    let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
    let row = |i: usize| EmphasisRow {
        question: question_text(train[i]),
        lesson: train[i].source.clone(),
        value: values[i],
        copies: copies[i],
        reasons: em[i].reasons.clone(),
    };
    let mut idx: Vec<usize> = (0..train.len()).filter(|&i| train[i].outcome_count() > 0).collect();
    idx.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    let top = idx.iter().take(5).map(|&i| row(i)).collect();
    let low = idx.iter().rev().take(5).map(|&i| row(i)).collect();
    let plan = EmphasisPlan {
        units: train.len(),
        copies: copies.iter().sum(),
        quizzed: quizzed.0,
        quiz_right: quizzed.1,
        raised: values.iter().filter(|&&v| v > mean * 1.05).count(),
        lowered: values.iter().filter(|&&v| v < mean * 0.95).count(),
        at_minimum: values.iter().filter(|&&v| v <= rewards.config.min_emphasis + 1e-6).count(),
        top,
        low,
    };
    (copies, plan)
}

// ----------------------------------------------------------------------------------------
// Reward-hacking checks and summaries
// ----------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlagKind {
    /// Credit keeps rising while held-out accuracy does not.
    RewardWithoutUnseenGain,
    /// One wrong answer given to many different questions.
    NearDuplicateAnswers,
    /// Answers that creep forward letter by letter without completing.
    PrefixProbing,
    /// Taught questions right, held-out questions not.
    MemorizedNotGeneralized,
    /// Recognised words used often without correct answers.
    RelevanceWithoutAnswers,
}

/// A pattern to investigate (not a verdict of cheating).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Flag {
    pub kind: FlagKind,
    pub detail: String,
}

impl RewardLedger {
    pub fn flags(&self) -> Vec<Flag> {
        let mut out = Vec::new();
        let recent: Vec<&SessionSummary> = self.sessions.iter().rev().take(3).collect();
        if recent.len() == 3 {
            let credit = |s: &SessionSummary| s.progress + s.retention;
            let (newest, oldest) = (recent[0], recent[2]);
            if recent.iter().all(|s| credit(s) > 0.) && newest.heldout_after <= oldest.heldout_before + 1e-6 {
                out.push(Flag {
                    kind: FlagKind::RewardWithoutUnseenGain,
                    detail: format!(
                        "credit earned in each of the last 3 sessions ({:.2} in total) while held-out accuracy went {:.0}% -> {:.0}%",
                        recent.iter().map(|s| credit(s)).sum::<f32>(),
                        oldest.heldout_before * 100.,
                        newest.heldout_after * 100.
                    ),
                });
            }
        }
        let window: Vec<&AuditEntry> = self.audit.iter().rev().take(300).collect();
        let mut by_answer: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        let mut questions = BTreeSet::new();
        for e in &window {
            questions.insert(e.question_id.as_str());
            if !e.grade.correct && !e.normalized.is_empty() {
                by_answer.entry(e.normalized.as_str()).or_default().insert(e.question_id.as_str());
            }
        }
        if let Some((a, qs)) = by_answer.iter().max_by_key(|(_, q)| q.len()) {
            if qs.len() >= 10 && qs.len() * 4 >= questions.len() {
                out.push(Flag {
                    kind: FlagKind::NearDuplicateAnswers,
                    detail: format!(
                        "the wrong answer \"{a}\" was given to {} of {} recent questions",
                        qs.len(),
                        questions.len()
                    ),
                });
            }
        }
        let mut creeping: BTreeMap<&str, usize> = BTreeMap::new();
        for e in self.audit.iter() {
            let c = creeping.entry(e.question_id.as_str()).or_default();
            match e.milestone.as_deref() {
                Some(m) if m.starts_with("prefix") => *c += 1,
                Some("complete") => *c = 0,
                _ => {}
            }
        }
        let probing = creeping.values().filter(|&&n| n >= 4).count();
        if probing > 0 {
            out.push(Flag {
                kind: FlagKind::PrefixProbing,
                detail: format!(
                    "{probing} question(s) earned 4 or more prefix milestones without being completed"
                ),
            });
        }
        if let Some(s) = self.sessions.last() {
            if let Some(qr) = s.quiz_rate {
                if qr - s.heldout_after >= 0.4 {
                    out.push(Flag {
                        kind: FlagKind::MemorizedNotGeneralized,
                        detail: format!(
                            "in {}: taught questions {:.0}% right, held-out questions {:.0}%",
                            s.lesson,
                            qr * 100.,
                            s.heldout_after * 100.
                        ),
                    });
                }
            }
        }
        let rel = window.iter().filter(|e| e.relevance > 0.).count();
        let right = window.iter().filter(|e| e.grade.correct).count();
        if rel >= 20 && (right as f32) < 0.1 * window.len() as f32 {
            out.push(Flag {
                kind: FlagKind::RelevanceWithoutAnswers,
                detail: format!("{rel} recent answers used words from the question's text, but only {right} of {} were right", window.len()),
            });
        }
        out
    }

    /// Clears a question's reward cycle (teacher's decision; recorded in the audit log as a
    /// note on the next answer).
    pub fn reset_cycle(&mut self, question_id: &str) -> bool {
        let Some(q) = self.questions.get(question_id) else { return false };
        self.cycles.remove(&q.cycle);
        true
    }

    pub fn summary(&self) -> LedgerSummary {
        let mut concepts: BTreeMap<String, ConceptRow> = BTreeMap::new();
        for q in self.questions.values() {
            let r = concepts
                .entry(q.template.clone())
                .or_insert_with(|| ConceptRow { template: q.template.clone(), ..ConceptRow::default() });
            r.questions += 1;
            if q.trained {
                if let Some(p) = q.last_percent {
                    r.taught.0 += usize::from(p >= 100.);
                    r.taught.1 += 1;
                }
                if let Some(p) = q.first_percent {
                    r.first_sight.0 += usize::from(p >= 100.);
                    r.first_sight.1 += 1;
                }
            }
            if q.unseen {
                if let Some(p) = q.last_percent {
                    r.unseen.0 += usize::from(p >= 100.);
                    r.unseen.1 += 1;
                }
            }
        }
        let mut concepts: Vec<ConceptRow> = concepts.into_values().collect();
        concepts.sort_by(|a, b| b.questions.cmp(&a.questions).then(a.template.cmp(&b.template)));
        LedgerSummary {
            questions: self.questions.len(),
            taught: self.questions.values().filter(|q| q.trained).count(),
            unseen: self.questions.values().filter(|q| q.unseen).count(),
            complete_cycles: self.cycles.values().filter(|c| c.neural.complete).count(),
            neural_right: self.questions.values().filter(|q| q.last_percent == Some(100.)).count(),
            memory_right: self.questions.values().filter(|q| q.last_memory_percent == Some(100.)).count(),
            forgotten: self.questions.values().filter(|q| q.forgotten).count(),
            stuck: self
                .questions
                .values()
                .filter(|q| q.failures >= EmphasisWeights::default().stuck_after)
                .count(),
            totals: self.totals.clone(),
            protected_refused: self.protected_refused,
            concepts,
            flags: self.flags(),
            sessions: self.sessions.iter().rev().take(20).cloned().collect(),
        }
    }

    /// Starts a session record; finish it with [`RewardLedger::end_session`].
    pub(crate) fn begin_session(&self) -> Totals {
        self.totals.clone()
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn end_session(
        &mut self,
        start: &Totals,
        model: &str,
        lesson: &str,
        steps: (u64, u64),
        learned: bool,
        weighted: bool,
        quiz_rate: Option<f32>,
        heldout: (f32, f32),
        loss: (f32, f32),
    ) {
        self.seq += 1;
        let t = &self.totals;
        self.sessions.push(SessionSummary {
            seq: self.seq,
            time: now(),
            model: model.to_string(),
            lesson: lesson.to_string(),
            step_start: steps.0,
            step_end: steps.1,
            learned,
            weighted,
            progress: t.progress - start.progress,
            retention: t.retention - start.retention,
            generalization: t.generalization - start.generalization,
            relevance: t.relevance - start.relevance,
            quiz_rate,
            heldout_before: heldout.0,
            heldout_after: heldout.1,
            heldout_loss_before: loss.0,
            heldout_loss_after: loss.1,
        });
        if self.sessions.len() > 500 {
            self.sessions.remove(0);
        }
    }
}

/// One concept (question pattern): taught questions right now, right at first sight, and
/// never-trained questions right: (right, asked).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConceptRow {
    pub template: String,
    pub questions: usize,
    pub taught: (usize, usize),
    pub first_sight: (usize, usize),
    pub unseen: (usize, usize),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LedgerSummary {
    pub questions: usize,
    pub taught: usize,
    pub unseen: usize,
    pub complete_cycles: usize,
    pub neural_right: usize,
    pub memory_right: usize,
    pub forgotten: usize,
    pub stuck: usize,
    pub totals: Totals,
    pub protected_refused: usize,
    pub concepts: Vec<ConceptRow>,
    pub flags: Vec<Flag>,
    pub sessions: Vec<SessionSummary>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::lessons::qa;

    fn item(q: &str, a: &str) -> Item {
        let mut it = crate::learn::parse_lesson(
            &format!("[Dialogue Lesson]\n[examples]\n{q}  {a}\n[/examples]\n[/Dialogue Lesson]\n"),
            "t.txt",
        )
        .unwrap()
        .remove(0);
        it.held_out = false;
        it.segments = qa(q, a);
        it
    }
    fn rewards(prefix: bool) -> Rewards {
        Rewards::new(
            RewardConfig { letter_prefix: prefix, ..RewardConfig::default() },
            RewardLedger::default(),
            "m",
        )
    }
    fn say(r: &mut Rewards, it: &Item, reply: &str, step: u64, checkpoint: bool) -> AuditEntry {
        r.record(Evidence {
            item: it,
            answer: Answer::Reply(reply),
            origin: Origin::Training,
            step,
            checkpoint,
            teacher: None,
        })
        .unwrap()
    }

    #[test]
    fn grader_judges_meaning_and_admits_uncertainty() {
        let g = |e: &str, r: &str| grade_reply(e, None, None, r, false);
        assert_eq!(g("Hallway", "Hallway").percent, Some(100.));
        let n = g("Hallway.", "HALLWAY!!!");
        assert!(n.correct && n.method == Method::Normalized);
        assert_eq!(g("Hallway", "").reason, "no answer");
        assert_eq!(g("Hallway", "Office").percent, Some(0.), "nothing in common: certainly wrong");
        assert_eq!(g("8", "8 apples").percent, Some(75.), "numbers count as content");
        assert_eq!(g("8", "7").percent, Some(0.));
        assert_eq!(g("The door is open.", "The door is not open.").method, Method::Mismatch);
        let c = g("Hallway", "In the hallway.");
        assert_eq!((c.percent, c.correct), (Some(75.), false));
        let u = g("Max likes to run in the big park.", "Max likes the park.");
        assert!(u.percent.is_none() && u.method == Method::Uncertain, "{u:?}");
        assert_eq!(
            grade_reply("Yes.", None, Some("No."), "No.", false).reason,
            "gave the contradicted answer"
        );
        let k =
            AnswerKey::parse("A robin is an animal. | must: robin animal | never: not", GradeMode::Sentence)
                .unwrap();
        assert_eq!(
            grade_reply("A robin is an animal.", Some(&k), None, "The robin flies.", false).percent,
            Some(30.)
        );
        assert_eq!(
            grade_reply("A robin is an animal.", Some(&k), None, "A robin is not an animal.", false).percent,
            Some(0.)
        );
        // Letters count only in spelling grading.
        assert_eq!(g("Hallway", "Hal").percent, Some(0.));
        assert_eq!(grade_reply("Hallway", None, None, "Hal", true).method, Method::Spelling);
    }

    #[test]
    fn progress_milestones_are_credited_once() {
        let mut r = rewards(true);
        let mut it = item("Where is Daniel?", "Hallway");
        it.spelling = true;
        let steps: Vec<(&str, f32)> = vec![
            ("How is Daniel?", 0.10),
            ("Hello", 0.),
            ("Hal", 0.18),
            ("Haley", 0.),
            ("Hallway", 0.72),
            ("HALLWAY!!!", 0.),
            ("hallway", 0.),
        ];
        for (reply, want) in steps {
            let e = say(&mut r, &it, reply, 1, true);
            assert!((e.progress - want).abs() < 1e-4, "{reply}: {} != {want} ({:?})", e.progress, e.withheld);
        }
        let e = say(&mut r, &it, "Hallway", 2, true);
        assert!(e.grade.correct && e.progress == 0., "still correct, no new progress");
        assert!((r.ledger.totals.progress - 1.0).abs() < 1e-4, "capped at the budget");
        // A reworded copy shares the cycle.
        let mut again = item("Where's Daniel?", "Hallway");
        again.spelling = true;
        assert_eq!(cycle_key(&again), cycle_key(&it));
        assert_eq!(say(&mut r, &again, "Hallway", 1, true).progress, 0.);
        // Without the experiment, letters earn nothing.
        let mut plain = rewards(false);
        assert_eq!(say(&mut plain, &it, "Hal", 1, true).progress, 0.);
        // An answer from a model still in training guides practice but earns nothing.
        let mut fresh = rewards(true);
        let e = say(&mut fresh, &it, "Hal", 1, false);
        assert_eq!(e.progress, 0.);
        assert!(e.withheld[0].contains("during training"));
        assert_eq!(fresh.ledger.questions[&question_id(&it)].last_percent.map(|p| p.round()), Some(43.));
    }

    #[test]
    fn retention_shrinks_and_unseen_questions_count_more() {
        let mut r = rewards(false);
        let it = item("Where is Daniel?", "Hallway");
        let bonus: Vec<f32> = [
            (10, "Hallway"),
            (10, "Hallway"),
            (20, "Hallway"),
            (30, "Hallway"),
            (40, "Office"),
            (50, "Hallway"),
        ]
        .iter()
        .map(|(s, a)| say(&mut r, &it, a, *s, true).retention)
        .collect();
        assert_eq!(bonus[..4], [0.6, 0., 0.2, 0.1]);
        assert_eq!(bonus[4], 0.);
        assert!(r.ledger.questions[&question_id(&it)].checkpoints[3].event.starts_with("lost"));
        assert!((bonus[5] - 0.05).abs() < 1e-6, "relearned continues the decay");
        let mut held = item("Where is Mary?", "Kitchen");
        held.held_out = true;
        let e = r
            .record(Evidence {
                item: &held,
                answer: Answer::Reply("Kitchen"),
                origin: Origin::HeldOut,
                step: 50,
                checkpoint: true,
                teacher: None,
            })
            .unwrap();
        assert_eq!((e.generalization, e.progress), (1.0, 0.));
    }

    #[test]
    fn memory_answers_are_separate_and_protected_questions_are_refused() {
        let mut r = rewards(false);
        let it = item("Where is Daniel?", "Hallway");
        let mut with = it.clone();
        with.context = Some(format!("{MEMORY}Daniel went to the hallway.\n"));
        assert_eq!(say(&mut r, &with, "Hallway", 1, true).progress, 0.5, "memory budget is smaller");
        assert_eq!(
            say(&mut r, &it, "Hallway", 1, true).progress,
            1.0,
            "the network alone still earns its own"
        );
        let q = &r.ledger.questions[&question_id(&it)];
        assert_eq!((q.last_percent, q.last_memory_percent), (Some(100.), Some(100.)));
        let exam = item("Where is Bob?", "Garden");
        r.protect([&exam]);
        assert!(r
            .record(Evidence {
                item: &exam,
                answer: Answer::Reply("Garden"),
                origin: Origin::Training,
                step: 1,
                checkpoint: true,
                teacher: None
            })
            .is_none());
        assert_eq!(r.ledger.protected_refused, 1);
        assert!(!r.ledger.questions.contains_key(&question_id(&exam)));
    }

    #[test]
    fn relevance_is_diagnostic_and_capped() {
        let mut r = rewards(false);
        let mut it =
            item("Daniel went to the office. Then Daniel went to the hallway. Where is Daniel?", "Hallway");
        it.kind = Kind::Reasoning;
        let e = say(&mut r, &it, "Office.", 1, true);
        assert_eq!(e.recognized, ["office"]);
        assert!((e.relevance - 0.01).abs() < 1e-6);
        assert_eq!(say(&mut r, &it, "The office!", 1, true).relevance, 0., "same word, no new credit");
        let e = say(&mut r, &it, "Banana", 1, true);
        assert_eq!(e.invented, ["banana"]);
    }

    #[test]
    fn teacher_grades_and_notes_are_recorded() {
        let mut r = rewards(false);
        let it = item("Is Max a dog?", "Yes, Max is a dog.");
        let e = r
            .record(Evidence {
                item: &it,
                answer: Answer::Reply("Max dog"),
                origin: Origin::Practice,
                step: 3,
                checkpoint: false,
                teacher: Some((40., "Right idea, not a sentence.")),
            })
            .unwrap();
        assert_eq!((e.grade.method, e.grade.percent, e.progress), (Method::Teacher, Some(40.), 0.4));
        assert_eq!(r.ledger.questions[&question_id(&it)].notes, ["Right idea, not a sentence."]);
    }

    #[test]
    fn emphasis_follows_need_and_never_excludes() {
        let mut r = rewards(false);
        let known = item("Is Max a dog?", "Yes.");
        let gap = item("Where is Daniel?", "Hallway");
        for step in [1, 2, 3] {
            say(&mut r, &known, "Yes.", step, true);
            say(&mut r, &gap, "Office", step, true);
        }
        let none = item("What is red?", "A colour.");
        let c = Concepts::of(&r.ledger, &[&known, &gap, &none]);
        let ek = r.ledger.emphasis(&r.config, &known, &c);
        let eg = r.ledger.emphasis(&r.config, &gap, &c);
        let en = r.ledger.emphasis(&r.config, &none, &c);
        assert!(eg.value > 2. && eg.reasons.iter().any(|x| x.contains("0%")), "{eg:?}");
        assert_eq!(ek.value, r.config.min_emphasis, "{ek:?}");
        assert!(ek.reasons.iter().any(|x| x.contains("never excluded")));
        assert_eq!(en.value, 1.0);
        let mut rng = Rng::new(1);
        let mut total = [0; 3];
        for _ in 0..200 {
            for (t, c) in total.iter_mut().zip(plan_copies(&[ek.value, eg.value, en.value], &mut rng)) {
                *t += c;
            }
        }
        assert!(total[0] > 0, "the known unit is still trained");
        assert!(total[1] > total[2] && total[2] > total[0]);
        let sum: usize = total.iter().sum();
        assert!((sum as f32 / 200. - 3.).abs() < 0.2, "same number of units as ordinary training");
    }
}
