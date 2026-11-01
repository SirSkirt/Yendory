//! Teaching sessions: Foundation candidate training, rehearsal of earlier lessons, retention
//! guards and checkpoint selection, with a typed record of every candidate in every epoch.
//! The rules are written out in docs/LEARNING.md.
//!
//! Two models are kept: the ACCEPTED model (the best validated checkpoint) and the WORKING model
//! that training continues from (weights, Adam state, step, tokenizer). With Foundation training
//! on, four candidates are trained from the working model on identical batches — Right (Dense),
//! Left (Transformer), Interconnect and Joint — differing only in which parameter group the
//! optimizer may update; every pathway still runs in every forward pass. Each candidate is
//! scored as a whole model; the best ELIGIBLE candidate that improves on the accepted model
//! becomes the accepted model and the working model.
//!
//! A flat epoch (candidates safe, but none better than the accepted model by the required
//! amount) does not throw the epoch's learning away: training continues from the best safe
//! candidate, so improvement can build up over several epochs until it is large enough to be
//! accepted. When no candidate is safe (held-out loss rose too far, earlier lessons or answers
//! lost, Dense unused), the working model is restored to the accepted model. After `patience`
//! epochs in a row without an accepted candidate the session ends, and the accepted model is
//! the result. Teacher answers are training targets and evaluation keys only; they are never
//! given to the model at inference.
use super::eval::{evaluate, evaluate_confirmed, sample_per_kind, Confirmed, Report};
use super::lessons::{grow_vocabulary, Item, Kind};
use super::reward::{Origin, Rewards, Totals};
use super::{encode_item, lr_schedule, Rng, Row};
use crate::model::{Ablation, Bicameral, Focus};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionConfig {
    pub epochs: u32,
    pub batch_size: usize,
    pub learning_rate: f32,
    /// Four candidates (Right, Left, Interconnect, Joint) per epoch instead of Joint only. With
    /// an education stage: each part the stage may train, on its own and together.
    pub foundation: bool,
    /// Experimental bicameral education stage; when set it chooses the candidates (see
    /// `EducationStage::foci`) and turns on the Dense measurements and guard.
    #[serde(default)]
    pub education: Option<EducationStage>,
    /// In bicameral education, reject a candidate whose Dense reliance falls below this share of
    /// the accepted model's (the held-out loss gap when Dense's input to the transformer is cut).
    #[serde(default)]
    pub keep_dense_reliance: f32,
    /// Stop after this many epochs in a row without an accepted candidate ("flat epochs"; the
    /// working model keeps training through them).
    pub patience: u32,
    /// Share of batch rows drawn from archived earlier lessons (when an archive exists).
    pub rehearsal_share: f32,
    /// Held-out probes evaluated per kind (`None` = all held-out units of the lesson).
    pub eval_limit: Option<usize>,
    /// A candidate's held-out loss may exceed the lesson-start and the last accepted loss by at
    /// most this much.
    pub heldout_loss_tolerance: f32,
    /// Archived earlier lessons' held-out loss may exceed its session baseline by at most this.
    pub retention_loss_tolerance: f32,
    /// Earlier-lesson held-out units whose answers the baseline gets right must stay right.
    pub retention_panel: usize,
    /// Held-out answers of THIS lesson that the accepted model gets right must stay right.
    pub protect_correct_answers: bool,
    /// With `protect_correct_answers`: judge answers by their net change instead. A candidate is
    /// rejected only when it loses more held-out answers the accepted model gave than it gains
    /// (so gaining three and losing one is allowed). Off: no answer may be lost.
    #[serde(default)]
    pub protect_answers_net: bool,
    /// Earlier lessons' held-out loss may rise by at most this share of its session baseline
    /// (e.g. 0.03 = 3%) instead of the fixed `retention_loss_tolerance`.
    #[serde(default)]
    pub retention_relative: Option<f32>,
    /// Earlier-lesson held-out units per kind used for that loss (more text, less noise).
    #[serde(default = "default_retention_units")]
    pub retention_units: usize,
    /// Grow learned word units from this lesson's training text (ids stay stable).
    pub grow_vocabulary: bool,
    /// Vocabulary limit (learned word units). `None`: the model's allocated slots. A larger
    /// limit adds slots as words are learned, so the model never carries unused ones.
    #[serde(default)]
    pub word_limit: Option<usize>,
    pub min_word_count: usize,
    pub seed: u64,
    /// How training units are scheduled into batches.
    #[serde(default)]
    pub strategy: Strategy,
    /// What a course does when a lesson is not passed (diagnosed retries, then later in the queue).
    #[serde(default)]
    pub retry: RetryPolicy,
    /// A lesson PASSES when this share of its held-out questions is answered exactly right.
    /// Below it, accepted improvements count as progress, not as mastery.
    #[serde(default = "default_pass_mark")]
    pub pass_mark: f32,
    /// A candidate that answers no more questions correctly must lower the held-out loss by at
    /// least this much to be accepted: smaller changes are noise, not learning.
    #[serde(default = "default_min_loss_improvement")]
    pub min_loss_improvement: f32,
    /// Dropout rate while training (0 = off): applied to the transformer blocks' attention and
    /// feed-forward outputs during the session's training steps only, never when scoring.
    #[serde(default)]
    pub dropout: f32,
    /// Share of each epoch's rows drawn from general reading text (`reading`) instead of the
    /// lesson (0 = off). Rows per epoch stay as without it, so the training budget is the same.
    #[serde(default)]
    pub reading_share: f32,
    /// General reading text for `reading_share`: documents, e.g. a corpus the model has read.
    /// Not saved with settings.
    #[serde(skip)]
    pub reading: Option<std::sync::Arc<Vec<String>>>,
    /// Where each accepted epoch's model is written while the session runs (see
    /// [`save_checkpoint`]), so a power cut loses at most the epoch in progress. `None`: kept in
    /// memory only until the session ends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_dir: Option<std::path::PathBuf>,
    /// Epochs of this session already done before an interruption: the session runs the
    /// remaining epochs with the learning rate where the interrupted one was (continuing its
    /// schedule instead of starting again at the full rate). 0: a new session.
    #[serde(default)]
    pub resume_epochs: u32,
    /// With `resume_epochs`: the share of the scheduled learning rate the interrupted session
    /// had reached by backing off (see `backoff`). `None`: the full scheduled rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_scale: Option<f32>,
    /// Off by default here; on by default for courses and the Classroom. While no epoch has been
    /// accepted in this session, [`BACKOFF_EPOCHS`] epochs in a row whose candidates were worse
    /// than the accepted model (they lost answers or earlier lessons) halve the learning rate,
    /// up to [`BACKOFF_TIMES`] times; the flat-epoch count starts again after each.
    #[serde(default)]
    pub backoff: bool,
    /// The first epoch's best candidate (most held-out answers, then lowest held-out loss) is
    /// accepted even when the safeguards reject it, and the earlier-lesson guards are measured
    /// from it afterwards. For continuing from a model the safeguards keep refusing to move.
    #[serde(default)]
    pub override_first_epoch: bool,
    /// Weight blending (0 = off): each epoch also judges, as one more candidate, the model this
    /// share of the way from the epoch's starting model to its most promising candidate (the
    /// most held-out answers, then the lowest held-out loss). No extra training steps.
    #[serde(default)]
    pub blend: f32,
    /// Off by default here; on by default for courses and the Classroom. When every candidate
    /// of an epoch is rejected only because earlier lessons were forgotten (their held-out loss or answers), training continues from
    /// the one with the lowest held-out loss on this lesson instead of restarting from the
    /// accepted model, so rehearsal can win the earlier lessons back. Acceptance still needs
    /// every guard: the accepted model never forgets more than the guards allow.
    #[serde(default)]
    pub recover_earlier: bool,
    /// Experimental (default off): earlier-lesson answers are judged by their net change, like
    /// `protect_answers_net` for this lesson's: a candidate is rejected only when it loses more
    /// earlier-lesson answers (of the baseline's) than it gains held-out answers of this lesson
    /// over the accepted model. Off: no earlier-lesson answer may be lost.
    #[serde(default)]
    pub earlier_answers_net: bool,
    /// Lessons taught together (`course::teach_group`) only; off by default here, on by default
    /// for courses and the Classroom: the saved earlier lessons outside the group (the archive)
    /// are trained in the same mixed session and their held-out questions join the group's
    /// judging. They also stay guarded as earlier lessons, so a checkpoint is accepted only when
    /// their held-out loss and answers pass the earlier-lesson guards.
    #[serde(default)]
    pub train_earlier: bool,
    /// Experimental (0 = off): probation. When no candidate of an epoch is accepted, training
    /// continues from the most promising rejected candidate (most held-out answers, then the
    /// lowest held-out loss; never one whose loss is not finite) for up to this many epochs in
    /// a row, instead of restarting from the accepted model, so a change that first costs
    /// something has room to recover. Since 0.8.6 it takes priority over the flat-epoch rule
    /// (continuing from a safe but flat candidate), which used to leave it no chance to act;
    /// with probation off the flat-epoch rule works as before. When the branch reaches its
    /// limit, training restarts from the accepted model (not from a flat candidate). Nothing about
    /// acceptance changes: every candidate is still judged against the accepted model by every
    /// guard, and the accepted model (with its optimizer state) changes only when one passes.
    /// After this many epochs without an accepted candidate, the branch is discarded and
    /// training restarts from the accepted model. Every step is recorded (`EpochRecord::probation`).
    #[serde(default)]
    pub probation_epochs: u32,
    /// New word units come only from the lessons being taught (on by default), not from saved
    /// earlier lessons trained alongside them (`train_earlier`): the model already reads those
    /// letter by letter, and turning their words into new units costs answers it had (measured
    /// on step 19,000: 380 words from all lessons cost 67 of its 417 earlier answers, 156 from
    /// the new course alone 9). Off: words come from everything trained, as before 0.7.2.
    #[serde(default = "yes")]
    pub words_from_new_lessons: bool,
    /// Experimental (off by default): when words are added, the earlier-lesson answers are
    /// guarded as the model gave them before the words were added, so answers lost to the new
    /// words must be won back before a checkpoint is accepted. Off: they are guarded as the model
    /// gives them after the words are added (that loss is reported, not guarded).
    #[serde(default)]
    pub guard_before_growth: bool,
    /// Experimental (`None` by default): the earlier lessons' answer check takes up to this many
    /// held-out questions from EVERY earlier lesson, instead of `retention_panel` questions in
    /// all, so a lesson the session damages is seen and one changed answer weighs less.
    #[serde(default)]
    pub retention_per_lesson: Option<usize>,
    /// Experimental (off by default): the earlier lessons' answer check is judged by its net
    /// change: a candidate is rejected only when it loses more of the answers the baseline gave
    /// than it newly gets right among the same questions (widespread forgetting, not a few
    /// answers changing). Off: no answer the baseline gave may be lost (see also
    /// `earlier_answers_net`).
    #[serde(default)]
    pub earlier_panel_net: bool,
    /// Experimental (off by default): with the larger earlier-lesson check
    /// (`retention_per_lesson`), judge the earlier lessons by their held-out ANSWERS: the check
    /// is judged by its net change (as `earlier_panel_net`), and their held-out loss may rise up
    /// to `earlier_loss_cap` instead of `retention_loss_tolerance`, so a small loss rise with the
    /// answers kept no longer rejects a candidate. A rise above the cap still does. Without the
    /// larger check it changes nothing (a dozen questions are too few to judge by).
    #[serde(default)]
    pub earlier_by_answers: bool,
    /// With `earlier_by_answers`: the most the earlier lessons' held-out loss may rise above its
    /// session baseline (a substantial regression is rejected whatever the answers).
    #[serde(default = "default_earlier_loss_cap")]
    pub earlier_loss_cap: f32,
    /// For experiments: called at the end of every epoch with the epoch, the accepted model and
    /// the working model training continues from. Not saved with settings.
    #[serde(skip)]
    pub epoch_hook: Option<EpochHook>,
}

/// See `SessionConfig::epoch_hook`.
#[derive(Clone)]
pub struct EpochHook(pub std::sync::Arc<EpochFn>);
/// The epoch, the accepted model and the working model.
pub type EpochFn = dyn Fn(u32, &Bicameral, &Bicameral) + Send + Sync;
impl std::fmt::Debug for EpochHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EpochHook")
    }
}
impl PartialEq for EpochHook {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}
fn yes() -> bool {
    true
}
fn default_pass_mark() -> f32 {
    0.5
}

fn default_earlier_loss_cap() -> f32 {
    0.25
}

fn default_retention_units() -> usize {
    16
}
fn default_min_loss_improvement() -> f32 {
    0.01
}

/// How a lesson's training units are scheduled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// Every training unit (including explanation prose), in a new random order each epoch.
    #[default]
    Standard,
    /// Question/answer units only (no prose), easiest (shortest) first: neighbouring units are
    /// shuffled within small groups so the order still varies between epochs.
    Focused,
}

/// What a course does when a lesson is not passed. Each retry is chosen from a diagnosis of the
/// previous attempt (see `course::diagnose`); the learning rate is lowered only when training
/// was diagnosed as unstable. After the retries, the lesson moves to the end of the queue.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Retries after the initial attempt (0 to 3).
    pub retries: u32,
    /// Move a lesson that is still not passed to the end of the queue.
    pub requeue: bool,
    /// Lowest learning rate a retry may use.
    pub min_learning_rate: f32,
}
impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy { retries: 3, requeue: true, min_learning_rate: 1e-6 }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        SessionConfig {
            epochs: 4,
            batch_size: 16,
            learning_rate: 1e-3,
            foundation: true,
            patience: 4,
            rehearsal_share: 0.10,
            eval_limit: Some(40),
            heldout_loss_tolerance: 0.08,
            retention_loss_tolerance: 0.02,
            retention_panel: 8,
            protect_correct_answers: true,
            protect_answers_net: true,
            retention_relative: None,
            retention_units: default_retention_units(),
            grow_vocabulary: true,
            education: None,
            keep_dense_reliance: 0.5,
            word_limit: None,
            min_word_count: 12,
            seed: 1,
            strategy: Strategy::Standard,
            retry: RetryPolicy::default(),
            pass_mark: default_pass_mark(),
            min_loss_improvement: default_min_loss_improvement(),
            dropout: 0.,
            reading_share: 0.,
            reading: None,
            blend: 0.,
            recover_earlier: false,
            earlier_answers_net: false,
            train_earlier: false,
            checkpoint_dir: None,
            resume_epochs: 0,
            resume_scale: None,
            backoff: false,
            override_first_epoch: false,
            probation_epochs: 0,
            epoch_hook: None,
            words_from_new_lessons: true,
            guard_before_growth: false,
            retention_per_lesson: None,
            earlier_panel_net: false,
            earlier_by_answers: false,
            earlier_loss_cap: default_earlier_loss_cap(),
        }
    }
}

/// Reading windows for general-reading rehearsal: each document encoded once.
struct ReadingRows {
    docs: Vec<Vec<crate::TokenId>>,
    window: usize,
}
impl ReadingRows {
    fn new(model: &Bicameral, texts: &[String]) -> Result<Self, String> {
        let mut docs = Vec::new();
        for t in texts {
            // Normalised as corpus reading is (typographic quotes and dashes to ASCII).
            let (text, _) = super::corpus::normalize_ascii(t.trim());
            let ids = model.tokenizer.encode(&text)?;
            if ids.len() >= 8 {
                docs.push(ids);
            }
        }
        Ok(ReadingRows { docs, window: model.config.max_tokens.min(64) })
    }
    /// A random window of a random document, every token trained at full weight.
    fn row(&self, rng: &mut Rng) -> Row {
        let doc = &self.docs[rng.below(self.docs.len())];
        let n = doc.len().min(self.window + 1);
        let start = rng.below(doc.len() - n + 1);
        let w = &doc[start..start + n];
        Row { ids: w[..n - 1].to_vec(), targets: w[1..].to_vec(), mask: vec![1.; n - 1] }
    }
}

pub fn focus_name(f: Focus) -> &'static str {
    match f {
        Focus::Joint => "joint",
        Focus::Left => "transformer",
        Focus::Right => "dense",
        Focus::Interconnect => "interconnect",
        Focus::DenseAndInterconnect => "dense+interconnect",
    }
}

/// Experimental bicameral education: which parts of the brain a lesson or reading session may
/// change, so each hemisphere is taught its intended role. Only the update mask and the guards
/// differ from ordinary training; nothing is added to the network or its checkpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EducationStage {
    /// Stage A: Dense and the interconnect learn; the transformer is frozen, so knowledge the
    /// lesson needs can only be stored in Dense and passed on through the interconnect.
    DenseKnowledge,
    /// Stage B: the transformer learns (language, in-context reasoning); Dense and the
    /// interconnect are frozen but keep sending their signals.
    TransformerLanguage,
    /// Stage C: cooperation. Each epoch tries Dense + interconnect and the whole brain, and keeps
    /// the better one, as long as Dense's contribution does not collapse.
    Cooperative,
}
impl EducationStage {
    /// The training candidates of an epoch in this stage. With Foundation, every part the stage
    /// may train is also tried on its own, so each is evaluated in its role; every candidate is
    /// still scored as a whole brain.
    pub fn foci(self, foundation: bool) -> Vec<Focus> {
        match (self, foundation) {
            (EducationStage::DenseKnowledge, false) => vec![Focus::DenseAndInterconnect],
            (EducationStage::DenseKnowledge, true) => {
                vec![Focus::Right, Focus::Interconnect, Focus::DenseAndInterconnect]
            }
            (EducationStage::TransformerLanguage, _) => vec![Focus::Left],
            (EducationStage::Cooperative, false) => vec![Focus::DenseAndInterconnect, Focus::Joint],
            (EducationStage::Cooperative, true) => vec![
                Focus::Right,
                Focus::Left,
                Focus::Interconnect,
                Focus::DenseAndInterconnect,
                Focus::Joint,
            ],
        }
    }
}

/// How a whole model scores at one point of the session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Score {
    /// Teacher-forced loss on the lesson's held-out probes (answer tokens weighted as in training).
    pub heldout_loss: f32,
    /// Greedy exact-match replies on the held-out probes.
    pub exact_correct: usize,
    pub exact_total: usize,
    /// Reasoning: supported answer more likely than the contradicted one.
    pub prefers_supported: usize,
    pub prefers_total: usize,
    /// Teacher-forced loss on archived earlier lessons' held-out units (None without archive).
    pub retention_loss: Option<f32>,
    /// Earlier-lesson panel answers produced correctly.
    pub panel_correct: usize,
    pub panel_total: usize,
    /// Exact matches per kind of question: (correct, total).
    #[serde(default)]
    pub by_kind: std::collections::BTreeMap<String, (usize, usize)>,
    /// How much the held-out loss rises when Dense's input to the transformer is cut: Dense's
    /// measured contribution (measured in bicameral education only).
    #[serde(default)]
    pub dense_reliance: Option<f32>,
    /// Held-out loss of the transformer with Dense's input cut: how the language side does on
    /// its own (measured in bicameral education only).
    #[serde(default)]
    pub without_dense_loss: Option<f32>,
    /// Exact matches per lesson (the items' source file): (correct, total). Lessons taught
    /// together show each lesson's own progress with it.
    #[serde(default)]
    pub by_lesson: std::collections::BTreeMap<String, (usize, usize)>,
    /// Held-out replies stopped by their length limit before they ended (among the wrong ones).
    #[serde(default)]
    pub cut_off: usize,
}

/// Exact matches per lesson: each item's outcomes (one per question, one per conversation
/// turn), in the order `evaluate` reports them.
fn by_lesson(items: &[&Item], report: &Report) -> std::collections::BTreeMap<String, (usize, usize)> {
    let mut out: std::collections::BTreeMap<String, (usize, usize)> = std::collections::BTreeMap::new();
    let mut outcomes = report.outcomes.iter();
    for it in items {
        let e = out.entry(it.source.clone()).or_default();
        for o in outcomes.by_ref().take(it.outcome_count()) {
            e.0 += usize::from(o.correct);
            e.1 += 1;
        }
    }
    out
}
impl Score {
    /// Share of held-out questions answered exactly right.
    pub fn exact_rate(&self) -> f32 {
        if self.exact_total == 0 {
            0.
        } else {
            self.exact_correct as f32 / self.exact_total as f32
        }
    }
}

/// Why a candidate was not accepted (the readable text is in `CandidateRecord::reasons`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rejection {
    /// Held-out loss not finite or rose beyond the tolerance: training was unstable.
    HeldoutLossRose,
    /// Earlier lessons' held-out loss rose beyond the tolerance.
    EarlierLessonsLoss,
    /// An earlier-lesson answer the baseline produced was lost.
    EarlierAnswersLost,
    /// A held-out answer of this lesson that the accepted model produced was lost.
    AnswersLost,
    /// Eligible, but not better than the accepted model by a meaningful amount.
    NoImprovement,
    /// Bicameral education: the model stopped using Dense (its contribution collapsed).
    DenseUnused,
    /// Eligible and better than the accepted model, but another candidate of the same epoch was
    /// better (one candidate is selected per epoch).
    Outranked,
}
impl Rejection {
    pub const ALL: [Rejection; 7] = [
        Rejection::HeldoutLossRose,
        Rejection::EarlierLessonsLoss,
        Rejection::EarlierAnswersLost,
        Rejection::AnswersLost,
        Rejection::NoImprovement,
        Rejection::DenseUnused,
        Rejection::Outranked,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Rejection::HeldoutLossRose => "heldout_loss_rose",
            Rejection::EarlierLessonsLoss => "earlier_lessons_loss",
            Rejection::EarlierAnswersLost => "earlier_answers_lost",
            Rejection::AnswersLost => "answers_lost",
            Rejection::NoImprovement => "no_improvement",
            Rejection::DenseUnused => "dense_unused",
            Rejection::Outranked => "outranked",
        }
    }
}

/// Candidates of one or more sessions counted by outcome and by every rejection reason, in
/// total and per focus. Every candidate is either selected or has at least one reason; a
/// candidate rejected for several reasons counts once under each of them, and once under
/// `sole` when that reason was its only one.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CandidateTally {
    pub candidates: usize,
    pub selected: usize,
    /// Not selected, but training continued from it (the best safe candidate of a flat epoch).
    pub continued: usize,
    /// Candidates with this reason (among others).
    pub reasons: std::collections::BTreeMap<String, usize>,
    /// Candidates with this reason only.
    pub sole: std::collections::BTreeMap<String, usize>,
    pub by_focus: std::collections::BTreeMap<String, FocusTally>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FocusTally {
    pub candidates: usize,
    pub selected: usize,
    pub continued: usize,
    pub reasons: std::collections::BTreeMap<String, usize>,
}

impl CandidateTally {
    pub fn of(epochs: &[EpochRecord]) -> Self {
        let mut t = CandidateTally::default();
        t.add(epochs);
        t
    }
    /// Adds the final candidate records of these epochs.
    pub fn add(&mut self, epochs: &[EpochRecord]) {
        for c in epochs.iter().flat_map(|e| &e.candidates) {
            let f = self.by_focus.entry(c.focus.clone()).or_default();
            self.candidates += 1;
            f.candidates += 1;
            self.selected += usize::from(c.selected);
            f.selected += usize::from(c.selected);
            self.continued += usize::from(c.continued);
            f.continued += usize::from(c.continued);
            for r in &c.rejections {
                *self.reasons.entry(r.name().into()).or_default() += 1;
                *f.reasons.entry(r.name().into()).or_default() += 1;
            }
            if let [only] = c.rejections[..] {
                *self.sole.entry(only.name().into()).or_default() += 1;
            }
        }
    }
    pub fn merge(&mut self, other: &CandidateTally) {
        self.candidates += other.candidates;
        self.selected += other.selected;
        self.continued += other.continued;
        for (k, v) in &other.reasons {
            *self.reasons.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &other.sole {
            *self.sole.entry(k.clone()).or_default() += v;
        }
        for (focus, o) in &other.by_focus {
            let f = self.by_focus.entry(focus.clone()).or_default();
            f.candidates += o.candidates;
            f.selected += o.selected;
            f.continued += o.continued;
            for (k, v) in &o.reasons {
                *f.reasons.entry(k.clone()).or_default() += v;
            }
        }
    }
    /// Candidates with this reason.
    pub fn count(&self, r: Rejection) -> usize {
        self.reasons.get(r.name()).copied().unwrap_or(0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub epoch: u32,
    /// The part trained ("dense", "transformer", "interconnect", "joint"), or "blend:<part>"
    /// for a weight-blended candidate.
    pub focus: String,
    /// Mean training loss of the candidate's epoch (0 for a blend: it trains no steps).
    pub train_loss: f32,
    pub score: Score,
    pub eligible: bool,
    /// Why the candidate was not eligible, or did not improve.
    pub reasons: Vec<String>,
    #[serde(default)]
    pub rejections: Vec<Rejection>,
    pub selected: bool,
    pub step: u64,
    /// Not accepted, but training continued from it (the best safe candidate of a flat epoch).
    #[serde(default)]
    pub continued: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpochRecord {
    pub epoch: u32,
    pub candidates: Vec<CandidateRecord>,
    pub selected: Option<String>,
    pub accepted_step: u64,
    pub accepted_score: Score,
    /// A flat epoch: the candidate training continued from (not accepted).
    #[serde(default)]
    pub continued: Option<String>,
    /// No candidate was safe: the working model was restored to the accepted model.
    #[serde(default)]
    pub restored: bool,
    /// Epochs in a row without an accepted candidate, this one included.
    #[serde(default)]
    pub flat_epochs: u32,
    /// A few held-out replies of the accepted model (prompt, expected, reply).
    pub sample_replies: Vec<(String, String, String)>,
    /// The selected candidate was accepted by `override_first_epoch` despite the safeguards.
    #[serde(default)]
    pub overridden: bool,
    /// Probation (`probation_epochs`): what happened to the branch this epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probation: Option<Probation>,
}

/// Words added at the start of a session: what the model answered just before and just after
/// they were added (right, asked), with no training between, so a loss here comes from the new
/// words alone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrowthCheck {
    pub words: usize,
    /// The questions the session judges by.
    pub judged_before: (usize, usize),
    pub judged_after: (usize, usize),
    /// The earlier lessons' answer check.
    pub earlier_before: (usize, usize),
    pub earlier_after: (usize, usize),
    /// The earlier-lesson answers are guarded as they were before (`guard_before_growth`).
    pub guarded_before: bool,
}

/// One epoch of probation (see `SessionConfig::probation_epochs`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Probation {
    /// Training continues from a rejected candidate: its focus, the guards it failed (names
    /// and reasons), its probation epoch (1 = the branch starts) and the limit, and the safe
    /// but flat candidate the flat-epoch rule would otherwise have continued from (probation
    /// takes priority over it since 0.8.6).
    Continued {
        focus: String,
        epoch_in_branch: u32,
        rejections: Vec<Rejection>,
        reasons: Vec<String>,
        #[serde(default)]
        limit: u32,
        #[serde(default)]
        instead_of_flat: Option<String>,
    },
    /// A candidate of the branch passed every guard and was accepted, after this many epochs.
    Promoted { epochs: u32 },
    /// The branch reached its limit (or its best candidate's loss was not finite) without an
    /// accepted candidate: training restarts from the accepted model.
    Discarded { epochs: u32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Started {
        train_units: usize,
        heldout_units: usize,
        rehearsal_units: usize,
        new_words: usize,
        baseline: Score,
        /// How much room replies to the held-out questions get (answer lengths in tokens with
        /// the model's vocabulary, reply limits, answers that cannot fit).
        #[serde(default)]
        replies: super::eval::ReplyRoom,
        /// Words added at the start, and the answers just before and just after.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        growth: Option<GrowthCheck>,
    },
    Batch {
        epoch: u32,
        focus: String,
        batch: usize,
        batches: usize,
        loss: f32,
        learning_rate: f32,
    },
    Candidate(CandidateRecord),
    Epoch(EpochRecord),
    /// The learning rate was halved (`backoff`): `scale` is the share of the set rate now used.
    LearningRateLowered {
        epoch: u32,
        scale: f32,
        reason: String,
    },
    /// An accepted epoch's model written to the checkpoint folder (or why it could not be).
    Checkpoint {
        epoch: u32,
        step: u64,
        file: String,
        error: Option<String>,
    },
    /// Rewards (Stage 1): what the ledger recorded at the session start or end.
    Rewarded {
        /// "start" or "end".
        when: String,
        /// Taught units (and earlier lessons' units) asked, and how many were right.
        quizzed: usize,
        quiz_right: usize,
        /// Credit earned so far in this session.
        progress: f32,
        retention: f32,
        generalization: f32,
        /// Diagnostic only.
        relevance: f32,
        flags: Vec<super::reward::Flag>,
    },
    /// Reward-guided training (Stage 2): how this epoch's units were scheduled and why.
    Emphasis {
        epoch: u32,
        plan: super::reward::EmphasisPlan,
    },
    Finished {
        accepted: bool,
        epochs_run: u32,
        stopped: bool,
    },
}

pub struct SessionResult {
    /// The accepted model: the input model unchanged when no candidate was ever selected.
    pub model: Bicameral,
    pub learned: bool,
    pub baseline: Score,
    pub final_score: Score,
    pub epochs: Vec<EpochRecord>,
    pub stopped: bool,
}

struct Probes<'a> {
    heldout_items: Vec<&'a Item>,
    heldout_rows: Vec<Row>,
    panel_items: Vec<&'a Item>,
    retention_rows: Vec<Row>,
}

pub(super) fn rows_of(model: &Bicameral, items: &[&Item], rng: &mut Rng) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    for it in items {
        if let Some(r) = encode_item(&model.tokenizer, &it.segments, model.config.max_tokens, rng)? {
            rows.push(r);
        }
    }
    Ok(rows)
}

/// The parts of a candidate's score the GPU computes in GPU training (before the CPU writes
/// out the remaining answers): the losses, and the answers it confirms.
struct GpuParts {
    heldout_loss: f32,
    without_dense_loss: Option<f32>,
    retention_loss: Option<f32>,
    heldout: Confirmed,
    panel: Confirmed,
}

/// The losses (the same values as on the CPU) and the confirmed answers of a candidate, with
/// the weights on `trainer`'s GPU; `None` when training is not on the GPU or GPU checks are off
/// ([`super::set_gpu_checks`]).
fn gpu_parts(
    trainer: &super::Trainer,
    model: &Bicameral,
    p: &Probes,
    dense: bool,
) -> Result<Option<GpuParts>, String> {
    if !super::gpu_checks() || !trainer.on_gpu() {
        return Ok(None);
    }
    let loss = |rows: &[Row], abl: Ablation| -> Result<f32, String> {
        if rows.is_empty() {
            return batch_loss_and_gradients_loss(model, rows, abl);
        }
        trainer.gpu_loss(model, rows, abl).unwrap_or_else(|| batch_loss_and_gradients_loss(model, rows, abl))
    };
    Ok(Some(GpuParts {
        heldout_loss: loss(&p.heldout_rows, Ablation::default())?,
        without_dense_loss: if dense {
            Some(loss(&p.heldout_rows, Ablation { no_afferent: true, ..Ablation::default() })?)
        } else {
            None
        },
        retention_loss: if p.retention_rows.is_empty() {
            None
        } else {
            Some(loss(&p.retention_rows, Ablation::default())?)
        },
        heldout: super::eval::confirm(trainer, model, &p.heldout_items)?,
        panel: super::eval::confirm(trainer, model, &p.panel_items)?,
    }))
}

fn score(model: &Bicameral, p: &Probes, dense: bool) -> Result<(Score, Report, Report), String> {
    score_with(model, p, dense, None)
}

/// [`score`] with the parts the GPU already computed (the same results).
fn score_with(
    model: &Bicameral,
    p: &Probes,
    dense: bool,
    gpu: Option<GpuParts>,
) -> Result<(Score, Report, Report), String> {
    let abl = Ablation::default();
    let (heldout_loss, without_dense_loss, retention_loss, heldout, panel) = match gpu {
        Some(g) => (g.heldout_loss, g.without_dense_loss, g.retention_loss, g.heldout, g.panel),
        None => {
            let heldout_loss = batch_loss_and_gradients_loss(model, &p.heldout_rows, abl)?;
            let without_dense_loss = if dense {
                let cut = Ablation { no_afferent: true, ..Ablation::default() };
                Some(batch_loss_and_gradients_loss(model, &p.heldout_rows, cut)?)
            } else {
                None
            };
            let retention_loss = if p.retention_rows.is_empty() {
                None
            } else {
                Some(batch_loss_and_gradients_loss(model, &p.retention_rows, abl)?)
            };
            (heldout_loss, without_dense_loss, retention_loss, Confirmed::default(), Confirmed::default())
        }
    };
    let dense_reliance = without_dense_loss.map(|l| l - heldout_loss);
    let report = evaluate_confirmed(model, &p.heldout_items, abl, &heldout)?;
    let panel = evaluate_confirmed(model, &p.panel_items, abl, &panel)?;
    let overall = report.overall();
    Ok((
        Score {
            heldout_loss,
            exact_correct: overall.correct,
            exact_total: overall.total,
            prefers_supported: report.prefers_supported.correct,
            prefers_total: report.prefers_supported.total,
            retention_loss,
            panel_correct: panel.overall().correct,
            panel_total: panel.overall().total,
            by_kind: report.exact.iter().map(|(k, v)| (k.clone(), (v.correct, v.total))).collect(),
            dense_reliance,
            without_dense_loss,
            by_lesson: by_lesson(&p.heldout_items, &report),
            cut_off: report.cut_off,
        },
        report,
        panel,
    ))
}

/// One lesson's score on its own held-out questions (all of them): teacher-forced loss and
/// exact answers. Used for lessons taught together, where the session scores the group.
pub(super) fn lesson_score(model: &Bicameral, items: &[Item]) -> Result<Score, String> {
    let held: Vec<&Item> = items.iter().filter(|i| i.held_out).collect();
    let rows = rows_of(model, &held, &mut Rng::new(0x0005_C04E))?;
    let heldout_loss =
        if rows.is_empty() { 0. } else { batch_loss_and_gradients_loss(model, &rows, Ablation::default())? };
    let report = evaluate(model, &held, Ablation::default())?;
    let overall = report.overall();
    Ok(Score {
        heldout_loss,
        exact_correct: overall.correct,
        exact_total: overall.total,
        prefers_supported: report.prefers_supported.correct,
        prefers_total: report.prefers_supported.total,
        retention_loss: None,
        panel_correct: 0,
        panel_total: 0,
        by_kind: report.exact.iter().map(|(k, v)| (k.clone(), (v.correct, v.total))).collect(),
        dense_reliance: None,
        without_dense_loss: None,
        by_lesson: by_lesson(&held, &report),
        cut_off: report.cut_off,
    })
}

/// Teacher-forced mean loss of rows (no gradient is applied).
pub(super) fn batch_loss_and_gradients_loss(
    model: &Bicameral,
    rows: &[Row],
    abl: Ablation,
) -> Result<f32, String> {
    // Forward passes only: the same mean as the training loss, without gradients.
    super::batch_loss(model, rows, abl)
}

/// Flat epochs (with worse candidates, before any acceptance) before the learning rate halves.
pub const BACKOFF_EPOCHS: u32 = 2;
/// The learning rate halves at most this many times in a session.
pub const BACKOFF_TIMES: u32 = 3;

/// Accepted checkpoints kept in a checkpoint folder (the newest).
pub const CHECKPOINTS_KEPT: usize = 3;

/// Writes `model` as `accepted-step-<step>-epoch-<epoch>.nnova2` in `dir` (atomically: a
/// temporary file renamed into place, so a power cut leaves either the old files or the new
/// one), keeps the newest [`CHECKPOINTS_KEPT`] of them and a README saying how to recover.
/// Where a session was when a checkpoint was written (saved beside it as `<name>.json`), so an
/// interrupted session can be continued with the same learning-rate schedule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub epoch: u32,
    pub epochs: u32,
    /// The learning rate the session was given (its peak).
    pub learning_rate: f32,
    /// The learning rate of the epoch's last batch, and the back-off share in force.
    pub learning_rate_now: f32,
    pub lr_scale: f32,
    pub batch_size: usize,
    /// Training units of the session (to recognise the same lessons).
    pub train_units: usize,
}

/// The information saved beside a checkpoint file, if any.
pub fn checkpoint_info(path: &std::path::Path) -> Option<CheckpointInfo> {
    serde_json::from_slice(&std::fs::read(path.with_extension("json")).ok()?).ok()
}

pub fn save_checkpoint(
    dir: &std::path::Path,
    model: &Bicameral,
    info: &CheckpointInfo,
) -> Result<std::path::PathBuf, String> {
    let epoch = info.epoch;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let readme = dir.join("README.txt");
    if !readme.exists() {
        let _ = std::fs::write(&readme, CHECKPOINT_README);
    }
    let path = dir.join(format!("accepted-step-{:09}-epoch-{epoch:02}.nnova2", model.step));
    let tmp = dir.join(".writing.tmp");
    let bytes = model.to_bytes()?;
    std::fs::write(&tmp, &bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
    // The information first: a checkpoint file never appears without it.
    let json = serde_json::to_vec_pretty(info).map_err(|e| e.to_string())?;
    std::fs::write(path.with_extension("json"), json).map_err(|e| format!("{}: {e}", path.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut saved = checkpoint_files(dir);
    while saved.len() > CHECKPOINTS_KEPT {
        let old = saved.remove(0);
        let _ = std::fs::remove_file(old.with_extension("json"));
        let _ = std::fs::remove_file(old);
    }
    Ok(path)
}

/// The accepted checkpoints in `dir`, oldest first (by training step).
pub fn checkpoint_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map(|r| {
            r.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("accepted-step-") && n.ends_with(".nnova2"))
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

const CHECKPOINT_README: &str = "Accepted checkpoints of the training session in progress\n\
\n\
Yendory writes the model here each time an epoch is accepted, and saves the model itself\n\
(model.nnova2 in the folder above) when the session ends. Once the model is saved, these files\n\
are removed: anything in this folder is newer than model.nnova2.\n\
\n\
After a power cut or crash during training, either:\n\
\n\
A. Open Yendory Studio and the model, then press Restore (newest) on the Models page. The old\n\
model file is kept as model-before-restore.nnova2.\n\
\n\
B. By hand, with Studio closed: copy the newest file here (the highest step number) to the\n\
folder above, rename it to model.nnova2 (replacing the old one; keep a copy if you like), then\n\
delete this checkpoints folder. Studio does not start new training while checkpoints wait here.\n\
\n\
Nova Neuro models use the same file name (model.nnova2); the file's header says which kind it is.\n";

fn correct_set(r: &Report) -> Vec<bool> {
    r.outcomes.iter().map(|o| o.correct).collect()
}

/// Teaches one lesson (its items) to `model`. `archive` holds the items of earlier lessons for
/// rehearsal and retention checks. `observer` receives every event; `stop` ends the session at
/// the next batch boundary (the epoch in progress is discarded).
pub fn teach(
    model: &Bicameral,
    lesson: &[Item],
    archive: &[Item],
    cfg: &SessionConfig,
    observer: &mut dyn FnMut(&Event),
    stop: &AtomicBool,
) -> Result<SessionResult, String> {
    teach_with(model, lesson, None, archive, cfg, observer, stop)
}

/// The best candidate of an epoch so far: index, model, score, correct answers, report.
type Best = (usize, Bicameral, Score, Vec<bool>, Report);

/// A candidate's three reports: the score, the held-out replies and the earlier-lesson panel.
type Scored = (Score, Report, Report);

/// A candidate being scored on a CPU thread while the GPU trains the next one.
struct Pending<'s> {
    ci: usize,
    focus: Focus,
    train_loss: f32,
    scoring: std::thread::ScopedJoinHandle<'s, (Bicameral, Result<Scored, String>)>,
}

/// The safeguards on the earlier lessons for one candidate: their held-out loss `(now, at the
/// session start)` against the allowance, and the earlier-lesson answers the baseline gave that
/// are wrong now (`lost_panel`), against those newly right among the same questions
/// (`regained_panel`) or the answers gained in the lessons taught (`gained_here`), as the
/// settings say. Returns each rejection with its reason.
fn earlier_lessons_check(
    cfg: &SessionConfig,
    loss: Option<(f32, f32)>,
    lost_panel: usize,
    regained_panel: usize,
    gained_here: usize,
) -> Vec<(Rejection, String)> {
    let mut out = Vec::new();
    // `earlier_by_answers` (with the larger check): the earlier lessons are judged by their
    // answers, and their loss only by the cap.
    let by_answers = cfg.earlier_by_answers && cfg.retention_per_lesson.is_some();
    if let Some((now, before)) = loss {
        let allowed = match cfg.retention_relative {
            Some(share) => before * share,
            None => cfg.retention_loss_tolerance,
        };
        let allowed = if by_answers { allowed.max(cfg.earlier_loss_cap) } else { allowed };
        if now > before + allowed {
            out.push((
                Rejection::EarlierLessonsLoss,
                match cfg.retention_relative {
                    Some(share) if !by_answers => format!(
                        "earlier lessons' held-out loss {now:.4} exceeds {before:.4} + {:.0}%",
                        share * 100.
                    ),
                    _ => format!("earlier lessons' held-out loss {now:.4} exceeds {before:.4} + {allowed}"),
                },
            ));
        }
    }
    let panel_net = cfg.earlier_panel_net || by_answers;
    let panel_ok =
        (cfg.earlier_answers_net && lost_panel <= gained_here) || (panel_net && lost_panel <= regained_panel);
    if lost_panel > 0 && !panel_ok {
        out.push((
            Rejection::EarlierAnswersLost,
            if panel_net {
                format!(
                    "lost {lost_panel} earlier-lesson answer(s) the baseline produced and got {regained_panel} of the others right (net -{})",
                    lost_panel - regained_panel
                )
            } else if cfg.earlier_answers_net {
                format!(
                    "lost {lost_panel} earlier-lesson answer(s) the baseline produced, more than the {gained_here} gained in this lesson"
                )
            } else {
                format!("lost {lost_panel} earlier-lesson answer(s) the baseline produced")
            },
        ));
    }
    out
}

/// What a candidate is judged against during one epoch.
struct Judge<'a> {
    cfg: &'a SessionConfig,
    epoch: u32,
    baseline: &'a Score,
    base_panel_correct: &'a [bool],
    /// `override_first_epoch` in the session's first epoch: the best candidate so far, whatever
    /// the safeguards say.
    fallback: Option<&'a std::cell::RefCell<Option<Best>>>,
}

impl Judge<'_> {
    /// Judges a scored candidate: records why it is or is not eligible, reports it, and keeps
    /// it as the epoch's best when it improves on the accepted model and on earlier candidates.
    #[allow(clippy::too_many_arguments)]
    fn judge(
        &self,
        ci: usize,
        label: &str,
        train_loss: f32,
        cand: Bicameral,
        (s, report, panel): Scored,
        accepted_score: &Score,
        accepted_correct: &[bool],
        best: &mut Option<Best>,
        plateau: &mut Option<Best>,
        records: &mut Vec<CandidateRecord>,
        observer: &mut dyn FnMut(&Event),
        promising: Option<&mut Option<(usize, Bicameral, Score)>>,
    ) {
        // Weight blending: remember the most promising candidate (most held-out answers, then
        // lowest held-out loss).
        if let Some(p) = promising {
            let better = s.heldout_loss.is_finite()
                && p.as_ref().map_or(true, |(_, _, b)| {
                    s.exact_correct > b.exact_correct
                        || (s.exact_correct == b.exact_correct && s.heldout_loss < b.heldout_loss)
                });
            if better {
                *p = Some((ci, cand.clone(), s.clone()));
            }
        }
        let (cfg, epoch, baseline, base_panel_correct) =
            (self.cfg, self.epoch, self.baseline, self.base_panel_correct);
        let mut reasons = Vec::new();
        let mut rejections = Vec::new();
        let tol = cfg.heldout_loss_tolerance;
        if !s.heldout_loss.is_finite()
            || s.heldout_loss > baseline.heldout_loss + tol
            || s.heldout_loss > accepted_score.heldout_loss + tol
        {
            rejections.push(Rejection::HeldoutLossRose);
        }
        if !s.heldout_loss.is_finite() {
            reasons.push("held-out loss is not finite".to_string());
        }
        if s.heldout_loss > baseline.heldout_loss + tol {
            reasons.push(format!(
                "held-out loss {:.4} exceeds the lesson-start {:.4} + {tol}",
                s.heldout_loss, baseline.heldout_loss
            ));
        }
        if s.heldout_loss > accepted_score.heldout_loss + tol {
            reasons.push(format!(
                "held-out loss {:.4} exceeds the accepted {:.4} + {tol}",
                s.heldout_loss, accepted_score.heldout_loss
            ));
        }
        let panel_now = correct_set(&panel);
        let lost_panel = base_panel_correct.iter().zip(&panel_now).filter(|(b, n)| **b && !**n).count();
        let regained_panel = base_panel_correct.iter().zip(&panel_now).filter(|(b, n)| !**b && **n).count();
        let now_correct = correct_set(&report);
        let gained_here = accepted_correct.iter().zip(&now_correct).filter(|(b, n)| !**b && **n).count();
        for (r, why) in earlier_lessons_check(
            cfg,
            s.retention_loss.zip(baseline.retention_loss),
            lost_panel,
            regained_panel,
            gained_here,
        ) {
            rejections.push(r);
            reasons.push(why);
        }
        if cfg.protect_correct_answers {
            let lost = accepted_correct.iter().zip(&now_correct).filter(|(b, n)| **b && !**n).count();
            let gained = accepted_correct.iter().zip(&now_correct).filter(|(b, n)| !**b && **n).count();
            if cfg.protect_answers_net {
                if lost > gained {
                    rejections.push(Rejection::AnswersLost);
                    reasons.push(format!(
                        "lost {lost} held-out answer(s) the accepted model produced and gained {gained} (net -{})",
                        lost - gained
                    ));
                }
            } else if lost > 0 {
                rejections.push(Rejection::AnswersLost);
                reasons.push(format!("lost {lost} held-out answer(s) the accepted model produced"));
            }
        }
        // Bicameral education: keep the model using Dense. A candidate whose measured Dense
        // contribution falls below the kept share of the accepted model's is rejected.
        if let (Some(now), Some(before)) = (s.dense_reliance, accepted_score.dense_reliance) {
            if before > 0.01 && now < cfg.keep_dense_reliance * before {
                rejections.push(Rejection::DenseUnused);
                reasons.push(format!(
                    "Dense's contribution fell from {before:.4} to {now:.4} (held-out loss rise with Dense cut)"
                ));
            }
        }
        let eligible = reasons.is_empty();
        let improves = s.exact_correct > accepted_score.exact_correct
            || (s.exact_correct == accepted_score.exact_correct
                && s.heldout_loss < accepted_score.heldout_loss - cfg.min_loss_improvement);
        if eligible && !improves {
            rejections.push(Rejection::NoImprovement);
            reasons.push(format!(
                "no meaningful improvement (same or fewer exact answers, held-out loss not {} lower)",
                cfg.min_loss_improvement
            ));
        }
        let better_than_best = best.as_ref().map_or(true, |(_, _, b, _, _)| {
            s.exact_correct > b.exact_correct
                || (s.exact_correct == b.exact_correct && s.heldout_loss < b.heldout_loss)
        });
        // Safe, just not better enough yet: training may continue from it.
        let safe = rejections == [Rejection::NoImprovement];
        let record = CandidateRecord {
            epoch,
            focus: label.into(),
            train_loss,
            score: s.clone(),
            eligible,
            reasons,
            rejections,
            selected: false,
            step: cand.step,
            continued: false,
        };
        observer(&Event::Candidate(record.clone()));
        records.push(record);
        if let Some(f) = self.fallback {
            let mut f = f.borrow_mut();
            let better = s.heldout_loss.is_finite()
                && f.as_ref().map_or(true, |(_, _, b, _, _)| {
                    s.exact_correct > b.exact_correct
                        || (s.exact_correct == b.exact_correct && s.heldout_loss < b.heldout_loss)
                });
            if better {
                *f = Some((ci, cand.clone(), s.clone(), now_correct.clone(), report.clone()));
            }
        }
        let better_than_plateau = plateau.as_ref().map_or(true, |(_, _, b, _, _)| {
            s.exact_correct > b.exact_correct
                || (s.exact_correct == b.exact_correct && s.heldout_loss < b.heldout_loss)
        });
        // Rejected only for forgetting earlier lessons (with `recover_earlier`): training may
        // continue from it when no candidate is safe, so rehearsal can restore those lessons.
        let earlier =
            |r: &Rejection| matches!(r, Rejection::EarlierLessonsLoss | Rejection::EarlierAnswersLost);
        let recoverable = cfg.recover_earlier
            && records.last().is_some_and(|r| {
                r.rejections.iter().any(earlier)
                    && r.rejections.iter().all(|x| earlier(x) || *x == Rejection::NoImprovement)
            });
        let plateau_is_safe = plateau.as_ref().is_some_and(|(pc, ..)| {
            records.get(*pc).is_some_and(|r| r.rejections == [Rejection::NoImprovement])
        });
        let closer = plateau.as_ref().map_or(true, |(_, _, b, _, _)| s.heldout_loss < b.heldout_loss);
        if eligible && improves && better_than_best {
            *best = Some((ci, cand, s, now_correct, report));
        } else if (safe && (better_than_plateau || !plateau_is_safe))
            || (recoverable && !plateau_is_safe && closer && s.heldout_loss.is_finite())
        {
            *plateau = Some((ci, cand, s, now_correct, report));
        }
    }
}

/// Like [`teach`], but trains on `practice` (a chosen selection of the lesson's training units,
/// repeats allowed) instead of all of them. Held-out probes and vocabulary still come from the
/// whole lesson, so scores stay comparable between attempts.
pub fn teach_with(
    model: &Bicameral,
    lesson: &[Item],
    practice: Option<&[Item]>,
    archive: &[Item],
    cfg: &SessionConfig,
    observer: &mut dyn FnMut(&Event),
    stop: &AtomicBool,
) -> Result<SessionResult, String> {
    teach_rewarded(model, lesson, practice, archive, cfg, None, observer, stop)
}

/// What the ledger saw of the session so far.
fn rewarded_event(r: &Rewards, start: &Totals, when: &str, quiz: (usize, usize)) -> Event {
    let t = &r.ledger.totals;
    Event::Rewarded {
        when: when.into(),
        quizzed: quiz.0,
        quiz_right: quiz.1,
        progress: t.progress - start.progress,
        retention: t.retention - start.retention,
        generalization: t.generalization - start.generalization,
        relevance: t.relevance - start.relevance,
        flags: if when == "end" { r.ledger.flags() } else { Vec::new() },
    }
}

/// Picks an index with probability proportional to `cumulative` weights (a running sum).
fn weighted_pick(cumulative: &[f32], rng: &mut Rng) -> usize {
    let total = cumulative.last().copied().unwrap_or(0.);
    let x = rng.unit() as f32 * total;
    cumulative.partition_point(|&c| c <= x).min(cumulative.len().saturating_sub(1))
}

/// [`teach_with`] with rewards. With `rewards` recording (Stage 1), Nova's answers to the
/// lesson's held-out questions (start and end) and to a quiz of its taught units and of earlier
/// lessons' units (start) are graded into the ledger; nothing about training changes. With
/// reward-guided training on (Stage 2), each epoch's units are scheduled by their emphasis
/// (`reward::plan_epoch`) and rehearsal of earlier lessons favours what is being forgotten; the
/// targets, the candidates and every acceptance rule stay exactly as in ordinary training.
#[allow(clippy::too_many_arguments)]
pub fn teach_rewarded(
    model: &Bicameral,
    lesson: &[Item],
    practice: Option<&[Item]>,
    archive: &[Item],
    cfg: &SessionConfig,
    mut rewards: Option<&mut Rewards>,
    observer: &mut dyn FnMut(&Event),
    stop: &AtomicBool,
) -> Result<SessionResult, String> {
    if cfg.epochs == 0 || cfg.batch_size == 0 || !(cfg.learning_rate > 0. && cfg.learning_rate.is_finite()) {
        return Err("Choose at least one epoch, a batch size and a positive learning rate".into());
    }
    let train: Vec<&Item> = practice
        .unwrap_or(lesson)
        .iter()
        .filter(|it| it.trainable() && (cfg.strategy == Strategy::Standard || it.kind != Kind::Prose))
        .collect();
    let heldout: Vec<&Item> = lesson.iter().filter(|it| it.held_out && it.kind != Kind::Prose).collect();
    if train.is_empty() || heldout.is_empty() {
        return Err(
            "The lesson needs training units and held-out units (about 10% are held out by hash)".into()
        );
    }
    let mut start = model.clone();
    let mut new_words = 0;
    if cfg.grow_vocabulary {
        let limit = cfg.word_limit.unwrap_or(start.word_capacity()).max(start.tokenizer.words().len());
        // Words of the lessons being taught; saved earlier lessons trained alongside them are left
        // out (see `words_from_new_lessons`).
        let earlier: std::collections::HashSet<&str> = archive.iter().map(|i| i.source.as_str()).collect();
        let taught: Vec<Item> = if cfg.words_from_new_lessons {
            lesson.iter().filter(|i| !earlier.contains(i.source.as_str())).cloned().collect()
        } else {
            lesson.to_vec()
        };
        let words = grow_vocabulary(start.tokenizer.words(), &taught, limit, cfg.min_word_count);
        new_words = words.len() - start.tokenizer.words().len();
        if words.len() > start.word_capacity() {
            start = start.with_word_capacity(words.len(), cfg.seed ^ 0x5EED)?;
        }
        start.tokenizer.set_words(words);
    }

    if !(0. ..=1.).contains(&cfg.blend) {
        return Err("The blend share must be between 0 and 1".into());
    }
    if !(0. ..1.).contains(&cfg.dropout) {
        return Err("Dropout must be at least 0 and below 1".into());
    }
    if !(0. ..1.).contains(&cfg.reading_share)
        || (cfg.reading_share > 0. && cfg.reading_share + cfg.rehearsal_share >= 1.)
    {
        return Err("The reading and rehearsal shares must leave room for the lesson".into());
    }
    let reading = match (&cfg.reading, cfg.reading_share > 0.) {
        (Some(texts), true) => Some(ReadingRows::new(&start, texts)?).filter(|r| !r.docs.is_empty()),
        (None, true) => return Err("General-reading rehearsal needs reading text".into()),
        _ => None,
    };
    let mut rng = Rng::new(cfg.seed);
    let probe_items = sample_per_kind(&heldout, cfg.eval_limit, cfg.seed);
    let archive_train: Vec<&Item> = archive.iter().filter(|it| it.trainable()).collect();
    let archive_heldout: Vec<&Item> =
        archive.iter().filter(|it| it.held_out && it.kind != Kind::Prose).collect();
    let kinds = archive_heldout.iter().map(|it| it.kind).collect::<std::collections::BTreeSet<_>>().len();
    let panel_items = match cfg.retention_per_lesson {
        // Up to `n` questions of every earlier lesson, spread over its kinds.
        Some(n) => {
            let mut by_lesson: std::collections::BTreeMap<&str, Vec<&Item>> = Default::default();
            for &it in &archive_heldout {
                by_lesson.entry(it.source.as_str()).or_default().push(it);
            }
            let mut out = Vec::new();
            for (i, items) in by_lesson.values().enumerate() {
                let kinds = items.iter().map(|it| it.kind).collect::<std::collections::BTreeSet<_>>().len();
                let mut pick =
                    sample_per_kind(items, Some(n.div_ceil(kinds.max(1))), cfg.seed ^ 0xA5 ^ i as u64);
                pick.truncate(n);
                out.extend(pick);
            }
            out
        }
        None => {
            let per_kind = cfg.retention_panel.div_ceil(kinds.max(1));
            let mut p = sample_per_kind(&archive_heldout, Some(per_kind), cfg.seed ^ 0xA5);
            p.truncate(cfg.retention_panel);
            p
        }
    };
    let retention_items = sample_per_kind(&archive_heldout, Some(cfg.retention_units), cfg.seed ^ 0x5A);
    let probes = Probes {
        heldout_rows: rows_of(&start, &probe_items, &mut rng)?,
        heldout_items: probe_items,
        panel_items,
        retention_rows: rows_of(&start, &retention_items, &mut rng)?,
    };

    let dense = cfg.education.is_some();
    let (baseline, base_report, base_panel) = score(&start, &probes, dense)?;
    let mut base_panel_correct = correct_set(&base_panel);
    let replies = super::eval::reply_room(&start, &heldout)?;
    // Words added: what the model answered just before and just after (no training between).
    let growth = if new_words > 0 {
        let lesson_before = evaluate(model, &probes.heldout_items, Ablation::default())?;
        let panel_before = evaluate(model, &probes.panel_items, Ablation::default())?;
        if cfg.guard_before_growth {
            base_panel_correct = correct_set(&panel_before);
        }
        let n = |r: &Report| (r.overall().correct, r.overall().total);
        Some(GrowthCheck {
            words: new_words,
            judged_before: n(&lesson_before),
            judged_after: n(&base_report),
            earlier_before: n(&panel_before),
            earlier_after: n(&base_panel),
            guarded_before: cfg.guard_before_growth,
        })
    } else {
        None
    };
    observer(&Event::Started {
        train_units: train.len(),
        heldout_units: heldout.len(),
        rehearsal_units: archive_train.len(),
        new_words,
        baseline: baseline.clone(),
        replies,
        growth,
    });

    // Rewards: the start checkpoint's answers go into the ledger (evaluation only: nothing here
    // touches the training random streams, so ordinary training is unchanged).
    rewards = rewards.filter(|r| r.config.record);
    let lesson_name =
        lesson.iter().find(|i| !i.held_out).or(lesson.first()).map(|i| i.source.clone()).unwrap_or_default();
    let mut ledger_start = Totals::default();
    let mut start_quiz = (0, 0);
    let mut quiz_rate = None;
    if let Some(r) = rewards.as_deref_mut() {
        ledger_start = r.ledger.begin_session();
        r.record_report(&probes.heldout_items, &base_report, Origin::HeldOut, start.step, true);
        start_quiz = r.quiz(&start, &train, Origin::Training, true)?;
        quiz_rate = (start_quiz.0 > 0).then(|| start_quiz.1 as f32 / start_quiz.0 as f32);
        // Earlier lessons' units: retention across checkpoints.
        let limit = r.config.quiz_limit;
        r.config.quiz_limit = limit / 4;
        let earlier = r.quiz(&start, &archive_train, Origin::Training, true);
        r.config.quiz_limit = limit;
        let earlier = earlier?;
        observer(&rewarded_event(
            r,
            &ledger_start,
            "start",
            (start_quiz.0 + earlier.0, start_quiz.1 + earlier.1),
        ));
    }
    let weighted = rewards.as_ref().is_some_and(|r| r.weighted());

    let foci: Vec<Focus> = if let Some(stage) = cfg.education {
        stage.foci(cfg.foundation)
    } else if cfg.foundation {
        vec![Focus::Right, Focus::Left, Focus::Interconnect, Focus::Joint]
    } else {
        vec![Focus::Joint]
    };
    let batches_per_epoch = train.len().div_ceil(cfg.batch_size);
    let total_steps = batches_per_epoch as u64 * cfg.epochs as u64;
    let warmup = (total_steps / 10).clamp(1, 200);

    let mut accepted = start.clone();
    // Where training continues from: the accepted model, or the best safe candidate of the last
    // flat epochs (so improvement can build up until it is large enough to be accepted).
    let mut working = start.clone();
    let mut accepted_score = baseline.clone();
    let mut accepted_correct = correct_set(&base_report);
    let mut accepted_report = base_report.clone();
    let mut learned = false;
    let mut epochs = Vec::new();
    let mut stale = 0;
    let mut stopped = false;
    // The earlier-lesson guards' reference (the session start, or an overridden first epoch).
    let mut guard_baseline = baseline.clone();
    // Back-off: the share of the scheduled learning rate used, and how often it was halved.
    let mut lr_scale =
        if cfg.resume_epochs > 0 { cfg.resume_scale.unwrap_or(1.).clamp(1e-3, 1.) } else { 1. };
    let mut backoffs = (-lr_scale.log2()).round().max(0.) as u32;
    let first_epoch = cfg.resume_epochs.min(cfg.epochs.saturating_sub(1)) + 1;
    // Probation: epochs in a row the working model has continued from a rejected candidate.
    let mut on_probation = 0u32;

    'epochs: for epoch in first_epoch..=cfg.epochs {
        let overriding = cfg.override_first_epoch && epoch == first_epoch;
        // The most promising candidate is also wanted for probation.
        let probing = cfg.probation_epochs > 0;
        let fallback: std::cell::RefCell<Option<Best>> = std::cell::RefCell::new(None);
        // One batch plan per epoch, shared by every candidate.
        let mut plan_rng = Rng::new(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(epoch as u64));
        // Stage 2: copies of each unit by emphasis (same total), rehearsal by emphasis.
        let mut rehearsal_weights: Option<Vec<f32>> = None;
        let mut order: Vec<&Item> = match rewards.as_deref_mut().filter(|_| weighted) {
            Some(r) => {
                let quizzed =
                    if epoch == 1 { start_quiz } else { r.quiz(&working, &train, Origin::Session, false)? };
                let (copies, plan) = super::reward::plan_epoch(r, &train, &mut plan_rng, quizzed);
                observer(&Event::Emphasis { epoch, plan });
                if !archive_train.is_empty() {
                    let concepts = super::reward::Concepts::of(&r.ledger, &archive_train);
                    let mut sum = 0f32;
                    rehearsal_weights = Some(
                        archive_train
                            .iter()
                            .map(|it| {
                                sum += r.ledger.emphasis(&r.config, it, &concepts).value;
                                sum
                            })
                            .collect(),
                    );
                }
                train.iter().zip(copies).flat_map(|(it, n)| std::iter::repeat(*it).take(n)).collect()
            }
            None => train.clone(),
        };
        match cfg.strategy {
            Strategy::Standard => {
                for i in (1..order.len()).rev() {
                    order.swap(i, plan_rng.below(i + 1));
                }
            }
            Strategy::Focused => {
                let size = |it: &Item| it.segments.iter().map(|s| s.text.len()).sum::<usize>();
                order.sort_by_key(|it| size(it));
                for group in order.chunks_mut(cfg.batch_size * 2) {
                    for i in (1..group.len()).rev() {
                        group.swap(i, plan_rng.below(i + 1));
                    }
                }
            }
        }
        let mut plan: Vec<Vec<Row>> = Vec::with_capacity(batches_per_epoch);
        let mut next = order.into_iter();
        // General-reading rehearsal: reading rows take the place of lesson rows. The epoch has
        // the rows it has without reading (its lesson units and, on average, the rehearsal rows
        // drawn with them), so the training budget is unchanged; the lesson units left over
        // differ between epochs because the order is shuffled every epoch.
        let mut budget = reading.as_ref().map(|_| {
            let rehearsal = if archive_train.is_empty() { 0. } else { cfg.rehearsal_share as f64 };
            (train.len() as f64 / (1. - rehearsal)).round() as usize
        });
        'plan: loop {
            let mut rows = Vec::with_capacity(cfg.batch_size);
            while rows.len() < cfg.batch_size {
                if let (Some(r), Some(left)) = (&reading, budget.as_mut()) {
                    if *left == 0 {
                        break;
                    }
                    *left -= 1;
                    if plan_rng.unit() < cfg.reading_share as f64 {
                        rows.push(r.row(&mut plan_rng));
                        continue;
                    }
                }
                let it = if !archive_train.is_empty() && plan_rng.unit() < cfg.rehearsal_share as f64 {
                    match &rehearsal_weights {
                        Some(w) => archive_train[weighted_pick(w, &mut plan_rng)],
                        None => archive_train[plan_rng.below(archive_train.len())],
                    }
                } else {
                    match next.next() {
                        Some(it) => it,
                        None => break,
                    }
                };
                if let Some(r) =
                    encode_item(&start.tokenizer, &it.segments, start.config.max_tokens, &mut plan_rng)?
                {
                    rows.push(r);
                }
            }
            if rows.is_empty() {
                break 'plan;
            }
            let last = rows.len() < cfg.batch_size;
            plan.push(rows);
            if last {
                break 'plan;
            }
        }

        let mut records = Vec::new();
        let mut best: Option<Best> = None;
        let mut plateau: Option<Best> = None;
        let blending = cfg.blend > 0.;
        let mut promising: Option<(usize, Bicameral, Score)> = None;
        // With GPU training the CPU scores each candidate while the GPU trains the next one
        // (every candidate of an epoch starts from the same working model). Candidates are
        // still judged one by one in order, so the result is the same as scoring inline.
        let interrupted = std::thread::scope(|scope| -> Result<bool, String> {
            let mut pending: Option<Pending> = None;
            for (ci, &focus) in foci.iter().enumerate() {
                let mut cand = working.clone();
                let mut trainer = super::Trainer::new(&cand);
                let mut losses = 0f64;
                for (b, rows) in plan.iter().enumerate() {
                    if stop.load(Ordering::Acquire) {
                        return Ok(true);
                    }
                    let step = (epoch as u64 - 1) * batches_per_epoch as u64 + b as u64;
                    let lr = lr_schedule(step, cfg.learning_rate, warmup, total_steps, 0.1) * lr_scale;
                    // Dropout masks depend on the epoch and batch, so every candidate of an
                    // epoch sees the same ones.
                    let abl = if cfg.dropout > 0. {
                        Ablation {
                            dropout: Some(crate::model::Dropout {
                                rate: cfg.dropout,
                                seed: cfg.seed ^ 0xD20F_0D20 ^ (step + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                                row: 0,
                            }),
                            ..Ablation::default()
                        }
                    } else {
                        Ablation::default()
                    };
                    let loss = trainer.step(&mut cand, rows, abl, lr, focus)?;
                    losses += loss as f64;
                    observer(&Event::Batch {
                        epoch,
                        focus: focus_name(focus).into(),
                        batch: b + 1,
                        batches: plan.len(),
                        loss,
                        learning_rate: lr,
                    });
                }
                trainer.sync(&mut cand)?;
                let train_loss = (losses / plan.len().max(1) as f64) as f32;
                // The losses and the answers the GPU confirms, while the CPU may still be
                // writing the previous candidate's answers.
                let gpu = gpu_parts(&trainer, &cand, &probes, dense)?;
                if let Some(p) = pending.take() {
                    let (cand, scored) = p.scoring.join().map_err(|_| "Scoring a candidate failed")?;
                    let j = Judge {
                        cfg,
                        epoch,
                        baseline: &guard_baseline,
                        base_panel_correct: &base_panel_correct,
                        fallback: (overriding || probing).then_some(&fallback),
                    };
                    j.judge(
                        p.ci,
                        focus_name(p.focus),
                        p.train_loss,
                        cand,
                        scored?,
                        &accepted_score,
                        &accepted_correct,
                        &mut best,
                        &mut plateau,
                        &mut records,
                        observer,
                        blending.then_some(&mut promising),
                    );
                }
                let probes = &probes;
                if trainer.on_gpu() {
                    // The CPU writes out the answers the GPU did not confirm.
                    let scoring = scope.spawn(move || {
                        let scored = score_with(&cand, probes, dense, gpu);
                        (cand, scored)
                    });
                    pending = Some(Pending { ci, focus, train_loss, scoring });
                } else {
                    let scored = score(&cand, probes, dense)?;
                    let j = Judge {
                        cfg,
                        epoch,
                        baseline: &guard_baseline,
                        base_panel_correct: &base_panel_correct,
                        fallback: (overriding || probing).then_some(&fallback),
                    };
                    j.judge(
                        ci,
                        focus_name(focus),
                        train_loss,
                        cand,
                        scored,
                        &accepted_score,
                        &accepted_correct,
                        &mut best,
                        &mut plateau,
                        &mut records,
                        observer,
                        blending.then_some(&mut promising),
                    );
                }
            }
            if let Some(p) = pending.take() {
                let (cand, scored) = p.scoring.join().map_err(|_| "Scoring a candidate failed")?;
                let j = Judge {
                    cfg,
                    epoch,
                    baseline: &guard_baseline,
                    base_panel_correct: &base_panel_correct,
                    fallback: (overriding || probing).then_some(&fallback),
                };
                j.judge(
                    p.ci,
                    focus_name(p.focus),
                    p.train_loss,
                    cand,
                    scored?,
                    &accepted_score,
                    &accepted_correct,
                    &mut best,
                    &mut plateau,
                    &mut records,
                    observer,
                    blending.then_some(&mut promising),
                );
            }
            Ok(false)
        })?;
        if interrupted {
            stopped = true;
            break 'epochs;
        }
        // Weight blending: one more candidate, part of the way from the epoch's starting model
        // to its most promising candidate, judged like the others (it trains no steps).
        if let Some((pci, cand, _)) = promising.take() {
            let blended = working.blend_toward(&cand, cfg.blend)?;
            let scored = score(&blended, &probes, dense)?;
            let label = format!("blend:{}", records[pci].focus);
            let j = Judge {
                cfg,
                epoch,
                baseline: &guard_baseline,
                base_panel_correct: &base_panel_correct,
                fallback: (overriding || probing).then_some(&fallback),
            };
            j.judge(
                records.len(),
                &label,
                0.,
                blended,
                scored,
                &accepted_score,
                &accepted_correct,
                &mut best,
                &mut plateau,
                &mut records,
                observer,
                None,
            );
        }

        // Override (first epoch): its best candidate is accepted despite the safeguards.
        let mut overridden = false;
        if best.is_none() && overriding {
            if let Some(fb) = fallback.borrow_mut().take() {
                records[fb.0].reasons.push("accepted anyway: first-epoch override".into());
                best = Some(fb);
                overridden = true;
            }
        }
        let (mut continued, mut restored) = (None, false);
        let mut probation = None;
        if best.is_none() {
            let promising = fallback.borrow_mut().take().filter(|b| b.2.heldout_loss.is_finite());
            match (plateau.take(), promising) {
                // Probation, when on, comes first: the branch continues from the most promising
                // candidate even when a safe but flat one exists.
                (flat, Some((ci, cand, ..))) if probing && on_probation < cfg.probation_epochs => {
                    on_probation += 1;
                    records[ci].continued = true;
                    continued = Some(records[ci].focus.clone());
                    probation = Some(Probation::Continued {
                        focus: records[ci].focus.clone(),
                        epoch_in_branch: on_probation,
                        rejections: records[ci].rejections.clone(),
                        reasons: records[ci].reasons.clone(),
                        limit: cfg.probation_epochs,
                        instead_of_flat: flat.map(|(fi, ..)| records[fi].focus.clone()),
                    });
                    working = cand;
                }
                // The branch reached its limit without an accepted candidate: back to the
                // accepted model (below), not on from a flat candidate.
                (Some((ci, cand, _, _, _)), _) if on_probation == 0 => {
                    records[ci].continued = true;
                    continued = Some(records[ci].focus.clone());
                    working = cand;
                }
                (_, _) => {
                    if on_probation > 0 {
                        probation = Some(Probation::Discarded { epochs: on_probation });
                        on_probation = 0;
                    }
                    restored = working.step != accepted.step;
                    working = accepted.clone();
                }
            }
        } else if on_probation > 0 {
            probation = Some(Probation::Promoted { epochs: on_probation });
            on_probation = 0;
        }
        let selected = best.map(|(ci, cand, s, correct, report)| {
            records[ci].selected = true;
            let winner = records[ci].focus.clone();
            // Candidates that improved on the accepted model but lost to this one.
            for (i, r) in records.iter_mut().enumerate() {
                if i != ci && r.rejections.is_empty() {
                    r.rejections.push(Rejection::Outranked);
                    r.reasons.push(format!("improved, but the {winner} candidate was better"));
                }
            }
            working = cand.clone();
            accepted = cand;
            accepted_score = s;
            accepted_correct = correct;
            accepted_report = report.clone();
            learned = true;
            (records[ci].focus.clone(), report)
        });
        let sample_replies = selected
            .as_ref()
            .map(|(_, report)| {
                report
                    .outcomes
                    .iter()
                    .take(3)
                    .map(|o| (o.prompt.clone(), o.expected.clone(), o.reply.clone()))
                    .collect()
            })
            .unwrap_or_default();
        if overridden {
            // The guards now protect what the overridden model knows.
            guard_baseline.retention_loss = accepted_score.retention_loss;
            guard_baseline.panel_correct = accepted_score.panel_correct;
            base_panel_correct = correct_set(&evaluate(&accepted, &probes.panel_items, Ablation::default())?);
        }
        stale = if selected.is_some() { 0 } else { stale + 1 };
        let record = EpochRecord {
            epoch,
            candidates: records,
            selected: selected.map(|(f, _)| f),
            accepted_step: accepted.step,
            accepted_score: accepted_score.clone(),
            continued,
            restored,
            flat_epochs: stale,
            sample_replies,
            overridden,
            probation,
        };
        observer(&Event::Epoch(record.clone()));
        if let Some(h) = &cfg.epoch_hook {
            (h.0)(epoch, &accepted, &working);
        }
        // Back-off: no epoch accepted yet and the last epochs' candidates were worse than the
        // accepted model: smaller steps.
        let worse = |r: &CandidateRecord| {
            r.rejections.iter().any(|x| {
                matches!(
                    x,
                    Rejection::AnswersLost | Rejection::EarlierLessonsLoss | Rejection::EarlierAnswersLost
                )
            })
        };
        if cfg.backoff
            && !learned
            && stale >= BACKOFF_EPOCHS
            && backoffs < BACKOFF_TIMES
            && record.candidates.iter().all(worse)
        {
            lr_scale *= 0.5;
            backoffs += 1;
            stale = 0;
            observer(&Event::LearningRateLowered {
                epoch,
                scale: lr_scale,
                reason: format!(
                    "{BACKOFF_EPOCHS} epochs in a row made the model worse (answers or earlier lessons lost): steps are halved"
                ),
            });
        }
        if let (Some(dir), true) = (&cfg.checkpoint_dir, record.selected.is_some()) {
            let info = CheckpointInfo {
                epoch,
                epochs: cfg.epochs,
                learning_rate: cfg.learning_rate,
                learning_rate_now: lr_schedule(
                    epoch as u64 * batches_per_epoch as u64 - 1,
                    cfg.learning_rate,
                    warmup,
                    total_steps,
                    0.1,
                ) * lr_scale,
                lr_scale,
                batch_size: cfg.batch_size,
                train_units: train.len(),
            };
            let (file, error) = match save_checkpoint(dir, &accepted, &info) {
                Ok(path) => (path.display().to_string(), None),
                Err(e) => (String::new(), Some(e)),
            };
            observer(&Event::Checkpoint { epoch, step: accepted.step, file, error });
        }
        epochs.push(record);
        if stale >= cfg.patience {
            break;
        }
    }

    if let Some(r) = rewards {
        if learned && !stopped {
            r.record_report(&probes.heldout_items, &accepted_report, Origin::HeldOut, accepted.step, true);
        }
        let name = r.model.clone();
        r.ledger.end_session(
            &ledger_start,
            &name,
            &lesson_name,
            (model.step, if learned { accepted.step } else { model.step }),
            learned,
            weighted,
            quiz_rate,
            (baseline.exact_rate(), accepted_score.exact_rate()),
            (baseline.heldout_loss, accepted_score.heldout_loss),
        );
        observer(&rewarded_event(r, &ledger_start, "end", start_quiz));
    }
    observer(&Event::Finished { accepted: learned, epochs_run: epochs.len() as u32, stopped });
    Ok(SessionResult {
        model: if learned { accepted } else { model.clone() },
        learned,
        baseline,
        final_score: accepted_score,
        epochs,
        stopped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The earlier lessons' safeguards: the loss allowance is a setting (0.02 unless changed);
    /// judging by answers needs the larger check, still rejects a lost net of answers, and
    /// rejects a loss rise above its cap whatever the answers.
    #[test]
    fn earlier_lessons_are_judged_by_the_allowance_or_by_answers() {
        let rejected = |cfg: &SessionConfig, now: f32, lost: usize, regained: usize| {
            earlier_lessons_check(cfg, Some((now, 1.12)), lost, regained, 0)
                .into_iter()
                .map(|(r, _)| r)
                .collect::<Vec<_>>()
        };
        let usual = SessionConfig::default();
        assert_eq!(usual.retention_loss_tolerance, 0.02, "the default is unchanged");
        assert!(!usual.earlier_by_answers);
        // The case from the owner's model: 1.12 -> 1.19 with the answers kept.
        assert_eq!(rejected(&usual, 1.19, 0, 0), [Rejection::EarlierLessonsLoss]);
        assert!(rejected(&usual, 1.13, 0, 0).is_empty());
        let wider = SessionConfig { retention_loss_tolerance: 0.08, ..usual.clone() };
        assert!(rejected(&wider, 1.19, 0, 0).is_empty());
        assert_eq!(rejected(&wider, 1.21, 0, 0), [Rejection::EarlierLessonsLoss]);
        // Answers are still guarded with the wider allowance: one lost answer rejects (the
        // usual check allows no loss).
        assert_eq!(rejected(&wider, 1.15, 1, 3), [Rejection::EarlierAnswersLost]);

        // By answers: without the larger check, nothing changes.
        let alone = SessionConfig { earlier_by_answers: true, ..usual.clone() };
        assert_eq!(rejected(&alone, 1.19, 0, 0), [Rejection::EarlierLessonsLoss]);
        let by_answers =
            SessionConfig { earlier_by_answers: true, retention_per_lesson: Some(4), ..usual.clone() };
        assert!(rejected(&by_answers, 1.19, 0, 0).is_empty(), "a small rise with the answers kept");
        assert!(rejected(&by_answers, 1.30, 4, 5).is_empty(), "answers judged by net change");
        assert_eq!(rejected(&by_answers, 1.30, 5, 4), [Rejection::EarlierAnswersLost]);
        assert_eq!(
            rejected(&by_answers, 1.12 + by_answers.earlier_loss_cap + 0.01, 0, 0),
            [Rejection::EarlierLessonsLoss],
            "a substantial regression is rejected whatever the answers"
        );
    }
}
