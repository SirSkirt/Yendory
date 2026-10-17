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
use super::eval::{evaluate, sample_per_kind, Report};
use super::lessons::{grow_vocabulary, Item, Kind};
use super::{batch_loss_and_gradients, encode_item, lr_schedule, Rng, Row};
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
}
fn default_pass_mark() -> f32 {
    0.5
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
            protect_answers_net: false,
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
        }
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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub epoch: u32,
    pub focus: String,
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

fn score(model: &Bicameral, p: &Probes, dense: bool) -> Result<(Score, Report, Report), String> {
    let abl = Ablation::default();
    let heldout_loss = batch_loss_and_gradients_loss(model, &p.heldout_rows, abl)?;
    let without_dense_loss = if dense {
        let cut = Ablation { no_afferent: true, ..Ablation::default() };
        Some(batch_loss_and_gradients_loss(model, &p.heldout_rows, cut)?)
    } else {
        None
    };
    let dense_reliance = without_dense_loss.map(|l| l - heldout_loss);
    let report = evaluate(model, &p.heldout_items, abl)?;
    let panel = evaluate(model, &p.panel_items, abl)?;
    let retention_loss = if p.retention_rows.is_empty() {
        None
    } else {
        Some(batch_loss_and_gradients_loss(model, &p.retention_rows, abl)?)
    };
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
        },
        report,
        panel,
    ))
}

/// Teacher-forced mean loss of rows (no gradient is applied).
pub(super) fn batch_loss_and_gradients_loss(
    model: &Bicameral,
    rows: &[Row],
    abl: Ablation,
) -> Result<f32, String> {
    if rows.is_empty() {
        return Ok(0.);
    }
    let mut total = 0f64;
    let mut weight = 0f64;
    for chunk in rows.chunks(32) {
        let (l, _) = batch_loss_and_gradients(model, chunk, abl)?;
        let w: f64 = chunk.iter().map(|r| r.mask.iter().sum::<f32>() as f64).sum();
        total += l as f64 * w.max(1.);
        weight += w.max(1.);
    }
    Ok((total / weight) as f32)
}

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

/// What a candidate is judged against during one epoch.
struct Judge<'a> {
    cfg: &'a SessionConfig,
    epoch: u32,
    baseline: &'a Score,
    base_panel_correct: &'a [bool],
}

impl Judge<'_> {
    /// Judges a scored candidate: records why it is or is not eligible, reports it, and keeps
    /// it as the epoch's best when it improves on the accepted model and on earlier candidates.
    #[allow(clippy::too_many_arguments)]
    fn judge(
        &self,
        ci: usize,
        focus: Focus,
        train_loss: f32,
        cand: Bicameral,
        (s, report, panel): Scored,
        accepted_score: &Score,
        accepted_correct: &[bool],
        best: &mut Option<Best>,
        plateau: &mut Option<Best>,
        records: &mut Vec<CandidateRecord>,
        observer: &mut dyn FnMut(&Event),
    ) {
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
        if let (Some(now), Some(before)) = (s.retention_loss, baseline.retention_loss) {
            let allowed = match cfg.retention_relative {
                Some(share) => before * share,
                None => cfg.retention_loss_tolerance,
            };
            if now > before + allowed {
                rejections.push(Rejection::EarlierLessonsLoss);
                reasons.push(match cfg.retention_relative {
                    Some(share) => format!(
                        "earlier lessons' held-out loss {now:.4} exceeds {before:.4} + {:.0}%",
                        share * 100.
                    ),
                    None => {
                        format!("earlier lessons' held-out loss {now:.4} exceeds {before:.4} + {allowed}")
                    }
                });
            }
        }
        let lost_panel =
            base_panel_correct.iter().zip(correct_set(&panel)).filter(|(b, n)| **b && !*n).count();
        if lost_panel > 0 {
            rejections.push(Rejection::EarlierAnswersLost);
            reasons.push(format!("lost {lost_panel} earlier-lesson answer(s) the baseline produced"));
        }
        let now_correct = correct_set(&report);
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
            focus: focus_name(focus).into(),
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
        let better_than_plateau = plateau.as_ref().map_or(true, |(_, _, b, _, _)| {
            s.exact_correct > b.exact_correct
                || (s.exact_correct == b.exact_correct && s.heldout_loss < b.heldout_loss)
        });
        if eligible && improves && better_than_best {
            *best = Some((ci, cand, s, now_correct, report));
        } else if safe && better_than_plateau {
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
    if cfg.epochs == 0 || cfg.batch_size == 0 || !(cfg.learning_rate > 0. && cfg.learning_rate.is_finite()) {
        return Err("Choose at least one epoch, a batch size and a positive learning rate".into());
    }
    let train: Vec<&Item> = practice
        .unwrap_or(lesson)
        .iter()
        .filter(|it| !it.held_out && (cfg.strategy == Strategy::Standard || it.kind != Kind::Prose))
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
        let words = grow_vocabulary(start.tokenizer.words(), lesson, limit, cfg.min_word_count);
        new_words = words.len() - start.tokenizer.words().len();
        if words.len() > start.word_capacity() {
            start = start.with_word_capacity(words.len(), cfg.seed ^ 0x5EED)?;
        }
        start.tokenizer.set_words(words);
    }

    let mut rng = Rng::new(cfg.seed);
    let probe_items = sample_per_kind(&heldout, cfg.eval_limit, cfg.seed);
    let archive_train: Vec<&Item> = archive.iter().filter(|it| !it.held_out).collect();
    let archive_heldout: Vec<&Item> =
        archive.iter().filter(|it| it.held_out && it.kind != Kind::Prose).collect();
    let kinds = archive_heldout.iter().map(|it| it.kind).collect::<std::collections::BTreeSet<_>>().len();
    let per_kind = cfg.retention_panel.div_ceil(kinds.max(1));
    let mut panel_items = sample_per_kind(&archive_heldout, Some(per_kind), cfg.seed ^ 0xA5);
    panel_items.truncate(cfg.retention_panel);
    let retention_items = sample_per_kind(&archive_heldout, Some(cfg.retention_units), cfg.seed ^ 0x5A);
    let probes = Probes {
        heldout_rows: rows_of(&start, &probe_items, &mut rng)?,
        heldout_items: probe_items,
        panel_items,
        retention_rows: rows_of(&start, &retention_items, &mut rng)?,
    };

    let dense = cfg.education.is_some();
    let (baseline, base_report, base_panel) = score(&start, &probes, dense)?;
    let base_panel_correct = correct_set(&base_panel);
    observer(&Event::Started {
        train_units: train.len(),
        heldout_units: heldout.len(),
        rehearsal_units: archive_train.len(),
        new_words,
        baseline: baseline.clone(),
    });

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
    let mut learned = false;
    let mut epochs = Vec::new();
    let mut stale = 0;
    let mut stopped = false;

    'epochs: for epoch in 1..=cfg.epochs {
        // One batch plan per epoch, shared by every candidate.
        let mut plan_rng = Rng::new(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(epoch as u64));
        let mut order: Vec<&Item> = train.clone();
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
        'plan: loop {
            let mut rows = Vec::with_capacity(cfg.batch_size);
            while rows.len() < cfg.batch_size {
                let it = if !archive_train.is_empty() && plan_rng.unit() < cfg.rehearsal_share as f64 {
                    archive_train[plan_rng.below(archive_train.len())]
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
                    let lr = lr_schedule(step, cfg.learning_rate, warmup, total_steps, 0.1);
                    let loss = trainer.step(&mut cand, rows, Ablation::default(), lr, focus)?;
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
                if let Some(p) = pending.take() {
                    let (cand, scored) = p.scoring.join().map_err(|_| "Scoring a candidate failed")?;
                    let j =
                        Judge { cfg, epoch, baseline: &baseline, base_panel_correct: &base_panel_correct };
                    j.judge(
                        p.ci,
                        p.focus,
                        p.train_loss,
                        cand,
                        scored?,
                        &accepted_score,
                        &accepted_correct,
                        &mut best,
                        &mut plateau,
                        &mut records,
                        observer,
                    );
                }
                let probes = &probes;
                if trainer.on_gpu() {
                    let scoring = scope.spawn(move || {
                        let scored = score(&cand, probes, dense);
                        (cand, scored)
                    });
                    pending = Some(Pending { ci, focus, train_loss, scoring });
                } else {
                    let scored = score(&cand, probes, dense)?;
                    let j =
                        Judge { cfg, epoch, baseline: &baseline, base_panel_correct: &base_panel_correct };
                    j.judge(
                        ci,
                        focus,
                        train_loss,
                        cand,
                        scored,
                        &accepted_score,
                        &accepted_correct,
                        &mut best,
                        &mut plateau,
                        &mut records,
                        observer,
                    );
                }
            }
            if let Some(p) = pending.take() {
                let (cand, scored) = p.scoring.join().map_err(|_| "Scoring a candidate failed")?;
                let j = Judge { cfg, epoch, baseline: &baseline, base_panel_correct: &base_panel_correct };
                j.judge(
                    p.ci,
                    p.focus,
                    p.train_loss,
                    cand,
                    scored?,
                    &accepted_score,
                    &accepted_correct,
                    &mut best,
                    &mut plateau,
                    &mut records,
                    observer,
                );
            }
            Ok(false)
        })?;
        if interrupted {
            stopped = true;
            break 'epochs;
        }

        let (mut continued, mut restored) = (None, false);
        if best.is_none() {
            match plateau.take() {
                Some((ci, cand, _, _, _)) => {
                    records[ci].continued = true;
                    continued = Some(focus_name(foci[ci]).to_string());
                    working = cand;
                }
                None => {
                    restored = working.step != accepted.step;
                    working = accepted.clone();
                }
            }
        }
        let selected = best.map(|(ci, cand, s, correct, report)| {
            records[ci].selected = true;
            working = cand.clone();
            accepted = cand;
            accepted_score = s;
            accepted_correct = correct;
            learned = true;
            (focus_name(foci[ci]).to_string(), report)
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
        };
        observer(&Event::Epoch(record.clone()));
        epochs.push(record);
        if stale >= cfg.patience {
            break;
        }
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
