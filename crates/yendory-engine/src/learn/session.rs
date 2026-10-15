//! Teaching sessions: Foundation candidate training, rehearsal of earlier lessons, retention
//! guards and checkpoint selection, with a typed record of every candidate in every epoch.
//! The rules are written out in docs/LEARNING.md.
//!
//! Each epoch starts from the currently ACCEPTED complete model (weights, Adam state, step,
//! tokenizer). With Foundation training on, four candidates are trained on identical batches —
//! Right (Dense), Left (Transformer), Interconnect and Joint — differing only in which parameter
//! group the optimizer may update; every pathway still runs in every forward pass. Each
//! candidate is scored as a whole model; the best ELIGIBLE candidate that improves becomes the
//! accepted model. When nothing qualifies, the accepted model is kept (rollback). Teacher answers
//! are training targets and evaluation keys only; they are never given to the model at inference.
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
    /// Four candidates (Right, Left, Interconnect, Joint) per epoch instead of Joint only.
    pub foundation: bool,
    /// Stop after this many epochs in a row without a selected candidate.
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
    /// Grow learned word units from this lesson's training text (ids stay stable).
    pub grow_vocabulary: bool,
    pub min_word_count: usize,
    pub seed: u64,
    /// How training units are scheduled into batches.
    #[serde(default)]
    pub strategy: Strategy,
    /// What a course does when a lesson is not learned (retries, then later in the queue).
    #[serde(default)]
    pub retry: RetryPolicy,
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

/// Neural Nova's Foundation V2 retry plan, applied by `teach_course` when a lesson is not learned:
///
/// | attempt | learning rate | strategy | epochs |
/// |---|---|---|---|
/// | initial | chosen rate | standard | chosen |
/// | retry 1 | rate x 0.5 (gentler) | standard | chosen |
/// | retry 2 | chosen rate | focused, new order | 12 |
/// | retry 3 | rate x 0.25 (gentler again) | focused | chosen + 6 |
///
/// After the retries, the lesson moves to the end of the queue; it is tried again (all attempts)
/// once at least one other lesson has been learned since, because a different lesson may teach
/// the skill it needs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Retries after the initial attempt (0 to 3).
    pub retries: u32,
    /// Move a lesson that is still not learned to the end of the queue.
    pub requeue: bool,
    /// Lowest learning rate a retry may use.
    pub min_learning_rate: f32,
}
impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy { retries: 3, requeue: true, min_learning_rate: 1e-6 }
    }
}

/// The settings of attempt `attempt` (0 = initial) under the retry plan.
pub fn attempt_config(base: &SessionConfig, attempt: u32) -> SessionConfig {
    let floor = base.retry.min_learning_rate;
    let (rate, strategy, epochs) = match attempt {
        0 => (base.learning_rate, base.strategy, base.epochs),
        1 => (base.learning_rate * 0.5, base.strategy, base.epochs),
        2 => (base.learning_rate, Strategy::Focused, 12),
        _ => (base.learning_rate * 0.25, Strategy::Focused, base.epochs + 6),
    };
    SessionConfig {
        learning_rate: rate.max(floor).min(base.learning_rate),
        strategy,
        epochs,
        seed: base.seed.wrapping_add(attempt as u64 * 7919),
        ..base.clone()
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        SessionConfig {
            epochs: 4,
            batch_size: 16,
            learning_rate: 1e-3,
            foundation: true,
            patience: 2,
            rehearsal_share: 0.10,
            eval_limit: Some(40),
            heldout_loss_tolerance: 0.08,
            retention_loss_tolerance: 0.02,
            retention_panel: 8,
            protect_correct_answers: true,
            grow_vocabulary: true,
            min_word_count: 12,
            seed: 1,
            strategy: Strategy::Standard,
            retry: RetryPolicy::default(),
        }
    }
}

pub fn focus_name(f: Focus) -> &'static str {
    match f {
        Focus::Joint => "joint",
        Focus::Left => "transformer",
        Focus::Right => "dense",
        Focus::Interconnect => "interconnect",
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
    pub selected: bool,
    pub step: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpochRecord {
    pub epoch: u32,
    pub candidates: Vec<CandidateRecord>,
    pub selected: Option<String>,
    pub accepted_step: u64,
    pub accepted_score: Score,
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

fn rows_of(model: &Bicameral, items: &[&Item], rng: &mut Rng) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    for it in items {
        if let Some(r) = encode_item(&model.tokenizer, &it.segments, model.config.max_tokens, rng)? {
            rows.push(r);
        }
    }
    Ok(rows)
}

fn score(model: &Bicameral, p: &Probes) -> Result<(Score, Report, Report), String> {
    let abl = Ablation::default();
    let heldout_loss = batch_loss_and_gradients_loss(model, &p.heldout_rows)?;
    let report = evaluate(model, &p.heldout_items, abl)?;
    let panel = evaluate(model, &p.panel_items, abl)?;
    let retention_loss = if p.retention_rows.is_empty() {
        None
    } else {
        Some(batch_loss_and_gradients_loss(model, &p.retention_rows)?)
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
        },
        report,
        panel,
    ))
}

/// Teacher-forced mean loss of rows (no gradient is applied).
fn batch_loss_and_gradients_loss(model: &Bicameral, rows: &[Row]) -> Result<f32, String> {
    if rows.is_empty() {
        return Ok(0.);
    }
    let mut total = 0f64;
    let mut weight = 0f64;
    for chunk in rows.chunks(32) {
        let (l, _) = batch_loss_and_gradients(model, chunk, Ablation::default())?;
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
    if cfg.epochs == 0 || cfg.batch_size == 0 || !(cfg.learning_rate > 0. && cfg.learning_rate.is_finite()) {
        return Err("Choose at least one epoch, a batch size and a positive learning rate".into());
    }
    let train: Vec<&Item> = lesson
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
        let words =
            grow_vocabulary(&start.tokenizer.words, lesson, crate::tokenizer::MAX_WORDS, cfg.min_word_count);
        let words: Vec<String> = words.into_iter().take(start.config.vocab.saturating_sub(128)).collect();
        new_words = words.len() - start.tokenizer.words.len();
        start.tokenizer.words = words;
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
    let retention_items = sample_per_kind(&archive_heldout, Some(16), cfg.seed ^ 0x5A);
    let probes = Probes {
        heldout_rows: rows_of(&start, &probe_items, &mut rng)?,
        heldout_items: probe_items,
        panel_items,
        retention_rows: rows_of(&start, &retention_items, &mut rng)?,
    };

    let (baseline, base_report, base_panel) = score(&start, &probes)?;
    let base_panel_correct = correct_set(&base_panel);
    observer(&Event::Started {
        train_units: train.len(),
        heldout_units: heldout.len(),
        rehearsal_units: archive_train.len(),
        new_words,
        baseline: baseline.clone(),
    });

    let foci: Vec<Focus> = if cfg.foundation {
        vec![Focus::Right, Focus::Left, Focus::Interconnect, Focus::Joint]
    } else {
        vec![Focus::Joint]
    };
    let batches_per_epoch = train.len().div_ceil(cfg.batch_size);
    let total_steps = batches_per_epoch as u64 * cfg.epochs as u64;
    let warmup = (total_steps / 10).clamp(1, 200);

    let mut accepted = start.clone();
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
        let mut best: Option<(usize, Bicameral, Score, Vec<bool>, Report)> = None;
        for (ci, &focus) in foci.iter().enumerate() {
            let mut cand = accepted.clone();
            let mut losses = 0f64;
            for (b, rows) in plan.iter().enumerate() {
                if stop.load(Ordering::Acquire) {
                    stopped = true;
                    break 'epochs;
                }
                let step = (epoch as u64 - 1) * batches_per_epoch as u64 + b as u64;
                let lr = lr_schedule(step, cfg.learning_rate, warmup, total_steps, 0.1);
                let (loss, grads) = batch_loss_and_gradients(&cand, rows, Ablation::default())?;
                cand.adam_step(&grads, lr, focus)?;
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
            let (s, report, panel) = score(&cand, &probes)?;
            let mut reasons = Vec::new();
            let tol = cfg.heldout_loss_tolerance;
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
                if now > before + cfg.retention_loss_tolerance {
                    reasons.push(format!(
                        "earlier lessons' held-out loss {now:.4} exceeds {before:.4} + {}",
                        cfg.retention_loss_tolerance
                    ));
                }
            }
            let lost_panel =
                base_panel_correct.iter().zip(correct_set(&panel)).filter(|(b, n)| **b && !*n).count();
            if lost_panel > 0 {
                reasons.push(format!("lost {lost_panel} earlier-lesson answer(s) the baseline produced"));
            }
            let now_correct = correct_set(&report);
            if cfg.protect_correct_answers {
                let lost = accepted_correct.iter().zip(&now_correct).filter(|(b, n)| **b && !**n).count();
                if lost > 0 {
                    reasons.push(format!("lost {lost} held-out answer(s) the accepted model produced"));
                }
            }
            let eligible = reasons.is_empty();
            let improves = s.exact_correct > accepted_score.exact_correct
                || (s.exact_correct == accepted_score.exact_correct
                    && s.heldout_loss < accepted_score.heldout_loss);
            if eligible && !improves {
                reasons.push("no improvement over the accepted model".into());
            }
            let better_than_best = best.as_ref().map_or(true, |(_, _, b, _, _)| {
                s.exact_correct > b.exact_correct
                    || (s.exact_correct == b.exact_correct && s.heldout_loss < b.heldout_loss)
            });
            let record = CandidateRecord {
                epoch,
                focus: focus_name(focus).into(),
                train_loss: (losses / plan.len().max(1) as f64) as f32,
                score: s.clone(),
                eligible,
                reasons,
                selected: false,
                step: cand.step,
            };
            observer(&Event::Candidate(record.clone()));
            records.push(record);
            if eligible && improves && better_than_best {
                best = Some((ci, cand, s, now_correct, report));
            }
        }

        let selected = best.map(|(ci, cand, s, correct, report)| {
            records[ci].selected = true;
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
