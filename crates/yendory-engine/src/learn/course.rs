//! Courses: several lesson files taught one after another (e.g. the files of a ZIP). Reading
//! the ZIP is the frontend's job; the Engine receives the files and records exactly which
//! content was taught, when, and with what result.
//!
//! A lesson is judged by a **pass mark** (share of its held-out questions answered exactly
//! right), not by whether some change was accepted. An attempt that does not pass is diagnosed
//! (unstable, forgetting earlier lessons, not yet learned, or memorised without generalising)
//! and the next attempt applies the remedy for that cause. See docs/LEARNING.md.
use super::eval::evaluate;
use super::lessons::{parse_lesson, Item, Kind};
use super::session::{teach_with, Event, Rejection, Score, SessionConfig, SessionResult, Strategy};
use crate::fingerprint::sha256_hex;
use crate::model::{Ablation, Bicameral};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CourseFile {
    pub name: String,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileOutcome {
    /// The pass mark was reached: the lesson is mastered (by its held-out questions).
    Passed,
    /// Records written before `Unresolved` existed: improvements were kept, the pass mark was
    /// not reached.
    Progressed,
    /// Not passed after its attempts; moved to the end of the queue to try again after other
    /// lessons. A later record for the same file gives the final outcome.
    Deferred,
    /// Records written before `Unresolved` existed: nothing qualified, the previous model was kept.
    KeptPrevious,
    /// The same content was already taught earlier in this course; not taught again.
    Duplicate,
    /// The file could not be taught (parse error, no held-out units, ...); see `error`.
    Failed,
    /// Teaching was stopped during this file; its epoch in progress was discarded.
    Stopped,
    /// Records written before pass marks existed: a change was accepted (passed or not unknown).
    Learned,
    /// Not passed after its retries and its one move to the end of the queue: the lesson's
    /// attempts are used up. Any progress it made is kept (see `FileRecord::changed`).
    Unresolved,
}

/// Why an attempt did not pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cause {
    /// Most candidates made held-out loss worse: the learning steps were too large.
    Unstable,
    /// Most candidates were rejected for harming earlier lessons.
    Forgetting,
    /// Most candidates were rejected for losing answers of this lesson the model already gave:
    /// new learning overwrites what it knows.
    Interference,
    /// The model cannot yet answer the lesson's own training questions.
    NotLearned,
    /// It answers the training questions but not the held-out ones: memorised, not generalised.
    NotGeneralising,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Diagnosis {
    pub cause: Cause,
    /// Held-out questions answered exactly right (share), against the pass mark.
    pub heldout_rate: f32,
    /// Training questions answered exactly right (share).
    pub train_rate: f32,
    /// Kinds of question below the pass mark on held-out questions.
    pub weak_kinds: Vec<String>,
    /// Training units answered wrong (they are what the remedy practises when not learned).
    pub wrong_training_units: usize,
    /// What the next attempt does about it, in words.
    pub remedy: String,
}

/// One attempt at a lesson.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttemptRecord {
    /// 0 = initial attempt, 1..=3 = retries.
    pub attempt: u32,
    pub learning_rate: f32,
    pub epochs: u32,
    pub strategy: Strategy,
    pub rehearsal_share: f32,
    /// Training units used (after a "not learned" diagnosis: the practice selection).
    pub practice_units: usize,
    /// Some change was accepted in this attempt.
    pub learned: bool,
    /// The pass mark was reached after this attempt.
    #[serde(default)]
    pub passed: bool,
    pub epochs_run: usize,
    pub final_score: Score,
    /// The diagnosis this attempt responded to (None for the initial attempt).
    #[serde(default)]
    pub because: Option<Diagnosis>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FileRecord {
    pub index: usize,
    pub name: String,
    /// SHA-256 of the file's bytes: proves which content was taught.
    pub sha256: String,
    pub bytes: usize,
    pub train_units: usize,
    pub heldout_units: usize,
    pub start_step: u64,
    pub end_step: u64,
    pub outcome: FileOutcome,
    pub error: Option<String>,
    pub epochs_run: usize,
    pub baseline: Option<Score>,
    pub final_score: Option<Score>,
    /// 1 for the first time through the queue, 2 after the lesson was moved to the end.
    #[serde(default = "first_pass")]
    pub pass: u32,
    #[serde(default)]
    pub attempts: Vec<AttemptRecord>,
    /// The pass mark this lesson was judged by.
    #[serde(default)]
    pub pass_mark: f32,
}
fn first_pass() -> u32 {
    1
}
impl FileRecord {
    /// Some change from this file was accepted (in this pass).
    pub fn changed(&self) -> bool {
        self.attempts.iter().any(|a| a.learned)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CourseEvent {
    FileStarted {
        index: usize,
        files: usize,
        name: String,
        sha256: String,
        #[serde(default = "first_pass")]
        pass: u32,
    },
    AttemptStarted {
        index: usize,
        attempt: u32,
        attempts: u32,
        learning_rate: f32,
        epochs: u32,
        strategy: Strategy,
        #[serde(default)]
        remedy: Option<String>,
    },
    /// Why the attempt just finished did not pass, and what happens next.
    Diagnosed {
        index: usize,
        attempt: u32,
        diagnosis: Diagnosis,
    },
    Session {
        index: usize,
        event: Event,
    },
    /// Sent when a file's pass ends: once per pass, so a deferred file has two records.
    FileFinished(FileRecord),
}

pub struct CourseResult {
    pub model: Bicameral,
    /// Items of every lesson whose training changed the model (the given archive plus this
    /// course's), for rehearsal and retention checks in later sessions.
    pub archive: Vec<Item>,
    /// The final record of every file, in file order.
    pub files: Vec<FileRecord>,
    pub stopped: bool,
}

/// Called with the new model and archive after each file that changed the model.
pub type OnLearned<'a> = dyn FnMut(&Bicameral, &[Item]) -> Result<(), String> + 'a;

/// Orders files by name, like the reference's `sorted(glob(...))` (lesson files are numbered).
pub fn sort_files(files: &mut [CourseFile]) {
    files.sort_by(|a, b| a.name.cmp(&b.name));
}

/// The teaching order. A lesson that is not passed goes to the end of the queue, remembering how
/// many changes had been made to the model at that moment; it is tried again only if the model
/// has changed since (otherwise a retry would repeat the same result).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Queue {
    waiting: std::collections::VecDeque<(usize, Option<usize>)>,
    changes: usize,
}
pub enum Next {
    /// Teach this lesson; `pass` is 1 the first time, 2 after a deferral.
    Teach { index: usize, pass: u32 },
    /// Deferred, and the model has not changed since: this lesson is finished.
    Exhausted { index: usize },
}
impl Queue {
    pub fn new(files: usize) -> Self {
        Queue { waiting: (0..files).map(|i| (i, None)).collect(), changes: 0 }
    }
    pub fn pop(&mut self) -> Option<Next> {
        let (index, deferred_at) = self.waiting.pop_front()?;
        Some(match deferred_at {
            None => Next::Teach { index, pass: 1 },
            Some(at) if self.changes > at => Next::Teach { index, pass: 2 },
            Some(_) => Next::Exhausted { index },
        })
    }
    /// The model changed (a lesson passed or progressed).
    pub fn changed(&mut self) {
        self.changes += 1;
    }
    pub fn defer(&mut self, index: usize) {
        self.waiting.push_back((index, Some(self.changes)));
    }
}

/// How the next attempt is set up.
#[derive(Clone)]
struct Plan {
    cfg: SessionConfig,
    practice: Option<Vec<Item>>,
    remedy: Option<String>,
}

/// Works out why `result` did not pass and plans the next attempt. Returns None when another
/// attempt is not expected to help (memorised twice without generalising): the lesson should
/// wait for other lessons instead.
fn diagnose(
    model: &Bicameral,
    items: &[Item],
    result: &SessionResult,
    previous: &Plan,
    base: &SessionConfig,
    generalising_tried: bool,
) -> Result<(Diagnosis, Option<Plan>), String> {
    // Rejected candidates across the attempt: what stopped them?
    let candidates: Vec<_> =
        result.epochs.iter().flat_map(|e| e.candidates.iter()).filter(|c| !c.selected).collect();
    let share = |r: Rejection| {
        if candidates.is_empty() {
            0.
        } else {
            candidates.iter().filter(|c| c.rejections.contains(&r)).count() as f32 / candidates.len() as f32
        }
    };
    let unstable = share(Rejection::HeldoutLossRose);
    let forgetting = share(Rejection::EarlierLessonsLoss).max(share(Rejection::EarlierAnswersLost));
    let interference = share(Rejection::AnswersLost);

    // Can the (current) model answer the lesson's own training questions?
    let training: Vec<&Item> = items.iter().filter(|i| !i.held_out && i.kind != Kind::Prose).collect();
    let report = evaluate(model, &training, Ablation::default())?;
    let correct_by_unit: Vec<bool> = {
        // Outcomes come in item order; conversations contribute one outcome per turn.
        let mut v = Vec::new();
        let mut o = report.outcomes.iter();
        for it in &training {
            let turns = it.outcome_count();
            let ok = (0..turns).filter_map(|_| o.next()).all(|x| x.correct);
            v.push(ok);
        }
        v
    };
    let train_rate = report.overall().rate().unwrap_or(0.) as f32;
    let score = &result.final_score;
    let weak_kinds: Vec<String> = score
        .by_kind
        .iter()
        .filter(|(_, (c, t))| *t > 0 && (*c as f32 / *t as f32) < base.pass_mark)
        .map(|(k, _)| k.clone())
        .collect();
    let wrong = correct_by_unit.iter().filter(|ok| !**ok).count();

    let cause = if unstable >= 0.5 {
        Cause::Unstable
    } else if forgetting >= 0.5 {
        Cause::Forgetting
    } else if interference >= 0.5 {
        Cause::Interference
    } else if train_rate < base.pass_mark {
        Cause::NotLearned
    } else {
        Cause::NotGeneralising
    };
    let mut next = previous.clone();
    next.cfg.seed = previous.cfg.seed.wrapping_add(7919);
    let remedy = match cause {
        Cause::Unstable => {
            next.cfg.learning_rate = (previous.cfg.learning_rate * 0.5).max(base.retry.min_learning_rate);
            format!(
                "gentler steps: learning rate {:.1e} -> {:.1e}",
                previous.cfg.learning_rate, next.cfg.learning_rate
            )
        }
        Cause::Forgetting => {
            next.cfg.rehearsal_share = (previous.cfg.rehearsal_share * 2.).clamp(0.25, 0.4);
            format!(
                "more rehearsal of earlier lessons: {:.0}% of each batch",
                next.cfg.rehearsal_share * 100.
            )
        }
        Cause::Interference => {
            // Keep what it knows in every batch while it learns the rest, with smaller steps.
            let mut practice = Vec::new();
            for (it, ok) in training.iter().zip(&correct_by_unit) {
                for _ in 0..(if *ok { 1 } else { 2 }) {
                    practice.push((*it).clone());
                }
            }
            next.practice = Some(practice);
            next.cfg.strategy = Strategy::Standard;
            next.cfg.learning_rate = (previous.cfg.learning_rate * 0.5).max(base.retry.min_learning_rate);
            next.cfg.epochs = base.epochs.max(8);
            format!(
                "balanced practice: the {wrong} question(s) it gets wrong twice and every question it knows once, \
                 learning rate {:.1e}, {} epochs",
                next.cfg.learning_rate, next.cfg.epochs
            )
        }
        Cause::NotLearned => {
            // Practise exactly what it gets wrong (twice), weak kinds twice more, and a quarter of
            // what it already knows so that is not lost.
            let mut practice = Vec::new();
            for (k, (it, ok)) in training.iter().zip(&correct_by_unit).enumerate() {
                let weak = weak_kinds
                    .iter()
                    .any(|w| w == it.kind.name() || (w == "conv_turn" && it.kind == Kind::Conversation));
                let copies = if !ok { 2 + usize::from(weak) } else { usize::from(k % 4 == 0) };
                for _ in 0..copies {
                    practice.push((*it).clone());
                }
            }
            next.practice = Some(practice);
            next.cfg.strategy = Strategy::Focused;
            next.cfg.epochs = base.epochs.max(12);
            format!(
                "practise the {wrong} training question(s) it gets wrong{}, easiest first, {} epochs",
                if weak_kinds.is_empty() {
                    String::new()
                } else {
                    format!(" (extra weight: {})", weak_kinds.join(", "))
                },
                next.cfg.epochs
            )
        }
        Cause::NotGeneralising => {
            if generalising_tried {
                let d = Diagnosis {
                    cause,
                    heldout_rate: score.exact_rate(),
                    train_rate,
                    weak_kinds,
                    wrong_training_units: wrong,
                    remedy: "stop retrying: it knows the training questions but still not new ones; wait for other lessons".into(),
                };
                return Ok((d, None));
            }
            // Broader study: every unit including the explanations, new order, more epochs.
            next.practice = None;
            next.cfg.strategy = Strategy::Standard;
            next.cfg.epochs = base.epochs + 6;
            format!("broader study: all units and explanations, new order, {} epochs", next.cfg.epochs)
        }
    };
    next.remedy = Some(remedy.clone());
    let d = Diagnosis {
        cause,
        heldout_rate: score.exact_rate(),
        train_rate,
        weak_kinds,
        wrong_training_units: wrong,
        remedy,
    };
    Ok((d, Some(next)))
}

/// Teaches the files in order. Each lesson must reach `cfg.pass_mark`; an attempt that does not
/// is diagnosed and up to `cfg.retry.retries` remedies are tried. A lesson still not passed
/// moves to the end of the queue and is tried once more after the model has changed. A file
/// that cannot be taught at all is recorded and skipped; it does not end the course. Files whose
/// training changed the model join the archive, so later files rehearse and protect them.
/// `on_learned` receives the new model and archive after each such file; an error from it ends
/// the course.
pub fn teach_course(
    model: &Bicameral,
    files: &[CourseFile],
    archive: &[Item],
    cfg: &SessionConfig,
    observer: &mut dyn FnMut(&CourseEvent),
    on_learned: &mut OnLearned,
    stop: &AtomicBool,
) -> Result<CourseResult, String> {
    let mut current = model.clone();
    let mut archive = archive.to_vec();
    let mut finals: Vec<Option<FileRecord>> = vec![None; files.len()];
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut parsed: Vec<Option<Vec<Item>>> = vec![None; files.len()];
    let mut queue = Queue::new(files.len());
    let mut stopped = false;
    let retries = cfg.retry.retries.min(3);
    while let Some(next) = queue.pop() {
        if stop.load(Ordering::Acquire) {
            stopped = true;
            break;
        }
        let (index, pass) = match next {
            Next::Teach { index, pass } => (index, pass),
            Next::Exhausted { index } => {
                if let Some(r) = finals[index].as_mut() {
                    r.outcome = FileOutcome::Unresolved;
                    r.error = Some(
                        "not passed; the model did not change after it was moved to the end of the queue"
                            .into(),
                    );
                    observer(&CourseEvent::FileFinished(r.clone()));
                }
                continue;
            }
        };
        let file = &files[index];
        let sha256 = sha256_hex(file.text.as_bytes());
        observer(&CourseEvent::FileStarted {
            index,
            files: files.len(),
            name: file.name.clone(),
            sha256: sha256.clone(),
            pass,
        });
        let mut record = FileRecord {
            index,
            name: file.name.clone(),
            sha256: sha256.clone(),
            bytes: file.text.len(),
            train_units: 0,
            heldout_units: 0,
            start_step: current.step,
            end_step: current.step,
            outcome: FileOutcome::Failed,
            error: None,
            epochs_run: 0,
            baseline: None,
            final_score: None,
            pass,
            attempts: Vec::new(),
            pass_mark: cfg.pass_mark,
        };
        let duplicate =
            (pass == 1).then(|| seen.iter().find(|(h, _)| *h == sha256).map(|(_, n)| n.clone())).flatten();
        if let Some(first) = duplicate {
            record.outcome = FileOutcome::Duplicate;
            record.error = Some(format!("same content as {first}"));
        } else {
            if pass == 1 {
                seen.push((sha256.clone(), file.name.clone()));
                parsed[index] = match parse_lesson(&file.text, &file.name) {
                    Ok(items) => Some(items),
                    Err(e) => {
                        record.error = Some(e);
                        None
                    }
                };
            }
            if let Some(items) = parsed[index].clone() {
                record.train_units = items.iter().filter(|i| !i.held_out).count();
                record.heldout_units = items.iter().filter(|i| i.held_out).count();
                let judgeable =
                    record.train_units > 0 && items.iter().any(|i| i.held_out && i.kind != Kind::Prose);
                if !judgeable {
                    record.error = Some(
                        "The lesson needs training units and held-out units (about 10% are held out by hash)"
                            .into(),
                    );
                } else {
                    teach_lesson(
                        &mut current,
                        &mut archive,
                        &items,
                        index,
                        retries,
                        cfg,
                        &mut record,
                        &mut stopped,
                        &mut queue,
                        observer,
                        on_learned,
                        stop,
                    )?;
                    if record.outcome != FileOutcome::Passed && !stopped && record.error.is_none() {
                        record.outcome = if pass == 1 && cfg.retry.requeue {
                            queue.defer(index);
                            FileOutcome::Deferred
                        } else {
                            FileOutcome::Unresolved
                        };
                    }
                }
            }
        }
        record.end_step = current.step;
        observer(&CourseEvent::FileFinished(record.clone()));
        finals[index] = Some(record);
        if stopped {
            break;
        }
    }
    let files = finals.into_iter().flatten().collect();
    Ok(CourseResult { model: current, archive, files, stopped })
}

/// All attempts at one lesson in one pass. Sets `record.outcome` to Passed when the pass mark is
/// reached (or Stopped); otherwise leaves the outcome to the caller.
#[allow(clippy::too_many_arguments)]
pub(super) fn teach_lesson(
    current: &mut Bicameral,
    archive: &mut Vec<Item>,
    items: &[Item],
    index: usize,
    retries: u32,
    cfg: &SessionConfig,
    record: &mut FileRecord,
    stopped: &mut bool,
    queue: &mut Queue,
    observer: &mut dyn FnMut(&CourseEvent),
    on_learned: &mut OnLearned,
    stop: &AtomicBool,
) -> Result<(), String> {
    let source = items.first().map(|i| i.source.clone()).unwrap_or_default();
    let mut plan = Plan { cfg: cfg.clone(), practice: None, remedy: None };
    let mut because: Option<Diagnosis> = None;
    let mut generalising_tried = false;
    let mut last_cause: Option<Cause> = None;
    for attempt in 0..=retries {
        // Earlier lessons only: this lesson's own units are not rehearsal or retention material.
        let others: Vec<Item> = archive.iter().filter(|i| i.source != source).cloned().collect();
        observer(&CourseEvent::AttemptStarted {
            index,
            attempt,
            attempts: retries + 1,
            learning_rate: plan.cfg.learning_rate,
            epochs: plan.cfg.epochs,
            strategy: plan.cfg.strategy,
            remedy: plan.remedy.clone(),
        });
        let mut forward = |e: &Event| observer(&CourseEvent::Session { index, event: e.clone() });
        let r = match teach_with(
            current,
            items,
            plan.practice.as_deref(),
            &others,
            &plan.cfg,
            &mut forward,
            stop,
        ) {
            Ok(r) => r,
            Err(e) => {
                record.error = Some(e);
                return Ok(());
            }
        };
        record.epochs_run += r.epochs.len();
        if record.baseline.is_none() {
            record.baseline = Some(r.baseline.clone());
        }
        record.final_score = Some(r.final_score.clone());
        let passed = !r.stopped && r.final_score.exact_rate() >= cfg.pass_mark;
        record.attempts.push(AttemptRecord {
            attempt,
            learning_rate: plan.cfg.learning_rate,
            epochs: plan.cfg.epochs,
            strategy: plan.cfg.strategy,
            rehearsal_share: plan.cfg.rehearsal_share,
            practice_units: plan.practice.as_ref().map_or(record.train_units, |p| p.len()),
            learned: r.learned,
            passed,
            epochs_run: r.epochs.len(),
            final_score: r.final_score.clone(),
            because: because.take(),
        });
        if r.stopped {
            *stopped = true;
            record.outcome = FileOutcome::Stopped;
            return Ok(());
        }
        if r.learned {
            *current = r.model.clone();
            if !archive.iter().any(|i| i.source == source) {
                archive.extend(items.iter().cloned());
            }
            on_learned(current, archive)?;
            queue.changed();
        }
        if passed {
            record.outcome = FileOutcome::Passed;
            return Ok(());
        }
        if attempt == retries {
            return Ok(());
        }
        let (mut diagnosis, mut next) = diagnose(current, items, &r, &plan, cfg, generalising_tried)?;
        // The same cause again after an attempt that changed nothing, and the remedy would set up
        // the same attempt again: repeating it would only repeat the result. Wait for other
        // lessons instead. (A remedy that changes something, e.g. a further halved rate, may run.)
        let same_plan = next.as_ref().is_some_and(|n| {
            n.cfg.learning_rate == plan.cfg.learning_rate
                && n.cfg.strategy == plan.cfg.strategy
                && n.cfg.epochs == plan.cfg.epochs
                && n.cfg.rehearsal_share == plan.cfg.rehearsal_share
        });
        if last_cause == Some(diagnosis.cause) && !r.learned && same_plan {
            diagnosis.remedy =
                "stop retrying: the remedy for this cause already failed without any change; wait for other lessons"
                    .into();
            next = None;
        }
        observer(&CourseEvent::Diagnosed { index, attempt, diagnosis: diagnosis.clone() });
        generalising_tried |= diagnosis.cause == Cause::NotGeneralising;
        last_cause = Some(diagnosis.cause);
        match next {
            Some(p) => plan = p,
            None => return Ok(()),
        }
        because = Some(diagnosis);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deferred_lessons_return_only_after_the_model_changed() {
        let mut q = Queue::new(3);
        // Lesson 0 is not passed and deferred; lesson 1 changes the model; lesson 2 is deferred.
        assert!(matches!(q.pop(), Some(Next::Teach { index: 0, pass: 1 })));
        q.defer(0);
        assert!(matches!(q.pop(), Some(Next::Teach { index: 1, pass: 1 })));
        q.changed();
        assert!(matches!(q.pop(), Some(Next::Teach { index: 2, pass: 1 })));
        q.defer(2);
        // Lesson 0 comes back (the model changed after it was deferred) ...
        assert!(matches!(q.pop(), Some(Next::Teach { index: 0, pass: 2 })));
        // ... lesson 2 does not: nothing changed after it was deferred.
        assert!(matches!(q.pop(), Some(Next::Exhausted { index: 2 })));
        assert!(q.pop().is_none());
    }
}
