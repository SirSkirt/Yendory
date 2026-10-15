//! Courses: several lesson files taught one after another (e.g. the files of a ZIP). Reading
//! the ZIP is the frontend's job; the Engine receives the files and records exactly which
//! content was taught, when, and with what result.
use super::lessons::{parse_lesson, Item, Kind};
use super::session::{attempt_config, teach, Event, Score, SessionConfig, Strategy};
use crate::fingerprint::sha256_hex;
use crate::model::Bicameral;
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
    /// A candidate was selected; the model changed.
    Learned,
    /// Not learned after all attempts; moved to the end of the queue to try again after other
    /// lessons. A later record for the same file gives the final outcome.
    Deferred,
    /// Not learned after all attempts (and after its later retry, if any); the model before this
    /// file was kept.
    KeptPrevious,
    /// The same content was already taught earlier in this course; not taught again.
    Duplicate,
    /// The file could not be taught (parse error, no held-out units, ...); see `error`.
    Failed,
    /// Teaching was stopped during this file; its epoch in progress was discarded.
    Stopped,
}

/// One attempt at a lesson under the retry plan (`session::attempt_config`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttemptRecord {
    /// 0 = initial attempt, 1..=3 = retries.
    pub attempt: u32,
    pub learning_rate: f32,
    pub epochs: u32,
    pub strategy: Strategy,
    pub learned: bool,
    pub epochs_run: usize,
    pub final_score: Score,
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
}
fn first_pass() -> u32 {
    1
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
    /// Items of every lesson learned so far (the given archive plus this course's learned files),
    /// for rehearsal and retention checks in later sessions.
    pub archive: Vec<Item>,
    /// The final record of every file, in file order.
    pub files: Vec<FileRecord>,
    pub stopped: bool,
}

/// Called with the new model and archive after each learned file.
pub type OnLearned<'a> = dyn FnMut(&Bicameral, &[Item]) -> Result<(), String> + 'a;

/// Orders files by name, like the reference's `sorted(glob(...))` (lesson files are numbered).
pub fn sort_files(files: &mut [CourseFile]) {
    files.sort_by(|a, b| a.name.cmp(&b.name));
}

/// The teaching order. A lesson that is not learned goes to the end of the queue, remembering how
/// many lessons had been learned at that moment; it is tried again only if more have been
/// learned since (otherwise the model is unchanged and a retry would repeat the same result).
#[derive(Debug)]
pub struct Queue {
    waiting: std::collections::VecDeque<(usize, Option<usize>)>,
    learned: usize,
}
pub enum Next {
    /// Teach this lesson; `pass` is 1 the first time, 2 after a deferral.
    Teach { index: usize, pass: u32 },
    /// Deferred, and nothing new was learned since: this lesson is finished as not learned.
    Exhausted { index: usize },
}
impl Queue {
    pub fn new(files: usize) -> Self {
        Queue { waiting: (0..files).map(|i| (i, None)).collect(), learned: 0 }
    }
    pub fn pop(&mut self) -> Option<Next> {
        let (index, deferred_at) = self.waiting.pop_front()?;
        Some(match deferred_at {
            None => Next::Teach { index, pass: 1 },
            Some(at) if self.learned > at => Next::Teach { index, pass: 2 },
            Some(_) => Next::Exhausted { index },
        })
    }
    pub fn learned(&mut self) {
        self.learned += 1;
    }
    pub fn defer(&mut self, index: usize) {
        self.waiting.push_back((index, Some(self.learned)));
    }
}

/// Teaches the files in order. Each lesson gets the initial attempt plus the retries of
/// `cfg.retry` (gentler learning rate, focused strategy, more epochs); a lesson still not learned
/// moves to the end of the queue and is tried once more after other lessons have been learned.
/// A file that cannot be taught at all is recorded and skipped; it does not end the course.
/// Learned files join the archive, so later files rehearse and protect them. `on_learned`
/// receives the new model and archive right after each learned file, so a frontend can save
/// progress immediately; an error from it ends the course.
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
                    r.outcome = FileOutcome::KeptPrevious;
                    r.error = Some(
                        "not learned; no other lesson was learned after it was moved to the end of the queue"
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
                let mut learned = false;
                // A lesson without training units or held-out questions cannot be judged: no attempt.
                let judgeable =
                    record.train_units > 0 && items.iter().any(|i| i.held_out && i.kind != Kind::Prose);
                if !judgeable {
                    record.error = Some(
                        "The lesson needs training units and held-out units (about 10% are held out by hash)"
                            .into(),
                    );
                }
                for attempt in 0..=retries {
                    if !judgeable {
                        break;
                    }
                    let acfg = attempt_config(cfg, attempt);
                    observer(&CourseEvent::AttemptStarted {
                        index,
                        attempt,
                        attempts: retries + 1,
                        learning_rate: acfg.learning_rate,
                        epochs: acfg.epochs,
                        strategy: acfg.strategy,
                    });
                    let mut forward = |e: &Event| observer(&CourseEvent::Session { index, event: e.clone() });
                    let r = match teach(&current, &items, &archive, &acfg, &mut forward, stop) {
                        Ok(r) => r,
                        Err(e) => {
                            record.error = Some(e);
                            break;
                        }
                    };
                    record.epochs_run += r.epochs.len();
                    if record.baseline.is_none() {
                        record.baseline = Some(r.baseline.clone());
                    }
                    record.final_score = Some(r.final_score.clone());
                    record.attempts.push(AttemptRecord {
                        attempt,
                        learning_rate: acfg.learning_rate,
                        epochs: acfg.epochs,
                        strategy: acfg.strategy,
                        learned: r.learned,
                        epochs_run: r.epochs.len(),
                        final_score: r.final_score.clone(),
                    });
                    if r.stopped {
                        stopped = true;
                        record.outcome = FileOutcome::Stopped;
                        break;
                    }
                    if r.learned {
                        learned = true;
                        current = r.model;
                        archive.extend(items.iter().cloned());
                        on_learned(&current, &archive)?;
                        queue.learned();
                        record.outcome = FileOutcome::Learned;
                        break;
                    }
                }
                if !learned && !stopped && record.error.is_none() {
                    if pass == 1 && cfg.retry.requeue {
                        record.outcome = FileOutcome::Deferred;
                        queue.defer(index);
                    } else {
                        record.outcome = FileOutcome::KeptPrevious;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deferred_lessons_return_only_after_progress() {
        let mut q = Queue::new(3);
        // Lesson 0 fails and is deferred; lesson 1 is learned; lesson 2 fails and is deferred.
        assert!(matches!(q.pop(), Some(Next::Teach { index: 0, pass: 1 })));
        q.defer(0);
        assert!(matches!(q.pop(), Some(Next::Teach { index: 1, pass: 1 })));
        q.learned();
        assert!(matches!(q.pop(), Some(Next::Teach { index: 2, pass: 1 })));
        q.defer(2);
        // Lesson 0 comes back (something was learned after it was deferred) ...
        assert!(matches!(q.pop(), Some(Next::Teach { index: 0, pass: 2 })));
        // ... lesson 2 does not: nothing was learned after it was deferred.
        assert!(matches!(q.pop(), Some(Next::Exhausted { index: 2 })));
        assert!(q.pop().is_none());
    }

    #[test]
    fn retry_plan_matches_foundation_v2() {
        let base = SessionConfig { learning_rate: 1e-3, epochs: 4, ..SessionConfig::default() };
        let plan: Vec<_> = (0..4).map(|a| attempt_config(&base, a)).collect();
        assert_eq!(plan[0].learning_rate, 1e-3);
        assert_eq!(plan[1].learning_rate, 5e-4);
        assert_eq!((plan[2].learning_rate, plan[2].epochs, plan[2].strategy), (1e-3, 12, Strategy::Focused));
        assert_eq!(
            (plan[3].learning_rate, plan[3].epochs, plan[3].strategy),
            (2.5e-4, 10, Strategy::Focused)
        );
        // Rates below 0.00002 are allowed, down to the floor.
        let low = SessionConfig { learning_rate: 1e-5, ..base };
        assert_eq!(attempt_config(&low, 3).learning_rate, 2.5e-6);
        let tiny = SessionConfig { learning_rate: 2e-6, ..base };
        assert_eq!(attempt_config(&tiny, 3).learning_rate, 1e-6);
    }
}
