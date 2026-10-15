//! Courses: several lesson files taught one after another (e.g. the files of a ZIP). Reading
//! the ZIP is the frontend's job; the Engine receives the files and records exactly which
//! content was taught, when, and with what result.
use super::lessons::{parse_lesson, Item};
use super::session::{teach, Event, Score, SessionConfig};
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
    /// No candidate qualified; the model before this file was kept.
    KeptPrevious,
    /// The same content was already taught earlier in this course; not taught again.
    Duplicate,
    /// The file could not be taught (parse error, no held-out units, ...); see `error`.
    Failed,
    /// Teaching was stopped during this file; its epoch in progress was discarded.
    Stopped,
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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CourseEvent {
    FileStarted { index: usize, files: usize, name: String, sha256: String },
    Session { index: usize, event: Event },
    FileFinished(FileRecord),
}

pub struct CourseResult {
    pub model: Bicameral,
    /// Items of every lesson learned so far (the given archive plus this course's learned files),
    /// for rehearsal and retention checks in later sessions.
    pub archive: Vec<Item>,
    pub files: Vec<FileRecord>,
    pub stopped: bool,
}

/// Orders files by name, like the reference's `sorted(glob(...))` (lesson files are numbered).
pub fn sort_files(files: &mut [CourseFile]) {
    files.sort_by(|a, b| a.name.cmp(&b.name));
}

/// Teaches each file in order. A file that cannot be taught is recorded and skipped; it does not
/// end the course. Learned files join the archive, so later files rehearse and protect them.
pub fn teach_course(
    model: &Bicameral,
    files: &[CourseFile],
    archive: &[Item],
    cfg: &SessionConfig,
    observer: &mut dyn FnMut(&CourseEvent),
    stop: &AtomicBool,
) -> Result<CourseResult, String> {
    let mut current = model.clone();
    let mut archive = archive.to_vec();
    let mut records: Vec<FileRecord> = Vec::new();
    let mut stopped = false;
    for (index, file) in files.iter().enumerate() {
        if stop.load(Ordering::Acquire) {
            stopped = true;
            break;
        }
        let sha256 = sha256_hex(file.text.as_bytes());
        observer(&CourseEvent::FileStarted {
            index,
            files: files.len(),
            name: file.name.clone(),
            sha256: sha256.clone(),
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
        };
        if let Some(first) = records.iter().find(|r| r.sha256 == sha256) {
            record.outcome = FileOutcome::Duplicate;
            record.error = Some(format!("same content as {}", first.name));
        } else {
            match parse_lesson(&file.text, &file.name) {
                Err(e) => record.error = Some(e),
                Ok(items) => {
                    record.train_units = items.iter().filter(|i| !i.held_out).count();
                    record.heldout_units = items.iter().filter(|i| i.held_out).count();
                    let mut forward = |e: &Event| observer(&CourseEvent::Session { index, event: e.clone() });
                    match teach(&current, &items, &archive, cfg, &mut forward, stop) {
                        Err(e) => record.error = Some(e),
                        Ok(r) => {
                            record.epochs_run = r.epochs.len();
                            record.baseline = Some(r.baseline.clone());
                            record.final_score = Some(r.final_score.clone());
                            record.outcome = if r.stopped {
                                FileOutcome::Stopped
                            } else if r.learned {
                                FileOutcome::Learned
                            } else {
                                FileOutcome::KeptPrevious
                            };
                            stopped = r.stopped;
                            if r.learned {
                                current = r.model;
                                archive.extend(items);
                            }
                        }
                    }
                }
            }
        }
        record.end_step = current.step;
        observer(&CourseEvent::FileFinished(record.clone()));
        records.push(record);
        if stopped {
            break;
        }
    }
    Ok(CourseResult { model: current, archive, files: records, stopped })
}
