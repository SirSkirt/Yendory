//! Running a course: optional pre-test, lessons taught in their stages, the course grade at the
//! end (70% lessons re-graded on the final model, 30% final exam), and up to three remediation
//! attempts with different strategies when the course is failed. Progress is saved after every
//! lesson and every remediation session so an interrupted run resumes where it stopped.
//! The rules are written out in docs/LEARNING.md ("Courses").
use super::course::{teach_group, teach_lesson, CourseEvent, FileOutcome, FileRecord, Next, Queue};
use super::curriculum::{PreparedCourse, PreparedLesson, Section};
use super::eval::{evaluate, Report};
use super::lessons::{Item, Kind};
use super::session::{
    batch_loss_and_gradients_loss, rows_of, EducationStage, Event, RetryPolicy, SessionConfig, Strategy,
};
use super::Rng;
use crate::model::{Ablation, Bicameral};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

/// Share of the course grade from the lessons; the final exam gives the rest.
pub const LESSON_SHARE: f32 = 0.7;
/// Remediation attempts per course run.
pub const REMEDIATION_ATTEMPTS: u32 = 3;
/// Lowest learning rate a remediation attempt uses.
pub const MIN_LEARNING_RATE: f32 = 1e-6;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SectionGrade {
    pub section: Section,
    /// Share of the exam.
    pub weight: f32,
    pub questions: usize,
    pub score: f32,
    /// The same questions with Dense's input to the transformer cut (recorded, not graded).
    pub without_dense: f32,
    pub report: Report,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExamGrade {
    pub sections: Vec<SectionGrade>,
    pub score: f32,
    pub without_dense: f32,
    /// Held-out loss rise on the exam questions when Dense's input is cut (recorded, not graded).
    pub dense_reliance: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LessonGrade {
    pub name: String,
    pub stage: Option<EducationStage>,
    pub pass_mark: f32,
    pub questions: usize,
    /// Right after the lesson was taught.
    pub when_taught: Option<f32>,
    /// Re-graded on the model at the end of the course (counts for the grade).
    pub course_end: f32,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CourseGrade {
    pub lessons: Vec<LessonGrade>,
    pub lesson_average: f32,
    pub exam: ExamGrade,
    /// 70% lesson average + 30% exam.
    pub grade: f32,
    /// The average of the lesson pass marks.
    pub required: f32,
    pub exam_minimum: Option<f32>,
    pub passed: bool,
    /// Lessons below their own pass mark (recorded even when the course passes).
    pub failed_lessons: Vec<String>,
    pub step: u64,
}

/// Why lessons fell short at the end of the course.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CourseCause {
    /// Passed when taught, below the mark at the end of the course.
    Forgotten,
    /// It cannot answer the lesson's own training questions.
    NotLearned,
    /// It answers the training questions but not the evaluation questions.
    NotGeneralising,
    /// The previous remediation attempt made the course grade worse.
    Unstable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemedyStrategy {
    /// Wrong answers twice, extra practice, a quarter of what it knows; easiest first.
    TargetedPractice,
    /// The whole lesson plus extra practice with 40% rehearsal of the course's other lessons.
    Consolidation,
    /// All units and explanations plus a varied share of the practice, more epochs.
    BroaderStudy,
    /// Smaller batches, Foundation candidates (each part of the stage on its own), more epochs.
    FineGrained,
}
impl RemedyStrategy {
    fn preference(cause: CourseCause) -> [RemedyStrategy; 4] {
        use RemedyStrategy::*;
        match cause {
            CourseCause::NotLearned => [TargetedPractice, FineGrained, BroaderStudy, Consolidation],
            CourseCause::Forgotten => [Consolidation, TargetedPractice, FineGrained, BroaderStudy],
            CourseCause::NotGeneralising => [BroaderStudy, Consolidation, FineGrained, TargetedPractice],
            CourseCause::Unstable => [FineGrained, TargetedPractice, Consolidation, BroaderStudy],
        }
    }
    pub fn describe(self) -> &'static str {
        match self {
            RemedyStrategy::TargetedPractice => {
                "targeted practice: wrong answers twice, extra practice material, a quarter of what it knows, easiest first"
            }
            RemedyStrategy::Consolidation => {
                "consolidation: the whole lesson and its extra practice with 40% rehearsal of the course's other lessons"
            }
            RemedyStrategy::BroaderStudy => {
                "broader study: all examples and explanations with a varied share of the extra practice, more epochs"
            }
            RemedyStrategy::FineGrained => {
                "fine-grained: smaller batches, each part of the stage tried on its own (Foundation), more epochs"
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RemediationRecord {
    /// 1..=3.
    pub attempt: u32,
    pub cause: CourseCause,
    pub strategy: RemedyStrategy,
    pub learning_rate: f32,
    /// Lessons practised (indices into the course), in course order.
    pub targets: Vec<usize>,
    pub target_names: Vec<String>,
    /// Target sessions finished so far (for resuming).
    pub done: usize,
    pub description: String,
    pub grade_after: Option<f32>,
    /// The attempt lowered the course grade; the best checkpoint was restored.
    pub reverted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Pretest,
    Teaching,
    Remediation,
    Passed,
    /// Not passed after three remediation attempts: the plan stops on the best checkpoint.
    Failed,
    /// The course took longer than its time limit: the plan stops on the best checkpoint.
    TimedOut,
}

/// The state of one run of a course: everything needed to resume it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CourseRun {
    pub course_id: String,
    pub name: String,
    pub fingerprint: String,
    /// This run was granted by a manual override of an unchanged failed course.
    pub overridden: bool,
    pub phase: Phase,
    pub pretest: Option<ExamGrade>,
    pub queue: Queue,
    /// Final record of each lesson's teaching.
    pub records: Vec<Option<FileRecord>>,
    pub when_taught: Vec<Option<f32>>,
    /// The course grade after teaching and after each remediation attempt.
    pub grades: Vec<CourseGrade>,
    pub remediation: Vec<RemediationRecord>,
    pub best_grade: Option<f32>,
    /// Practice items of each lesson already added to the rehearsal archive.
    pub practice_archived: Vec<bool>,
    pub elapsed_secs: f64,
    /// Model step when the run was last saved (to detect a model changed elsewhere).
    pub saved_step: u64,
    pub seed: u64,
}
impl CourseRun {
    pub fn new(prep: &PreparedCourse, overridden: bool, seed: u64) -> Self {
        let n = prep.lessons.len();
        CourseRun {
            course_id: prep.id.clone(),
            name: prep.name.clone(),
            fingerprint: prep.fingerprint.clone(),
            overridden,
            phase: Phase::Pretest,
            pretest: None,
            queue: Queue::new(n),
            records: vec![None; n],
            when_taught: vec![None; n],
            grades: Vec::new(),
            remediation: Vec::new(),
            best_grade: None,
            practice_archived: vec![false; n],
            elapsed_secs: 0.,
            saved_step: 0,
            seed,
        }
    }
    pub fn finished(&self) -> bool {
        matches!(self.phase, Phase::Passed | Phase::Failed | Phase::TimedOut)
    }
}

/// A model with its rehearsal archive: the best checkpoint of a course run.
#[derive(Clone)]
pub struct Snapshot {
    pub model: Bicameral,
    pub archive: Vec<Item>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProgramEvent {
    CourseStarted {
        name: String,
        lessons: usize,
        resumed: bool,
    },
    Pretest(ExamGrade),
    /// Teaching events of lesson `index` (attempts, diagnoses, sessions, file records).
    Lesson {
        index: usize,
        stage: Option<EducationStage>,
        event: CourseEvent,
    },
    LessonTaught {
        index: usize,
        name: String,
        outcome: FileOutcome,
        when_taught: f32,
        /// The lesson's record: held-out and own-question progress apart from the pass mark.
        #[serde(default)]
        record: Option<Box<FileRecord>>,
    },
    Graded {
        after_attempt: u32,
        grade: CourseGrade,
    },
    RemediationStarted(RemediationRecord),
    /// Session events of a remediation session on lesson `index`.
    Remediation {
        attempt: u32,
        index: usize,
        name: String,
        event: Event,
    },
    RemediationFinished(RemediationRecord),
    CourseFinished {
        phase: Phase,
        best_grade: Option<f32>,
    },
}

/// Where a course run reports to and saves.
pub trait CourseHost {
    fn event(&mut self, e: &ProgramEvent);
    /// Saves the model, archive and run state, in that order.
    fn save(&mut self, model: &Bicameral, archive: &[Item], run: &CourseRun) -> Result<(), String>;
    /// Saves the best checkpoint of the run.
    fn save_best(&mut self, best: &Snapshot) -> Result<(), String>;
    /// Study Mode for the course's lessons (see `memory::Studying`); none by default.
    fn study(&mut self) -> Option<&mut dyn crate::memory::StudyHook> {
        None
    }
    /// The reward ledger and settings for this run (none by default). It is taken when the run
    /// starts and given back with [`CourseHost::return_rewards`] when it ends, whatever the
    /// outcome; the course's final exam is protected from it.
    fn take_rewards(&mut self) -> Option<super::reward::Rewards> {
        None
    }
    fn return_rewards(&mut self, _rewards: super::reward::Rewards) {}
}

#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Vocabulary limit of the model (see `SessionConfig::word_limit`).
    pub word_limit: Option<usize>,
    /// General reading text for `CourseSettings::reading_share` (documents).
    pub reading: Option<std::sync::Arc<Vec<String>>>,
    /// Where accepted epochs are written while lessons are taught (see
    /// `SessionConfig::checkpoint_dir`).
    pub checkpoint_dir: Option<std::path::PathBuf>,
}

pub struct RunResult {
    pub model: Bicameral,
    pub archive: Vec<Item>,
    pub stopped: bool,
}

/// Share of `items`' questions answered correctly (conversation turns count separately).
fn rate(model: &Bicameral, items: &[&Item], abl: Ablation) -> Result<(f32, Report), String> {
    let report = evaluate(model, items, abl)?;
    let o = report.overall();
    Ok((if o.total == 0 { 0. } else { o.correct as f32 / o.total as f32 }, report))
}

pub fn lesson_score(model: &Bicameral, lesson: &PreparedLesson) -> Result<f32, String> {
    Ok(rate(model, &lesson.eval_items(), Ablation::default())?.0)
}

pub fn grade_exam(model: &Bicameral, prep: &PreparedCourse) -> Result<ExamGrade, String> {
    let cut = Ablation { no_afferent: true, ..Ablation::default() };
    let mut sections = Vec::new();
    for s in &prep.exam {
        let items: Vec<&Item> = s.items.iter().collect();
        let (score, report) = rate(model, &items, Ablation::default())?;
        let (without_dense, _) = rate(model, &items, cut)?;
        sections.push(SectionGrade {
            section: s.section,
            weight: s.weight,
            questions: report.overall().total,
            score,
            without_dense,
            report,
        });
    }
    let all: Vec<&Item> = prep.exam.iter().flat_map(|s| &s.items).collect();
    let rows = rows_of(model, &all, &mut Rng::new(1))?;
    let dense_reliance = batch_loss_and_gradients_loss(model, &rows, cut)?
        - batch_loss_and_gradients_loss(model, &rows, Ablation::default())?;
    Ok(ExamGrade {
        score: sections.iter().map(|s| s.weight * s.score).sum(),
        without_dense: sections.iter().map(|s| s.weight * s.without_dense).sum(),
        sections,
        dense_reliance,
    })
}

pub fn grade_course(
    model: &Bicameral,
    prep: &PreparedCourse,
    run: &CourseRun,
) -> Result<CourseGrade, String> {
    let mut lessons = Vec::new();
    for (i, l) in prep.lessons.iter().enumerate() {
        let course_end = lesson_score(model, l)?;
        lessons.push(LessonGrade {
            name: l.name.clone(),
            stage: l.stage,
            pass_mark: l.pass_mark,
            questions: l.eval_items().len(),
            when_taught: run.when_taught.get(i).copied().flatten(),
            course_end,
            passed: course_end >= l.pass_mark,
        });
    }
    let lesson_average = if lessons.is_empty() {
        0.
    } else {
        lessons.iter().map(|l| l.course_end).sum::<f32>() / lessons.len() as f32
    };
    let exam = grade_exam(model, prep)?;
    let grade = LESSON_SHARE * lesson_average + (1. - LESSON_SHARE) * exam.score;
    let exam_ok = prep.exam_minimum.map_or(true, |m| exam.score + 1e-6 >= m);
    Ok(CourseGrade {
        failed_lessons: lessons.iter().filter(|l| !l.passed).map(|l| l.name.clone()).collect(),
        passed: grade + 1e-6 >= prep.required_grade && exam_ok,
        lessons,
        lesson_average,
        exam,
        grade,
        required: prep.required_grade,
        exam_minimum: prep.exam_minimum,
        step: model.step,
    })
}

fn session_cfg(prep: &PreparedCourse, l: &PreparedLesson, opts: &RunOptions, seed: u64) -> SessionConfig {
    let s = &prep.settings;
    SessionConfig {
        epochs: s.epochs,
        batch_size: s.batch_size,
        learning_rate: s.learning_rate,
        foundation: s.foundation,
        rehearsal_share: if s.rehearsal { 0.10 } else { 0. },
        protect_correct_answers: s.protect_answers,
        protect_answers_net: s.protect_answers_net,
        retention_relative: s.retention_relative,
        retention_units: s.retention_units.max(1),
        pass_mark: l.pass_mark,
        retry: RetryPolicy { retries: s.retries.min(3), requeue: s.requeue, ..RetryPolicy::default() },
        education: l.stage,
        patience: s.flat_epochs.max(1),
        word_limit: opts.word_limit,
        seed,
        dropout: s.dropout,
        reading_share: s.reading_share,
        reading: opts.reading.clone(),
        blend: s.blend,
        checkpoint_dir: opts.checkpoint_dir.clone(),
        train_earlier: s.train_earlier,
        recover_earlier: s.recover_earlier,
        backoff: s.backoff,
        override_first_epoch: s.override_first_epoch,
        ..SessionConfig::default()
    }
}

/// Plans the next remediation attempt from the latest course grade.
fn plan_remediation(
    model: &Bicameral,
    prep: &PreparedCourse,
    run: &CourseRun,
    grade: &CourseGrade,
    attempt: u32,
) -> Result<RemediationRecord, String> {
    let mut targets: Vec<usize> = (0..prep.lessons.len()).filter(|&i| !grade.lessons[i].passed).collect();
    if targets.is_empty() {
        // Every lesson reached its mark but the course grade did not (the exam): practise the
        // lessons with the lowest course-end scores, chosen from lesson results only.
        let avg = grade.lesson_average;
        targets = (0..prep.lessons.len()).filter(|&i| grade.lessons[i].course_end <= avg).collect();
    }
    let mut counts = [0usize; 3];
    for &i in &targets {
        let l = &prep.lessons[i];
        let end = grade.lessons[i].course_end;
        let cause = if run.when_taught[i].is_some_and(|w| w >= l.pass_mark) && end < l.pass_mark {
            CourseCause::Forgotten
        } else {
            let train: Vec<&Item> = l.train_items().into_iter().filter(|it| it.kind != Kind::Prose).collect();
            if rate(model, &train, Ablation::default())?.0 < l.pass_mark {
                CourseCause::NotLearned
            } else {
                CourseCause::NotGeneralising
            }
        };
        counts[match cause {
            CourseCause::Forgotten => 0,
            CourseCause::NotLearned => 1,
            _ => 2,
        }] += 1;
    }
    let mut cause = [CourseCause::Forgotten, CourseCause::NotLearned, CourseCause::NotGeneralising]
        [(0..3).max_by_key(|&k| (counts[k], 3 - k)).unwrap_or(1)];
    if run.remediation.last().is_some_and(|r| r.reverted) {
        cause = CourseCause::Unstable;
    }
    let used: Vec<RemedyStrategy> = run.remediation.iter().map(|r| r.strategy).collect();
    let strategy = RemedyStrategy::preference(cause)
        .into_iter()
        .find(|s| !used.contains(s))
        .unwrap_or(RemedyStrategy::FineGrained);
    let learning_rate = (prep.settings.learning_rate * 0.5f32.powi(attempt as i32)).max(MIN_LEARNING_RATE);
    Ok(RemediationRecord {
        attempt,
        cause,
        strategy,
        learning_rate,
        target_names: targets.iter().map(|&i| prep.lessons[i].name.clone()).collect(),
        targets,
        done: 0,
        description: format!("{}; learning rate {}", strategy.describe(), super::rate_text(learning_rate)),
        grade_after: None,
        reverted: false,
    })
}

/// The practice selection and settings of one remediation session.
fn remedy_session(
    model: &Bicameral,
    prep: &PreparedCourse,
    lesson: &PreparedLesson,
    rec: &RemediationRecord,
    opts: &RunOptions,
    seed: u64,
) -> Result<(SessionConfig, Vec<Item>), String> {
    let mut cfg = session_cfg(prep, lesson, opts, seed);
    cfg.learning_rate = rec.learning_rate;
    cfg.protect_correct_answers = true;
    let base_epochs = prep.settings.epochs;
    let train: Vec<&Item> = lesson.train_items();
    let practice = &lesson.practice;
    let mut items: Vec<Item> = Vec::new();
    match rec.strategy {
        RemedyStrategy::TargetedPractice => {
            let questions: Vec<&Item> = train.iter().copied().filter(|it| it.kind != Kind::Prose).collect();
            let report = evaluate(model, &questions, Ablation::default())?;
            // One outcome per item, except conversations (one per turn): an item is right when
            // all its outcomes are.
            let mut outcomes = report.outcomes.iter();
            for (k, it) in questions.iter().enumerate() {
                let n = it.outcome_count();
                let ok = outcomes.by_ref().take(n).all(|o| o.correct);
                let copies = if ok { usize::from(k % 4 == 0) } else { 2 };
                items.extend(std::iter::repeat((*it).clone()).take(copies));
            }
            items.extend(practice.iter().filter(|it| it.kind != Kind::Prose).cloned());
            cfg.strategy = Strategy::Focused;
            cfg.epochs = base_epochs.max(8);
            cfg.rehearsal_share = 0.25;
        }
        RemedyStrategy::Consolidation => {
            items.extend(train.iter().map(|it| (*it).clone()));
            items.extend(practice.iter().cloned());
            cfg.strategy = Strategy::Standard;
            cfg.epochs = base_epochs.max(6);
            cfg.rehearsal_share = 0.4;
        }
        RemedyStrategy::BroaderStudy => {
            items.extend(train.iter().map(|it| (*it).clone()));
            let mut rng = Rng::new(seed ^ 0xB0AD);
            items.extend(practice.iter().filter(|_| rng.unit() < 2. / 3.).cloned());
            cfg.strategy = Strategy::Standard;
            cfg.epochs = base_epochs + 6;
        }
        RemedyStrategy::FineGrained => {
            items.extend(train.iter().map(|it| (*it).clone()));
            items.extend(practice.iter().cloned());
            cfg.strategy = Strategy::Standard;
            cfg.batch_size = (prep.settings.batch_size / 2).max(4);
            cfg.foundation = true;
            cfg.epochs = base_epochs + 4;
        }
    }
    if items.iter().all(|it| it.kind == Kind::Prose) {
        items.extend(train.iter().map(|it| (*it).clone()));
    }
    Ok((cfg, items))
}

fn time_up(prep: &PreparedCourse, run: &CourseRun, started: std::time::Instant, elapsed0: f64) -> bool {
    let elapsed = elapsed0 + started.elapsed().as_secs_f64();
    prep.settings.time_limit_hours.is_some_and(|h| elapsed > h as f64 * 3600.) && !run.finished()
}

/// Runs (or resumes) a course. `best` is the run's best checkpoint saved earlier (when resuming
/// a remediation). Returns the model to continue with: after a failed or timed-out course, the
/// best checkpoint of the run.
pub fn run_course(
    model: &Bicameral,
    archive: &[Item],
    prep: &PreparedCourse,
    run: &mut CourseRun,
    best: Option<Snapshot>,
    opts: &RunOptions,
    host: &mut dyn CourseHost,
    stop: &AtomicBool,
) -> Result<RunResult, String> {
    let mut rewards = host.take_rewards();
    if let Some(r) = rewards.as_mut() {
        // The final exam stays independent: its questions are never recorded or credited.
        r.protect(prep.exam.iter().flat_map(|s| &s.items));
    }
    let result = run_course_inner(model, archive, prep, run, best, opts, host, rewards.as_mut(), stop);
    if let Some(r) = rewards {
        host.return_rewards(r);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn run_course_inner(
    model: &Bicameral,
    archive: &[Item],
    prep: &PreparedCourse,
    run: &mut CourseRun,
    best: Option<Snapshot>,
    opts: &RunOptions,
    host: &mut dyn CourseHost,
    mut rewards: Option<&mut super::reward::Rewards>,
    stop: &AtomicBool,
) -> Result<RunResult, String> {
    if run.fingerprint != prep.fingerprint || run.records.len() != prep.lessons.len() {
        return Err("The course changed since this run started; start a new run of the revised course".into());
    }
    let mut current = model.clone();
    let mut archive = archive.to_vec();
    let mut best = best;
    let started = std::time::Instant::now();
    let elapsed0 = run.elapsed_secs;
    host.event(&ProgramEvent::CourseStarted {
        name: prep.name.clone(),
        lessons: prep.lessons.len(),
        resumed: run.phase != Phase::Pretest,
    });
    let save = |host: &mut dyn CourseHost, m: &Bicameral, a: &[Item], run: &mut CourseRun| {
        run.elapsed_secs = elapsed0 + started.elapsed().as_secs_f64();
        run.saved_step = m.step;
        host.save(m, a, run)
    };
    loop {
        if !run.finished() && time_up(prep, run, started, elapsed0) {
            run.phase = Phase::TimedOut;
            if let Some(b) = &best {
                current = b.model.clone();
                archive = b.archive.clone();
            }
            save(host, &current, &archive, run)?;
            continue;
        }
        match run.phase {
            Phase::Pretest => {
                if prep.pretest {
                    let g = grade_exam(&current, prep)?;
                    host.event(&ProgramEvent::Pretest(g.clone()));
                    run.pretest = Some(g);
                }
                run.phase = Phase::Teaching;
                save(host, &current, &archive, run)?;
            }
            Phase::Teaching => {
                // Lessons taught together: the "together" lessons of one stage not yet taught,
                // in one mixed session (the stage of the earliest such lesson first).
                let waiting = run.queue.waiting_first();
                let pending: Vec<usize> = (0..prep.lessons.len())
                    .filter(|&i| prep.lessons[i].together && run.records[i].is_none() && waiting.contains(&i))
                    .collect();
                if let Some(&first) = pending.first() {
                    let stage = prep.lessons[first].stage;
                    let group: Vec<usize> =
                        pending.into_iter().filter(|&i| prep.lessons[i].stage == stage).collect();
                    if group.len() >= 2 {
                        if teach_course_group(
                            &mut current,
                            &mut archive,
                            prep,
                            run,
                            &group,
                            opts,
                            host,
                            rewards.as_deref_mut(),
                            stop,
                        )? {
                            save(host, &current, &archive, run)?;
                            return Ok(RunResult { model: current, archive, stopped: true });
                        }
                        save(host, &current, &archive, run)?;
                        continue;
                    }
                }
                let before = run.queue.clone();
                let Some(next) = run.queue.pop() else {
                    let grade = grade_course(&current, prep, run)?;
                    host.event(&ProgramEvent::Graded { after_attempt: 0, grade: grade.clone() });
                    run.grades.push(grade.clone());
                    run.best_grade = Some(grade.grade);
                    if grade.passed {
                        run.phase = Phase::Passed;
                    } else {
                        let b = Snapshot { model: current.clone(), archive: archive.clone() };
                        host.save_best(&b)?;
                        best = Some(b);
                        run.phase = Phase::Remediation;
                    }
                    save(host, &current, &archive, run)?;
                    continue;
                };
                let (index, pass) = match next {
                    Next::Teach { index, pass } => (index, pass),
                    Next::Exhausted { index } => {
                        if let Some(r) = run.records[index].as_mut() {
                            r.outcome = FileOutcome::Unresolved;
                            r.error = Some(
                                "not passed; the model did not change after it was moved to the end of the queue"
                                    .into(),
                            );
                        }
                        continue;
                    }
                };
                let lesson = &prep.lessons[index];
                let cfg =
                    session_cfg(prep, lesson, opts, run.seed.wrapping_add(index as u64 * 7919 + pass as u64));
                let mut record = FileRecord {
                    index,
                    name: lesson.name.clone(),
                    sha256: lesson.sha256.clone(),
                    bytes: 0,
                    train_units: lesson.items.iter().filter(|i| !i.held_out).count(),
                    heldout_units: lesson.items.iter().filter(|i| i.held_out).count(),
                    start_step: current.step,
                    end_step: current.step,
                    outcome: FileOutcome::Failed,
                    error: None,
                    epochs_run: 0,
                    baseline: None,
                    final_score: None,
                    pass,
                    attempts: Vec::new(),
                    pass_mark: lesson.pass_mark,
                    own_before: super::course::own_rate(&current, &lesson.items).ok(),
                    own_after: None,
                    with_memory: None,
                };
                let mut stopped = false;
                let stage = lesson.stage;
                let studied = match host.study() {
                    Some(st) => Some(st.before_lesson(&current, &lesson.name, &lesson.items)?),
                    None => None,
                };
                let taught = match studied {
                    Some((taught, report, copies)) => {
                        if let Some(report) = report {
                            host.event(&ProgramEvent::Lesson {
                                index,
                                stage,
                                event: CourseEvent::Studied { index, report, copies },
                            });
                        }
                        taught
                    }
                    None => lesson.items.clone(),
                };
                teach_lesson(
                    &mut current,
                    &mut archive,
                    &taught,
                    index,
                    cfg.retry.retries,
                    &cfg,
                    &mut record,
                    &mut stopped,
                    &mut run.queue,
                    rewards.as_deref_mut(),
                    &mut |e| host.event(&ProgramEvent::Lesson { index, stage, event: e.clone() }),
                    &mut |_, _| Ok(()),
                    stop,
                )?;
                if stopped {
                    // The lesson starts again when the run resumes (accepted changes are kept).
                    run.queue = before;
                    save(host, &current, &archive, run)?;
                    return Ok(RunResult { model: current, archive, stopped: true });
                }
                if let Some(e) = record.error.clone() {
                    return Err(format!("{}: {e}", lesson.name));
                }
                record.own_after = super::course::own_rate(&current, &lesson.items).ok();
                record.with_memory = host.study().and_then(|st| st.after_lesson(&current, &lesson.items));
                if record.outcome != FileOutcome::Passed {
                    record.outcome = if pass == 1 && prep.settings.requeue {
                        run.queue.defer(index);
                        FileOutcome::Deferred
                    } else {
                        FileOutcome::Unresolved
                    };
                }
                record.end_step = current.step;
                let score = lesson_score(&current, lesson)?;
                run.when_taught[index] = Some(score);
                host.event(&ProgramEvent::LessonTaught {
                    index,
                    name: lesson.name.clone(),
                    outcome: record.outcome,
                    when_taught: score,
                    record: Some(Box::new(record.clone())),
                });
                run.records[index] = Some(record);
                save(host, &current, &archive, run)?;
            }
            Phase::Remediation => {
                let finished = run.remediation.iter().filter(|r| r.grade_after.is_some()).count() as u32;
                if run.remediation.last().map_or(true, |r| r.grade_after.is_some()) {
                    if finished >= REMEDIATION_ATTEMPTS {
                        run.phase = Phase::Failed;
                        if let Some(b) = &best {
                            current = b.model.clone();
                            archive = b.archive.clone();
                        }
                        save(host, &current, &archive, run)?;
                        continue;
                    }
                    let grade = run.grades.last().cloned().ok_or("no course grade to remediate")?;
                    let rec = plan_remediation(&current, prep, run, &grade, finished + 1)?;
                    host.event(&ProgramEvent::RemediationStarted(rec.clone()));
                    run.remediation.push(rec);
                    save(host, &current, &archive, run)?;
                }
                let k = run.remediation.len() - 1;
                while run.remediation[k].done < run.remediation[k].targets.len() {
                    if time_up(prep, run, started, elapsed0) {
                        break;
                    }
                    let rec = run.remediation[k].clone();
                    let index = rec.targets[rec.done];
                    let lesson = &prep.lessons[index];
                    // Lessons taught together are practised together: the following targets
                    // of the same stage that are "together" lessons join this session.
                    let group: Vec<usize> = if lesson.together {
                        rec.targets[rec.done..]
                            .iter()
                            .copied()
                            .take_while(|&i| {
                                prep.lessons[i].together && prep.lessons[i].stage == lesson.stage
                            })
                            .collect()
                    } else {
                        vec![index]
                    };
                    let seed = run.seed ^ (rec.attempt as u64 * 0x9E37) ^ (index as u64 * 7919);
                    let (cfg, mut practice) = remedy_session(&current, prep, lesson, &rec, opts, seed)?;
                    let mut items = lesson.items.clone();
                    for &j in &group[1..] {
                        let (_, more) =
                            remedy_session(&current, prep, &prep.lessons[j], &rec, opts, seed ^ j as u64)?;
                        practice.extend(more);
                        items.extend(prep.lessons[j].items.iter().cloned());
                    }
                    let names: Vec<&str> = group.iter().map(|&j| prep.lessons[j].name.as_str()).collect();
                    let others: Vec<Item> =
                        archive.iter().filter(|i| !names.contains(&i.source.as_str())).cloned().collect();
                    // Together lessons also practise the saved earlier lessons in the same mix
                    // (they stay guarded as earlier lessons).
                    if lesson.together && cfg.train_earlier {
                        items.extend(others.iter().cloned());
                        practice.extend(others.iter().filter(|i| !i.held_out).cloned());
                    }
                    let name = names.join(" + ");
                    let attempt = rec.attempt;
                    let r = super::session::teach_rewarded(
                        &current,
                        &items,
                        Some(&practice),
                        &others,
                        &cfg,
                        rewards.as_deref_mut(),
                        &mut |e| {
                            host.event(&ProgramEvent::Remediation {
                                attempt,
                                index,
                                name: name.clone(),
                                event: e.clone(),
                            })
                        },
                        stop,
                    )?;
                    if r.learned {
                        current = r.model;
                        for &j in &group {
                            let l = &prep.lessons[j];
                            if !archive.iter().any(|i| i.source == l.name) {
                                archive.extend(l.items.iter().cloned());
                            }
                            if !run.practice_archived[j] {
                                archive.extend(l.practice.iter().cloned());
                                run.practice_archived[j] = true;
                            }
                        }
                    }
                    if r.stopped || stop.load(Ordering::Acquire) {
                        save(host, &current, &archive, run)?;
                        return Ok(RunResult { model: current, archive, stopped: true });
                    }
                    run.remediation[k].done += group.len();
                    save(host, &current, &archive, run)?;
                }
                if run.remediation[k].done < run.remediation[k].targets.len() {
                    continue; // time limit
                }
                let attempt = run.remediation[k].attempt;
                let grade = grade_course(&current, prep, run)?;
                host.event(&ProgramEvent::Graded { after_attempt: attempt, grade: grade.clone() });
                run.grades.push(grade.clone());
                run.remediation[k].grade_after = Some(grade.grade);
                let best_grade = run.best_grade.unwrap_or(f32::NEG_INFINITY);
                if grade.passed || grade.grade >= best_grade {
                    run.best_grade = Some(grade.grade);
                    let b = Snapshot { model: current.clone(), archive: archive.clone() };
                    host.save_best(&b)?;
                    best = Some(b);
                } else if let Some(b) = &best {
                    current = b.model.clone();
                    archive = b.archive.clone();
                    run.remediation[k].reverted = true;
                }
                if grade.passed {
                    run.phase = Phase::Passed;
                }
                host.event(&ProgramEvent::RemediationFinished(run.remediation[k].clone()));
                save(host, &current, &archive, run)?;
            }
            Phase::Passed | Phase::Failed | Phase::TimedOut => {
                host.event(&ProgramEvent::CourseFinished { phase: run.phase, best_grade: run.best_grade });
                return Ok(RunResult { model: current, archive, stopped: false });
            }
        }
        if stop.load(Ordering::Acquire) && !run.finished() {
            return Ok(RunResult { model: current, archive, stopped: true });
        }
    }
}

/// Teaches a group of a course's lessons together (one mixed session, see
/// [`teach_group`]) and records each lesson like a lesson taught on its own. Returns true when
/// teaching was stopped (the group is taught again when the run resumes; accepted changes are
/// kept).
#[allow(clippy::too_many_arguments)]
fn teach_course_group(
    current: &mut Bicameral,
    archive: &mut Vec<Item>,
    prep: &PreparedCourse,
    run: &mut CourseRun,
    group: &[usize],
    opts: &RunOptions,
    host: &mut dyn CourseHost,
    rewards: Option<&mut super::reward::Rewards>,
    stop: &AtomicBool,
) -> Result<bool, String> {
    let first = &prep.lessons[group[0]];
    let cfg = session_cfg(prep, first, opts, run.seed.wrapping_add(group[0] as u64 * 7919 + 1));
    let stage = first.stage;
    let mut lessons: Vec<(usize, Vec<Item>)> = Vec::new();
    let mut records: Vec<FileRecord> = Vec::new();
    for &index in group {
        let lesson = &prep.lessons[index];
        records.push(FileRecord {
            index,
            name: lesson.name.clone(),
            sha256: lesson.sha256.clone(),
            bytes: 0,
            train_units: lesson.items.iter().filter(|i| !i.held_out).count(),
            heldout_units: lesson.items.iter().filter(|i| i.held_out).count(),
            start_step: current.step,
            end_step: current.step,
            outcome: FileOutcome::Failed,
            error: None,
            epochs_run: 0,
            baseline: None,
            final_score: None,
            pass: 1,
            attempts: Vec::new(),
            pass_mark: lesson.pass_mark,
            own_before: super::course::own_rate(current, &lesson.items).ok(),
            own_after: None,
            with_memory: None,
        });
        let taught = match host.study() {
            Some(st) => {
                let (taught, report, copies) = st.before_lesson(current, &lesson.name, &lesson.items)?;
                if let Some(report) = report {
                    host.event(&ProgramEvent::Lesson {
                        index,
                        stage,
                        event: CourseEvent::Studied { index, report, copies },
                    });
                }
                taught
            }
            None => lesson.items.clone(),
        };
        lessons.push((index, taught));
    }
    let mut stopped = false;
    let lead = group[0];
    teach_group(
        current,
        archive,
        &lessons,
        cfg.retry.retries,
        &cfg,
        &mut records,
        &mut stopped,
        rewards,
        &mut |e| {
            let index = match e {
                CourseEvent::FileFinished(r) => r.index,
                _ => lead,
            };
            host.event(&ProgramEvent::Lesson { index, stage, event: e.clone() })
        },
        &mut |_, _| Ok(()),
        stop,
    )?;
    if stopped {
        return Ok(true);
    }
    if let Some(e) = records.iter().find_map(|r| r.error.clone()) {
        return Err(format!(
            "{}: {e}",
            group.iter().map(|&i| prep.lessons[i].name.as_str()).collect::<Vec<_>>().join(" + ")
        ));
    }
    run.queue.remove(group);
    if records.iter().any(|r| r.changed()) {
        run.queue.changed();
    }
    for mut record in records {
        let index = record.index;
        let lesson = &prep.lessons[index];
        record.own_after = super::course::own_rate(current, &lesson.items).ok();
        record.with_memory = host.study().and_then(|st| st.after_lesson(current, &lesson.items));
        if record.outcome != FileOutcome::Passed {
            record.outcome = FileOutcome::Unresolved;
            record.error = Some("below its pass mark after being taught together with the others".into());
        }
        record.end_step = current.step;
        let score = lesson_score(current, lesson)?;
        run.when_taught[index] = Some(score);
        host.event(&ProgramEvent::LessonTaught {
            index,
            name: lesson.name.clone(),
            outcome: record.outcome,
            when_taught: score,
            record: Some(Box::new(record.clone())),
        });
        run.records[index] = Some(record);
    }
    Ok(false)
}

/// What ended a course run, for the restart rules.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    pub fingerprint: String,
    pub phase: Phase,
    pub overridden: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Restart {
    /// A new or revised course: a fresh run with three remediation attempts.
    Fresh,
    /// This exact version failed before: one manual override is allowed (warned and recorded).
    NeedsOverride,
    /// This exact version failed again after its override: revise the course first.
    Refused,
}

/// Whether a course with `fingerprint` may start a new run, given earlier runs. A failed or
/// timed-out run of the same version uses up a fresh start; one override per version follows.
pub fn restart_policy(history: &[RunSummary], fingerprint: &str) -> Restart {
    let failed: Vec<&RunSummary> = history
        .iter()
        .filter(|r| r.fingerprint == fingerprint && matches!(r.phase, Phase::Failed | Phase::TimedOut))
        .collect();
    if failed.is_empty() {
        Restart::Fresh
    } else if failed.iter().any(|r| r.overridden) {
        Restart::Refused
    } else {
        Restart::NeedsOverride
    }
}
