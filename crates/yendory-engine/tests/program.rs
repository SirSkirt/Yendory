//! Course Builder and course runs: file names, exam pools, practice material, grading, the
//! three remediation attempts, the best checkpoint, resuming and the restart rules.
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use yendory_engine::learn::course::{CourseFile, FileOutcome};
use yendory_engine::learn::curriculum::{prepare, read_name, CourseDef, FileRole, Section, StageFrom};
use yendory_engine::learn::lessons::{canonical, Item, Kind};
use yendory_engine::learn::program::{
    restart_policy, run_course, CourseHost, CourseRun, Phase, ProgramEvent, RemedyStrategy, Restart,
    RunOptions, RunSummary, Snapshot,
};
use yendory_engine::learn::session::EducationStage;
use yendory_engine::{Bicameral, Config};

fn file(name: &str, text: String) -> CourseFile {
    CourseFile { name: name.into(), text }
}

const THINGS: [(&str, &str); 12] = [
    ("cat", "animal"),
    ("dog", "animal"),
    ("cow", "animal"),
    ("rose", "plant"),
    ("oak", "plant"),
    ("fern", "plant"),
    ("car", "vehicle"),
    ("bus", "vehicle"),
    ("van", "vehicle"),
    ("apple", "fruit"),
    ("pear", "fruit"),
    ("plum", "fruit"),
];

fn dense_lesson(n: usize) -> String {
    let mut s = String::from("[Understanding Lesson]\nThings belong to kinds.\n[examples]\n");
    for i in 0..n {
        let (a, b) = THINGS[i % THINGS.len()];
        s.push_str(&format!("What kind of thing is a {a} number {i}?  {b}\n"));
    }
    s + "[/examples]\n[/Understanding Lesson]\n"
}

fn sentence_lesson(n: usize, verb: &str) -> String {
    let mut s = String::from("[Dialogue Lesson]\n[examples]\n");
    for i in 0..n {
        let (a, b) = THINGS[i % THINGS.len()];
        s.push_str(&format!("Tell me about {a} {i} {verb}  A {a} is a {b}. | The {a} is a {b}.\n"));
    }
    s + "[/examples]\n[/Dialogue Lesson]\n"
}

#[test]
fn file_names_give_order_stage_and_role() {
    let n = read_name("03 Dense - colours.txt");
    assert_eq!(
        (n.role, n.number, n.stage),
        (FileRole::Lesson, Some(3), Some(EducationStage::DenseKnowledge))
    );
    let n = read_name("Lesson 12 CO-OP talking.txt");
    assert_eq!((n.number, n.stage), (Some(12), Some(EducationStage::Cooperative)));
    let n = read_name("course/2 Transformer Practice.txt");
    assert_eq!(
        (n.role, n.number, n.stage),
        (FileRole::Practice, Some(2), Some(EducationStage::TransformerLanguage))
    );
    assert_eq!(read_name("Final Exam - Basic English I.txt").role, FileRole::Exam);
    assert_eq!(read_name("05 condensed notes.txt").stage, None, "whole words only");
    assert!(read_name("05 Transformer dense fog.txt").problem.is_some());

    let mut c = CourseDef::new("c", "Basic English I");
    let notes = c.add_files(vec![
        file("10 Dense - weather.txt", dense_lesson(30)),
        file("2 Co-op - talk.txt", sentence_lesson(30, "c")),
        file("2 Dense - nouns.txt", dense_lesson(31)),
        file("2 Transformer - sentences.txt", sentence_lesson(30, "t")),
        file("2 Dense Practice - nouns.txt", dense_lesson(5)),
        file("notes.txt", sentence_lesson(30, "n")),
        file("Final Exam.txt", String::new()),
    ]);
    let names: Vec<&str> = c.lessons.iter().map(|l| l.file.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "2 Dense - nouns.txt",
            "2 Transformer - sentences.txt",
            "2 Co-op - talk.txt",
            "10 Dense - weather.txt",
            "notes.txt"
        ]
    );
    assert_eq!(c.lessons[0].practice.len(), 1, "{notes:?}");
    assert_eq!(c.lessons[4].stage_from, StageFrom::Default);
    assert!(c.exam.is_some());
}

#[test]
fn exam_questions_are_separate_and_practice_never_leaks() {
    let mut c = CourseDef::new("c", "Kinds");
    // The practice file repeats lesson questions: those matching evaluation or exam questions
    // must be excluded.
    c.add_files(vec![
        file("1 Dense - kinds.txt", dense_lesson(240)),
        file("1 Dense Practice.txt", dense_lesson(240)),
        file("2 Transformer - sentences.txt", sentence_lesson(240, "t")),
    ]);
    let p = prepare(&c);
    assert!(p.check.problems.is_empty(), "{:?}", p.check.problems);
    let eval: BTreeSet<String> = p
        .lessons
        .iter()
        .flat_map(|l| l.eval_items())
        .map(|i| canonical(i.prompt.as_deref().unwrap()))
        .collect();
    let train: BTreeSet<String> = p
        .lessons
        .iter()
        .flat_map(|l| l.train_items())
        .filter(|i| i.kind != Kind::Prose)
        .map(|i| canonical(i.prompt.as_deref().unwrap()))
        .collect();
    let exam: BTreeSet<String> =
        p.exam.iter().flat_map(|s| &s.items).map(|i| canonical(i.prompt.as_deref().unwrap())).collect();
    assert!(!exam.is_empty() && !eval.is_empty());
    assert!(exam.is_disjoint(&eval) && exam.is_disjoint(&train) && eval.is_disjoint(&train));
    let practice: BTreeSet<String> = p.lessons[0]
        .practice
        .iter()
        .filter(|i| i.kind != Kind::Prose)
        .map(|i| canonical(i.prompt.as_deref().unwrap()))
        .collect();
    assert!(practice.is_disjoint(&eval) && practice.is_disjoint(&exam));
    assert!(p.check.lessons[0].practice_excluded > 0);
    // Dense questions are graded by keyword, with the other answers as conflicts.
    let key = p.lessons[0].items.iter().find_map(|i| i.key.clone()).unwrap();
    assert!(key.conflicts.iter().any(|c| c == "plant"));
    // Sentence lessons train on the first accepted form.
    let it: &Item = p.lessons[1].items.iter().find(|i| i.kind == Kind::Dialogue).unwrap();
    assert!(it.segments[1].text.starts_with("A "));
    // No Co-op lessons: the exam is Dense and Transformer, half each.
    let sections: Vec<(Section, f32)> = p.exam.iter().map(|s| (s.section, s.weight)).collect();
    assert_eq!(sections.len(), 2);
    assert!(sections.iter().all(|(_, w)| (w - 0.5).abs() < 1e-3), "{sections:?}");
}

#[test]
fn small_courses_warn_and_teacher_exams_are_checked() {
    let mut c = CourseDef::new("c", "Tiny");
    let exam = format!(
        "[Dense]\n{}[/Dense]\n[Co-op]\n[Dialogue Lesson]\n[examples]\nHow are you today?  I am well.\n[/examples]\n[/Dialogue Lesson]\n[/Co-op]\n",
        dense_lesson(3)
    );
    c.add_files(vec![file("1 Dense - kinds.txt", dense_lesson(40)), file("Exam.txt", exam)]);
    let p = prepare(&c);
    // The exam's Dense questions are the lesson's first questions: not independent.
    assert!(p.check.warnings.iter().any(|w| w.contains("left out of the exam")), "{:?}", p.check.warnings);
    assert_eq!(p.exam.len(), 1);
    assert_eq!(p.exam[0].section, Section::Coop);
    assert!(p.check.warnings.iter().any(|w| w.contains("only")), "few questions are flagged");
}

#[test]
fn fingerprints_and_restart_rules() {
    let mut c = CourseDef::new("c", "Kinds");
    c.add_files(vec![file("1 Dense - kinds.txt", dense_lesson(60))]);
    let f0 = c.fingerprint();
    c.settings.learning_rate = 5e-4;
    c.settings.epochs = 9;
    assert_eq!(c.fingerprint(), f0, "training settings are not a revision");
    c.lessons[0].pass_mark = 0.6;
    let f1 = c.fingerprint();
    assert_ne!(f1, f0);
    c.add_files(vec![file("1 Dense Practice.txt", dense_lesson(5))]);
    assert_ne!(c.fingerprint(), f1, "new practice material is a revision");

    let failed = |overridden| RunSummary { fingerprint: f0.clone(), phase: Phase::Failed, overridden };
    assert_eq!(restart_policy(&[], &f0), Restart::Fresh);
    assert_eq!(restart_policy(&[failed(false)], &f0), Restart::NeedsOverride);
    assert_eq!(restart_policy(&[failed(false), failed(true)], &f0), Restart::Refused);
    assert_eq!(restart_policy(&[failed(false), failed(true)], &f1), Restart::Fresh);
}

struct Host {
    events: Vec<ProgramEvent>,
    saves: usize,
    stop_after: Option<usize>,
    stop: Arc<AtomicBool>,
    run: Option<CourseRun>,
    best: Option<Snapshot>,
    model: Option<Bicameral>,
}
impl CourseHost for Host {
    fn event(&mut self, e: &ProgramEvent) {
        if !matches!(e, ProgramEvent::Lesson { .. } | ProgramEvent::Remediation { .. }) {
            self.events.push(e.clone());
        }
    }
    fn save(&mut self, model: &Bicameral, _: &[Item], run: &CourseRun) -> Result<(), String> {
        assert_eq!(run.saved_step, model.step);
        self.saves += 1;
        self.run = Some(run.clone());
        self.model = Some(model.clone());
        if self.stop_after == Some(self.saves) {
            self.stop.store(true, Ordering::Release);
        }
        Ok(())
    }
    fn save_best(&mut self, best: &Snapshot) -> Result<(), String> {
        self.best = Some(best.clone());
        Ok(())
    }
}
fn host(stop: &Arc<AtomicBool>, stop_after: Option<usize>) -> Host {
    Host { events: Vec::new(), saves: 0, stop_after, stop: stop.clone(), run: None, best: None, model: None }
}

fn small_course(mark: f32) -> CourseDef {
    let mut c = CourseDef::new("c", "Kinds");
    c.default_pass_mark = mark;
    c.pretest = true;
    c.settings.epochs = 1;
    c.settings.retries = 0;
    c.settings.requeue = false;
    c.settings.foundation = false;
    c.settings.learning_rate = 2e-3;
    c.add_files(vec![
        file("1 Dense - kinds.txt", dense_lesson(70)),
        file("1 Dense Practice - kinds.txt", dense_lesson(30).replace("number", "no.")),
        file("1 Transformer - sentences.txt", sentence_lesson(70, "t")),
    ]);
    c
}

#[test]
fn a_failed_course_gets_three_different_remediations_then_stops_on_its_best_checkpoint() {
    let model = Bicameral::with_config(Config::default(), 3);
    let prep = prepare(&small_course(1.0));
    assert!(prep.check.problems.is_empty(), "{:?}", prep.check.problems);
    let mut run = CourseRun::new(&prep, false, 7);
    let stop = Arc::new(AtomicBool::new(false));
    let mut h = host(&stop, None);
    let r = run_course(&model, &[], &prep, &mut run, None, &RunOptions::default(), &mut h, &stop).unwrap();
    assert!(!r.stopped);
    assert_eq!(run.phase, Phase::Failed);
    assert!(run.pretest.is_some());
    assert_eq!(run.remediation.len(), 3);
    let strategies: BTreeSet<_> = run.remediation.iter().map(|r| r.strategy as u8).collect();
    assert_eq!(strategies.len(), 3, "each attempt uses a different strategy");
    for (k, rec) in run.remediation.iter().enumerate() {
        let expect = 2e-3 * 0.5f32.powi(k as i32 + 1);
        assert!((rec.learning_rate - expect).abs() < 1e-9, "the learning rate halves each attempt");
        assert!(rec.grade_after.is_some());
    }
    // Grade composition and recorded failures.
    let g = run.grades.last().unwrap();
    assert!((g.grade - (0.7 * g.lesson_average + 0.3 * g.exam.score)).abs() < 1e-5);
    assert!((g.required - 1.0).abs() < 1e-6);
    assert!(!g.failed_lessons.is_empty());
    assert!(g.lessons.iter().all(|l| l.when_taught.is_some()));
    // The model returned is the best checkpoint.
    let best = h.best.as_ref().unwrap();
    assert_eq!(r.model.weights, best.model.weights);
    let best_grade = run.grades.iter().map(|g| g.grade).fold(f32::NEG_INFINITY, f32::max);
    assert!((run.best_grade.unwrap() - best_grade).abs() < 1e-6);
    let used: Vec<RemedyStrategy> = run.remediation.iter().map(|r| r.strategy).collect();
    assert!(!used.is_empty());
    let lessons_taught = h
        .events
        .iter()
        .filter(|e| matches!(e, ProgramEvent::LessonTaught { outcome: FileOutcome::Unresolved, .. }))
        .count();
    assert!(lessons_taught >= 1);
}

#[test]
fn a_passed_course_records_weak_lessons_and_an_interrupted_run_resumes() {
    let model = Bicameral::with_config(Config::default(), 4);
    let mut c = small_course(0.0);
    c.lessons[1].pass_mark = 1.0; // a high mark for the transformer lesson
    c.lessons[0].pass_mark = 0.0;
    let prep = prepare(&c);
    assert!(prep.required_grade > 0.4 && prep.required_grade < 0.6);
    let stop = Arc::new(AtomicBool::new(false));
    let mut run = CourseRun::new(&prep, false, 3);
    // Stop after the second save (after the pre-test and the first lesson).
    let mut h = host(&stop, Some(2));
    let r = run_course(&model, &[], &prep, &mut run, None, &RunOptions::default(), &mut h, &stop).unwrap();
    assert!(r.stopped);
    assert_eq!(run.phase, Phase::Teaching);
    assert!(run.when_taught[0].is_some() && run.when_taught[1].is_none());

    // Resume from what was saved.
    stop.store(false, Ordering::Release);
    let mut saved = h.run.clone().unwrap();
    let saved_model = h.model.clone().unwrap();
    assert_eq!(saved.saved_step, saved_model.step);
    let mut h2 = host(&stop, None);
    let r2 =
        run_course(&saved_model, &r.archive, &prep, &mut saved, None, &RunOptions::default(), &mut h2, &stop)
            .unwrap();
    assert!(!r2.stopped);
    let taught: Vec<usize> = h2
        .events
        .iter()
        .filter_map(|e| if let ProgramEvent::LessonTaught { index, .. } = e { Some(*index) } else { None })
        .collect();
    assert_eq!(taught, [1], "only the lesson not yet taught runs after resuming");
    let g = saved.grades.last().unwrap();
    if saved.phase == Phase::Passed {
        assert!(g.passed);
    }
    // Lessons below their own mark are recorded whether or not the course passes.
    let below: Vec<String> =
        g.lessons.iter().filter(|l| l.course_end < l.pass_mark).map(|l| l.name.clone()).collect();
    assert_eq!(g.failed_lessons, below);
    assert!(matches!(h2.events[0], ProgramEvent::CourseStarted { resumed: true, .. }));
}
