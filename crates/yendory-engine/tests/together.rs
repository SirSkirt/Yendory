//! Lessons taught together (the default): one mixed session over the chosen lessons, each lesson
//! then graded on its own held-out questions; lessons not chosen are taught one at a time after.
//! In courses, "together" lessons of the same stage form one group, and remediation practises
//! them together too.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use yendory_engine::learn::builtin::builtin_courses;
use yendory_engine::learn::course::{teach_course_together, CourseEvent, CourseFile, FileOutcome};
use yendory_engine::learn::curriculum::{prepare, CourseDef};
use yendory_engine::learn::lessons::Item;
use yendory_engine::learn::program::{run_course, CourseHost, CourseRun, ProgramEvent, RunOptions, Snapshot};
use yendory_engine::learn::session::SessionConfig;
use yendory_engine::{Bicameral, Config};

fn english(n: usize) -> Vec<CourseFile> {
    builtin_courses().remove(0).lessons.into_iter().take(n).collect()
}

#[test]
fn classroom_teaches_the_chosen_lessons_together_then_the_rest() {
    let files = english(4);
    let model = Bicameral::with_config(Config::default(), 5);
    let cfg = SessionConfig {
        epochs: 3,
        patience: 3,
        foundation: false,
        learning_rate: 2e-3,
        eval_limit: None,
        ..SessionConfig::default()
    };
    let mut events = Vec::new();
    let together = [true, true, true, false];
    let r = teach_course_together(
        &model,
        &files,
        &together,
        &[],
        &cfg,
        None,
        None,
        &mut |e| events.push(e.clone()),
        &mut |_, _| Ok(()),
        &AtomicBool::new(false),
    )
    .unwrap();
    let groups: Vec<&Vec<usize>> = events
        .iter()
        .filter_map(|e| if let CourseEvent::GroupStarted { indices, .. } = e { Some(indices) } else { None })
        .collect();
    assert_eq!(groups, [&vec![0, 1, 2]], "one group of the three chosen lessons");
    // The fourth lesson is taught on its own, after the group.
    let group_end =
        events.iter().rposition(|e| matches!(e, CourseEvent::FileFinished(f) if f.index == 2)).unwrap();
    let alone = events.iter().position(|e| matches!(e, CourseEvent::Session { index: 3, .. })).unwrap();
    assert!(alone > group_end);
    assert_eq!(r.files.len(), 4);
    for f in &r.files[..3] {
        let (b, e) = (f.baseline.as_ref().unwrap(), f.final_score.as_ref().unwrap());
        println!(
            "{}: {}/{} -> {}/{} ({:?})",
            f.name, b.exact_correct, b.exact_total, e.exact_correct, e.exact_total, f.outcome
        );
        assert_eq!(e.exact_total, f.heldout_units, "each lesson is graded on its own held-out questions");
        assert!(matches!(f.outcome, FileOutcome::Passed | FileOutcome::Unresolved));
        assert!(!f.attempts.is_empty());
    }
    let learned: usize = r.files[..3].iter().map(|f| f.final_score.as_ref().unwrap().exact_correct).sum();
    assert!(learned > 0, "the group learns something in three epochs");
    // Every lesson of the group joins the rehearsal archive.
    for f in &files[..3] {
        assert!(r.archive.iter().any(|i: &Item| i.source == f.name));
    }
}

struct Host {
    groups: Vec<Vec<usize>>,
    taught: Vec<usize>,
    alone_sessions: Vec<usize>,
    stop: Arc<AtomicBool>,
}
impl CourseHost for Host {
    fn event(&mut self, e: &ProgramEvent) {
        match e {
            ProgramEvent::Lesson { event: CourseEvent::GroupStarted { indices, .. }, .. } => {
                self.groups.push(indices.clone())
            }
            ProgramEvent::Lesson { index, event: CourseEvent::AttemptStarted { .. }, .. } => {
                self.alone_sessions.push(*index)
            }
            ProgramEvent::LessonTaught { index, .. } => self.taught.push(*index),
            _ => {}
        }
    }
    fn save(&mut self, _: &Bicameral, _: &[Item], _: &CourseRun) -> Result<(), String> {
        let _ = self.stop.load(Ordering::Acquire);
        Ok(())
    }
    fn save_best(&mut self, _: &Snapshot) -> Result<(), String> {
        Ok(())
    }
}

#[test]
fn courses_group_together_lessons_of_a_stage_and_keep_their_fingerprint() {
    let mut c = CourseDef::new("c", "English");
    c.settings.epochs = 2;
    c.settings.retries = 0;
    c.settings.foundation = false;
    c.settings.learning_rate = 2e-3;
    c.add_files(english(3));
    assert!(c.lessons.iter().all(|l| l.together), "together is the default");
    let fp = c.fingerprint();
    c.lessons[2].together = false;
    assert_ne!(c.fingerprint(), fp, "teaching a lesson on its own is a different course");
    c.lessons[2].together = true;
    assert_eq!(c.fingerprint(), fp, "the default leaves the fingerprint as before");
    c.lessons[2].together = false;
    let prep = prepare(&c);
    assert!(prep.check.problems.is_empty(), "{:?}", prep.check.problems);
    let model = Bicameral::with_config(Config::default(), 6);
    let mut run = CourseRun::new(&prep, false, 1);
    let stop = Arc::new(AtomicBool::new(false));
    let mut h =
        Host { groups: Vec::new(), taught: Vec::new(), alone_sessions: Vec::new(), stop: stop.clone() };
    run_course(&model, &[], &prep, &mut run, None, &RunOptions::default(), &mut h, &stop).unwrap();
    assert_eq!(h.groups.first(), Some(&vec![0, 1]), "lessons 1 and 2 together, lesson 3 on its own");
    assert_eq!(&h.taught[..3], &[0, 1, 2]);
    assert!(h.alone_sessions.contains(&2));
    assert!(run.when_taught.iter().all(|w| w.is_some()));
    assert!(run.records.iter().all(|r| r.as_ref().is_some_and(|r| r.final_score.is_some())));
}
