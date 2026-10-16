//! Course teaching: per-file records, duplicates, failures that do not end the course.
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use yendory_engine::fingerprint::sha256_hex;
use yendory_engine::learn::course::{sort_files, teach_course, Cause, CourseEvent, CourseFile, FileOutcome};
use yendory_engine::learn::session::{RetryPolicy, SessionConfig, Strategy};
use yendory_engine::Bicameral;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/compat")
}

#[test]
fn course_records_every_file() {
    let model = Bicameral::load(root().join("fixtures/nova2-step15000.nnova2")).unwrap();
    let a = std::fs::read_to_string(root().join("fixtures/lessons/sample_course_a.txt")).unwrap();
    let b = std::fs::read_to_string(root().join("fixtures/lessons/sample_course_b.txt")).unwrap();
    let mut files = vec![
        CourseFile { name: "03_copy_of_01.txt".into(), text: a.clone() },
        CourseFile { name: "02_tiny.txt".into(), text: b },
        CourseFile { name: "01_sample.txt".into(), text: a.clone() },
        CourseFile {
            name: "04_not_ascii.txt".into(),
            text: "[Dialogue Lesson]\ncaf\u{e9}\n[/Dialogue Lesson]\n".into(),
        },
    ];
    sort_files(&mut files);
    let cfg = SessionConfig {
        epochs: 1,
        batch_size: 8,
        foundation: false,
        eval_limit: Some(6),
        // Pass mark 0: the lesson passes on its first attempt (this test is about records).
        pass_mark: 0.0,
        ..SessionConfig::default()
    };
    let mut events = Vec::new();
    let mut saved = 0;
    let r = teach_course(
        &model,
        &files,
        &[],
        &cfg,
        &mut |e| events.push(e.clone()),
        &mut |_, _| {
            saved += 1;
            Ok(())
        },
        &AtomicBool::new(false),
    )
    .unwrap();

    let names: Vec<&str> = r.files.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["01_sample.txt", "02_tiny.txt", "03_copy_of_01.txt", "04_not_ascii.txt"]);
    let first = &r.files[0];
    assert_eq!(first.sha256, sha256_hex(a.as_bytes()));
    assert_eq!(first.outcome, FileOutcome::Passed);
    assert_eq!(first.attempts.len(), 1);
    assert!(first.train_units > 0 && first.heldout_units > 0 && first.baseline.is_some());
    assert_eq!(first.start_step, model.step);
    // 02_tiny has two units and no held-out unit: recorded as failed, the course continues.
    assert_eq!(r.files[1].outcome, FileOutcome::Failed);
    assert!(r.files[1].error.as_deref().unwrap().contains("held-out"));
    assert_eq!(r.files[1].start_step, first.end_step);
    assert_eq!(r.files[1].end_step, first.end_step);
    // Identical content is not taught twice.
    assert_eq!(r.files[2].outcome, FileOutcome::Duplicate);
    assert_eq!(r.files[2].sha256, first.sha256);
    assert_eq!(r.files[3].outcome, FileOutcome::Failed);
    // The archive holds exactly the learned files' items.
    let learned = first.changed();
    assert_eq!(r.archive.is_empty(), !learned);
    assert_eq!(saved, usize::from(learned));
    assert_eq!(r.model.step, first.end_step);
    let finished = events.iter().filter(|e| matches!(e, CourseEvent::FileFinished(_))).count();
    assert_eq!(finished, 4);
    assert!(events.iter().any(|e| matches!(e, CourseEvent::Session { index: 0, .. })));
}

/// Training that makes things worse is diagnosed as unstable: each retry halves the learning
/// rate (the only diagnosis that lowers it). Never passing, the lesson moves to the end of the
/// queue and finishes unchanged because nothing else changed the model.
#[test]
fn unstable_training_gets_gentler_steps_then_waits() {
    let model = Bicameral::load(root().join("fixtures/nova2-step15000.nnova2")).unwrap();
    let text = std::fs::read_to_string(root().join("fixtures/lessons/sample_course_a.txt")).unwrap();
    let files = vec![CourseFile { name: "01.txt".into(), text }];
    let cfg = SessionConfig {
        epochs: 1,
        batch_size: 16,
        learning_rate: 5.0,
        foundation: false,
        eval_limit: Some(4),
        ..SessionConfig::default()
    };
    let mut events = Vec::new();
    let r = teach_course(
        &model,
        &files,
        &[],
        &cfg,
        &mut |e| events.push(e.clone()),
        &mut |_, _| Ok(()),
        &AtomicBool::new(false),
    )
    .unwrap();
    let attempts: Vec<(u32, f32)> = events
        .iter()
        .filter_map(|e| match e {
            CourseEvent::AttemptStarted { attempt, learning_rate, .. } => Some((*attempt, *learning_rate)),
            _ => None,
        })
        .collect();
    assert_eq!(attempts, [(0, 5.0), (1, 2.5), (2, 1.25), (3, 0.625)]);
    let causes: Vec<Cause> = events
        .iter()
        .filter_map(|e| {
            if let CourseEvent::Diagnosed { diagnosis, .. } = e {
                Some(diagnosis.cause)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(causes, [Cause::Unstable, Cause::Unstable, Cause::Unstable]);
    let finished: Vec<FileOutcome> = events
        .iter()
        .filter_map(|e| if let CourseEvent::FileFinished(f) = e { Some(f.outcome) } else { None })
        .collect();
    assert_eq!(finished, [FileOutcome::Deferred, FileOutcome::Unresolved]);
    assert_eq!(r.files[0].attempts.len(), 4);
    assert!(r.files[0].attempts[1].because.as_ref().is_some_and(|d| d.cause == Cause::Unstable));
    assert_eq!(r.model.weights, model.weights);
}

/// Progress that does not reach the pass mark is "unresolved", not "passed", and a model that
/// cannot answer its own training questions is sent to practise exactly the ones it gets wrong.
#[test]
fn progress_is_not_mastery_and_gaps_are_practised() {
    let model = Bicameral::load(root().join("fixtures/nova2-step15000.nnova2")).unwrap();
    let text = std::fs::read_to_string(root().join("fixtures/lessons/sample_course_a.txt")).unwrap();
    let files = vec![CourseFile { name: "01.txt".into(), text }];
    let cfg = SessionConfig {
        epochs: 1,
        batch_size: 16,
        foundation: false,
        eval_limit: Some(6),
        pass_mark: 0.95,
        retry: RetryPolicy { retries: 3, requeue: false, ..RetryPolicy::default() },
        ..SessionConfig::default()
    };
    let mut events = Vec::new();
    let r = teach_course(
        &model,
        &files,
        &[],
        &cfg,
        &mut |e| events.push(e.clone()),
        &mut |_, _| Ok(()),
        &AtomicBool::new(false),
    )
    .unwrap();
    let f = &r.files[0];
    assert!(f.final_score.as_ref().unwrap().exact_rate() < 0.95);
    assert_eq!(f.outcome, FileOutcome::Unresolved);
    let d = events
        .iter()
        .find_map(|e| {
            if let CourseEvent::Diagnosed { diagnosis, .. } = e {
                Some(diagnosis.clone())
            } else {
                None
            }
        })
        .expect("a diagnosis after the first attempt");
    assert_eq!(d.cause, Cause::NotLearned, "{d:?}");
    assert!(d.wrong_training_units > 0 && d.train_rate < 0.95);
    let retry = &f.attempts[1];
    assert_eq!(retry.strategy, Strategy::Focused);
    assert_eq!(retry.epochs, 12);
    assert_eq!(retry.learning_rate, cfg.learning_rate, "not unstable: the rate is not lowered");
    assert!(retry.practice_units >= 2 * d.wrong_training_units);
    // A remedy that failed without any change is never run again unchanged.
    for w in f.attempts.windows(2) {
        let repeated = !w[0].learned
            && w[0].because.as_ref().map(|d| d.cause) == w[1].because.as_ref().map(|d| d.cause)
            && (w[0].learning_rate, w[0].strategy, w[0].epochs, w[0].rehearsal_share)
                == (w[1].learning_rate, w[1].strategy, w[1].epochs, w[1].rehearsal_share);
        assert!(!repeated, "attempt {} repeated attempt {} unchanged", w[1].attempt, w[0].attempt);
    }
    let summary: Vec<_> = f
        .attempts
        .iter()
        .map(|a| (a.because.as_ref().map(|d| d.cause), a.learned, a.final_score.exact_correct))
        .collect();
    eprintln!("attempts: {summary:?}");
    eprintln!(
        "diagnosis: {d:?}\nscores: {:?} -> {:?}",
        f.baseline.as_ref().map(|s| s.exact_rate()),
        f.final_score.as_ref().map(|s| s.exact_rate())
    );
}
