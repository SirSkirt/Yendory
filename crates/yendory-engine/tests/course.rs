//! Course teaching: per-file records, duplicates, failures that do not end the course.
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use yendory_engine::fingerprint::sha256_hex;
use yendory_engine::learn::course::{sort_files, teach_course, CourseEvent, CourseFile, FileOutcome};
use yendory_engine::learn::session::SessionConfig;
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
        ..SessionConfig::default()
    };
    let mut events = Vec::new();
    let r = teach_course(&model, &files, &[], &cfg, &mut |e| events.push(e.clone()), &AtomicBool::new(false))
        .unwrap();

    let names: Vec<&str> = r.files.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["01_sample.txt", "02_tiny.txt", "03_copy_of_01.txt", "04_not_ascii.txt"]);
    let first = &r.files[0];
    assert_eq!(first.sha256, sha256_hex(a.as_bytes()));
    assert!(matches!(first.outcome, FileOutcome::Learned | FileOutcome::KeptPrevious));
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
    let learned = first.outcome == FileOutcome::Learned;
    assert_eq!(r.archive.is_empty(), !learned);
    assert_eq!(r.model.step, first.end_step);
    let finished = events.iter().filter(|e| matches!(e, CourseEvent::FileFinished(_))).count();
    assert_eq!(finished, 4);
    assert!(events.iter().any(|e| matches!(e, CourseEvent::Session { index: 0, .. })));
}
