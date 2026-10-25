//! Accepted epochs are written to the checkpoint folder while a session runs (atomically, the
//! newest three kept, with a README), so a power cut loses at most the epoch in progress; the
//! newest checkpoint is the session's final model.
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::builtin::builtin_courses;
use yendory_engine::learn::lessons::parse_lesson;
use yendory_engine::learn::session::{checkpoint_files, teach, Event, SessionConfig, CHECKPOINTS_KEPT};
use yendory_engine::{Bicameral, Config};

#[test]
fn accepted_epochs_are_written_and_the_newest_is_the_final_model() {
    let file = builtin_courses().remove(0).lessons.remove(0);
    let items = parse_lesson(&file.text, &file.name).unwrap();
    let dir = std::env::temp_dir().join(format!("yendory-checkpoints-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = SessionConfig {
        epochs: 6,
        patience: 6,
        foundation: false,
        learning_rate: 3e-3,
        checkpoint_dir: Some(dir.clone()),
        ..SessionConfig::default()
    };
    let model = Bicameral::with_config(Config::default(), 7);
    let mut written = Vec::new();
    let r = teach(
        &model,
        &items,
        &[],
        &cfg,
        &mut |e| {
            if let Event::Checkpoint { epoch, step, file, error } = e {
                assert!(error.is_none(), "{error:?}");
                written.push((*epoch, *step, file.clone()));
            }
        },
        &AtomicBool::new(false),
    )
    .unwrap();
    let accepted = r.epochs.iter().filter(|e| e.selected.is_some()).count();
    println!("{accepted} accepted epochs, {} checkpoints written", written.len());
    assert!(accepted > 0, "the lesson is learnable in six epochs");
    assert_eq!(written.len(), accepted, "one checkpoint per accepted epoch");
    let files = checkpoint_files(&dir);
    assert_eq!(files.len(), accepted.min(CHECKPOINTS_KEPT), "only the newest are kept");
    assert!(dir.join("README.txt").exists());
    assert!(!dir.join(".writing.tmp").exists());
    // Steps only grow, so the files sort oldest to newest; the newest is the final model.
    let newest = Bicameral::load(files.last().unwrap()).unwrap();
    assert_eq!(newest.step, r.model.step);
    assert_eq!(newest.weights, r.model.weights);
    assert_eq!(newest.tokenizer.words(), r.model.tokenizer.words());
    for f in &files {
        Bicameral::load(f).unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}
