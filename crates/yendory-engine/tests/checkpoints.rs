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

use yendory_engine::learn::session::{checkpoint_info, SessionResult, BACKOFF_EPOCHS};

fn lesson() -> Vec<yendory_engine::learn::lessons::Item> {
    let file = builtin_courses().remove(0).lessons.remove(0);
    parse_lesson(&file.text, &file.name).unwrap()
}

fn run(model: &Bicameral, cfg: &SessionConfig, events: &mut Vec<Event>) -> SessionResult {
    teach(model, &lesson(), &[], cfg, &mut |e| events.push(e.clone()), &AtomicBool::new(false)).unwrap()
}

fn batch_rates(events: &[Event]) -> Vec<(u32, usize, f32)> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Batch { epoch, batch, learning_rate, .. } => Some((*epoch, *batch, *learning_rate)),
            _ => None,
        })
        .collect()
}

/// A session continued from a checkpoint runs the remaining epochs with the interrupted
/// session's learning rates, not a new schedule from the full rate.
#[test]
fn a_continued_session_keeps_the_interrupted_schedule() {
    let dir = std::env::temp_dir().join(format!("yendory-resume-{}", std::process::id()));
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
    let mut full = Vec::new();
    run(&model, &cfg, &mut full);
    // The checkpoint of epoch 4 (the newest three are 4, 5 and 6) and where the session was.
    let files = checkpoint_files(&dir);
    let at4 = files.iter().find(|f| f.to_string_lossy().ends_with("epoch-04.nnova2")).unwrap();
    let info = checkpoint_info(at4).unwrap();
    assert_eq!((info.epoch, info.epochs, info.lr_scale), (4, 6, 1.));
    let resumed_model = Bicameral::load(at4).unwrap();
    let cfg2 = SessionConfig {
        resume_epochs: info.epoch,
        resume_scale: Some(info.lr_scale),
        checkpoint_dir: None,
        ..cfg
    };
    let mut rest = Vec::new();
    let r = run(&resumed_model, &cfg2, &mut rest);
    assert_eq!(r.epochs.first().unwrap().epoch, 5, "it continues with epoch 5");
    assert_eq!(r.epochs.len(), 2);
    let (a, b) = (batch_rates(&full), batch_rates(&rest));
    let later: Vec<_> = a.into_iter().filter(|(e, _, _)| *e > 4).collect();
    assert_eq!(later, b, "the same learning rate at every batch of epochs 5 and 6");
    let last4 = batch_rates(&full).into_iter().rfind(|(e, _, _)| *e == 4).unwrap().2;
    assert!((info.learning_rate_now - last4).abs() < 1e-9, "{} vs {last4}", info.learning_rate_now);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A well-taught model and steps far too large: every epoch makes it worse. With back-off the
/// learning rate halves after two such epochs; with the first-epoch override, epoch 1 is
/// accepted anyway and the session continues from it.
#[test]
fn backoff_halves_the_rate_and_the_override_accepts_the_first_epoch() {
    let base = SessionConfig {
        epochs: 6,
        patience: 6,
        foundation: false,
        learning_rate: 3e-3,
        ..SessionConfig::default()
    };
    let taught = run(&Bicameral::with_config(Config::default(), 7), &base, &mut Vec::new()).model;
    let harsh = SessionConfig { epochs: 6, patience: 8, learning_rate: 0.05, seed: 3, ..base.clone() };

    // Without back-off: nothing is accepted and the rate follows the schedule.
    let mut plain = Vec::new();
    let r0 = run(&taught, &harsh, &mut plain);
    println!(
        "no back-off: accepted {:?}",
        r0.epochs.iter().map(|e| e.selected.is_some()).collect::<Vec<_>>()
    );
    assert!(!r0.learned, "steps this large make every candidate worse");

    let mut backed = Vec::new();
    let r1 = run(&taught, &SessionConfig { backoff: true, ..harsh.clone() }, &mut backed);
    let lowered: Vec<(u32, f32)> = backed
        .iter()
        .filter_map(|e| {
            if let Event::LearningRateLowered { epoch, scale, .. } = e {
                Some((*epoch, *scale))
            } else {
                None
            }
        })
        .collect();
    println!(
        "back-off: lowered {lowered:?}, accepted {:?}",
        r1.epochs.iter().map(|e| e.selected.is_some()).collect::<Vec<_>>()
    );
    assert_eq!(lowered.first(), Some(&(BACKOFF_EPOCHS, 0.5)), "halved after {BACKOFF_EPOCHS} worse epochs");
    // The next epoch runs at half the scheduled rate at every batch.
    let (a, b) = (batch_rates(&plain), batch_rates(&backed));
    let next = BACKOFF_EPOCHS + 1;
    for ((e1, b1, l1), (e2, b2, l2)) in a.iter().zip(&b).filter(|((e, _, _), _)| *e == next) {
        assert_eq!((e1, b1), (e2, b2));
        assert!((l2 - 0.5 * l1).abs() < 1e-9, "{l2} vs half of {l1}");
    }

    // The override: epoch 1's best candidate is accepted although the safeguards reject it.
    let r2 = run(&taught, &SessionConfig { override_first_epoch: true, ..harsh }, &mut Vec::new());
    let first = &r2.epochs[0];
    assert!(first.overridden && first.selected.is_some(), "{first:?}");
    let chosen = first.candidates.iter().find(|c| c.selected).unwrap();
    assert!(!chosen.eligible, "it was accepted despite the safeguards");
    assert!(r2.learned);
    assert!(r2.epochs[1..].iter().all(|e| !e.overridden), "only the first epoch");
}

/// Probation: with every candidate rejected, training continues from the most promising
/// rejected one for at most `probation_epochs` epochs in a row, then the branch is discarded
/// and training restarts from the accepted model. The accepted model never changes.
#[test]
fn probation_continues_rejected_candidates_then_discards_the_branch() {
    use yendory_engine::learn::session::{EpochHook, Probation};
    let base = SessionConfig {
        epochs: 6,
        patience: 6,
        foundation: false,
        learning_rate: 3e-3,
        ..SessionConfig::default()
    };
    let taught = run(&Bicameral::with_config(Config::default(), 7), &base, &mut Vec::new()).model;
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let hook = {
        let seen = seen.clone();
        EpochHook(std::sync::Arc::new(move |epoch: u32, accepted: &Bicameral, working: &Bicameral| {
            seen.lock().unwrap().push((epoch, accepted.step, working.step));
        }))
    };
    let harsh = SessionConfig {
        epochs: 5,
        patience: 8,
        learning_rate: 0.05,
        seed: 3,
        probation_epochs: 2,
        epoch_hook: Some(hook),
        ..base
    };
    let r = run(&taught, &harsh, &mut Vec::new());
    let states: Vec<_> = r.epochs.iter().map(|e| e.probation.clone()).collect();
    println!("{states:?}");
    assert!(!r.learned, "steps this large make every candidate worse");
    assert_eq!(r.model.weights, taught.weights, "the accepted model is unchanged");
    assert_eq!(r.model.step, taught.step);
    let continued = |e: &Option<Probation>| match e {
        Some(Probation::Continued { epoch_in_branch, rejections, .. }) => {
            assert!(!rejections.is_empty());
            Some(*epoch_in_branch)
        }
        _ => None,
    };
    assert_eq!(continued(&states[0]), Some(1));
    assert_eq!(continued(&states[1]), Some(2));
    assert_eq!(states[2], Some(Probation::Discarded { epochs: 2 }));
    assert!(r.epochs[2].restored && !r.epochs[0].restored && !r.epochs[1].restored);
    assert_eq!(continued(&states[3]), Some(1), "a new branch");
    // The working model is the branch while on probation, and the accepted model after it.
    let seen = seen.lock().unwrap().clone();
    assert!(seen.iter().all(|&(_, accepted, _)| accepted == taught.step));
    assert!(seen[0].2 > taught.step && seen[1].2 > seen[0].2, "the branch keeps training: {seen:?}");
    assert_eq!(seen[2].2, taught.step, "discarded: back to the accepted model");
    // Off (the default): every epoch restarts from the accepted model, as before.
    let off =
        run(&taught, &SessionConfig { probation_epochs: 0, epoch_hook: None, ..harsh }, &mut Vec::new());
    assert!(off.epochs.iter().all(|e| e.probation.is_none()));
    assert!(off.epochs.iter().all(|e| e.continued.is_none() && !e.restored), "never left the accepted model");
}
