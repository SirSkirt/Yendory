//! Teaching-session behaviour: Foundation candidates, selection, rollback, stop, records.
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::session::{teach, Event, Rejection, SessionConfig};
use yendory_engine::learn::{parse_lesson, Item, Kind};
use yendory_engine::Bicameral;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/compat")
}
fn model() -> Bicameral {
    Bicameral::load(root().join("fixtures/nova2-step15000.nnova2")).unwrap()
}
/// Lesson = reasoning + conversation of the sample course; archive = its dialogue + understanding.
fn course() -> (Vec<Item>, Vec<Item>) {
    let text = std::fs::read_to_string(root().join("fixtures/lessons/sample_course_a.txt")).unwrap();
    let items = parse_lesson(&text, "sample_course_a.txt").unwrap();
    let (lesson, archive) =
        items.into_iter().partition(|it| matches!(it.kind, Kind::Reasoning | Kind::Conversation));
    (lesson, archive)
}
fn quick() -> SessionConfig {
    SessionConfig {
        epochs: 2,
        batch_size: 8,
        learning_rate: 1e-3,
        eval_limit: Some(8),
        ..SessionConfig::default()
    }
}

#[test]
fn foundation_candidates_start_equal_and_records_are_complete() {
    let (lesson, archive) = course();
    let start = model();
    let mut events = Vec::new();
    let r =
        teach(&start, &lesson, &archive, &quick(), &mut |e| events.push(e.clone()), &AtomicBool::new(false))
            .unwrap();
    assert!(matches!(events.first(), Some(Event::Started { rehearsal_units, .. }) if *rehearsal_units > 0));
    assert!(matches!(events.last(), Some(Event::Finished { .. })));
    assert!(!r.epochs.is_empty());
    let mut previous_step = start.step;
    for e in &r.epochs {
        let foci: Vec<&str> = e.candidates.iter().map(|c| c.focus.as_str()).collect();
        assert_eq!(foci, ["dense", "transformer", "interconnect", "joint"]);
        // Every candidate starts from the same accepted model and sees the same batches.
        let steps: Vec<u64> = e.candidates.iter().map(|c| c.step).collect();
        assert!(steps.iter().all(|&s| s == steps[0] && s > previous_step), "{steps:?}");
        assert!(e.candidates.iter().all(|c| c.score.retention_loss.is_some()));
        assert!(e.candidates.iter().filter(|c| c.selected).count() <= 1);
        assert_eq!(e.selected.is_some(), e.candidates.iter().any(|c| c.selected));
        for c in &e.candidates {
            assert_eq!(c.eligible, c.rejections.iter().all(|x| *x == Rejection::NoImprovement));
            assert!(!c.selected || (c.eligible && c.reasons.is_empty()));
        }
        previous_step = e.accepted_step;
    }
    let base = &r.baseline;
    let fin = &r.final_score;
    if r.learned {
        assert!(
            fin.exact_correct > base.exact_correct
                || (fin.exact_correct == base.exact_correct && fin.heldout_loss < base.heldout_loss)
        );
        assert_eq!(r.model.step, r.epochs.last().unwrap().accepted_step);
    } else {
        assert_eq!(r.model.weights, start.weights);
    }
    eprintln!("baseline {base:?}\nfinal    {fin:?}\nlearned {}", r.learned);
}

#[test]
fn same_seed_same_session() {
    let (lesson, archive) = course();
    let cfg = SessionConfig { epochs: 1, foundation: false, ..quick() };
    let a = teach(&model(), &lesson, &archive, &cfg, &mut |_| {}, &AtomicBool::new(false)).unwrap();
    let b = teach(&model(), &lesson, &archive, &cfg, &mut |_| {}, &AtomicBool::new(false)).unwrap();
    assert_eq!(a.epochs, b.epochs);
    assert_eq!(a.model.weights, b.model.weights);
}

#[test]
fn nothing_eligible_keeps_the_original_model() {
    let (lesson, archive) = course();
    let start = model();
    // An absurd learning rate wrecks every candidate; the guards must reject all of them.
    let cfg = SessionConfig { epochs: 1, learning_rate: 5.0, ..quick() };
    let r = teach(&start, &lesson, &archive, &cfg, &mut |_| {}, &AtomicBool::new(false)).unwrap();
    assert!(!r.learned);
    assert!(r.epochs[0].candidates.iter().all(|c| !c.eligible && !c.selected));
    assert_eq!(r.model.weights, start.weights);
    assert_eq!(r.model.momentum, start.momentum);
    assert_eq!(r.model.step, start.step);
    assert_eq!(r.model.tokenizer, start.tokenizer);
}

#[test]
fn stop_discards_the_epoch_in_progress() {
    let (lesson, archive) = course();
    let start = model();
    let stop = AtomicBool::new(false);
    let mut batches = 0;
    let r = teach(
        &start,
        &lesson,
        &archive,
        &quick(),
        &mut |e| {
            if matches!(e, Event::Batch { .. }) {
                batches += 1;
                if batches == 3 {
                    stop.store(true, std::sync::atomic::Ordering::Release);
                }
            }
        },
        &stop,
    )
    .unwrap();
    assert!(r.stopped && !r.learned && r.epochs.is_empty());
    assert_eq!(r.model.weights, start.weights);
}
