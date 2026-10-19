//! Teaching-session behaviour: Foundation candidates, selection, rollback, stop, records.
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::session::{teach, CandidateTally, Event, Rejection, SessionConfig};
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
            assert_eq!(
                c.eligible,
                c.rejections.iter().all(|x| matches!(x, Rejection::NoImprovement | Rejection::Outranked))
            );
            assert!(!c.selected || (c.eligible && c.reasons.is_empty()));
            // Every candidate is either selected or has at least one recorded reason.
            assert_ne!(c.selected, !c.rejections.is_empty(), "{c:?}");
            assert_eq!(c.rejections.is_empty(), c.reasons.is_empty());
            if c.rejections.contains(&Rejection::Outranked) {
                assert_eq!(c.rejections, [Rejection::Outranked]);
                assert!(e.selected.is_some());
            }
        }
        previous_step = e.accepted_step;
    }
    // The tally counts every candidate and every reason exactly.
    let tally = CandidateTally::of(&r.epochs);
    let all: Vec<_> = r.epochs.iter().flat_map(|e| &e.candidates).collect();
    assert_eq!(tally.candidates, all.len());
    assert_eq!(tally.selected, all.iter().filter(|c| c.selected).count());
    assert_eq!(tally.continued, all.iter().filter(|c| c.continued).count());
    for reason in Rejection::ALL {
        let n = all.iter().filter(|c| c.rejections.contains(&reason)).count();
        assert_eq!(tally.count(reason), n, "{reason:?}");
        let sole = all.iter().filter(|c| c.rejections == [reason]).count();
        assert_eq!(tally.sole.get(reason.name()).copied().unwrap_or(0), sole);
    }
    assert_eq!(tally.by_focus.values().map(|f| f.candidates).sum::<usize>(), all.len());
    eprintln!("{}", serde_json::to_string(&tally).unwrap());
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

#[test]
fn bicameral_education_stages_choose_candidates_and_measure_dense() {
    use yendory_engine::learn::session::EducationStage;
    use yendory_engine::Focus;
    let (lesson, archive) = course();
    let start = model();
    let cfg = SessionConfig {
        epochs: 1,
        foundation: false,
        education: Some(EducationStage::Cooperative),
        ..quick()
    };
    let mut focuses = Vec::new();
    let r = teach(
        &start,
        &lesson,
        &archive,
        &cfg,
        &mut |e| {
            if let Event::Candidate(c) = e {
                focuses.push(c.focus.clone());
                assert!(c.score.dense_reliance.is_some(), "Dense reliance is measured in education");
            }
        },
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(focuses, ["dense+interconnect", "joint"]);
    assert!(r.baseline.dense_reliance.is_some());
    // With Foundation, each part the stage may train is tried on its own as well.
    let cfg = SessionConfig {
        epochs: 1,
        foundation: true,
        education: Some(EducationStage::DenseKnowledge),
        ..quick()
    };
    let mut focuses = Vec::new();
    let r = teach(
        &start,
        &lesson,
        &archive,
        &cfg,
        &mut |e| {
            if let Event::Candidate(c) = e {
                focuses.push(c.focus.clone());
                assert!(c.score.without_dense_loss.is_some());
            }
        },
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(focuses, ["dense", "interconnect", "dense+interconnect"]);
    assert_eq!(
        r.model.group_values(Focus::Left),
        start.group_values(Focus::Left),
        "stage A never trains the transformer"
    );

    let cfg = SessionConfig { epochs: 1, education: Some(EducationStage::TransformerLanguage), ..quick() };
    let r = teach(&start, &lesson, &archive, &cfg, &mut |_| {}, &AtomicBool::new(false)).unwrap();
    assert_eq!(r.model.group_values(Focus::Right), start.group_values(Focus::Right));
    assert_eq!(r.model.group_values(Focus::Interconnect), start.group_values(Focus::Interconnect));
    // Ordinary teaching does not pay for the Dense measurement.
    let plain = SessionConfig { epochs: 1, foundation: false, ..quick() };
    let r = teach(&start, &lesson, &archive, &plain, &mut |_| {}, &AtomicBool::new(false)).unwrap();
    assert!(r.baseline.dense_reliance.is_none());
}

/// Flat epochs: candidates that are safe but not better by the required amount are not thrown
/// away. Training continues from the best of them (weights and Adam state), so improvement builds
/// up over epochs until it is large enough to be accepted; the result is always the best
/// accepted checkpoint.
#[test]
fn training_continues_through_flat_epochs() {
    let text = std::fs::read_to_string(root().join("fixtures/lessons/sample_course_a.txt")).unwrap();
    let lesson = parse_lesson(&text, "a.txt").unwrap();
    let start = Bicameral::with_config(yendory_engine::Config::default(), 21);
    // Slow steps and a threshold no single epoch reaches.
    let cfg = SessionConfig {
        epochs: 10,
        foundation: false,
        learning_rate: 2e-4,
        min_loss_improvement: 0.3,
        patience: 10,
        eval_limit: Some(8),
        ..SessionConfig::default()
    };
    let mut events = Vec::new();
    let r =
        teach(&start, &lesson, &[], &cfg, &mut |e| events.push(e.clone()), &AtomicBool::new(false)).unwrap();
    let epochs: Vec<_> =
        r.epochs.iter().map(|e| (e.selected.is_some(), e.continued.is_some(), e.flat_epochs)).collect();
    println!("epochs (accepted, continued, flat): {epochs:?}");

    let per_epoch = r.epochs[0].candidates[0].step - start.step;
    // A flat stretch followed by an accepted epoch: the accepted model includes the flat
    // epochs' training (it was not restarted from the last accepted model).
    let flat = r.epochs.iter().position(|e| e.continued.is_some()).expect("a flat epoch");
    let after = flat + r.epochs[flat..].iter().position(|e| e.selected.is_some()).expect("accepted after it");
    let before_step = if flat == 0 { start.step } else { r.epochs[flat - 1].accepted_step };
    for e in &r.epochs[flat..after] {
        assert!(e.continued.is_some() && !e.restored, "flat epochs continue training");
        assert_eq!(e.accepted_step, before_step, "nothing is accepted during the flat stretch");
    }
    assert_eq!(r.epochs[after].accepted_step, before_step + per_epoch * (after - flat + 1) as u64);
    assert_eq!(r.epochs[after].flat_epochs, 0);
    assert!(r.learned);
    assert_eq!(r.final_score, r.epochs.last().unwrap().accepted_score);
}

fn stories() -> std::sync::Arc<Vec<String>> {
    std::sync::Arc::new(
        (0..60)
            .map(|i| format!("Once upon a time, a little dog named Rex {i} found a red ball in the park. He said, \u{201c}I am happy!\u{201d} and played all day."))
            .collect(),
    )
}

/// Dropout and general-reading rehearsal are off by default: with both off (and reading text
/// present but unused) a session trains exactly as before, bit for bit.
#[test]
fn training_switches_off_change_nothing() {
    let (lesson, archive) = course();
    let start = model();
    let run = |cfg: &SessionConfig| {
        teach(&start, &lesson, &archive, cfg, &mut |_| {}, &AtomicBool::new(false)).unwrap()
    };
    let base = run(&quick());
    let off = run(&SessionConfig { dropout: 0., reading_share: 0., reading: Some(stories()), ..quick() });
    assert_eq!(base.model.weights, off.model.weights);
    assert_eq!(base.epochs, off.epochs);
}

/// Dropout trains differently (and still selects safely); general reading replaces lesson rows
/// without changing the number of batches per epoch.
#[test]
fn dropout_and_reading_rehearsal_train() {
    let (lesson, _) = course();
    let start = model();
    // No archive (no lesson rehearsal), so an epoch's rows are exactly its lesson units.
    let run = |cfg: &SessionConfig| {
        let mut batches = Vec::new();
        let r = teach(
            &start,
            &lesson,
            &[],
            cfg,
            &mut |e| {
                if let Event::Batch { epoch, focus, batches: n, .. } = e {
                    if focus == "joint" {
                        batches.push((*epoch, *n));
                    }
                }
            },
            &AtomicBool::new(false),
        )
        .unwrap();
        batches.dedup();
        (r, batches)
    };
    let (base, base_batches) = run(&quick());
    let (dropped, _) = run(&SessionConfig { dropout: 0.1, ..quick() });
    assert_ne!(dropped.epochs[0].candidates[0].train_loss, base.epochs[0].candidates[0].train_loss);
    let (read, read_batches) =
        run(&SessionConfig { reading_share: 0.2, reading: Some(stories()), ..quick() });
    assert_eq!(read_batches, base_batches, "the same number of batches per epoch");
    assert_ne!(read.epochs[0].candidates[0].train_loss, base.epochs[0].candidates[0].train_loss);
    // Reading without text is refused rather than silently ignored.
    let missing = teach(
        &start,
        &lesson,
        &[],
        &SessionConfig { reading_share: 0.2, ..quick() },
        &mut |_| {},
        &AtomicBool::new(false),
    );
    assert!(missing.is_err());
    for r in [&dropped, &read] {
        for e in &r.epochs {
            for c in &e.candidates {
                assert_ne!(c.selected, !c.rejections.is_empty());
            }
        }
    }
}
