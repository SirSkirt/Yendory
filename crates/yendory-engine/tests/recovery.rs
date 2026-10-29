//! Recovery snapshots: a reading or a conversation interrupted (crash or stop) and resumed from
//! its last snapshot ends exactly where an uninterrupted run ends (on the CPU, bit for bit), and
//! a snapshot is refused for other settings, other data or a changed model.
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use yendory_engine::learn::corpus::{
    pretrain_mix_with, Category, Corpus, MixPart, PretrainConfig, PretrainEvent, ReadingState,
};
use yendory_engine::learn::recovery::{Recovery, RecoveryEvery, Resume};
use yendory_engine::{Bicameral, Config};

fn stories() -> String {
    let names = ["Tom", "Lily", "Ben", "Mia", "Sam", "Anna"];
    let things = ["ball", "box", "hat", "kite", "cake"];
    let mut out = String::new();
    for i in 0..300usize {
        let n = names[i % names.len()];
        let t = things[(i / 3) % things.len()];
        out.push_str(&format!(
            "{n} had a {t}. One day {n} lost the {t}. Then {n} found the {t} under a tree. ({i})\n<|endoftext|>\n"
        ));
    }
    out
}

fn parts() -> Vec<MixPart> {
    vec![MixPart {
        name: "stories".into(),
        category: Category::General,
        corpus: Corpus::prepare(&stories()),
        share: 1.,
        stage: None,
    }]
}

fn cfg() -> PretrainConfig {
    PretrainConfig {
        tokens: 12_000,
        batch_size: 4,
        window: 32,
        evaluations: 6,
        eval_windows: 8,
        min_loss_improvement: 0.,
        ..Default::default()
    }
}

fn start() -> Bicameral {
    let c = Config {
        d: 16,
        heads: 2,
        ff: 24,
        blocks: 2,
        split: 1,
        r1: 12,
        r2: 10,
        max_tokens: 32,
        ..Config::default()
    };
    Bicameral::with_config(c, 5)
}

/// What a run leaves: the final model, its evaluations' losses, and the snapshots it took, each
/// with the accepted model as it was saved when the snapshot was taken.
struct Run {
    model: Bicameral,
    losses: Vec<(u64, f32, bool)>,
    tokens: u64,
    snapshots: Vec<(ReadingState, Bicameral, Bicameral, bool)>,
    accepted: Vec<Bicameral>,
}

fn read(
    model: &Bicameral,
    cfg: &PretrainConfig,
    every: u64,
    resume: Option<Resume<ReadingState>>,
    stop_at: Option<u64>,
) -> Result<Run, String> {
    let saved = RefCell::new(model.clone());
    let accepted = RefCell::new(Vec::new());
    let snapshots = RefCell::new(Vec::new());
    let stop = AtomicBool::new(false);
    let mut on_snapshot = |s: yendory_engine::learn::recovery::Snapshot<'_, ReadingState>| {
        snapshots.borrow_mut().push((s.state.clone(), s.model.clone(), saved.borrow().clone(), s.stopped));
        Ok(())
    };
    let mut recovery = Recovery {
        every: RecoveryEvery { seconds: 0, steps: every },
        on_snapshot: Some(&mut on_snapshot),
        resume,
    };
    let r = pretrain_mix_with(
        model,
        &parts(),
        cfg,
        &mut |e| {
            if let (PretrainEvent::Batch { step, .. }, Some(at)) = (e, stop_at) {
                if *step == at {
                    stop.store(true, Ordering::Release);
                }
            }
        },
        &mut |m| {
            *saved.borrow_mut() = m.clone();
            accepted.borrow_mut().push(m.clone());
            Ok(())
        },
        &stop,
        &mut recovery,
    )?;
    drop(recovery);
    Ok(Run {
        model: r.model,
        losses: r.evaluations.iter().map(|e| (e.step, e.held_out_loss, e.accepted)).collect(),
        tokens: r.tokens,
        snapshots: snapshots.into_inner(),
        accepted: accepted.into_inner(),
    })
}

fn same(a: &Bicameral, b: &Bicameral) -> bool {
    a.step == b.step
        && a.weights == b.weights
        && a.momentum == b.momentum
        && a.variance == b.variance
        && a.tokenizer.words() == b.tokenizer.words()
}

#[test]
fn a_resumed_reading_ends_exactly_like_an_uninterrupted_one() {
    let m = start();
    let full = read(&m, &cfg(), 20, None, None).unwrap();
    assert!(full.snapshots.len() >= 3, "{}", full.snapshots.len());
    assert!(full.losses.iter().any(|l| l.2), "something was accepted");
    // A crash after each snapshot: resumed from it, with the model saved at that time.
    for (state, working, saved, stopped) in &full.snapshots {
        assert!(!stopped);
        assert_eq!(working.step, m.step + state.step, "the working model is at the snapshot's step");
        let resumed = read(
            saved,
            &cfg(),
            0,
            Some(Resume { state: state.clone(), model: working.clone(), extra: None }),
            None,
        )
        .unwrap();
        assert!(same(&resumed.model, &full.model), "resumed after step {}", state.step);
        assert_eq!(resumed.losses, full.losses);
        assert_eq!(resumed.tokens, full.tokens);
    }
}

#[test]
fn a_stopped_reading_continues_where_it_stopped() {
    let m = start();
    let full = read(&m, &cfg(), 0, None, None).unwrap();
    let stopped = read(&m, &cfg(), 0, None, Some(37)).unwrap();
    let (state, working, saved, was_stopped) = stopped.snapshots.last().cloned().unwrap();
    assert!(was_stopped && state.step == 37, "{}", state.step);
    let resumed = read(&saved, &cfg(), 0, Some(Resume { state, model: working, extra: None }), None).unwrap();
    assert!(same(&resumed.model, &full.model));
    assert_eq!(resumed.losses, full.losses);
}

/// A checkpoint accepted after the last snapshot (and saved before the crash) is not lost: the
/// resumed run measures it again and keeps it as the best, and ends where the uninterrupted run
/// ends.
#[test]
fn a_checkpoint_accepted_after_the_snapshot_is_kept() {
    let m = start();
    let full = read(&m, &cfg(), 10, None, None).unwrap();
    // A snapshot followed by an accepted checkpoint.
    let (state, working, _, _) = full
        .snapshots
        .iter()
        .find(|(s, ..)| full.accepted.iter().any(|a| a.step > m.step + s.step))
        .cloned()
        .expect("an acceptance after a snapshot");
    let newer = full.accepted.iter().find(|a| a.step > working.step).unwrap().clone();
    let resumed = read(&newer, &cfg(), 0, Some(Resume { state, model: working, extra: None }), None).unwrap();
    assert!(same(&resumed.model, &full.model));
}

#[test]
fn a_snapshot_is_refused_for_other_settings_data_or_model() {
    let m = start();
    let full = read(&m, &cfg(), 20, None, None).unwrap();
    let (state, working, saved, _) = full.snapshots[1].clone();
    let resume = || Some(Resume { state: state.clone(), model: working.clone(), extra: None });
    let other = PretrainConfig { learning_rate: 2e-3, ..cfg() };
    let err = read(&saved, &other, 0, resume(), None).err().unwrap();
    assert!(err.contains("other settings"), "{err}");
    let mut changed = state.clone();
    changed.data ^= 1;
    let err =
        read(&saved, &cfg(), 0, Some(Resume { state: changed, model: working.clone(), extra: None }), None)
            .err()
            .unwrap();
    assert!(err.contains("corpora changed"), "{err}");
    // A model that is neither the one saved nor a later checkpoint of this run.
    let err = read(&m, &cfg(), 0, resume(), None).err();
    if full.snapshots[1].0.accepted_step.is_some() {
        assert!(err.unwrap().contains("changed after"));
    }
}

mod conversation {
    use super::*;
    use yendory_engine::learn::conversation::{
        converse_with, parse_dialogue, ConversationConfig, ConversationSet, ConversationState,
    };

    fn sets() -> Vec<ConversationSet> {
        let dialogues = (0..120)
            .map(|i| {
                parse_dialogue(&format!(
                    "Ann: Do you like the color number {i}?\nBen: Yes, I like it.\nAnn: Why?\nBen: Because it is bright.\nAnn: Is number {i} your favourite?\nBen: Yes, number {i} is."
                ))
                .unwrap()
            })
            .collect();
        vec![ConversationSet { name: "x".into(), dialogues, nova: None, hold_out: true }]
    }

    fn cfg() -> ConversationConfig {
        ConversationConfig {
            tokens: 16_000,
            batch_size: 4,
            learning_rate: 3e-3,
            evaluations: 6,
            pool_size: 32,
            context_tests: 1,
            understanding_tests: 1,
            keep_balanced: true,
            min_loss_improvement: 0.3,
            competency_loss: 5.1,
            stage_patience: 1,
            ..ConversationConfig::default()
        }
    }

    fn model() -> Bicameral {
        let c = Config {
            d: 16,
            heads: 2,
            ff: 24,
            blocks: 2,
            split: 1,
            r1: 12,
            r2: 10,
            max_tokens: 256,
            ..Config::default()
        };
        Bicameral::with_config(c, 3)
    }

    type Snap = (ConversationState, Bicameral, Option<Bicameral>, Bicameral, bool);

    struct Out {
        model: Bicameral,
        balanced: Option<Bicameral>,
        checks: Vec<(u64, f32, bool, usize)>,
        snapshots: Vec<Snap>,
    }

    fn run(
        model: &Bicameral,
        every: u64,
        resume: Option<Resume<ConversationState>>,
        stop_at: Option<u64>,
    ) -> Out {
        let saved = RefCell::new(model.clone());
        let snapshots = RefCell::new(Vec::new());
        let stop = AtomicBool::new(false);
        let mut on_snapshot = |s: yendory_engine::learn::recovery::Snapshot<'_, ConversationState>| {
            snapshots.borrow_mut().push((
                s.state.clone(),
                s.model.clone(),
                s.extra.cloned(),
                saved.borrow().clone(),
                s.stopped,
            ));
            Ok(())
        };
        let mut recovery = Recovery {
            every: RecoveryEvery { seconds: 0, steps: every },
            on_snapshot: Some(&mut on_snapshot),
            resume,
        };
        let r = converse_with(
            model,
            &sets(),
            &[],
            &[],
            &cfg(),
            &mut |e| {
                if let (
                    yendory_engine::learn::conversation::ConversationEvent::Batch { step, .. },
                    Some(at),
                ) = (e, stop_at)
                {
                    if *step == at {
                        stop.store(true, Ordering::Release);
                    }
                }
            },
            &mut |m| {
                *saved.borrow_mut() = m.clone();
                Ok(())
            },
            &stop,
            &mut recovery,
        )
        .unwrap();
        drop(recovery);
        Out {
            model: r.model,
            balanced: r.balanced.map(|(m, _)| m),
            checks: r.checks.iter().map(|c| (c.step, c.held_out_loss, c.accepted, c.stage)).collect(),
            snapshots: snapshots.into_inner(),
        }
    }

    #[test]
    fn a_resumed_conversation_run_ends_exactly_like_an_uninterrupted_one() {
        let m = model();
        let full = run(&m, 7, None, None);
        assert!(full.snapshots.len() >= 3, "{}", full.snapshots.len());
        // The run moves through the curriculum's stages, and snapshots fall between a stage's
        // last check and its new practice pool.
        assert!(full.checks.iter().any(|c| c.3 > 0), "{:?}", full.checks);
        assert!(full.snapshots.iter().any(|s| s.0.stage > 0));
        assert!(full.checks.iter().any(|c| c.2), "something was accepted");
        for (state, working, extra, saved, _) in full.snapshots.iter().cloned() {
            let at = state.step;
            let resumed = run(&saved, 0, Some(Resume { state, model: working, extra }), None);
            assert!(same(&resumed.model, &full.model), "resumed after step {at}");
            assert_eq!(resumed.checks, full.checks, "resumed after step {at}");
            assert_eq!(
                resumed.balanced.map(|b| b.weights),
                full.balanced.as_ref().map(|b| b.weights.clone())
            );
        }
        // Stopped part-way, then continued.
        let stopped = run(&m, 0, None, Some(23));
        let (state, working, extra, saved, was_stopped) = stopped.snapshots.last().cloned().unwrap();
        assert!(was_stopped && state.step == 23);
        let resumed = run(&saved, 0, Some(Resume { state, model: working, extra }), None);
        assert!(same(&resumed.model, &full.model));
        assert_eq!(resumed.checks, full.checks);
    }
}
