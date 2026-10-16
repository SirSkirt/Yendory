//! Corpus pre-training: held-out split, vocabulary, best-checkpoint selection, stop and rollback.
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::corpus::{pretrain, Corpus, PretrainConfig, PretrainEvent};
use yendory_engine::{Bicameral, Config};

/// A small synthetic story corpus with regular grammar (TinyStories-like format).
fn stories() -> String {
    let names = ["Tom", "Lily", "Ben", "Mia", "Sam", "Anna", "Max", "Sue"];
    let animals = ["cat", "dog", "bird", "fish", "frog", "duck"];
    let things = ["ball", "box", "hat", "kite", "cake", "book", "boat"];
    let colours = ["red", "blue", "green", "big", "small", "yellow"];
    let feelings = ["happy", "sad", "excited", "tired", "scared"];
    let mut out = String::new();
    for i in 0..600usize {
        let n = names[i % names.len()];
        let a = animals[(i / 3) % animals.len()];
        let t = things[(i / 7) % things.len()];
        let c = colours[(i / 2) % colours.len()];
        let f = feelings[(i / 5) % feelings.len()];
        out.push_str(&format!(
            "Once upon a time, there was a {a} named {n}. {n} had a {c} {t}. One day, {n} lost the {t}. \
             {n} was very {f}. Then {n} found the {t} under a tree. {n} said, \u{201C}Thank you, tree!\u{201D} \
             The end. ({i})\n<|endoftext|>\n"
        ));
    }
    out
}

fn fresh() -> Bicameral {
    Bicameral::with_config(Config::default(), 11)
}

fn quick() -> PretrainConfig {
    PretrainConfig {
        tokens: 30_000,
        batch_size: 8,
        window: 64,
        evaluations: 5,
        eval_windows: 16,
        ..Default::default()
    }
}

#[test]
fn reading_lowers_held_out_loss_and_keeps_the_best_checkpoint() {
    let corpus = Corpus::prepare(&stories());
    assert!(!corpus.held_out.is_empty() && corpus.train.len() > corpus.held_out.len() * 5);
    assert_eq!(corpus.dropped_chars, 0, "typographic quotes are converted, not dropped");
    let start = fresh();
    let mut events = Vec::new();
    let mut saved = Vec::new();
    let r = pretrain(
        &start,
        &corpus,
        &quick(),
        &mut |e| events.push(e.clone()),
        &mut |m| {
            saved.push(m.step);
            Ok(())
        },
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(r.accepted);
    assert!(r.best_loss < r.baseline_loss - 0.5, "{} -> {}", r.baseline_loss, r.best_loss);
    assert!(r.new_words > 0 && r.model.tokenizer.words.len() <= 192);
    assert!(r.model.tokenizer.words.contains(&"named".to_string()));
    // Names are kept character-level, like in lessons.
    assert!(!r.model.tokenizer.words.contains(&"Lily".to_string()));
    let accepted: Vec<_> = r.evaluations.iter().filter(|e| e.accepted).collect();
    assert_eq!(saved.len(), accepted.len());
    let last = accepted.last().unwrap();
    assert_eq!(r.model.step, last.step);
    assert_eq!(saved.last(), Some(&last.step));
    assert!((last.held_out_loss - r.best_loss).abs() < 1e-6);
    assert!(matches!(events.first(), Some(PretrainEvent::Started { .. })));
    assert!(matches!(events.last(), Some(PretrainEvent::Finished { accepted: true, .. })));
    // The reading run is reproducible from its seed.
    let again =
        pretrain(&start, &corpus, &quick(), &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false)).unwrap();
    assert_eq!(again.model.weights, r.model.weights);
}

#[test]
fn stopping_or_failing_keeps_the_original_model() {
    let corpus = Corpus::prepare(&stories());
    let start = fresh();
    let r =
        pretrain(&start, &corpus, &quick(), &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(true)).unwrap();
    assert!(r.stopped && !r.accepted);
    assert_eq!(r.model.weights, start.weights);
    assert_eq!(r.model.tokenizer, start.tokenizer);
    // An absurd learning rate never produces an accepted checkpoint.
    let wild = PretrainConfig { learning_rate: 5.0, ..quick() };
    match pretrain(&start, &corpus, &wild, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false)) {
        Ok(r) => {
            assert!(!r.accepted);
            assert_eq!(r.model.weights, start.weights);
        }
        Err(e) => assert!(e.contains("diverged"), "{e}"),
    }
}

#[test]
fn a_corpus_without_held_out_documents_is_refused() {
    let corpus = Corpus::prepare("Just one short document that is long enough.");
    assert!(
        pretrain(&fresh(), &corpus, &quick(), &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false)).is_err()
    );
}
