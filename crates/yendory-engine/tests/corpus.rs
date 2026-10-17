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
    assert!(r.new_words > 0 && r.model.tokenizer.words().len() <= 192);
    assert!(r.model.tokenizer.words().contains(&"named".to_string()));
    // Names are kept character-level, like in lessons.
    assert!(!r.model.tokenizer.words().contains(&"Lily".to_string()));
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

/// Reading a mix: batches follow the shares, every corpus is checked on its own held-out text,
/// and bicameral reading changes only the parts each corpus's stage may train.
#[test]
fn reading_a_mix_interleaves_and_checks_each_corpus() {
    use yendory_engine::learn::corpus::{pretrain_mix, Category, MixPart};
    use yendory_engine::learn::generate::{generate, Family, GenerateConfig};
    use yendory_engine::learn::session::EducationStage;
    use yendory_engine::Focus;
    let reasoning = generate(&GenerateConfig {
        families: vec![Family::Arithmetic, Family::World],
        difficulty: 1,
        bytes: 60_000,
        seed: 2,
    })
    .unwrap();
    let parts = |stages: [Option<EducationStage>; 2]| {
        vec![
            MixPart {
                name: "stories".into(),
                category: Category::General,
                corpus: Corpus::prepare(&stories()),
                share: 3.,
                stage: stages[0],
            },
            MixPart {
                name: "practice".into(),
                category: Category::Reasoning,
                corpus: Corpus::prepare(&reasoning),
                share: 1.,
                stage: stages[1],
            },
        ]
    };
    let start = fresh();
    let mut batches = [0usize; 2];
    let mut evaluated = Vec::new();
    let r = pretrain_mix(
        &start,
        &parts([None, None]),
        &quick(),
        &mut |e| match e {
            PretrainEvent::Batch { part, .. } => batches[*part] += 1,
            PretrainEvent::Evaluated(ev) => evaluated.push(ev.clone()),
            _ => {}
        },
        &mut |_| Ok(()),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(r.accepted && r.best_loss < r.baseline_loss);
    let total = (batches[0] + batches[1]) as f64;
    assert!((batches[0] as f64 / total - 0.75).abs() < 0.02, "{batches:?}");
    let last = evaluated.last().unwrap();
    assert_eq!(last.parts.len(), 2);
    assert!(last.parts[1].sample_prompt.starts_with("U: ") && last.parts[1].sample_prompt.ends_with("N: "));
    let weighted = 0.75 * last.parts[0].held_out_loss + 0.25 * last.parts[1].held_out_loss;
    assert!((weighted - last.held_out_loss).abs() < 1e-4);

    // Bicameral: both corpora in Stage A leave the transformer untouched.
    let a = pretrain_mix(
        &start,
        &parts([Some(EducationStage::DenseKnowledge), Some(EducationStage::DenseKnowledge)]),
        &PretrainConfig { grow_vocabulary: false, ..quick() },
        &mut |_| {},
        &mut |_| Ok(()),
        &AtomicBool::new(false),
    )
    .unwrap();
    if a.accepted {
        assert_eq!(a.model.group_values(Focus::Left), start.group_values(Focus::Left));
        assert_ne!(a.model.group_values(Focus::Right), start.group_values(Focus::Right));
    }
}
