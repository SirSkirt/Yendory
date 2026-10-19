//! Nova Neuro through the real learning pipeline: reading, a teaching session with Foundation
//! candidates, measurements and the thinking tree, saving and loading.
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::corpus::{pretrain, Corpus, PretrainConfig};
use yendory_engine::learn::parse_lesson;
use yendory_engine::learn::session::{teach, SessionConfig};
use yendory_engine::model::neuro::presets;
use yendory_engine::{Ablation, Bicameral, ThinkOptions};

fn neuro() -> Bicameral {
    let p = presets().into_iter().next().unwrap();
    Bicameral::with_design(p.config, Some(p.design), 7)
}

fn stories() -> String {
    let names = ["Tom", "Lily", "Ben", "Mia", "Sam", "Anna"];
    let rooms = ["kitchen", "garden", "hallway", "office", "bedroom"];
    let mut out = String::new();
    for i in 0..300usize {
        let n = names[i % names.len()];
        let a = rooms[(i / 2) % rooms.len()];
        let b = rooms[(i / 3 + 1) % rooms.len()];
        out.push_str(&format!(
            "{n} went to the {a}. Then {n} moved to the {b}. Where is {n}? {n} is in the {b}.\n<|endoftext|>\n"
        ));
    }
    out
}

#[test]
fn neuro_reads_learns_a_lesson_and_can_be_measured() {
    let start = neuro();
    assert_eq!(start.architecture(), "neuro");
    let corpus = Corpus::prepare(&stories());
    let cfg = PretrainConfig {
        tokens: 12_000,
        batch_size: 8,
        window: 48,
        evaluations: 3,
        eval_windows: 8,
        ..Default::default()
    };
    let r = pretrain(&start, &corpus, &cfg, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false)).unwrap();
    assert!(r.accepted && r.best_loss < r.baseline_loss - 0.3, "{} -> {}", r.baseline_loss, r.best_loss);
    let model = r.model;
    assert!(model.neuro.is_some(), "reading keeps the design");

    // A lesson with Foundation candidates (modules, language core, communication, joint).
    let mut text = String::from("[Dialogue Lesson]\n[examples]\n");
    for (i, n) in ["Tom", "Lily", "Ben", "Mia", "Sam", "Anna", "Max", "Kim", "Joe", "Liz"].iter().enumerate()
    {
        for room in ["kitchen", "garden", "hallway"] {
            text.push_str(&format!("{n} is in the {room}. Where is {n} {i}?  The {room}.\n"));
        }
    }
    text.push_str("[/examples]\n[/Dialogue Lesson]\n[Evaluation]\nUma is in the garden. Where is Uma?  The garden.\n[/Evaluation]\n");
    let items = parse_lesson(&text, "rooms.txt").unwrap();
    let scfg = SessionConfig {
        epochs: 1,
        batch_size: 8,
        learning_rate: 2e-3,
        heldout_loss_tolerance: 1.,
        ..SessionConfig::default()
    };
    let s = teach(&model, &items, &[], &scfg, &mut |_| {}, &AtomicBool::new(false)).unwrap();
    assert_eq!(s.epochs[0].candidates.len(), 4, "Foundation tries four candidates");
    assert!(s.model.neuro.is_some());

    // Measurements and the thinking tree come from the real forward pass.
    let m = s.model.measure("Where is Tom?").unwrap();
    let nt = m.trace.neuro.as_ref().expect("Neuro activity is measured");
    assert_eq!(nt.modules, ["working memory", "associative memory"]);
    assert_eq!(nt.rounds.len(), s.model.config.loops);
    for r in &nt.rounds {
        assert_eq!(r.gates.len(), 2);
        assert!(r.gates.iter().all(|g| (0. ..=1.).contains(g)));
        assert!(!r.patterns.is_empty());
    }
    assert_eq!(nt.memory_norm.len(), 2);
    let tree =
        s.model.think("Where is Tom?", &ThinkOptions { max_tokens: 4, ..ThinkOptions::default() }).unwrap();
    assert!(!tree.steps.is_empty());

    // Saving and loading keeps every weight and the design; generation is identical.
    let back = Bicameral::from_bytes(&s.model.to_bytes().unwrap()).unwrap();
    assert_eq!(back.weights, s.model.weights);
    let ids = back.tokenizer.encode("Where is Tom?").unwrap();
    assert_eq!(
        back.generate_cached_ids(&ids, 8, Ablation::default()).unwrap(),
        s.model.generate_cached_ids(&ids, 8, Ablation::default()).unwrap()
    );
}
