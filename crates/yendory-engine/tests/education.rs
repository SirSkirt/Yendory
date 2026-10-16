//! Experimental bicameral education: each stage changes only the parts it is meant to teach,
//! Dense's contribution is measured, and the guard protects it.
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::corpus::{pretrain, Corpus, PretrainConfig};
use yendory_engine::learn::session::EducationStage;
use yendory_engine::{Bicameral, Focus};

fn facts() -> String {
    let pairs = [
        ("apple", "fruit"),
        ("banana", "fruit"),
        ("pear", "fruit"),
        ("dog", "animal"),
        ("cat", "animal"),
        ("horse", "animal"),
        ("rose", "flower"),
        ("tulip", "flower"),
        ("car", "vehicle"),
        ("bus", "vehicle"),
    ];
    let mut out = String::new();
    for i in 0..300 {
        let (a, b) = pairs[i % pairs.len()];
        out.push_str(&format!("A {a} is a kind of {b}. Every {a} is a {b}. ({i})\n\n"));
    }
    out
}

fn read(model: &Bicameral, stage: Option<EducationStage>) -> yendory_engine::learn::corpus::PretrainResult {
    let cfg = PretrainConfig {
        tokens: 12_000,
        batch_size: 8,
        window: 48,
        evaluations: 4,
        eval_windows: 12,
        grow_vocabulary: false,
        education: stage,
        learning_rate: 3e-3,
        ..Default::default()
    };
    pretrain(model, &Corpus::prepare(&facts()), &cfg, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false))
        .unwrap()
}

#[test]
fn each_stage_changes_only_its_hemispheres() {
    let start = Bicameral::with_config(Default::default(), 6);
    let a = read(&start, Some(EducationStage::DenseKnowledge));
    assert!(a.accepted && a.best_loss < a.baseline_loss);
    assert_eq!(
        a.model.group_values(Focus::Left),
        start.group_values(Focus::Left),
        "Stage A froze the transformer"
    );
    assert_ne!(a.model.group_values(Focus::Right), start.group_values(Focus::Right));
    assert_ne!(a.model.group_values(Focus::Interconnect), start.group_values(Focus::Interconnect));

    let b = read(&a.model, Some(EducationStage::TransformerLanguage));
    assert!(b.accepted);
    assert_ne!(b.model.group_values(Focus::Left), a.model.group_values(Focus::Left));
    assert_eq!(b.model.group_values(Focus::Right), a.model.group_values(Focus::Right), "Stage B froze Dense");
    assert_eq!(b.model.group_values(Focus::Interconnect), a.model.group_values(Focus::Interconnect));
}

#[test]
fn dense_learning_alone_makes_the_model_rely_on_dense() {
    let start = Bicameral::with_config(Default::default(), 6);
    let a = read(&start, Some(EducationStage::DenseKnowledge));
    let first = a.evaluations.first().unwrap().dense_reliance;
    let best = a.evaluations.iter().rfind(|e| e.accepted).unwrap().dense_reliance;
    // With the transformer frozen, every improvement had to come through Dense: cutting
    // Dense's input to the transformer now costs much more held-out loss than before.
    assert!(best > 0.05 && best > first, "reliance {first} -> {best}");
}
