//! Visualizer measurements come from the real forward pass and do not change it.
use std::path::PathBuf;
use yendory_engine::{Ablation, Bicameral, KvCache};

fn model() -> Bicameral {
    Bicameral::load(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/compat/fixtures/nova2-step15000.nnova2"),
    )
    .unwrap()
}

#[test]
fn tracing_does_not_change_outputs() {
    let m = model();
    let ids = m.tokenizer.encode("U: The key is in the box. Where is the key?\nN: ").unwrap();
    let plain = KvCache::new(&m, Ablation::default()).prefill(&m, &ids).unwrap();
    let mut traced = KvCache::with_trace(&m, Ablation::default());
    assert_eq!(traced.prefill(&m, &ids).unwrap(), plain);
}

#[test]
fn measurements_are_consistent() {
    let m = model();
    let prompt = "U: The key is in the box. Where is the key?\nN: ";
    let r = m.measure(prompt).unwrap();
    let c = &m.config;
    let n = m.tokenizer.encode(prompt).unwrap().len();
    assert_eq!(r.tokens.len(), n);
    assert_eq!(r.trace.stages.len(), c.split + (c.blocks - c.split) * c.loops);
    for s in &r.trace.stages {
        assert_eq!(s.attention.len(), c.heads);
        for head in &s.attention {
            assert_eq!(head.len(), n);
            assert!((head.iter().sum::<f32>() - 1.).abs() < 1e-4);
        }
    }
    assert_eq!(r.trace.gates.len(), 1 + c.loops);
    assert!(r.gate_means.iter().all(|&g| (0. ..=1.).contains(&g)));
    assert_eq!(r.trace.right_trunk.len(), c.r1);
    assert_eq!(r.trace.right_upper.len(), c.loops);
    assert!(r.trace.right_upper.iter().all(|z| z.len() == c.r2));
    let total: f32 = r.next_tokens.iter().map(|t| t.1).sum();
    assert!(total > 0. && total <= 1.0001);
    for tv in [r.ablation.without_right_to_left, r.ablation.without_left_to_right] {
        assert!((0. ..=1.).contains(&tv));
    }
    // This trained model answers this prompt with "In the box." (tests/compat); its first
    // token is the character "I".
    assert_eq!(r.next_tokens[0].0, "I");
    assert_eq!(m.measure(prompt).unwrap(), r, "measurements are deterministic");
    eprintln!(
        "gates {:?}  right->left TV {:.3}  left->right TV {:.3}",
        r.gate_means, r.ablation.without_right_to_left, r.ablation.without_left_to_right
    );
}

/// The claim in docs/STATUS.md: silencing the right hemisphere and cutting the afferent path
/// give the same output.
#[test]
fn no_right_equals_no_afferent() {
    let m = model();
    let ids = m.tokenizer.encode("U: hello\nN: ").unwrap();
    let a = KvCache::new(&m, Ablation { no_right: true, ..Ablation::default() }).prefill(&m, &ids).unwrap();
    let b =
        KvCache::new(&m, Ablation { no_afferent: true, ..Ablation::default() }).prefill(&m, &ids).unwrap();
    assert_eq!(a, b);
}
