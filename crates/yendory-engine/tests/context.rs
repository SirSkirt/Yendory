//! Selectable context length: no parameter depends on it, and long contexts train and generate.
use yendory_engine::model::MAX_CONTEXT;
use yendory_engine::{Ablation, Bicameral, Config, Focus, KvCache};

fn long_text(tokens: usize, m: &Bicameral) -> Vec<yendory_engine::TokenId> {
    let mut text = String::new();
    let mut i = 0;
    while m.tokenizer.encode(&text).unwrap().len() < tokens + 1 {
        text.push_str(&format!("U: Where is key {i}?\nN: Key {i} is in box {}.\n", i % 7));
        i += 1;
    }
    m.tokenizer.encode(&text).unwrap()[..tokens + 1].to_vec()
}

#[test]
fn changing_the_context_keeps_every_weight() {
    let m = Bicameral::new();
    let long = m.with_context(4096).unwrap();
    assert_eq!(long.weights, m.weights);
    assert_eq!(long.parameter_count(), m.parameter_count());
    let ids = m.tokenizer.encode("U: Where is the key?\nN: ").unwrap();
    assert_eq!(long.logits(&ids, Ablation::default()).unwrap(), m.logits(&ids, Ablation::default()).unwrap());
    let back = Bicameral::from_bytes(&long.to_bytes().unwrap()).unwrap();
    assert_eq!(back.config.max_tokens, 4096);
    assert!(m.with_context(MAX_CONTEXT + 1).is_err());
    assert!(m.with_context(0).is_err());
}

#[test]
fn a_long_context_trains_and_the_cache_matches_the_full_pass() {
    let mut m = Bicameral::with_config(Config { max_tokens: 1024, ..Config::default() }, 4);
    let ids = long_text(900, &m);
    let (x, y) = (&ids[..900], &ids[1..]);
    let mask = vec![1.; 900];
    let before = m.loss_and_gradients(x, y, &mask, Ablation::default()).unwrap().0;
    for _ in 0..4 {
        let (_, g) = m.loss_and_gradients(x, y, &mask, Ablation::default()).unwrap();
        m.adam_step(&g, 3e-3, Focus::Joint).unwrap();
    }
    let after = m.loss_and_gradients(x, y, &mask, Ablation::default()).unwrap().0;
    assert!(after < before - 0.5, "{before} -> {after}");
    // KV-cached reading of 900 tokens gives the full forward pass's last-position logits.
    let full = m.logits(x, Ablation::default()).unwrap();
    let cached = KvCache::new(&m, Ablation::default()).prefill(&m, x).unwrap();
    let v = m.config.vocab;
    for (a, b) in full[full.len() - v..].iter().zip(&cached) {
        assert!((a - b).abs() < 2e-3 * (1. + a.abs()), "{a} vs {b}");
    }
}
