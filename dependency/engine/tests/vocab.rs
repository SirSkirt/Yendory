//! Selectable vocabulary: capacity from the configuration, ids beyond 16 bits, resizing.
use yendory_engine::model::{MAX_PARAMETERS, MAX_VOCAB};
use yendory_engine::tokenizer::BASE;
use yendory_engine::{Ablation, Bicameral, Config, Focus, TokenId};

fn words(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("w{i:06}")).collect()
}

#[test]
fn a_large_vocabulary_trains_generates_and_round_trips() {
    // Nova-2 size with room for 100,000 words, 70,001 of them in use: ids go past 65,535.
    let cfg = Config { vocab: BASE + 100_000, ..Config::default() };
    Bicameral::check_config(&cfg).unwrap();
    let mut m = Bicameral::with_config(cfg, 9);
    assert_eq!(m.word_capacity(), 100_000);
    let mut w = words(70_000);
    w.push("zebra".into());
    m.tokenizer.set_words(w);
    let text = "U: Where is the zebra?\nN: The zebra is in w069999.\n";
    let ids = m.tokenizer.encode(text).unwrap();
    assert!(ids.iter().any(|&t| t as usize >= 65_536 + BASE), "{ids:?}");
    let x = &ids[..ids.len() - 1];
    let mask = vec![1.; x.len()];
    let before = m.loss_and_gradients(x, &ids[1..], &mask, Ablation::default()).unwrap().0;
    for _ in 0..8 {
        m.train_text(text, 3e-3, Focus::Joint).unwrap();
    }
    let after = m.loss_and_gradients(x, &ids[1..], &mask, Ablation::default()).unwrap().0;
    assert!(after < before - 1.0, "{before} -> {after}");
    // The cached generation can emit a word id beyond 16 bits.
    let prompt = m.tokenizer.encode("U: Where is the zebra?\nN: The").unwrap();
    let out = m.generate_cached_ids(&prompt, 8, Ablation::default()).unwrap();
    assert!(out.contains(&((BASE + 70_000) as TokenId)), "{:?}", m.tokenizer.decode(&out));
    let bytes = m.to_bytes().unwrap();
    let back = Bicameral::from_bytes(&bytes).unwrap();
    assert_eq!(back.tokenizer, m.tokenizer);
    assert_eq!(back.to_bytes().unwrap(), bytes);
}

#[test]
fn resizing_keeps_everything_learned() {
    let mut m = Bicameral::new();
    m.tokenizer.set_words(vec!["the".into(), "key".into(), "box".into()]);
    for _ in 0..3 {
        m.train_text("U: Where is the key?\nN: In the box.\n", 5e-3, Focus::Joint).unwrap();
    }
    let ids = m.tokenizer.encode("U: Where is the key?\nN: ").unwrap();
    let old = m.logits(&ids, Ablation::default()).unwrap();
    let big = m.with_word_capacity(5_000, 3).unwrap();
    assert_eq!((big.word_capacity(), big.step, big.tokenizer.words().len()), (5_000, m.step, 3));
    let new = big.logits(&ids, Ablation::default()).unwrap();
    // Every existing id scores exactly as before; the new slots are extra.
    let (vo, vn) = (m.config.vocab, big.config.vocab);
    for p in 0..ids.len() {
        assert_eq!(&old[p * vo..(p + 1) * vo], &new[p * vn..p * vn + vo]);
    }
    // Back down to the original size gives the original model back.
    let small = big.with_word_capacity(256, 1).unwrap();
    assert_eq!(small.weights, m.weights);
    assert_eq!(small.momentum, m.momentum);
    assert!(m.with_word_capacity(2, 1).is_err(), "cannot drop words in use");
}

#[test]
fn growth_respects_each_models_capacity_and_limits_hold() {
    use yendory_engine::learn::lessons::grow_vocabulary_from_texts;
    let tiny = Bicameral::with_config(Config { vocab: BASE + 10, ..Config::default() }, 1);
    assert_eq!(tiny.word_capacity(), 10);
    let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu ".repeat(20);
    let grown = grow_vocabulary_from_texts(&[], [text.as_str()], tiny.word_capacity(), 12);
    assert_eq!(grown.len(), 10);
    // A checkpoint cannot carry more words than its vocabulary holds.
    let mut over = tiny.clone();
    over.tokenizer.set_words(words(11));
    assert!(Bicameral::from_bytes(&over.to_bytes().unwrap()).is_err());
    // Limits: a vocabulary beyond MAX_VOCAB, or one pushing the model past MAX_PARAMETERS.
    assert!(Bicameral::check_config(&Config { vocab: MAX_VOCAB + 1, ..Config::default() }).is_err());
    let too_big = MAX_PARAMETERS / Config::default().d + 1;
    assert!(Bicameral::check_config(&Config { vocab: too_big, ..Config::default() }).is_err());
    assert!(Bicameral::check_config(&Config { vocab: BASE + 1_000_000, ..Config::default() }).is_ok());
}

#[test]
fn a_vocabulary_limit_adds_slots_only_for_learned_words() {
    use std::sync::atomic::AtomicBool;
    use yendory_engine::learn::corpus::{pretrain, Corpus, PretrainConfig};
    let text: String = (0..400)
        .map(|i| format!("Once upon a time a little dog named Rex found a red ball in the garden. ({i})\n\n"))
        .collect();
    let corpus = Corpus::prepare(&text);
    // No word slots at all to begin with; a limit of one million.
    let start = Bicameral::with_config(Config { vocab: BASE, ..Config::default() }, 2);
    let cfg = PretrainConfig {
        tokens: 6_000,
        batch_size: 8,
        window: 32,
        evaluations: 3,
        eval_windows: 8,
        max_words: 1_000_000,
        word_limit: Some(1_000_000),
        ..Default::default()
    };
    let r = pretrain(&start, &corpus, &cfg, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false)).unwrap();
    assert!(r.accepted && r.new_words > 5, "{}", r.new_words);
    // Exactly as many slots as words learned: nothing unused to pay for.
    assert_eq!(r.model.word_capacity(), r.model.tokenizer.words().len());
    assert_eq!(r.model.config.vocab, BASE + r.new_words);
    // Without a limit the allocated slots are the limit (Nova-2 behaviour).
    let fixed = PretrainConfig { word_limit: None, ..cfg };
    let r = pretrain(&start, &corpus, &fixed, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false)).unwrap();
    assert_eq!(r.model.tokenizer.words().len(), 0);
}
