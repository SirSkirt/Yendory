//! Conversation replies: greedy by default (identical to exam replies), sampled replies
//! reproducible from their seed, and the sampling rules on known probabilities.
use yendory_engine::learn::eval::reply;
use yendory_engine::learn::sampling::{reply_with, Sampling};
use yendory_engine::learn::Rng;
use yendory_engine::model::{Ablation, Bicameral, Config};

fn model() -> Bicameral {
    let cfg = Config {
        d: 16,
        heads: 2,
        ff: 24,
        blocks: 2,
        split: 1,
        r1: 12,
        r2: 10,
        max_tokens: 64,
        ..Config::default()
    };
    Bicameral::with_config(cfg, 5)
}

#[test]
fn default_reply_is_the_greedy_exam_reply_and_streams_every_token() {
    let m = model();
    let prompt = "U: hello there\nN: ";
    let mut streamed = String::new();
    let r = reply_with(&m, prompt, 20, &Sampling::default(), |t| {
        streamed.push_str(t);
        true
    })
    .unwrap();
    assert_eq!(r.text, reply(&m, prompt, 20, Ablation::default()).unwrap());
    assert_eq!(streamed, r.text);
    assert!(!r.stopped);
    assert!(r.tokens >= 1 && r.tokens <= 20);
}

#[test]
fn sampled_replies_repeat_with_the_same_seed_and_stop_on_request() {
    let m = model();
    let prompt = "U: tell me a story\nN: ";
    let s = Sampling { temperature: 1.5, seed: 42, ..Sampling::default() };
    let a = reply_with(&m, prompt, 16, &s, |_| true).unwrap();
    let b = reply_with(&m, prompt, 16, &s, |_| true).unwrap();
    assert_eq!(a, b);
    // Only the most likely token allowed: the same as greedy.
    let k1 = Sampling { temperature: 1.5, top_k: 1, seed: 9, ..Sampling::default() };
    assert_eq!(
        reply_with(&m, prompt, 16, &k1, |_| true).unwrap().text,
        reply(&m, prompt, 16, Ablation::default()).unwrap()
    );
    let mut n = 0;
    let stopped = reply_with(&m, prompt, 16, &s, |_| {
        n += 1;
        n < 3
    })
    .unwrap();
    assert!(stopped.stopped && !stopped.finished);
    assert_eq!(stopped.tokens, 3);
}

#[test]
fn sampling_rules_on_known_probabilities() {
    let p = [0.5f32, 0.3, 0.2, 0.];
    let greedy = Sampling::default();
    let mut rng = Rng::new(1);
    assert_eq!(greedy.pick(&p, &[], &mut rng), 0);
    // A token already written is 5 times less likely: 0.1 < 0.3.
    let penal = Sampling { repetition_penalty: 5., ..Sampling::default() };
    assert_eq!(penal.pick(&p, &[0, 0], &mut rng), 1);
    let count = |s: Sampling| {
        let mut rng = Rng::new(s.seed);
        let mut c = [0usize; 4];
        for _ in 0..20_000 {
            c[s.pick(&p, &[], &mut rng) as usize] += 1;
        }
        c
    };
    let c = count(Sampling { temperature: 1., seed: 3, ..Sampling::default() });
    assert_eq!(c[3], 0, "a token of probability 0 is never drawn");
    for (i, want) in [0.5, 0.3, 0.2].iter().enumerate() {
        let got = c[i] as f64 / 20_000.;
        assert!((got - want).abs() < 0.02, "token {i}: {got} vs {want}");
    }
    let c = count(Sampling { temperature: 1., top_k: 2, seed: 4, ..Sampling::default() });
    assert_eq!(c[2], 0);
    assert!((c[0] as f64 / 20_000. - 0.625).abs() < 0.02);
    // 0.5 alone is short of 0.6, so the two most likely tokens remain.
    let c = count(Sampling { temperature: 1., top_p: 0.6, seed: 5, ..Sampling::default() });
    assert_eq!(c[2], 0);
    assert!(c[1] > 0);
    // A low temperature favours the likely token: 0.5^4 / (0.5^4 + 0.3^4 + 0.2^4) ~ 0.86.
    let c = count(Sampling { temperature: 0.25, seed: 6, ..Sampling::default() });
    assert!((c[0] as f64 / 20_000. - 0.8601).abs() < 0.02);
}
