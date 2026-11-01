//! Investigation: what vocabulary growth does to text a model has already learned.
//! 1. Teach lesson A with no new words (every word spelled with characters).
//! 2. Grow the vocabulary from lesson B, as teaching B would (new slots, random init).
//! 3. Measure lesson A's held-out loss before and after the growth (no training between),
//!    and how many of lesson A's tokens changed.
//! 4. Train on lesson A again and measure how fast it recovers.
//!
//! cargo run --release --example vocab_probe -- <lesson A> <lesson B> [epochs]
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::lessons::grow_vocabulary;
use yendory_engine::learn::session::{teach, SessionConfig};
use yendory_engine::learn::{batch_loss_and_gradients, encode_item, parse_lesson, Item, Rng};
use yendory_engine::{Ablation, Bicameral, Config};

fn heldout_loss(m: &Bicameral, items: &[Item]) -> (f32, usize) {
    let mut rng = Rng::new(1);
    let rows: Vec<_> = items
        .iter()
        .filter(|i| i.held_out)
        .filter_map(|i| encode_item(&m.tokenizer, &i.segments, m.config.max_tokens, &mut rng).unwrap())
        .collect();
    let tokens = rows.iter().map(|r| r.ids.len()).sum();
    (batch_loss_and_gradients(m, &rows, Ablation::default()).unwrap().0, tokens)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a = parse_lesson(&std::fs::read_to_string(&args[1]).unwrap(), "a").unwrap();
    let b = parse_lesson(&std::fs::read_to_string(&args[2]).unwrap(), "b").unwrap();
    let epochs = args.get(3).and_then(|e| e.parse().ok()).unwrap_or(8);
    let quiet = &mut |_: &_| {};
    let no_growth = SessionConfig {
        epochs,
        grow_vocabulary: false,
        foundation: false,
        seed: 3,
        ..SessionConfig::default()
    };
    let start = Bicameral::with_config(Config { vocab: 128, ..Config::default() }, 7);
    let taught = teach(&start, &a, &[], &no_growth, quiet, &AtomicBool::new(false)).unwrap().model;
    let (before, tokens_before) = heldout_loss(&taught, &a);
    println!("after teaching A ({epochs} epochs, characters only): A held-out loss {before:.4} over {tokens_before} tokens");

    // Grow the vocabulary from B's training text, as teaching B would.
    let words = grow_vocabulary(taught.tokenizer.words(), &b, 7500, 12);
    let mut grown = taught.with_word_capacity(words.len(), 0x5EED).unwrap();
    grown.tokenizer.set_words(words.clone());
    let (after, tokens_after) = heldout_loss(&grown, &a);
    let in_a: Vec<&String> = words
        .iter()
        .filter(|w| {
            a.iter().any(|i| {
                i.segments.iter().any(|s| {
                    s.text.split(|c: char| !c.is_ascii_alphanumeric() && c != '\'').any(|x| x == w.as_str())
                })
            })
        })
        .collect();
    println!(
        "grew {} words from B ({} of them occur in A, e.g. {:?}); A's held-out text: {tokens_before} -> {tokens_after} tokens",
        words.len(),
        in_a.len(),
        in_a.iter().take(8).collect::<Vec<_>>()
    );
    println!("A held-out loss before growth {before:.4}, right after growth (no training) {after:.4}");
    // Per token, loss is not comparable when the token count changes; compare total nats too.
    println!(
        "total held-out nats: {:.1} -> {:.1}",
        before * tokens_before as f32,
        after * tokens_after as f32
    );
    // Recovery: plain training steps on A's training units (same steps for both), measured on
    // A's held-out text. With growth the new word tokens start untrained; without, nothing changed.
    let train = |m: &Bicameral, epochs: usize| -> Bicameral {
        let mut m = m.clone();
        let mut rng = Rng::new(5);
        let rows: Vec<_> = a
            .iter()
            .filter(|i| !i.held_out)
            .filter_map(|i| encode_item(&m.tokenizer, &i.segments, m.config.max_tokens, &mut rng).unwrap())
            .collect();
        for _ in 0..epochs {
            for batch in rows.chunks(16) {
                let (_, g) = batch_loss_and_gradients(&m, batch, Ablation::default()).unwrap();
                m.adam_step(&g, 1e-3, yendory_engine::Focus::Joint).unwrap();
            }
        }
        m
    };
    for e in [1, 2, 4, 8] {
        let (g, _) = heldout_loss(&train(&grown, e), &a);
        let (c, _) = heldout_loss(&train(&taught, e), &a);
        println!("{e} more epoch(s) on A: with growth {g:.4} (total {:.0} nats), without growth {c:.4} (total {:.0} nats)", g * tokens_after as f32, c * tokens_before as f32);
    }
}
