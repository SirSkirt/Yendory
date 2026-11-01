//! Where a conversation check spends its time, on the CPU: loss-only forward passes (the
//! practice pool and the held-out probes) at the stages' lengths, a training step, and writing
//! replies. cargo run --release --example check_profile -- <model.nnova2> [rows]
use std::time::Instant;
use yendory_engine::learn::eval::reply;
use yendory_engine::learn::{row_losses, Row, Trainer};
use yendory_engine::{Ablation, Bicameral, Focus};

const SENTENCES: [&str; 8] = [
    "I went to the market this morning and bought some fresh bread for the family.",
    "That sounds lovely, did you also find the apples you were looking for last week?",
    "Yes, and the farmer told me the weather had been perfect for them this year.",
    "We should bake a pie together on Sunday when your cousins come to visit us.",
    "Honestly I think the best part of the trip was the long walk by the river.",
    "If you need help with the garden later I can bring my tools over after lunch.",
    "The new teacher explained the lesson so clearly that everyone understood it.",
    "Let me know what time works for you and I will make sure to be ready.",
];

/// A conversation of `turns` turns, each `per_turn` sentences long, as one row.
fn row(m: &Bicameral, turns: usize, per_turn: usize, k: usize) -> Row {
    let mut text = String::new();
    for t in 0..turns {
        text.push_str(if t % 2 == 0 { "U: " } else { "N: " });
        for s in 0..per_turn {
            text.push_str(SENTENCES[(k + t * 3 + s) % SENTENCES.len()]);
            text.push(' ');
        }
        text.push('\n');
    }
    let mut ids = m.tokenizer.encode(&text).unwrap();
    ids.truncate(m.config.max_tokens + 1);
    let targets = ids[1..].to_vec();
    ids.pop();
    let mask = vec![1.; targets.len()];
    Row { ids, targets, mask }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = Bicameral::load(&a[1]).unwrap();
    let n: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(64);
    println!(
        "model: {} parameters, neuro {}, context {}, threads {}",
        m.parameter_count(),
        m.neuro.is_some(),
        m.config.max_tokens,
        yendory_engine::learn::cpu_threads()
    );
    // `mixed` as a third argument runs only the mixed-pool comparison.
    let only_mixed = a.get(3).is_some_and(|x| x == "mixed");
    // Cost of one forward pass by length (is it linear, or worse?).
    for len in [64usize, 128, 256, 512, 1024].into_iter().filter(|l| *l <= m.config.max_tokens && !only_mixed)
    {
        let ids: Vec<u32> = (0..len).map(|i| (i % 100) as u32 + 32).collect();
        let r = Row { targets: ids.clone(), mask: vec![1.; len], ids };
        let t = Instant::now();
        let reps = 4;
        for _ in 0..reps {
            m.loss(&r.ids, &r.targets, &r.mask, Ablation::default()).unwrap();
        }
        let s = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "one forward pass of {len:>4} tokens: {:>7.1} ms ({:>6.0} tokens/s on one thread)",
            s * 1e3,
            len as f64 / s
        );
    }
    // The practice pool at each stage's length (2 turns, 6 turns, whole), short and long turns.
    for (stage, turns) in [("short exchanges", 2usize), ("a few turns", 6), ("whole conversations", 12)]
        .into_iter()
        .filter(|_| !only_mixed)
    {
        for per_turn in [1usize, 6] {
            let rows: Vec<Row> = (0..n).map(|k| row(&m, turns, per_turn, k)).collect();
            let tokens: usize = rows.iter().map(|r| r.ids.len()).sum();
            let t = Instant::now();
            row_losses(&m, &rows, Ablation::default()).unwrap();
            let s = t.elapsed().as_secs_f64();
            println!(
                "pool, {stage:<20} {per_turn} sentence(s)/turn: {n} rows, {:>5} tokens each, {:>7.1} s, {:>6.0} tokens/s; 4,096 rows would take {:>6.0} s",
                tokens / n,
                s,
                tokens as f64 / s,
                s * 4096. / n as f64
            );
        }
    }
    // A mixed pool as the real one is laid out: datasets one after another, so the long
    // conversations sit together at the end. Fixed slices per thread (the old way) against
    // shared longest-first work (the new way); the losses must be identical.
    let rows: Vec<Row> =
        (0..n).map(|k| if k < n * 3 / 4 { row(&m, 2, 1, k) } else { row(&m, 6, 6, k) }).collect();
    let tokens: usize = rows.iter().map(|r| r.ids.len()).sum();
    let threads = yendory_engine::learn::cpu_threads().max(1);
    let t = Instant::now();
    let chunk = rows.len().div_ceil(threads);
    let old: Vec<f32> = std::thread::scope(|s| {
        let hs: Vec<_> = rows
            .chunks(chunk)
            .map(|c| {
                s.spawn(|| {
                    c.iter()
                        .map(|r| m.loss(&r.ids, &r.targets, &r.mask, Ablation::default()).unwrap())
                        .collect::<Vec<f32>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let s_old = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let new = row_losses(&m, &rows, Ablation::default()).unwrap();
    let s_new = t.elapsed().as_secs_f64();
    println!(
        "mixed pool, {n} rows ({} short, {} long), {tokens} tokens: fixed slices {s_old:.1} s, shared work {s_new:.1} s ({:.2}x); identical losses: {}",
        n * 3 / 4,
        n - n * 3 / 4,
        s_old / s_new,
        old.iter().zip(&new).all(|(a, b)| a.to_bits() == b.to_bits())
    );
    if only_mixed {
        return;
    }
    // A training step (batch of 16 rows of a few turns).
    let rows: Vec<Row> = (0..16).map(|k| row(&m, 6, 1, k)).collect();
    let tokens: usize = rows.iter().map(|r| r.ids.len()).sum();
    let mut model = m.clone();
    let mut trainer = Trainer::new(&model);
    let t = Instant::now();
    for _ in 0..2 {
        trainer.step(&mut model, &rows, Ablation::default(), 1e-5, Focus::Joint).unwrap();
    }
    let s = t.elapsed().as_secs_f64() / 2.;
    println!(
        "training step, 16 rows x {} tokens: {:.1} s, {:.0} tokens/s",
        tokens / 16,
        s,
        tokens as f64 / s
    );
    // Writing replies (KV cache), from a 200-token prompt.
    let prompt = format!("{}N: ", (0..4).map(|k| format!("U: {}\n", SENTENCES[k])).collect::<String>());
    let t = Instant::now();
    let r = reply(&m, &prompt, 32, Ablation::default()).unwrap();
    let s = t.elapsed().as_secs_f64();
    println!(
        "one reply of up to 32 tokens ({} chars) after a {}-token prompt: {:.2} s",
        r.len(),
        m.tokenizer.encode(&prompt).unwrap().len(),
        s
    );
}
