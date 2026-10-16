//! Training and generation speed of each size preset on this machine:
//! `cargo run --release --example bench [preset ids…]`.
use std::time::Instant;
use yendory_engine::learn::{batch_loss_and_gradients, Row};
use yendory_engine::{size_presets, Ablation, Bicameral};

fn main() {
    let only: Vec<String> = std::env::args().skip(1).collect();
    for p in size_presets().into_iter().filter(|p| only.is_empty() || only.iter().any(|o| o == p.id)) {
        let m = Bicameral::with_config(p.config.clone(), 7);
        let n = 128;
        let rows: Vec<Row> = (0..16)
            .map(|r| Row {
                ids: (0..n).map(|i| ((i * 7 + r) % 120 + 1) as u16).collect(),
                targets: (0..n).map(|i| ((i * 7 + r + 1) % 120 + 1) as u16).collect(),
                mask: vec![1.; n],
            })
            .collect();
        batch_loss_and_gradients(&m, &rows[..2], Ablation::default()).unwrap();
        let t = Instant::now();
        batch_loss_and_gradients(&m, &rows, Ablation::default()).unwrap();
        let train = (16 * n) as f64 / t.elapsed().as_secs_f64();
        let t = Instant::now();
        let out =
            m.generate_cached_ids(&[85, 58, 32, 104, 105, 10, 78, 58, 32], 32, Ablation::default()).unwrap();
        let gen = out.len() as f64 / t.elapsed().as_secs_f64();
        println!(
            "{:<20} {:>11} params  training {:>7.0} tokens/s  generation {:>6.0} tokens/s",
            p.name,
            m.parameter_count(),
            train,
            gen
        );
    }
}
