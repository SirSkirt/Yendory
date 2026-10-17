//! Training benchmark on this machine: identical batches on every available device.
//! cargo run --release --features gpu --example train_bench -- [size] [context] [rows] [tokens] [steps]
use yendory_engine::learn::bench::{candidates, run, sample_rows};
use yendory_engine::{Bicameral, Config};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let size = a.get(1).map(String::as_str).unwrap_or("8m");
    let context: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let rows: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let tokens: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(context);
    let steps: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);
    let c = yendory_engine::size_presets().into_iter().find(|p| p.id == size).unwrap().config;
    let model = Bicameral::with_config(Config { max_tokens: context, ..c }, 3);
    let batch = sample_rows(&model, rows, tokens);
    let b = run(&model, &batch, steps, &candidates(), &mut |d| eprintln!("measuring {d}…"));
    println!("{}", serde_json::to_string_pretty(&b).unwrap());
}
