//! GPU vs CPU on a given size: loss, every gradient and two AdamW steps.
//! YENDORY_GPU_SOFTWARE=1 cargo run --release --features gpu --example gpu_check -- 8m 1024 600
use std::sync::Arc;
use yendory_engine::gpu::{Gpu, GpuModel};
use yendory_engine::learn::{batch_loss_and_gradients, Row};
use yendory_engine::{Ablation, Bicameral, Config, Focus};

fn worst(a: &[f32], b: &[f32]) -> (f32, f32) {
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs()));
    (a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max), scale)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size = args.get(1).map(String::as_str).unwrap_or("8m");
    let context: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let len: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(600);
    let c = yendory_engine::size_presets().into_iter().find(|p| p.id == size).unwrap().config;
    let mut model = Bicameral::with_config(Config { vocab: 128, max_tokens: context, ..c }, 3);
    // Grow a vocabulary as Studio does (slots added for learned words).
    let words: Vec<String> = (0..700).map(|i| format!("word{i:03}")).collect();
    model = model.with_word_capacity(words.len(), 9).unwrap();
    model.tokenizer.set_words(words);
    let text: String =
        (0..400).map(|i| format!("word{:03} cat sat on the mat {i}. ", (i * 37) % 700)).collect();
    let ids = model.tokenizer.encode(&text).unwrap();
    for _ in 0..2 {
        let row = Row { ids: ids[..200].to_vec(), targets: ids[1..201].to_vec(), mask: vec![1.; 200] };
        let (_, g) = batch_loss_and_gradients(&model, &[row], Ablation::default()).unwrap();
        model.adam_step(&g, 2e-3, Focus::Joint).unwrap();
    }
    let rows: Vec<Row> = (0..3)
        .map(|k| {
            let n = len - k * 97;
            let s = k * 50;
            Row { ids: ids[s..s + n].to_vec(), targets: ids[s + 1..s + n + 1].to_vec(), mask: vec![1.; n] }
        })
        .collect();
    let gpu = Arc::new(Gpu::open(true).unwrap());
    println!(
        "{} params on {}; rows {:?}",
        model.parameter_count(),
        gpu.name(),
        rows.iter().map(|r| r.ids.len()).collect::<Vec<_>>()
    );
    let t = std::time::Instant::now();
    let (cl, cg) = batch_loss_and_gradients(&model, &rows, Ablation::default()).unwrap();
    println!("CPU {:.1?}", t.elapsed());
    let g = GpuModel::upload(&gpu, &model).unwrap();
    let t = std::time::Instant::now();
    let (gl, gg) = g.loss_and_gradients(&model, &rows, Ablation::default()).unwrap();
    println!("GPU {:.1?}", t.elapsed());
    let (w, s) = worst(&gg, &cg);
    println!(
        "loss CPU {cl} GPU {gl}; gradients: worst difference {w:.3e} (largest {s:.3e}, ratio {:.2e})",
        w / s
    );
    let mut cpu = model.clone();
    let mut on_gpu = GpuModel::upload(&gpu, &model).unwrap();
    for _ in 0..2 {
        let (_, gr) = batch_loss_and_gradients(&cpu, &rows, Ablation::default()).unwrap();
        cpu.adam_step(&gr, 1e-3, Focus::Joint).unwrap();
        on_gpu.train_step(&cpu, &rows, Ablation::default(), 1e-3, Focus::Joint).unwrap();
    }
    let mut back = model.clone();
    on_gpu.download(&mut back).unwrap();
    let (w, s) = worst(&back.weights, &cpu.weights);
    let moved = worst(&cpu.weights, &model.weights).0;
    println!("weights after 2 steps: worst difference {w:.3e} (largest weight {s:.3e}; steps moved weights by up to {moved:.3e})");
}
