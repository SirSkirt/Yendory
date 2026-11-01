//! Where a GPU training step's time goes: CPU preparation, GPU time per kernel, GPU memory.
//! cargo run --release --features gpu --example gpu_profile -- [size or model file] [context] [rows] [tokens] [api]
use yendory_engine::gpu::{shared_with, Api, GpuModel};
use yendory_engine::learn::Row;
use yendory_engine::{Ablation, Bicameral, Config, Focus};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let size = a.get(1).map(String::as_str).unwrap_or("8m");
    let context: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n_rows: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let len: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(context);
    let api = match a.get(5).map(String::as_str) {
        Some("vulkan") => Api::Vulkan,
        Some("dx12") => Api::Dx12,
        _ => Api::Auto,
    };
    // A size preset, or a model file (.nnova2 / .nneuro) as it is.
    let model = if size.ends_with(".nnova2") || size.ends_with(".nneuro") {
        Bicameral::load(size).unwrap()
    } else {
        let c = yendory_engine::size_presets().into_iter().find(|p| p.id == size).unwrap().config;
        Bicameral::with_config(Config { max_tokens: context, ..c }, 3)
    };
    let text: String = (0..len * n_rows).map(|i| (b'a' + (i * 7 % 26) as u8) as char).collect();
    let ids = model.tokenizer.encode(&text).unwrap();
    let rows: Vec<Row> = (0..n_rows)
        .map(|r| {
            let s = &ids[r * len..(r + 1) * len];
            Row { ids: s[..len - 1].to_vec(), targets: s[1..].to_vec(), mask: vec![1.; len - 1] }
        })
        .collect();
    let gpu = shared_with(api).unwrap();
    println!("{} · {} params · {} rows × {} tokens", gpu.name(), model.parameter_count(), n_rows, len - 1);
    let mut g = GpuModel::upload(&gpu, &model).unwrap();
    g.train_step(&model, &rows, Ablation::default(), 1e-3, Focus::Joint).unwrap(); // warm-up
    let t = std::time::Instant::now();
    let steps = 2;
    for _ in 0..steps {
        g.train_step(&model, &rows, Ablation::default(), 1e-3, Focus::Joint).unwrap();
    }
    let per = t.elapsed().as_secs_f64() / steps as f64;
    let tokens = n_rows * (len - 1);
    println!("normal step: {:.1} ms, {:.0} tokens/s", per * 1e3, tokens as f64 / per);
    let p = g.profile_step(&model, &rows, Ablation::default(), 1e-3, Focus::Joint).unwrap();
    println!(
        "profiled: CPU prepare {:.1} ms, GPU kernels {:.1} ms over {} dispatches (timestamps: {}), memory {:.0} MB, uploads {:.1} MB",
        p.cpu_prepare_ms, p.gpu_ms, p.dispatches, p.timestamps, p.memory_bytes as f64 / 1048576., p.upload_bytes as f64 / 1048576.
    );
    for k in &p.kernels {
        println!("  {:<18} {:>5} × {:>9.2} ms  ({:.1}%)", k.kernel, k.count, k.ms, 100. * k.ms / p.gpu_ms);
    }
}
