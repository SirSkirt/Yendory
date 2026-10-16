//! Writes a briefly trained model of each larger size preset, plus the Rust Engine's logits,
//! loss and gradients for a fixed prompt, so `reference/check_sizes.py` can recompute them with
//! the numpy reference: `cargo run --release --example size_compat -- <out dir>`.
use yendory_engine::{size_presets, Ablation, Bicameral, Focus};

fn main() {
    let out = std::path::PathBuf::from(std::env::args().nth(1).expect("output directory"));
    std::fs::create_dir_all(&out).unwrap();
    for p in size_presets().into_iter().filter(|p| p.id != "nova2") {
        let mut m = Bicameral::with_config(p.config, 5);
        for _ in 0..3 {
            m.train_text("U: Where is the key?\nN: In the box.\n", 5e-3, Focus::Joint).unwrap();
        }
        let ids =
            m.tokenizer.encode("U: The key is in the box. Where is the key?\nN: In the box.\n").unwrap();
        let (inputs, targets) = (&ids[..ids.len() - 1], &ids[1..]);
        let mask: Vec<f32> = (0..inputs.len()).map(|i| if i < 40 { 0.15 } else { 1. }).collect();
        let logits = m.logits(inputs, Ablation::default()).unwrap();
        let last = logits[logits.len() - m.config.vocab..].to_vec();
        let (loss, grads) = m.loss_and_gradients(inputs, targets, &mask, Ablation::default()).unwrap();
        let norm = grads.iter().map(|&g| (g as f64).powi(2)).sum::<f64>().sqrt();
        let step = (grads.len() / 3000).max(1);
        let sample: Vec<(usize, f32)> = (0..grads.len()).step_by(step).map(|i| (i, grads[i])).collect();
        std::fs::write(out.join(format!("{}.nnova2", p.id)), m.to_bytes().unwrap()).unwrap();
        let record = serde_json::json!({
            "preset": p.id, "parameters": m.parameter_count(), "ids": inputs, "targets": targets,
            "mask": mask, "last_logits": last, "loss": loss, "grad_norm": norm, "grad_sample": sample,
        });
        std::fs::write(out.join(format!("{}.json", p.id)), serde_json::to_vec(&record).unwrap()).unwrap();
        println!("wrote {} ({} parameters)", p.id, m.parameter_count());
    }
}
