//! Writes a briefly trained model of each larger size preset, and of Nova-2's size with a
//! 100,000-word vocabulary, plus the Rust Engine's logits, loss and gradients for a fixed
//! prompt, so `reference/check_sizes.py` can recompute them with the numpy reference:
//! `cargo run --release --example size_compat -- <out dir>`.
use yendory_engine::tokenizer::BASE;
use yendory_engine::{size_presets, Ablation, Bicameral, Config, Focus};

fn main() {
    let out = std::path::PathBuf::from(std::env::args().nth(1).expect("output directory"));
    std::fs::create_dir_all(&out).unwrap();
    let mut cases: Vec<(String, Config)> = size_presets()
        .into_iter()
        .filter(|p| p.id != "nova2")
        .map(|p| (p.id.to_string(), p.config))
        .collect();
    // Nova-2 size with room for 100,000 words, 70,003 in use: token ids go beyond 16 bits.
    cases.push(("vocab100k".into(), Config { vocab: BASE + 100_000, ..Config::default() }));
    for (id, config) in cases {
        let mut m = Bicameral::with_config(config, 5);
        if id == "vocab100k" {
            let mut words: Vec<String> = (0..70_000).map(|i| format!("w{i:06}")).collect();
            words.extend(["key".into(), "box".into(), "zebra".into()]);
            m.tokenizer.set_words(words);
        }
        for _ in 0..3 {
            m.train_text("U: Where is the key?\nN: In the box.\n", 5e-3, Focus::Joint).unwrap();
        }
        let ids = m
            .tokenizer
            .encode("U: The key is in the box. Where is the zebra and w069999?\nN: In the box.\n")
            .unwrap();
        let (inputs, targets) = (&ids[..ids.len() - 1], &ids[1..]);
        let mask: Vec<f32> = (0..inputs.len()).map(|i| if i < 40 { 0.15 } else { 1. }).collect();
        let logits = m.logits(inputs, Ablation::default()).unwrap();
        let last = logits[logits.len() - m.config.vocab..].to_vec();
        let (loss, grads) = m.loss_and_gradients(inputs, targets, &mask, Ablation::default()).unwrap();
        let norm = grads.iter().map(|&g| (g as f64).powi(2)).sum::<f64>().sqrt();
        let step = (grads.len() / 3000).max(1);
        let sample: Vec<(usize, f32)> = (0..grads.len()).step_by(step).map(|i| (i, grads[i])).collect();
        std::fs::write(out.join(format!("{id}.nnova2")), m.to_bytes().unwrap()).unwrap();
        let record = serde_json::json!({
            "preset": id, "parameters": m.parameter_count(), "ids": inputs, "targets": targets,
            "mask": mask, "last_logits": last, "loss": loss, "grad_norm": norm, "grad_sample": sample,
        });
        std::fs::write(out.join(format!("{id}.json")), serde_json::to_vec(&record).unwrap()).unwrap();
        println!(
            "wrote {id} ({} parameters, largest token id {})",
            m.parameter_count(),
            ids.iter().max().unwrap()
        );
    }
}
