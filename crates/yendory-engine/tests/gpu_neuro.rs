//! Nova Neuro trains on the GPU with the values the CPU computes: the same loss, the same
//! gradient of every parameter (working memory's scan, associative memory's softmax over
//! patterns, the controller's gates, the bus and the write-back) under every ablation, the same
//! AdamW steps for every focus, and the same reading run. Run with `--features gpu`; a software
//! GPU (lavapipe) is enough.
#![cfg(feature = "gpu")]
use std::sync::Arc;
use yendory_engine::gpu::{Gpu, GpuModel};
use yendory_engine::learn::{batch_loss_and_gradients, Row};
use yendory_engine::model::neuro::presets;
use yendory_engine::{Ablation, Bicameral, Focus};

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::open(true) {
        Ok(g) => {
            println!("GPU: {}", g.name());
            Some(Arc::new(g))
        }
        Err(e) => {
            println!("no GPU available ({e}); skipped");
            None
        }
    }
}

fn neuro(seed: u64) -> Bicameral {
    let p = presets().into_iter().next().unwrap();
    Bicameral::with_design(p.config, Some(p.design), seed)
}

/// Rows of different lengths (padded on the GPU), so the scan restarts at every sequence and
/// padding is exercised.
fn rows(model: &Bicameral) -> Vec<Row> {
    let texts = [
        "U: Where is Tom?\nN: Tom is in the kitchen.\n",
        "Tom went to the garden. Then Tom moved to the office. Where is Tom? Tom is in the office.",
        "U: hi\nN: Hello!\n",
    ];
    texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let ids = model.tokenizer.encode(t).unwrap();
            let n = ids.len() - 1;
            let mask = (0..n).map(|j| if i == 0 && j < 10 { 0.2 } else { 1. }).collect();
            Row { ids: ids[..n].to_vec(), targets: ids[1..].to_vec(), mask }
        })
        .collect()
}

fn close(name: &str, gpu: &[f32], cpu: &[f32]) {
    assert_eq!(gpu.len(), cpu.len());
    let scale = cpu.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    let (i, worst) = gpu
        .iter()
        .zip(cpu)
        .map(|(a, b)| (a - b).abs())
        .enumerate()
        .fold((0, 0f32), |m, (i, d)| if d > m.1 { (i, d) } else { m });
    println!("{name}: largest difference {worst:.3e} at {i} (largest value {scale:.3e})");
    assert!(worst <= 1e-4 * scale + 1e-7, "{name}: {} vs {} at {i}", gpu[i], cpu[i]);
}

/// A few CPU steps first, so no gradient is trivially zero (the write-back starts small).
fn warmed(seed: u64) -> Bicameral {
    let mut model = neuro(seed);
    let rows = rows(&model);
    for _ in 0..3 {
        let (_, g) = batch_loss_and_gradients(&model, &rows, Ablation::default()).unwrap();
        model.adam_step(&g, 3e-3, Focus::Joint).unwrap();
    }
    model
}

#[test]
fn neuro_gradients_and_steps_match_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let model = warmed(3);
    let rows = rows(&model);
    let ablations = [
        ("full", Ablation::default()),
        ("no working memory", Ablation { no_memory: true, ..Ablation::default() }),
        ("no associative memory", Ablation { no_associative: true, ..Ablation::default() }),
        ("no controller", Ablation { no_controller: true, ..Ablation::default() }),
        ("one round", Ablation { one_round: true, ..Ablation::default() }),
        ("no write-back", Ablation { no_afferent: true, ..Ablation::default() }),
        ("no bus read", Ablation { no_efferent: true, ..Ablation::default() }),
    ];
    let g = GpuModel::upload(&gpu, &model).unwrap();
    for (name, abl) in ablations {
        let (cpu_loss, cpu_grads) = batch_loss_and_gradients(&model, &rows, abl).unwrap();
        let (gpu_loss, gpu_grads) = g.loss_and_gradients(&model, &rows, abl).unwrap();
        println!("{name}: loss GPU {gpu_loss} CPU {cpu_loss}");
        assert!((gpu_loss - cpu_loss).abs() < 1e-4 * cpu_loss.abs().max(1.), "{name}");
        close(&format!("gradients ({name})"), &gpu_grads, &cpu_grads);
        let gpu_only = g.mean_loss(&model, &rows, abl, 2).unwrap();
        assert!(
            (gpu_only - cpu_loss).abs() < 1e-4 * cpu_loss.abs().max(1.),
            "{name}: {gpu_only} vs {cpu_loss}"
        );
    }
    // Every Foundation focus (language core, modules, communication, all): the same AdamW steps.
    for focus in [Focus::Joint, Focus::Left, Focus::Right, Focus::Interconnect] {
        let mut cpu = model.clone();
        let mut on_gpu = GpuModel::upload(&gpu, &model).unwrap();
        for _ in 0..2 {
            let (l, grads) = batch_loss_and_gradients(&cpu, &rows, Ablation::default()).unwrap();
            cpu.adam_step(&grads, 2e-3, focus).unwrap();
            let lg = on_gpu.train_step(&cpu, &rows, Ablation::default(), 2e-3, focus).unwrap();
            assert!((l - lg).abs() < 1e-4 * l.abs().max(1.), "{l} vs {lg}");
        }
        let mut back = model.clone();
        on_gpu.download(&mut back).unwrap();
        assert_eq!(back.step, cpu.step);
        close(&format!("weights after 2 steps ({focus:?})"), &back.weights, &cpu.weights);
        close("momentum", &back.momentum, &cpu.momentum);
        if focus != Focus::Joint {
            for other in [Focus::Left, Focus::Right, Focus::Interconnect] {
                if other != focus {
                    assert_eq!(back.group_values(other), model.group_values(other), "{other:?} stays frozen");
                }
            }
        }
    }
}

/// Reading a corpus on the GPU gives the CPU's run: the same checkpoints accepted, losses equal
/// to float rounding, and the design kept.
#[test]
fn neuro_reads_on_the_gpu() {
    use std::sync::atomic::AtomicBool;
    use yendory_engine::learn::corpus::{pretrain, Corpus, PretrainConfig};
    use yendory_engine::learn::set_gpu_training;
    std::env::set_var("YENDORY_GPU_SOFTWARE", "1");
    if yendory_engine::learn::gpu_name().is_err() {
        println!("no GPU available; skipped");
        return;
    }
    let text: String = (0..300)
        .map(|i| {
            let (a, b) = (["kitchen", "garden", "office"][i % 3], ["park", "hallway"][i % 2]);
            format!("Story {i}: Tom went to the {a}. Then Tom moved to the {b}. Where is Tom? Tom is in the {b}.\n<|endoftext|>\n")
        })
        .collect();
    let corpus = Corpus::prepare(&text);
    let cfg = PretrainConfig {
        tokens: 12_000,
        batch_size: 8,
        window: 48,
        evaluations: 3,
        eval_windows: 8,
        ..Default::default()
    };
    let start = neuro(9);
    let run = |gpu: bool| {
        set_gpu_training(gpu);
        let r =
            pretrain(&start, &corpus, &cfg, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false)).unwrap();
        set_gpu_training(false);
        r
    };
    let (cpu, gpu) = (run(false), run(true));
    println!("Neuro reading: CPU best {} GPU best {}", cpu.best_loss, gpu.best_loss);
    assert!(gpu.model.neuro.is_some(), "the design is kept");
    assert_eq!(cpu.evaluations.len(), gpu.evaluations.len());
    for (a, b) in cpu.evaluations.iter().zip(&gpu.evaluations) {
        assert_eq!(a.accepted, b.accepted);
        assert!(
            (a.held_out_loss - b.held_out_loss).abs() < 2e-3,
            "{} vs {}",
            a.held_out_loss,
            b.held_out_loss
        );
    }
    close("weights after reading", &gpu.model.weights, &cpu.model.weights);
}
