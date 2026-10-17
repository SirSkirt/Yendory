//! GPU training computes what the CPU computes: the same loss, the same gradient of every
//! parameter and the same AdamW step, within float rounding. Run with `--features gpu`; a
//! software GPU (lavapipe) is enough, so CI checks this without a graphics card.
#![cfg(feature = "gpu")]
use std::sync::Arc;
use yendory_engine::gpu::{Gpu, GpuModel};
use yendory_engine::learn::{batch_loss_and_gradients, Row};
use yendory_engine::{Ablation, Bicameral, Config, Focus};

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

fn rows(model: &Bicameral) -> Vec<Row> {
    let texts = [
        "U: Where is the key?\nN: In the box.\n",
        "Once upon a time, a small cat found a red hat under the old tree.",
        "U: hi\nN: Hello!\n",
    ];
    texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let ids = model.tokenizer.encode(t).unwrap();
            let n = ids.len() - 1;
            // Different loss weights per row, as lessons have.
            let mask = (0..n).map(|j| if i == 0 && j < 12 { 0.15 } else { 1. }).collect();
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

#[test]
fn gpu_gradients_and_steps_match_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let mut model = Bicameral::with_config(Config::default(), 3);
    // Train a little on the CPU first so no gradient is trivially zero.
    for _ in 0..3 {
        model.train_text("U: Where is the key?\nN: In the box.\n", 3e-3, Focus::Joint).unwrap();
    }
    let rows = rows(&model);
    for abl in [Ablation::default(), Ablation { no_afferent: true, ..Ablation::default() }] {
        let (cpu_loss, cpu_grads) = batch_loss_and_gradients(&model, &rows, abl).unwrap();
        let g = GpuModel::upload(&gpu, &model).unwrap();
        let (gpu_loss, gpu_grads) = g.loss_and_gradients(&model, &rows, abl).unwrap();
        println!("loss: GPU {gpu_loss} CPU {cpu_loss}");
        assert!((gpu_loss - cpu_loss).abs() < 1e-4 * cpu_loss.abs().max(1.));
        close("gradients", &gpu_grads, &cpu_grads);
    }
    // Two AdamW steps with a focus: the same weights and optimizer state.
    for focus in [Focus::Joint, Focus::DenseAndInterconnect] {
        let mut cpu = model.clone();
        let mut on_gpu = GpuModel::upload(&gpu, &model).unwrap();
        for _ in 0..2 {
            let (l, grads) = batch_loss_and_gradients(&cpu, &rows, Ablation::default()).unwrap();
            cpu.adam_step(&grads, 2e-3, focus).unwrap();
            let lg = on_gpu.train_step(&model_view(&cpu), &rows, Ablation::default(), 2e-3, focus).unwrap();
            assert!((l - lg).abs() < 1e-4 * l.abs().max(1.), "{l} vs {lg}");
        }
        let mut back = model.clone();
        on_gpu.download(&mut back).unwrap();
        assert_eq!(back.step, cpu.step);
        close(&format!("weights after 2 steps ({focus:?})"), &back.weights, &cpu.weights);
        close("momentum", &back.momentum, &cpu.momentum);
        if focus == Focus::DenseAndInterconnect {
            assert_eq!(
                back.group_values(Focus::Left),
                model.group_values(Focus::Left),
                "the transformer stays frozen"
            );
        }
    }
}

/// The GPU only reads the model's shape and tokenizer; its weights live on the GPU.
fn model_view(m: &Bicameral) -> Bicameral {
    m.clone()
}

/// Reading a corpus and teaching a lesson with GPU training give the same results as on the
/// CPU (the same checkpoints are accepted; losses agree to float rounding).
#[test]
fn reading_and_lessons_train_on_the_gpu() {
    use std::sync::atomic::AtomicBool;
    use yendory_engine::learn::corpus::{pretrain, Corpus, PretrainConfig};
    use yendory_engine::learn::{parse_lesson, set_gpu_training};
    std::env::set_var("YENDORY_GPU_SOFTWARE", "1");
    if yendory_engine::learn::gpu_name().is_err() {
        println!("no GPU available; skipped");
        return;
    }
    let text: String = (0..300)
        .map(|i| format!("Story {i}: a little cat named Tom found a red ball in the park and was happy.\n<|endoftext|>\n"))
        .collect();
    let corpus = Corpus::prepare(&text);
    let cfg = PretrainConfig {
        tokens: 20_000,
        batch_size: 8,
        window: 48,
        evaluations: 4,
        eval_windows: 8,
        ..Default::default()
    };
    let start = Bicameral::with_config(Config::default(), 9);
    let run = |gpu: bool| {
        set_gpu_training(gpu);
        let r =
            pretrain(&start, &corpus, &cfg, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false)).unwrap();
        set_gpu_training(false);
        r
    };
    let (cpu, gpu) = (run(false), run(true));
    println!("reading: CPU best {} GPU best {}", cpu.best_loss, gpu.best_loss);
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

    let lesson = parse_lesson(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/compat/fixtures/lessons/sample_course_a.txt"
        ))
        .unwrap(),
        "a",
    )
    .unwrap();
    let scfg = yendory_engine::learn::session::SessionConfig {
        epochs: 1,
        foundation: true,
        eval_limit: Some(6),
        ..Default::default()
    };
    let teach = |gpu: bool| {
        set_gpu_training(gpu);
        let r = yendory_engine::learn::session::teach(
            &start,
            &lesson,
            &[],
            &scfg,
            &mut |_| {},
            &AtomicBool::new(false),
        )
        .unwrap();
        set_gpu_training(false);
        r
    };
    let (c, g) = (teach(false), teach(true));
    let pick = |r: &yendory_engine::learn::session::SessionResult| {
        r.epochs.iter().map(|e| e.selected.clone()).collect::<Vec<_>>()
    };
    assert_eq!(pick(&c), pick(&g), "the same candidates are selected");
    assert!((c.final_score.heldout_loss - g.final_score.heldout_loss).abs() < 2e-3);
}
