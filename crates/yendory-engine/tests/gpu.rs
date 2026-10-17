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
        // The loss alone (held-out checks): forward passes only, on the CPU and on the GPU, in
        // parts of a few rows, the same mean as with gradients.
        let cpu_only = yendory_engine::learn::batch_loss(&model, &rows, abl).unwrap();
        assert!((cpu_only - cpu_loss).abs() < 1e-6 * cpu_loss.abs().max(1.), "{cpu_only} vs {cpu_loss}");
        let gpu_only = g.mean_loss(&model, &rows, abl, 2).unwrap();
        println!("loss only: GPU {gpu_only} CPU {cpu_only}");
        assert!((gpu_only - cpu_loss).abs() < 1e-4 * cpu_loss.abs().max(1.), "{gpu_only} vs {cpu_loss}");
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
    // Early stopping: checks run on the CPU while the GPU reads, yet reading stops at the
    // same check as when every check runs inline.
    let strict = PretrainConfig { evaluations: 12, patience: 1, min_loss_improvement: 0.05, ..cfg.clone() };
    let early = |gpu: bool| {
        set_gpu_training(gpu);
        let r = pretrain(&start, &corpus, &strict, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false))
            .unwrap();
        set_gpu_training(false);
        r
    };
    let (ce, ge) = (early(false), early(true));
    println!(
        "early stop: CPU {} checks ({}), GPU {} checks ({})",
        ce.evaluations.len(),
        ce.early_stop,
        ge.evaluations.len(),
        ge.early_stop
    );
    assert_eq!(ce.early_stop, ge.early_stop);
    assert_eq!(
        ce.evaluations.iter().map(|e| e.accepted).collect::<Vec<_>>(),
        ge.evaluations.iter().map(|e| e.accepted).collect::<Vec<_>>()
    );

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

/// Hybrid training (part of each batch on the GPU, the rest on the CPU, gradients added, the
/// CPU takes the AdamW step) trains the same model as the CPU: the same weights and optimizer
/// state after several steps, with a whole-brain and a Stage A (Dense + interconnect) focus.
#[test]
fn hybrid_steps_match_the_cpu() {
    use yendory_engine::learn::{set_device, Device, Trainer};
    std::env::set_var("YENDORY_GPU_SOFTWARE", "1");
    if yendory_engine::learn::gpu_name().is_err() {
        println!("no GPU available; skipped");
        return;
    }
    let mut start = Bicameral::with_config(Config::default(), 5);
    for _ in 0..2 {
        start.train_text("U: Where is the key?\nN: In the box.\n", 3e-3, Focus::Joint).unwrap();
    }
    let rows = rows(&start);
    for focus in [Focus::Joint, Focus::DenseAndInterconnect] {
        let train = |device: Device| {
            set_device(device);
            let mut m = start.clone();
            let mut t = Trainer::new(&m);
            let mut losses = Vec::new();
            for _ in 0..4 {
                losses.push(t.step(&mut m, &rows, Ablation::default(), 2e-3, focus).unwrap());
            }
            t.sync(&mut m).unwrap();
            let share = t.gpu_share();
            set_device(Device::Cpu);
            (m, losses, share)
        };
        let (cpu, cpu_losses, _) = train(Device::Cpu);
        let (hyb, hyb_losses, share) = train(Device::Hybrid(yendory_engine::gpu::Api::Auto));
        println!(
            "{focus:?}: losses CPU {cpu_losses:?} hybrid {hyb_losses:?}; GPU share after 4 steps {share:?}"
        );
        assert!(share.is_some(), "hybrid training was used");
        for (a, b) in cpu_losses.iter().zip(&hyb_losses) {
            assert!((a - b).abs() < 1e-4 * a.abs().max(1.), "{a} vs {b}");
        }
        assert_eq!(hyb.step, cpu.step);
        close(&format!("hybrid weights ({focus:?})"), &hyb.weights, &cpu.weights);
        close("hybrid momentum", &hyb.momentum, &cpu.momentum);
        if focus == Focus::DenseAndInterconnect {
            assert_eq!(hyb.group_values(Focus::Left), start.group_values(Focus::Left), "transformer frozen");
        }
    }
}
