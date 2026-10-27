//! GPU memory management (docs/GROWTH.md, section 3): micro-batches and the optimizer state in
//! system RAM take exactly the step the whole batch takes on the CPU; a step over the GPU
//! memory budget fails before anything runs and changes nothing; the trainer then retries with
//! less memory and reports what it did; the estimates used to plan steps cover what a step
//! really uses. Run with `--features gpu` (a software GPU is enough).
#![cfg(feature = "gpu")]
use std::sync::{Arc, Mutex};
use yendory_engine::gpu::{Gpu, GpuModel, OVER_BUDGET};
use yendory_engine::learn::memory::{self, MemoryMode, MemorySettings, StepShape};
use yendory_engine::learn::{batch_loss_and_gradients, set_device, Device, Row, Trainer};
use yendory_engine::model::neuro::presets;
use yendory_engine::{Ablation, Bicameral, Config, Dropout, Focus};

/// Tests that change the global device or memory settings run one at a time.
static GLOBALS: Mutex<()> = Mutex::new(());

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::open(true) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            println!("no GPU available ({e}); skipped");
            None
        }
    }
}

fn rows(model: &Bicameral, n: usize) -> Vec<Row> {
    let texts = [
        "U: Where is the key?\nN: In the box.\n",
        "Once upon a time, a small cat found a red hat under the old tree.",
        "U: hi\nN: Hello!\n",
        "The dog ran to the park and played with a ball all day long.",
        "U: What colour is the sky?\nN: Blue.\n",
    ];
    texts
        .iter()
        .cycle()
        .take(n)
        .enumerate()
        .map(|(i, t)| {
            let ids = model.tokenizer.encode(t).unwrap();
            let n = ids.len() - 1;
            let mask = (0..n).map(|j| if i == 0 && j < 12 { 0.15 } else { 1. }).collect();
            Row { ids: ids[..n].to_vec(), targets: ids[1..].to_vec(), mask }
        })
        .collect()
}

fn close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("{name}: largest difference {worst:.3e} (largest value {scale:.3e})");
    assert!(worst <= 2e-4 * scale + 1e-6, "{name}: {worst}");
}

fn models() -> Vec<(&'static str, Bicameral)> {
    let p = presets().into_iter().next().unwrap();
    let mut out = vec![
        ("bicameral", Bicameral::with_config(Config::default(), 3)),
        ("neuro", Bicameral::with_design(p.config, Some(p.design), 3)),
    ];
    for (_, m) in &mut out {
        for _ in 0..2 {
            m.train_text("U: Where is the key?\nN: In the box.\n", 3e-3, Focus::Joint).unwrap();
        }
    }
    out
}

/// One CPU step (the reference).
fn cpu_step(m: &Bicameral, rows: &[Row], abl: Ablation, focus: Focus) -> (f32, Bicameral) {
    let mut m = m.clone();
    let (loss, g) = batch_loss_and_gradients(&m, rows, abl).unwrap();
    m.adam_step(&g, 2e-3, focus).unwrap();
    (loss, m)
}

#[test]
fn micro_batches_and_optimizer_in_ram_take_the_same_step() {
    let Some(gpu) = gpu() else { return };
    let dropout = Ablation { dropout: Some(Dropout { rate: 0.1, seed: 5, row: 0 }), ..Ablation::default() };
    for (arch, start) in models() {
        let rows = rows(&start, 5);
        for (abl, focus) in [(Ablation::default(), Focus::Joint), (dropout, Focus::DenseAndInterconnect)] {
            let (cpu_loss, cpu) = cpu_step(&start, &rows, abl, focus);
            for (micro, in_ram) in [(5, false), (2, false), (1, false), (5, true), (2, true)] {
                let mut m = start.clone();
                let mut g = GpuModel::upload_with(&gpu, &m, in_ram).unwrap();
                let st = g.fit_step(&mut m, &rows, abl, 2e-3, focus, micro).unwrap();
                assert_eq!(g.step, start.step + 1);
                if in_ram {
                    // The CPU applied the step: the model is up to date without a download.
                    assert_eq!(m.step, start.step + 1);
                    assert_eq!(st.transfer_bytes, 2 * 4 * m.parameter_count() as u64);
                } else {
                    g.download(&mut m).unwrap();
                }
                let name = format!("{arch} {focus:?} micro {micro} optimizer in RAM {in_ram}");
                println!("{name}: loss {} vs CPU {cpu_loss}", st.loss);
                assert!((st.loss - cpu_loss).abs() <= 1e-4 * cpu_loss.abs().max(1.), "{name}");
                close(&format!("{name}: weights"), &m.weights, &cpu.weights);
                close(&format!("{name}: momentum"), &m.momentum, &cpu.momentum);
                close(&format!("{name}: variance"), &m.variance, &cpu.variance);
                // The GPU's weights match the model's for the next step.
                let mut back = m.clone();
                g.download(&mut back).unwrap();
                close(&format!("{name}: weights on the GPU"), &back.weights, &m.weights);
            }
        }
    }
}

#[test]
fn optimizer_in_ram_frees_gpu_memory_and_can_be_moved_mid_training() {
    let Some(gpu) = gpu() else { return };
    let (_, start) = models().remove(0);
    let rows = rows(&start, 3);
    let n = start.parameter_count() as u64;
    let on_gpu = GpuModel::upload(&gpu, &start).unwrap();
    let in_ram = GpuModel::upload_with(&gpu, &start, true).unwrap();
    println!(
        "model state on the GPU: {} bytes; with the optimizer in RAM: {}",
        on_gpu.fixed_bytes(),
        in_ram.fixed_bytes()
    );
    assert!(in_ram.fixed_bytes() + n * 4 <= on_gpu.fixed_bytes(), "at least 40% of the model state freed");
    // Two steps on the GPU, then the optimizer state moves to RAM and two more: the same as
    // four CPU steps.
    let mut cpu = start.clone();
    for _ in 0..4 {
        cpu = cpu_step(&cpu, &rows, Ablation::default(), Focus::Joint).1;
    }
    let mut m = start.clone();
    let mut g = GpuModel::upload(&gpu, &m).unwrap();
    for i in 0..4 {
        if i == 2 {
            g.move_optimizer_to_ram(&mut m).unwrap();
            assert!(g.optimizer_in_ram());
        }
        g.fit_step(&mut m, &rows, Ablation::default(), 2e-3, Focus::Joint, 3).unwrap();
    }
    assert_eq!(m.step, cpu.step);
    close("moved mid-training: weights", &m.weights, &cpu.weights);
    close("moved mid-training: momentum", &m.momentum, &cpu.momentum);
}

/// A step that needs more than the budget is refused before it runs: nothing changes, and the
/// same batch in smaller micro-batches fits.
#[test]
fn over_budget_steps_change_nothing_and_smaller_parts_fit() {
    let Some(gpu) = gpu() else { return };
    let (_, start) = models().remove(0);
    let rows = rows(&start, 4);
    // Working memory of one row and of the whole batch, measured.
    let measure = |micro: usize| {
        let mut m = start.clone();
        let mut g = GpuModel::upload(&gpu, &m).unwrap();
        g.fit_step(&mut m, &rows, Ablation::default(), 2e-3, Focus::Joint, micro).unwrap();
        assert_eq!(g.working_bytes(), g.memory_bytes() - g.fixed_bytes(), "accounting matches the pool");
        g.working_bytes()
    };
    let (one, all) = (measure(1), measure(4));
    println!("working memory: one row {one} bytes, four rows {all} bytes");
    assert!(all > one);
    let mut m = start.clone();
    let mut g = GpuModel::upload(&gpu, &m).unwrap();
    g.set_budget(Some(g.fixed_bytes() + (one + all) / 2));
    let err = g.fit_step(&mut m, &rows, Ablation::default(), 2e-3, Focus::Joint, 4).unwrap_err();
    println!("whole batch: {err}");
    assert!(err.contains(OVER_BUDGET), "{err}");
    assert_eq!(g.step, start.step, "no step was taken");
    let mut back = m.clone();
    g.download(&mut back).unwrap();
    assert_eq!(back.weights, start.weights, "the weights are unchanged");
    assert_eq!(back.momentum, start.momentum);
    // The same batch one row at a time fits and takes the CPU's step.
    let st = g.fit_step(&mut m, &rows, Ablation::default(), 2e-3, Focus::Joint, 1).unwrap();
    assert!(g.memory_bytes() <= g.budget().unwrap());
    g.download(&mut m).unwrap();
    let (cpu_loss, cpu) = cpu_step(&start, &rows, Ablation::default(), Focus::Joint);
    assert!((st.loss - cpu_loss).abs() <= 1e-4 * cpu_loss.max(1.));
    close("after the refused step", &m.weights, &cpu.weights);
}

/// The planning estimate covers what a step really uses, for both architectures.
#[test]
fn memory_estimates_cover_real_use() {
    let Some(gpu) = gpu() else { return };
    for (arch, start) in models() {
        for n in [1, 4] {
            let rows = rows(&start, n);
            let mut m = start.clone();
            let mut g = GpuModel::upload(&gpu, &m).unwrap();
            g.fit_step(&mut m, &rows, Ablation::default(), 2e-3, Focus::Joint, n).unwrap();
            let shape = StepShape {
                parameters: m.parameter_count(),
                activation_floats_per_token: Bicameral::activation_floats_for(&m.config, m.neuro.as_ref()),
                rows: n,
                tokens: rows.iter().map(|r| r.ids.len()).max().unwrap(),
                vocab: m.config.vocab,
            };
            let (used, estimate) = (g.working_bytes(), shape.working_bytes(n));
            println!(
                "{arch}, {n} rows of {} tokens: working memory {used} bytes, estimate {estimate}",
                shape.tokens
            );
            assert!(estimate >= used, "{arch}: the estimate must not be below the real use");
            assert!(estimate <= 6 * used, "{arch}: the estimate is far too high");
        }
    }
}

/// The trainer fits steps into the budget by itself (micro-batches, then the optimizer state
/// in RAM), trains the same model as the CPU, and reports what it did.
#[test]
fn trainer_adapts_to_the_budget_and_reports() {
    let _lock = GLOBALS.lock().unwrap();
    std::env::set_var("YENDORY_GPU_SOFTWARE", "1");
    if yendory_engine::learn::gpu_name().is_err() {
        println!("no GPU available; skipped");
        return;
    }
    let (_, start) = models().remove(0);
    let rows = rows(&start, 4);
    let n = start.parameter_count() as u64;
    let mut cpu = start.clone();
    for _ in 0..3 {
        cpu = cpu_step(&cpu, &rows, Ablation::default(), Focus::Joint).1;
    }
    let run = |settings: MemorySettings| {
        memory::set_memory_settings(settings);
        set_device(Device::Gpu(yendory_engine::gpu::Api::Auto));
        let mut m = start.clone();
        let mut t = Trainer::new(&m);
        assert!(t.on_gpu());
        for _ in 0..3 {
            t.step(&mut m, &rows, Ablation::default(), 2e-3, Focus::Joint).unwrap();
        }
        t.sync(&mut m).unwrap();
        set_device(Device::Cpu);
        let report = t.memory_report().unwrap();
        println!("{report:?}");
        assert_eq!(memory::memory_report().as_ref(), Some(&report), "the last report is published");
        assert!(yendory_engine::learn::take_device_notice().is_none(), "the GPU kept training");
        close(&report.strategy, &m.weights, &cpu.weights);
        report
    };
    let ram =
        MemorySettings { ram_limit_bytes: Some(1 << 40), ram_reserve_bytes: 0, ..MemorySettings::default() };
    // No budget: whole batches on the GPU.
    let r = run(MemorySettings { gpu_memory_bytes: None, ..ram.clone() });
    assert_eq!((r.micro_rows, r.optimizer_in_ram, r.refits), (None, false, 0));
    assert_eq!(r.steps, 3);
    // A budget the model state fits in with room for one or two rows: micro-batches.
    let fixed = GpuModel::fixed_bytes_for(n as usize, false);
    let room = |rows: u64| {
        let shape = StepShape {
            parameters: n as usize,
            activation_floats_per_token: Bicameral::activation_floats_for(&start.config, None),
            rows: 4,
            tokens: rows_max(&start),
            vocab: start.config.vocab,
        };
        fixed + shape.working_bytes(rows as usize)
    };
    let budget = |bytes: u64| Some(bytes * 100 / 90 + 100); // the 10% margin comes off
    let r = run(MemorySettings { gpu_memory_bytes: budget(room(2)), ..ram.clone() });
    assert!(r.micro_rows.is_some_and(|k| k < 4), "{r:?}");
    assert!(!r.optimizer_in_ram);
    // Too little for the optimizer state on the GPU: it goes to RAM, within the RAM limit.
    let tight = GpuModel::fixed_bytes_for(n as usize, true) + room(1) - fixed;
    let r = run(MemorySettings { gpu_memory_bytes: budget(tight), ..ram.clone() });
    assert!(r.optimizer_in_ram, "{r:?}");
    assert_eq!(r.ram_offload_bytes, 3 * 4 * n);
    assert!(r.transfer_bytes >= 3 * 2 * 4 * n);
    // The same budget, but no RAM allowed for offloading: not on the GPU (the CPU trains), and
    // the report says why.
    memory::set_memory_settings(MemorySettings {
        gpu_memory_bytes: budget(tight),
        ram_limit_bytes: Some(0),
        ..MemorySettings::default()
    });
    set_device(Device::Gpu(yendory_engine::gpu::Api::Auto));
    let mut m = start.clone();
    let mut t = Trainer::new(&m);
    for _ in 0..3 {
        t.step(&mut m, &rows, Ablation::default(), 2e-3, Focus::Joint).unwrap();
    }
    t.sync(&mut m).unwrap();
    set_device(Device::Cpu);
    let notice = yendory_engine::learn::take_device_notice().unwrap_or_default();
    println!("RAM refused: {notice}");
    assert!(!t.on_gpu() && notice.contains("RAM"), "{notice}");
    close("after moving to the CPU", &m.weights, &cpu.weights);
    // Manual: the chosen micro-batch size and placement are used as set.
    let r =
        run(MemorySettings { mode: MemoryMode::Manual, micro_rows: Some(3), optimizer_in_ram: true, ..ram });
    assert_eq!((r.micro_rows, r.optimizer_in_ram), (Some(3), true));
    memory::set_memory_settings(MemorySettings::default());
}

fn rows_max(m: &Bicameral) -> usize {
    rows(m, 4).iter().map(|r| r.ids.len()).max().unwrap()
}

/// A model grown past the GPU budget still trains on the GPU: the trainer moves the optimizer
/// state to system RAM and uses micro-batches, and trains the CPU's weights.
#[test]
fn a_grown_model_trains_past_the_budget_by_offloading() {
    let _lock = GLOBALS.lock().unwrap();
    std::env::set_var("YENDORY_GPU_SOFTWARE", "1");
    if yendory_engine::learn::gpu_name().is_err() {
        println!("no GPU available; skipped");
        return;
    }
    let (_, small) = models().remove(0);
    let (options, _) = small.growth_options();
    let start = small.grow(&options.iter().find(|o| o.id == "balanced").unwrap().growth, 3).unwrap();
    let rows = rows(&start, 4);
    let n = start.parameter_count();
    // The budget holds the small model with whole batches, but not the grown one.
    let shape = |m: &Bicameral| StepShape {
        parameters: m.parameter_count(),
        activation_floats_per_token: Bicameral::activation_floats_for(&m.config, m.neuro.as_ref()),
        rows: 4,
        tokens: rows_max(m),
        vocab: m.config.vocab,
    };
    let small_need =
        GpuModel::fixed_bytes_for(small.parameter_count(), false) + shape(&small).working_bytes(4);
    let grown_least = GpuModel::fixed_bytes_for(n, true) + shape(&start).working_bytes(1);
    let grown_whole = GpuModel::fixed_bytes_for(n, false) + shape(&start).working_bytes(4);
    println!(
        "small model {small_need} bytes; grown: {grown_least} at least, {grown_whole} with whole batches"
    );
    assert!(grown_whole > small_need);
    let budget = small_need.max(grown_least) + (1 << 20);
    assert!(budget < grown_whole, "the grown model's whole batches must not fit");
    memory::set_memory_settings(MemorySettings {
        gpu_memory_bytes: Some(budget * 100 / 90 + 100),
        ram_limit_bytes: Some(1 << 40),
        ram_reserve_bytes: 0,
        ..MemorySettings::default()
    });
    set_device(Device::Gpu(yendory_engine::gpu::Api::Auto));
    let mut m = start.clone();
    let mut t = Trainer::new(&m);
    for _ in 0..2 {
        t.step(&mut m, &rows, Ablation::default(), 2e-3, Focus::Joint).unwrap();
    }
    t.sync(&mut m).unwrap();
    set_device(Device::Cpu);
    memory::set_memory_settings(MemorySettings::default());
    let r = t.memory_report().unwrap();
    println!("{r:?}");
    assert!(t.on_gpu(), "still on the GPU");
    assert!(r.optimizer_in_ram || r.micro_rows.is_some(), "{r:?}");
    assert!(r.gpu_bytes <= budget, "within the budget: {} > {budget}", r.gpu_bytes);
    let mut cpu = start.clone();
    for _ in 0..2 {
        cpu = cpu_step(&cpu, &rows, Ablation::default(), Focus::Joint).1;
    }
    close("grown model under the budget", &m.weights, &cpu.weights);
}

/// Held-out checks of long rows fit the GPU's memory budget: they are scored a few rows at a
/// time, or on the CPU when not even one row fits, and give the CPU's loss either way.
#[test]
fn held_out_checks_fit_the_budget_or_run_on_the_cpu() {
    let _lock = GLOBALS.lock().unwrap();
    std::env::set_var("YENDORY_GPU_SOFTWARE", "1");
    if yendory_engine::learn::gpu_name().is_err() {
        println!("no GPU available; skipped");
        return;
    }
    let (_, start) = models().remove(0);
    let rows = rows(&start, 24);
    let cpu = yendory_engine::learn::batch_loss(&start, &rows, Ablation::default()).unwrap();
    let n = start.parameter_count();
    let shape = StepShape {
        parameters: n,
        activation_floats_per_token: Bicameral::activation_floats_for(&start.config, None),
        rows: rows.len(),
        tokens: rows_max(&start),
        vocab: start.config.vocab,
    };
    let ram =
        MemorySettings { ram_limit_bytes: Some(1 << 40), ram_reserve_bytes: 0, ..MemorySettings::default() };
    for (label, budget) in [
        ("no budget", None),
        (
            "room for two rows",
            Some((GpuModel::fixed_bytes_for(n, false) + shape.working_bytes(2)) * 100 / 90 + 100),
        ),
        ("room for no row", Some(GpuModel::fixed_bytes_for(n, true) * 100 / 90 + 100)),
    ] {
        memory::set_memory_settings(MemorySettings { gpu_memory_bytes: budget, ..ram.clone() });
        set_device(Device::Gpu(yendory_engine::gpu::Api::Auto));
        let t = Trainer::new(&start);
        let got = t.gpu_loss(&start, &rows, Ablation::default());
        set_device(Device::Cpu);
        let notice = yendory_engine::learn::take_device_notice();
        println!("{label}: {got:?} (CPU {cpu}); notice: {notice:?}");
        if !t.on_gpu() {
            continue;
        }
        let got = got.unwrap().unwrap();
        assert!((got - cpu).abs() <= 1e-4 * cpu.max(1.), "{label}: {got} vs {cpu}");
        // Each row's own loss, as the CPU gives it.
        let per_cpu = yendory_engine::learn::row_losses(&start, &rows, Ablation::default()).unwrap();
        set_device(Device::Gpu(yendory_engine::gpu::Api::Auto));
        let per_gpu =
            Trainer::new(&start).gpu_row_losses(&start, &rows, Ablation::default()).unwrap().unwrap();
        set_device(Device::Cpu);
        let _ = yendory_engine::learn::take_device_notice();
        assert_eq!(per_gpu.len(), rows.len());
        for (g, c) in per_gpu.iter().zip(&per_cpu) {
            assert!((g - c).abs() <= 1e-4 * c.max(1.), "{label}: row loss {g} vs {c}");
        }
    }
    memory::set_memory_settings(MemorySettings::default());
}

/// Conversation practice trains on the GPU, scores its practice pool there, and reports every
/// measure at every check.
#[test]
fn conversation_practice_runs_on_the_gpu() {
    use std::sync::atomic::AtomicBool;
    use yendory_engine::learn::conversation::{
        converse, parse_dialogue, ConversationConfig, ConversationSet,
    };
    let _lock = GLOBALS.lock().unwrap();
    std::env::set_var("YENDORY_GPU_SOFTWARE", "1");
    if yendory_engine::learn::gpu_name().is_err() {
        println!("no GPU available; skipped");
        return;
    }
    memory::set_memory_settings(MemorySettings::default());
    let model = Bicameral::with_config(
        Config {
            d: 16,
            heads: 2,
            ff: 24,
            blocks: 2,
            split: 1,
            r1: 12,
            r2: 10,
            max_tokens: 256,
            ..Config::default()
        },
        3,
    );
    let dialogues = (0..160)
        .map(|i| {
            parse_dialogue(&format!(
                "Ann: Do you like the color number {i}?\nBen: Yes, I like it.\nAnn: Why?\nBen: Because it is bright.\nAnn: Is number {i} your favourite?\nBen: Yes, number {i} is."
            ))
            .unwrap()
        })
        .collect();
    let sets = [ConversationSet { name: "x".into(), dialogues, nova: None, hold_out: true }];
    let cfg = ConversationConfig {
        tokens: 20_000,
        batch_size: 8,
        learning_rate: 3e-3,
        evaluations: 4,
        pool_size: 64,
        context_tests: 1,
        ..ConversationConfig::default()
    };
    set_device(Device::Gpu(yendory_engine::gpu::Api::Auto));
    let r = converse(&model, &sets, &[], &[], &cfg, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false));
    set_device(Device::Cpu);
    let r = r.unwrap();
    println!("checks: {}", r.checks.len());
    assert!(!r.checks.is_empty());
    assert!(r.checks.iter().all(|c| c.context.is_some() && c.stage_losses.len() == 3));
    assert!(r.best_loss <= r.baseline_loss);
}
