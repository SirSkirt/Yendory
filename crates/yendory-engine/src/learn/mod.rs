//! Learning: lesson parsing, training rows, batched gradients, the learning-rate schedule and
//! held-out evaluation. Reference behaviour: `reference/lessons.py` and
//! `reference/train_lessons.py` (the pipeline that trained the step-19,000 checkpoint).
pub mod autogrow;
pub mod bench;
pub mod builtin;
pub mod compare;
pub mod conversation;
pub mod corpus;
pub mod course;
pub mod curriculum;
pub mod designer;
pub mod eval;
pub mod examples;
pub mod generate;
pub mod grading;
pub mod lessons;
pub mod listening;
pub mod memory;
pub mod practice;
pub mod program;
pub mod reflect;
pub mod review;
pub mod reward;
pub mod sampling;
pub mod session;

use crate::model::{Ablation, Bicameral, Focus};
use crate::tokenizer::TokenId;
use crate::tokenizer::Tokenizer;
pub use lessons::{grow_vocabulary, parse_lesson, Item, Kind, Segment};
use serde::{Deserialize, Serialize};

/// Small deterministic generator (SplitMix64) for batch sampling and windows. Training runs are
/// reproducible from their seed; the stream is not Python's, so runs are not step-identical to
/// the reference scripts.
#[derive(Clone, Debug)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `0..n` (n > 0).
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Warmup then cosine decay to `floor * peak` (reference `lr_schedule`). `step` counts from the
/// start of the session.
/// A learning rate as a plain decimal with three significant digits, the way it is entered:
/// 0.0001, 0.00025, 0.001 (never "1.0e-4").
pub fn rate_text(rate: f32) -> String {
    if rate == 0. || !rate.is_finite() {
        return format!("{rate}");
    }
    let r = rate as f64;
    let rounded: f64 = format!("{r:.2e}").parse().unwrap_or(r);
    let decimals = (2 - rounded.abs().log10().floor() as i32).clamp(0, 20) as usize;
    let s = format!("{rounded:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

pub fn lr_schedule(step: u64, peak: f32, warmup: u64, total: u64, floor: f32) -> f32 {
    if step < warmup {
        return peak * (step + 1) as f32 / warmup as f32;
    }
    let progress = ((step - warmup) as f64 / total.saturating_sub(warmup).max(1) as f64).min(1.0);
    (peak as f64
        * (floor as f64 + (1.0 - floor as f64) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())))
        as f32
}

/// One training sequence: inputs, next-token targets and the loss weight of each target.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub ids: Vec<TokenId>,
    pub targets: Vec<TokenId>,
    pub mask: Vec<f32>,
}

/// Encodes an item's segments (each segment separately, as the reference does) into one token
/// sequence with per-token weights. Sequences longer than `max_tokens + 1` are cut to a random
/// window. Returns `None` for sequences too short to train on.
pub fn encode_item(
    tok: &Tokenizer,
    segments: &[Segment],
    max_tokens: usize,
    rng: &mut Rng,
) -> Result<Option<Row>, String> {
    let (mut ids, mut w) = (Vec::new(), Vec::new());
    for seg in segments {
        let t = tok.encode(&seg.text)?;
        w.extend(std::iter::repeat(seg.weight).take(t.len()));
        ids.extend(t);
    }
    if ids.len() > max_tokens + 1 {
        let s = rng.below(ids.len() - max_tokens - 1);
        ids = ids[s..s + max_tokens + 1].to_vec();
        w = w[s..s + max_tokens + 1].to_vec();
    }
    if ids.len() < 2 {
        return Ok(None);
    }
    // The weight of a target is the weight of the token being predicted.
    let targets = ids[1..].to_vec();
    let mask = w[1..].to_vec();
    ids.pop();
    Ok(Some(Row { ids, targets, mask }))
}

/// One thread's unnormalised loss, total weight and gradient sum.
type Partial = (f64, f64, Vec<f32>);

/// Mean loss and gradient over a batch, exactly as one padded batch in the reference: every
/// weighted target counts once and the sum is divided by the batch's total weight. Rows are
/// independent (causal attention), so they are computed in parallel threads.
pub fn batch_loss_and_gradients(
    model: &Bicameral,
    rows: &[Row],
    abl: Ablation,
) -> Result<(f32, Vec<f32>), String> {
    let (loss, weight, mut grads) = batch_sums(model, rows, abl)?;
    let denom = weight.max(1.);
    for g in &mut grads {
        *g = (*g as f64 / denom) as f32;
    }
    Ok(((loss / denom) as f32, grads))
}

/// The mean loss of a batch exactly as [`batch_loss_and_gradients`] reports it, without
/// computing gradients (forward passes only, rows in parallel threads).
pub fn batch_loss(model: &Bicameral, rows: &[Row], abl: Ablation) -> Result<f32, String> {
    if rows.is_empty() {
        return Ok(0.);
    }
    let threads = cpu_threads().min(rows.len());
    let chunk = rows.len().div_ceil(threads);
    let parts: Vec<Result<(f64, f64), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let (mut loss, mut weight) = (0f64, 0f64);
                    for row in part {
                        let sum: f32 = row.mask.iter().sum();
                        let l = model.loss(&row.ids, &row.targets, &row.mask, abl)?;
                        loss += l as f64 * sum.max(1.) as f64;
                        weight += sum as f64;
                    }
                    Ok((loss, weight))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("scoring thread panicked")).collect()
    });
    let (mut loss, mut weight) = (0f64, 0f64);
    for p in parts {
        let (l, w) = p?;
        loss += l;
        weight += w;
    }
    Ok((loss / weight.max(1.)) as f32)
}

/// Each row's own mean loss on the CPU (rows in parallel threads).
pub fn row_losses(model: &Bicameral, rows: &[Row], abl: Ablation) -> Result<Vec<f32>, String> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let threads = cpu_threads().max(1).min(rows.len());
    let chunk = rows.len().div_ceil(threads);
    let parts: Vec<Result<Vec<f32>, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter().map(|r| model.loss(&r.ids, &r.targets, &r.mask, abl)).collect()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("A scoring thread failed".into())))
            .collect()
    });
    let mut out = Vec::with_capacity(rows.len());
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}

/// Like [`batch_loss_and_gradients`] for a part of a batch: the loss and gradient sums are
/// divided by `denom`, the whole batch's total weight, so the parts of a batch add up to the
/// batch's mean (hybrid CPU + GPU training).
pub fn batch_part_loss_and_gradients(
    model: &Bicameral,
    rows: &[Row],
    abl: Ablation,
    denom: f64,
) -> Result<(f32, Vec<f32>), String> {
    let (loss, _, mut grads) = batch_sums(model, rows, abl)?;
    for g in &mut grads {
        *g = (*g as f64 / denom) as f32;
    }
    Ok(((loss / denom) as f32, grads))
}

/// The total weight of a batch's targets (the mean loss's denominator), at least 1.
pub fn batch_weight(rows: &[Row]) -> f64 {
    rows.iter().map(|r| r.mask.iter().map(|&m| m as f64).sum::<f64>()).sum::<f64>().max(1.)
}

/// Unnormalised loss, total weight and gradient sum of a batch.
fn batch_sums(model: &Bicameral, rows: &[Row], abl: Ablation) -> Result<Partial, String> {
    if rows.is_empty() {
        return Err("Empty training batch".into());
    }
    let threads = cpu_threads().min(rows.len());
    let chunk = rows.len().div_ceil(threads);
    let parts: Vec<Result<Partial, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(chunk)
            .enumerate()
            .map(|(ci, part)| {
                scope.spawn(move || {
                    let mut grads = vec![0f32; model.parameter_count()];
                    let (mut loss, mut weight) = (0f64, 0f64);
                    for (j, row) in part.iter().enumerate() {
                        let sum: f32 = row.mask.iter().sum();
                        // Undo the per-row normalisation (the tape divides by max(sum, 1)).
                        let scale = sum.max(1.);
                        // Dropout masks follow the row's place in the batch.
                        let l = model.accumulate_loss_and_gradients(
                            &row.ids,
                            &row.targets,
                            &row.mask,
                            abl.at_row(ci * chunk + j),
                            &mut grads,
                            scale,
                        )?;
                        loss += l as f64 * scale as f64;
                        weight += sum as f64;
                    }
                    Ok((loss, weight, grads))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("training thread panicked")).collect()
    });
    let mut grads = vec![0f32; model.parameter_count()];
    let (mut loss, mut weight) = (0f64, 0f64);
    for part in parts {
        let (l, w, g) = part?;
        loss += l;
        weight += w;
        for (a, b) in grads.iter_mut().zip(&g) {
            *a += b;
        }
    }
    Ok((loss, weight, grads))
}

/// Where training steps run. The GPU is used only when the Engine is built with the `gpu`
/// feature, the user chose it and a GPU could be opened; otherwise the CPU trains.
static CPU_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Limits the CPU threads training, scoring and checks use (0: all of the processor's).
pub fn set_cpu_threads(n: usize) {
    CPU_THREADS.store(n, std::sync::atomic::Ordering::Release);
}

static GPU_CHECKS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// GPU checks (on by default), for Reading, the Classroom, courses and Conversation in GPU
/// training: the Classroom's losses are computed on the GPU, the GPU confirms the answers
/// greedy writing certainly gives exactly (`eval::confirm`) so the CPU writes out only the
/// rest, and Reading keeps training while the CPU writes a check's samples. The results are the
/// same. Off: the CPU computes the Classroom's whole score and writes every answer, and Reading
/// waits for each check, as before.
pub fn set_gpu_checks(on: bool) {
    GPU_CHECKS.store(on, std::sync::atomic::Ordering::Release);
}

/// Whether GPU checks are on (see [`set_gpu_checks`]).
pub fn gpu_checks() -> bool {
    GPU_CHECKS.load(std::sync::atomic::Ordering::Acquire)
}

/// The CPU threads to use: the chosen limit, or every thread the processor has.
pub fn cpu_threads() -> usize {
    let all = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    match CPU_THREADS.load(std::sync::atomic::Ordering::Acquire) {
        0 => all,
        n => n.min(all),
    }
}

/// Where training steps run. Every device trains the same network and the same `.nnova2`
/// model: weights, optimizer state and step are the model's, wherever they are computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "kind", content = "api")]
pub enum Device {
    Cpu,
    /// The GPU through a graphics API (DirectX 12, Vulkan, Metal or automatic).
    #[cfg(feature = "gpu")]
    Gpu(crate::gpu::Api),
    /// Experimental: each batch is split between the GPU and the CPU (in proportion to their
    /// measured speed); their gradients are added and the CPU takes the AdamW step.
    #[cfg(feature = "gpu")]
    Hybrid(crate::gpu::Api),
}

static DEVICE: std::sync::Mutex<Device> = std::sync::Mutex::new(Device::Cpu);
static NOTICE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

pub fn set_device(d: Device) {
    *DEVICE.lock().unwrap() = d;
}
pub fn device() -> Device {
    *DEVICE.lock().unwrap()
}

/// Train on the GPU (automatic API) or on the CPU.
pub fn set_gpu_training(on: bool) {
    #[cfg(feature = "gpu")]
    set_device(if on { Device::Gpu(crate::gpu::Api::Auto) } else { Device::Cpu });
    #[cfg(not(feature = "gpu"))]
    let _ = on;
}

pub fn gpu_training() -> bool {
    device() != Device::Cpu
}

/// The message of a caught panic.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
fn panic_text(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "unknown error".into())
}

/// What happened to the chosen device during training (e.g. the GPU failed and training went
/// on on the CPU), once; `None` when nothing happened.
pub fn take_device_notice() -> Option<String> {
    NOTICE.lock().unwrap().take()
}
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
fn notice(text: String) {
    *NOTICE.lock().unwrap() = Some(text);
}

/// The GPU training would use ("NVIDIA GeForce RTX 2070 SUPER (Dx12)"), or why there is none.
pub fn gpu_name() -> Result<String, String> {
    #[cfg(feature = "gpu")]
    {
        crate::gpu::shared().map(|g| g.name())
    }
    #[cfg(not(feature = "gpu"))]
    {
        Err("this build has no GPU support".into())
    }
}

/// Splits batches between the GPU and the CPU in proportion to their measured speed.
#[cfg(feature = "gpu")]
struct Hybrid {
    gpu: crate::gpu::GpuModel,
    /// Share of a batch's tokens given to the GPU (adjusted after every step).
    gpu_share: f64,
}

/// Runs training steps on the chosen device. On the GPU the model's weights live on the card
/// between steps; [`Trainer::sync`] copies them back before the CPU reads them (checks,
/// scoring, saving). On the CPU each step is `batch_loss_and_gradients` + `adam_step`. In
/// hybrid training the model stays on the CPU and the GPU computes part of each gradient.
pub struct Trainer {
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::GpuModel>,
    #[cfg(feature = "gpu")]
    hybrid: Option<Hybrid>,
    dirty: bool,
    /// GPU training: each step's memory plan, adapted when a step does not fit.
    #[cfg(feature = "gpu")]
    fitter: memory::Fitter,
    #[cfg(feature = "gpu")]
    report: memory::MemoryReport,
}

impl Trainer {
    /// A trainer for `model` on the chosen device. When the GPU cannot be used (none, or the
    /// model does not fit) the CPU trains, and [`take_device_notice`] says why.
    pub fn new(model: &Bicameral) -> Trainer {
        #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
        let mut t = Trainer {
            #[cfg(feature = "gpu")]
            gpu: None,
            #[cfg(feature = "gpu")]
            hybrid: None,
            dirty: false,
            #[cfg(feature = "gpu")]
            fitter: memory::Fitter::default(),
            #[cfg(feature = "gpu")]
            report: memory::MemoryReport::default(),
        };
        #[cfg(feature = "gpu")]
        match device() {
            Device::Cpu => {}
            Device::Gpu(api) | Device::Hybrid(api) => {
                let s = memory::memory_settings();
                let hybrid = matches!(device(), Device::Hybrid(_));
                let n = model.parameter_count();
                let ram_ok = s.ram_allowance() >= 3 * n as u64 * 4;
                // The optimizer state starts in system RAM in hybrid training (the CPU applies
                // AdamW there anyway), when chosen (Manual), or when the model's own state
                // would take most of the GPU budget (Automatic).
                let in_ram = hybrid
                    || (s.mode == memory::MemoryMode::Manual && s.optimizer_in_ram)
                    || (s.mode == memory::MemoryMode::Automatic
                        && ram_ok
                        && s.gpu_budget()
                            .is_some_and(|b| crate::gpu::GpuModel::fixed_bytes_for(n, false) > b / 10 * 8));
                // A GPU failure while copying the model there (e.g. out of GPU memory) means
                // training on the CPU, with a notice, never a stop.
                let opened = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::gpu::shared_with(api)
                        .and_then(|g| crate::gpu::GpuModel::upload_with(&g, model, in_ram))
                }))
                .unwrap_or_else(|p| Err(panic_text(p)));
                match opened {
                    Ok(mut m) => {
                        m.set_budget(s.gpu_budget());
                        t.report = memory::MemoryReport {
                            device: m.gpu().name(),
                            optimizer_in_ram: m.optimizer_in_ram(),
                            gpu_bytes: m.memory_bytes(),
                            gpu_budget_bytes: m.budget(),
                            ..Default::default()
                        };
                        if hybrid {
                            t.hybrid = Some(Hybrid { gpu: m, gpu_share: 0.5 });
                        } else {
                            t.gpu = Some(m);
                        }
                    }
                    Err(e) => notice(format!("Training on the CPU: the GPU cannot be used ({e})")),
                }
            }
        }
        let _ = model;
        t
    }

    /// Training runs on the GPU alone (so the CPU is free for checks meanwhile).
    pub fn on_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// One step on a batch; returns its mean loss. On the GPU, `model`'s weights are not
    /// updated until [`Trainer::sync`] (its step counter is). If the GPU fails, the model is
    /// brought back from the GPU when possible and training continues on the CPU; if the GPU's
    /// state cannot be read back, the step fails and the model keeps its last synced state
    /// (nothing saved is touched).
    pub fn step(
        &mut self,
        model: &mut Bicameral,
        rows: &[Row],
        abl: Ablation,
        lr: f32,
        focus: Focus,
    ) -> Result<f32, String> {
        #[cfg(feature = "gpu")]
        if let Some(g) = self.gpu.as_mut() {
            let s = memory::memory_settings();
            let shape = memory::StepShape {
                parameters: model.parameter_count(),
                activation_floats_per_token: Bicameral::activation_floats_for(
                    &model.config,
                    model.neuro.as_ref(),
                ),
                rows: rows.len(),
                tokens: rows.iter().map(|r| r.ids.len()).max().unwrap_or(1),
                vocab: model.config.vocab,
            };
            let ram_ok = g.optimizer_in_ram() || s.ram_allowance() >= shape.ram_offload_bytes();
            let fixed = |ram| crate::gpu::GpuModel::fixed_bytes_for(shape.parameters, ram);
            let started = std::time::Instant::now();
            let mut failure = None;
            // Each retry uses less GPU memory than the last, and there are few ways to do so.
            for _ in 0..40 {
                let plan = match self.fitter.plan(
                    &s,
                    &shape,
                    fixed(false),
                    fixed(true),
                    ram_ok,
                    g.optimizer_in_ram(),
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                };
                if plan.optimizer_in_ram && !g.optimizer_in_ram() {
                    if let Err(e) = g.move_optimizer_to_ram(model) {
                        failure = Some(e);
                        break;
                    }
                    self.dirty = false;
                }
                let before = (g.step, model.step);
                let step = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    g.fit_step(model, rows, abl, lr, focus, plan.micro_rows)
                }))
                .unwrap_or_else(|p| Err(format!("GPU step failed: {}", panic_text(p))));
                match step {
                    Ok(st) => {
                        model.step = g.step;
                        self.dirty = !g.optimizer_in_ram();
                        let r = &mut self.report;
                        r.strategy = plan.describe(rows.len());
                        r.micro_rows = (plan.micro_rows < rows.len()).then_some(plan.micro_rows);
                        r.optimizer_in_ram = g.optimizer_in_ram();
                        r.gpu_bytes = g.memory_bytes();
                        r.gpu_budget_bytes = g.budget();
                        if r.steps % 10 == 0 {
                            r.gpu_allocated_bytes = g.gpu().allocated_bytes();
                        }
                        r.ram_offload_bytes = if r.optimizer_in_ram { shape.ram_offload_bytes() } else { 0 };
                        r.transfer_bytes += st.transfer_bytes;
                        r.transfer_seconds += st.transfer_seconds;
                        r.steps += 1;
                        r.tokens += rows.iter().map(|r| r.ids.len() as u64).sum::<u64>();
                        r.step_seconds += started.elapsed().as_secs_f64();
                        memory::publish_report(r);
                        return Ok(st.loss);
                    }
                    Err(e) if memory::is_out_of_memory(&e) && (g.step, model.step) == before => {
                        match self.fitter.after_out_of_memory(&s, &plan, &shape, ram_ok) {
                            Some(change) => {
                                self.report.refits += 1;
                                self.report.note = Some(format!(
                                    "A step did not fit in GPU memory ({}); training goes on with {change}",
                                    plan.describe(rows.len())
                                ));
                            }
                            None => {
                                failure = Some(format!(
                                    "{e}. It does not fit in GPU memory even as planned ({}){}",
                                    plan.describe(rows.len()),
                                    match s.mode {
                                        memory::MemoryMode::Manual => "; choose fewer rows per micro-batch, the optimizer state in RAM, or Automatic memory",
                                        memory::MemoryMode::GpuOnly => "; Automatic memory could move the optimizer state to system RAM",
                                        memory::MemoryMode::Automatic if !ram_ok => "; the system RAM limits do not allow moving more to RAM",
                                        _ => "",
                                    }
                                ));
                                break;
                            }
                        }
                    }
                    Err(e) if g.optimizer_in_ram() && model.step != before.1 => {
                        // The CPU applied the step (the model is up to date); the GPU stops.
                        self.gpu = None;
                        self.dirty = false;
                        notice(format!(
                            "The GPU failed ({e}) after the step was applied; training continues on the CPU"
                        ));
                        return Err(format!("The GPU failed ({e}) while the step was being applied"));
                    }
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                }
            }
            if let Some(e) = failure {
                let g = self.gpu.take().unwrap();
                // With the optimizer state in system RAM the model is already up to date.
                if self.dirty {
                    let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.download(model)))
                        .unwrap_or_else(|p| Err(panic_text(p)));
                    if let Err(read) = read {
                        return Err(format!(
                            "The GPU failed ({e}) and its training state could not be read back ({read}); \
                             the steps since the last check are lost, saved checkpoints are unchanged"
                        ));
                    }
                }
                self.dirty = false;
                self.report.note = Some(format!("Training moved to the CPU: {e}"));
                memory::publish_report(&self.report);
                notice(if memory::is_out_of_memory(&e) {
                    format!("Training continued on the CPU: {e}")
                } else {
                    format!("The GPU failed ({e}); training continued on the CPU")
                });
            }
        }
        #[cfg(feature = "gpu")]
        if let Some(h) = self.hybrid.as_mut() {
            let before = model.step;
            let step = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                hybrid_step(h, model, rows, abl, lr, focus)
            }))
            .unwrap_or_else(|p| Err(format!("hybrid step failed: {}", panic_text(p))));
            match step {
                Ok(loss) => {
                    // The step is applied on the CPU; the GPU gets the new weights for the next.
                    let upload =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.gpu.set_weights(model)))
                            .unwrap_or_else(|p| Err(panic_text(p)));
                    if let Err(e) = upload {
                        // The model on the CPU is up to date: the step counts, the GPU stops.
                        self.hybrid = None;
                        notice(format!(
                            "The GPU failed in hybrid training ({e}); training continued on the CPU"
                        ));
                    }
                    return Ok(loss);
                }
                Err(e) if model.step != before => {
                    return Err(format!(
                        "The GPU failed in hybrid training ({e}) while the step was being applied"
                    ));
                }
                Err(e) => {
                    // Nothing was applied and the model is on the CPU: nothing is lost.
                    self.hybrid = None;
                    notice(format!("The GPU failed in hybrid training ({e}); training continued on the CPU"));
                }
            }
        }
        let (loss, grads) = batch_loss_and_gradients(model, rows, abl)?;
        model.adam_step(&grads, lr, focus)?;
        Ok(loss)
    }

    /// Brings `model` up to date with the GPU (nothing to do on the CPU or in hybrid training).
    pub fn sync(&mut self, model: &mut Bicameral) -> Result<(), String> {
        #[cfg(feature = "gpu")]
        if let (Some(g), true) = (self.gpu.as_ref(), self.dirty) {
            g.download(model)?;
        }
        self.dirty = false;
        let _ = model;
        Ok(())
    }

    /// In GPU training: the mean loss of rows computed on the GPU with the weights there (the
    /// same value as [`batch_loss`] on the synced model, much faster). `None` otherwise.
    ///
    /// The rows are scored in parts the GPU's memory budget holds (at most 8 rows at a time):
    /// a check of long rows (whole conversations) needs much more working memory than a training
    /// window. A part that still does not fit is halved; when not even one row fits, the check
    /// runs on the CPU (with a notice) instead of failing.
    pub fn gpu_loss(&self, model: &Bicameral, rows: &[Row], abl: Ablation) -> Option<Result<f32, String>> {
        #[cfg(feature = "gpu")]
        {
            let g = self.gpu.as_ref()?;
            Some(self.in_parts(
                model,
                rows,
                "held-out check",
                |part| g.mean_loss(model, rows, abl, part),
                || batch_loss(model, rows, abl),
            ))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (model, rows, abl);
            None
        }
    }

    /// Each row's own mean loss on the GPU (`None` when training is not on the GPU), in parts
    /// that fit the memory budget like [`Trainer::gpu_loss`], or on the CPU when not even one
    /// row fits.
    pub fn gpu_row_losses(
        &self,
        model: &Bicameral,
        rows: &[Row],
        abl: Ablation,
    ) -> Option<Result<Vec<f32>, String>> {
        #[cfg(feature = "gpu")]
        {
            let g = self.gpu.as_ref()?;
            Some(self.in_parts(
                model,
                rows,
                "scoring",
                |part| g.row_losses(model, rows, abl, part),
                || row_losses(model, rows, abl),
            ))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (model, rows, abl);
            None
        }
    }

    /// The greedy check on the GPU (`None` when training is not on the GPU): for each row,
    /// whether greedy writing certainly gives its weighted targets (see
    /// [`crate::gpu::GpuModel::greedy_hits`]). Rows that do not fit in the GPU's memory are
    /// reported as not confirmed (the CPU then writes those answers).
    pub fn gpu_greedy(&self, model: &Bicameral, rows: &[Row]) -> Option<Result<Vec<bool>, String>> {
        #[cfg(feature = "gpu")]
        {
            let g = self.gpu.as_ref()?;
            Some(self.in_parts(
                model,
                rows,
                "answer check",
                |part| g.greedy_hits(model, rows, eval::GREEDY_MARGIN, part),
                || Ok(vec![false; rows.len()]),
            ))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (model, rows);
            None
        }
    }

    /// Runs `gpu(part)` with as many rows at a time as the GPU's free budget holds (at most 8),
    /// halving the part when it does not fit; when not even one row fits, `cpu()` runs instead
    /// (with a notice) and training stays on the GPU.
    #[cfg(feature = "gpu")]
    fn in_parts<T>(
        &self,
        model: &Bicameral,
        rows: &[Row],
        what: &str,
        gpu: impl Fn(usize) -> Result<T, String>,
        cpu: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let g = self.gpu.as_ref().expect("on the GPU");
        let shape = memory::StepShape {
            parameters: model.parameter_count(),
            activation_floats_per_token: Bicameral::activation_floats_for(
                &model.config,
                model.neuro.as_ref(),
            ),
            rows: rows.len(),
            tokens: rows.iter().map(|r| r.ids.len()).max().unwrap_or(1),
            vocab: model.config.vocab,
        };
        let mut part = match g.budget() {
            Some(b) => {
                let free = b.saturating_sub(g.memory_bytes());
                ((free / shape.working_bytes(1).max(1)) as usize).clamp(1, 8)
            }
            None => 8,
        };
        loop {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu(part)))
                .unwrap_or_else(|p| Err(format!("GPU {what} failed: {}", panic_text(p))));
            match r {
                Ok(v) => return Ok(v),
                Err(_) if part > 1 => {
                    g.clear_pool();
                    part /= 2;
                }
                Err(e) => {
                    g.clear_pool();
                    notice(format!(
                        "A {what} of {} rows of up to {} tokens did not fit in the GPU's memory ({e}); it ran on the CPU, and training stays on the GPU.",
                        rows.len(),
                        shape.tokens
                    ));
                    return cpu();
                }
            }
        }
    }

    /// GPU memory training uses (GPU and hybrid training).
    pub fn gpu_memory_bytes(&self) -> Option<u64> {
        #[cfg(feature = "gpu")]
        {
            self.gpu.as_ref().or(self.hybrid.as_ref().map(|h| &h.gpu)).map(|g| g.memory_bytes())
        }
        #[cfg(not(feature = "gpu"))]
        {
            None
        }
    }

    /// GPU training: how memory is used (strategy, GPU and RAM bytes, transfers), measured.
    pub fn memory_report(&self) -> Option<memory::MemoryReport> {
        #[cfg(feature = "gpu")]
        {
            self.gpu.is_some().then(|| self.report.clone())
        }
        #[cfg(not(feature = "gpu"))]
        {
            None
        }
    }

    /// Hybrid training: the share of each batch the GPU takes (measured speeds decide it).
    pub fn gpu_share(&self) -> Option<f64> {
        #[cfg(feature = "gpu")]
        {
            self.hybrid.as_ref().map(|h| h.gpu_share)
        }
        #[cfg(not(feature = "gpu"))]
        {
            None
        }
    }
}

/// One hybrid step: the GPU computes the gradient of the first rows, the CPU (at the same
/// time) that of the others, both divided by the whole batch's weight; their sum is the
/// batch's gradient, and the CPU takes the AdamW step (exactly as CPU training); the caller
/// then sends the new weights to the GPU. The split follows the measured speeds.
#[cfg(feature = "gpu")]
fn hybrid_step(
    h: &mut Hybrid,
    model: &mut Bicameral,
    rows: &[Row],
    abl: Ablation,
    lr: f32,
    focus: Focus,
) -> Result<f32, String> {
    let denom = batch_weight(rows);
    let tokens: Vec<usize> = rows.iter().map(|r| r.ids.len()).collect();
    let total: usize = tokens.iter().sum();
    // The GPU takes rows from the front until it has about its share of the tokens.
    let mut split = 0;
    let mut acc = 0;
    while split < rows.len() && ((acc + tokens[split] / 2) as f64) < h.gpu_share * total as f64 {
        acc += tokens[split];
        split += 1;
    }
    let (gpu_rows, cpu_rows) = rows.split_at(split);
    let model_ref: &Bicameral = model;
    let gpu = &h.gpu;
    let ((gpu_part, gpu_secs), (cpu_part, cpu_secs)) = std::thread::scope(|s| {
        let gpu_job = s.spawn(move || {
            let t = std::time::Instant::now();
            let r = if gpu_rows.is_empty() {
                Ok(None)
            } else {
                gpu.part_loss_and_gradients(model_ref, gpu_rows, abl, denom as f32).map(Some)
            };
            (r, t.elapsed().as_secs_f64())
        });
        let t = std::time::Instant::now();
        let cpu = if cpu_rows.is_empty() {
            Ok(None)
        } else {
            // The CPU's rows come after the GPU's in the batch (dropout masks follow the row).
            batch_part_loss_and_gradients(model_ref, cpu_rows, abl.at_row(split), denom).map(Some)
        };
        let cpu_secs = t.elapsed().as_secs_f64();
        (
            gpu_job.join().unwrap_or_else(|p| (Err(format!("GPU thread failed: {}", panic_text(p))), 0.)),
            (cpu, cpu_secs),
        )
    });
    let (gpu_part, cpu_part) = (gpu_part?, cpu_part?);
    let mut loss = 0f32;
    let mut grads = vec![0f32; model.parameter_count()];
    for (l, g) in [gpu_part, cpu_part].into_iter().flatten() {
        loss += l;
        for (a, b) in grads.iter_mut().zip(&g) {
            *a += b;
        }
    }
    model.adam_step(&grads, lr, focus)?;
    // Speeds in tokens per second, smoothed; the next split balances the two finishing times.
    let gpu_tokens: usize = tokens[..split].iter().sum();
    let cpu_tokens = total - gpu_tokens;
    if gpu_tokens > 0 && cpu_tokens > 0 && gpu_secs > 0. && cpu_secs > 0. {
        let (g, c) = (gpu_tokens as f64 / gpu_secs, cpu_tokens as f64 / cpu_secs);
        let target = g / (g + c);
        h.gpu_share = (0.7 * h.gpu_share + 0.3 * target).clamp(0.02, 0.98);
    } else if gpu_tokens == 0 {
        h.gpu_share = (h.gpu_share + 0.1).min(0.98);
    } else {
        h.gpu_share = (h.gpu_share - 0.1).max(0.02);
    }
    Ok(loss)
}

/// Appends one JSON line to a record file (epoch records, events). Records are plain data for
/// any frontend; they never feed back into training or inference.
pub fn append_jsonl(path: &std::path::Path, record: &impl serde::Serialize) -> Result<(), String> {
    use std::io::Write;
    let mut line = serde_json::to_vec(record).map_err(|e| e.to_string())?;
    line.push(b'\n');
    let mut f =
        std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(|e| e.to_string())?;
    f.write_all(&line).map_err(|e| e.to_string())
}
