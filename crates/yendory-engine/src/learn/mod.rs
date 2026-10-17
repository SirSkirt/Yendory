//! Learning: lesson parsing, training rows, batched gradients, the learning-rate schedule and
//! held-out evaluation. Reference behaviour: `reference/lessons.py` and
//! `reference/train_lessons.py` (the pipeline that trained the step-19,000 checkpoint).
pub mod corpus;
pub mod course;
pub mod curriculum;
pub mod eval;
pub mod generate;
pub mod grading;
pub mod lessons;
pub mod program;
pub mod session;

use crate::model::{Ablation, Bicameral, Focus};
use crate::tokenizer::TokenId;
use crate::tokenizer::Tokenizer;
pub use lessons::{grow_vocabulary, parse_lesson, Item, Kind, Segment};

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
    if rows.is_empty() {
        return Err("Empty training batch".into());
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(rows.len());
    let chunk = rows.len().div_ceil(threads);
    let parts: Vec<Result<Partial, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let mut grads = vec![0f32; model.parameter_count()];
                    let (mut loss, mut weight) = (0f64, 0f64);
                    for row in part {
                        let sum: f32 = row.mask.iter().sum();
                        // Undo the per-row normalisation (the tape divides by max(sum, 1)).
                        let scale = sum.max(1.);
                        let l = model.accumulate_loss_and_gradients(
                            &row.ids,
                            &row.targets,
                            &row.mask,
                            abl,
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
    let denom = weight.max(1.);
    for g in &mut grads {
        *g = (*g as f64 / denom) as f32;
    }
    Ok(((loss / denom) as f32, grads))
}

/// Where training steps run. The GPU is used only when the Engine is built with the `gpu`
/// feature, the user chose it and a GPU could be opened; otherwise the CPU trains.
static USE_GPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Train on the GPU (when there is one) or on the CPU.
pub fn set_gpu_training(on: bool) {
    USE_GPU.store(on, std::sync::atomic::Ordering::Release);
}

pub fn gpu_training() -> bool {
    USE_GPU.load(std::sync::atomic::Ordering::Acquire)
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

/// Runs training steps on the chosen device. On the GPU the model's weights live on the card
/// between steps; [`Trainer::sync`] copies them back before the CPU reads them (checks,
/// scoring, saving). On the CPU each step is `batch_loss_and_gradients` + `adam_step`.
pub struct Trainer {
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::GpuModel>,
    dirty: bool,
}

impl Trainer {
    /// A trainer for `model`, on the GPU when GPU training is on and the model fits.
    pub fn new(model: &Bicameral) -> Trainer {
        #[cfg(feature = "gpu")]
        if gpu_training() {
            if let Ok(gpu) = crate::gpu::shared() {
                if let Ok(m) = crate::gpu::GpuModel::upload(&gpu, model) {
                    return Trainer { gpu: Some(m), dirty: false };
                }
            }
        }
        let _ = model;
        Trainer {
            #[cfg(feature = "gpu")]
            gpu: None,
            dirty: false,
        }
    }

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
    /// updated until [`Trainer::sync`] (its step counter is).
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
            let loss = g.train_step(model, rows, abl, lr, focus)?;
            model.step = g.step;
            self.dirty = true;
            return Ok(loss);
        }
        let (loss, grads) = batch_loss_and_gradients(model, rows, abl)?;
        model.adam_step(&grads, lr, focus)?;
        Ok(loss)
    }

    /// Brings `model` up to date with the GPU (nothing to do on the CPU).
    pub fn sync(&mut self, model: &mut Bicameral) -> Result<(), String> {
        #[cfg(feature = "gpu")]
        if let (Some(g), true) = (self.gpu.as_ref(), self.dirty) {
            g.download(model)?;
        }
        self.dirty = false;
        let _ = model;
        Ok(())
    }
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
