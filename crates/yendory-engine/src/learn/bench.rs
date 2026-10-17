//! Training benchmark: the same batches trained on every available device (CPU, GPU through
//! each graphics API, hybrid CPU + GPU), measured in tokens per second, with GPU memory and a
//! check that every device computes the same loss as the CPU. The model is not changed (each
//! device trains its own copy).
use super::{batch_weight, device, set_device, Device, Row, Trainer};
use crate::model::{Ablation, Bicameral, Focus};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct DeviceResult {
    pub device: Device,
    /// "CPU (24 threads)", "NVIDIA GeForce RTX 2070 SUPER · Vulkan", ...
    pub name: String,
    /// Why the device could not be measured.
    pub error: Option<String>,
    pub tokens_per_second: f64,
    pub ms_per_step: f64,
    /// GPU memory used by training (weights, optimizer state, working buffers).
    pub gpu_memory_bytes: Option<u64>,
    /// Hybrid: the share of each batch the GPU took after balancing.
    pub gpu_share: Option<f64>,
    /// First step's loss minus the CPU's (a check that the device computes the same network).
    pub loss_difference: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Benchmark {
    pub rows: usize,
    pub tokens_per_row: usize,
    pub steps: usize,
    pub results: Vec<DeviceResult>,
    /// The fastest device that agrees with the CPU.
    pub fastest: Option<Device>,
    /// Where a GPU step's time goes, for the fastest GPU (kernel times, CPU preparation).
    #[cfg(feature = "gpu")]
    pub profile: Option<crate::gpu::StepProfile>,
}

/// The devices worth measuring on this machine.
pub fn candidates() -> Vec<Device> {
    #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
    let mut v = vec![Device::Cpu];
    #[cfg(feature = "gpu")]
    {
        use crate::gpu::Api;
        let apis: &[Api] = if cfg!(windows) {
            &[Api::Dx12, Api::Vulkan]
        } else if cfg!(target_os = "macos") {
            &[Api::Metal]
        } else {
            &[Api::Vulkan]
        };
        for &a in apis {
            v.push(Device::Gpu(a));
        }
        for &a in apis {
            v.push(Device::Hybrid(a));
        }
    }
    v
}

/// The device's display name, or why it cannot be used.
pub fn device_name(d: Device) -> Result<String, String> {
    match d {
        Device::Cpu => Ok(format!("CPU ({} threads)", super::cpu_threads())),
        #[cfg(feature = "gpu")]
        Device::Gpu(api) => {
            crate::gpu::shared_with(api).map(|g| format!("{} · {}", g.adapter_name(), g.api_name()))
        }
        #[cfg(feature = "gpu")]
        Device::Hybrid(api) => crate::gpu::shared_with(api).map(|g| {
            format!("CPU ({} threads) + {} · {}", super::cpu_threads(), g.adapter_name(), g.api_name())
        }),
    }
}

/// Trains `steps` steps (after one warm-up step) on each device with identical batches and
/// reports the speed of each. `progress` is called before each device.
pub fn run(
    model: &Bicameral,
    rows: &[Row],
    steps: usize,
    devices: &[Device],
    progress: &mut dyn FnMut(&str),
) -> Benchmark {
    let chosen = device();
    let tokens: usize = rows.iter().map(|r| r.ids.len()).sum();
    let _ = batch_weight(rows);
    let mut results = Vec::new();
    let mut cpu_loss = None;
    for &d in devices {
        let name = device_name(d);
        progress(name.as_deref().unwrap_or("device"));
        let mut r = DeviceResult {
            device: d,
            name: name.clone().unwrap_or_else(|_| format!("{d:?}")),
            error: name.err(),
            tokens_per_second: 0.,
            ms_per_step: 0.,
            gpu_memory_bytes: None,
            gpu_share: None,
            loss_difference: None,
        };
        if r.error.is_none() {
            set_device(d);
            let _ = super::take_device_notice();
            let mut m = model.clone();
            let mut t = Trainer::new(&m);
            if let Some(n) = super::take_device_notice() {
                r.error = Some(n);
            } else {
                let measured = (|| -> Result<(), String> {
                    let first = t.step(&mut m, rows, Ablation::default(), 1e-4, Focus::Joint)?;
                    if let Some(n) = super::take_device_notice() {
                        return Err(n);
                    }
                    if d == Device::Cpu {
                        cpu_loss = Some(first);
                    } else if let Some(c) = cpu_loss {
                        r.loss_difference = Some(first - c);
                    }
                    // Hybrid training balances its split over a few steps first.
                    if !matches!(d, Device::Cpu) && t.gpu_share().is_some() {
                        for _ in 0..3 {
                            t.step(&mut m, rows, Ablation::default(), 1e-4, Focus::Joint)?;
                        }
                    }
                    let started = std::time::Instant::now();
                    for _ in 0..steps {
                        t.step(&mut m, rows, Ablation::default(), 1e-4, Focus::Joint)?;
                    }
                    t.sync(&mut m)?;
                    let secs = started.elapsed().as_secs_f64();
                    r.ms_per_step = secs * 1e3 / steps.max(1) as f64;
                    r.tokens_per_second = (tokens * steps) as f64 / secs.max(1e-9);
                    r.gpu_share = t.gpu_share();
                    r.gpu_memory_bytes = t.gpu_memory_bytes();
                    Ok(())
                })();
                if let Err(e) = measured {
                    r.error = Some(e);
                }
            }
        }
        results.push(r);
    }
    set_device(chosen);
    let agrees = |r: &DeviceResult| r.loss_difference.map_or(r.device == Device::Cpu, |x| x.abs() < 1e-3);
    let fastest = results
        .iter()
        .filter(|r| r.error.is_none() && agrees(r))
        .max_by(|a, b| a.tokens_per_second.total_cmp(&b.tokens_per_second))
        .map(|r| r.device);
    #[cfg(feature = "gpu")]
    let profile = {
        let best_gpu = results
            .iter()
            .filter(|r| r.error.is_none() && matches!(r.device, Device::Gpu(_)))
            .max_by(|a, b| a.tokens_per_second.total_cmp(&b.tokens_per_second))
            .map(|r| r.device);
        match best_gpu {
            Some(Device::Gpu(api)) => {
                progress("profiling the GPU");
                crate::gpu::shared_with(api)
                    .and_then(|g| crate::gpu::GpuModel::upload(&g, model))
                    .and_then(|mut g| g.profile_step(model, rows, Ablation::default(), 1e-4, Focus::Joint))
                    .ok()
            }
            _ => None,
        }
    };
    Benchmark {
        rows: rows.len(),
        tokens_per_row: rows.first().map_or(0, |r| r.ids.len()),
        steps,
        results,
        fastest,
        #[cfg(feature = "gpu")]
        profile,
    }
}

/// Batches of the given shape for benchmarking, from text the model's tokenizer encodes.
pub fn sample_rows(model: &Bicameral, rows: usize, tokens: usize) -> Vec<Row> {
    let words = [
        "the ",
        "little ",
        "cat ",
        "found ",
        "a ",
        "red ",
        "ball ",
        "in ",
        "park ",
        "and ",
        "was ",
        "happy. ",
        "U: where is the key?\nN: In the box.\n",
    ];
    let mut text = String::new();
    let mut i = 7usize;
    let need = rows * (tokens + 1);
    let mut ids = Vec::new();
    while ids.len() < need {
        for _ in 0..256 {
            i = i.wrapping_mul(1103515245).wrapping_add(12345);
            text.push_str(words[(i >> 8) % words.len()]);
        }
        ids = model.tokenizer.encode(&text).unwrap_or_default();
    }
    (0..rows)
        .map(|r| {
            let s = &ids[r * (tokens + 1)..(r + 1) * (tokens + 1)];
            Row { ids: s[..tokens].to_vec(), targets: s[1..].to_vec(), mask: vec![1.; tokens] }
        })
        .collect()
}
