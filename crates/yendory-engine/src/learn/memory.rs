//! Training memory management (docs/GROWTH.md, section 3): what training may use on the GPU
//! and in system RAM, how a step is fitted into that, and what it actually used.
//!
//! In order of preference: everything on the GPU; micro-batches with gradient accumulation
//! (exact: the same gradient and step, less activation memory); the AdamW state in system RAM
//! (the GPU computes the gradient, the CPU takes the step, the new weights go back to the GPU).
//! A step that does not fit is reported, never forced: nothing is applied unless the whole
//! gradient was computed.
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryMode {
    /// Everything stays on the GPU (micro-batches only if automatic optimisation is on).
    GpuOnly,
    /// Micro-batches, then the optimizer state in system RAM, as needed and within the limits.
    #[default]
    Automatic,
    /// You choose the micro-batch size and where the optimizer state lives.
    Manual,
}

/// What training may use.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MemorySettings {
    pub mode: MemoryMode,
    /// The graphics card's memory in bytes (wgpu does not report it). `None`: unknown; the
    /// operating system's GPU budget (with the margin) still stops allocations that would not
    /// fit.
    pub gpu_memory_bytes: Option<u64>,
    /// Share of the GPU's memory left free (percent).
    pub gpu_margin_percent: u8,
    /// Reduce memory automatically (micro-batches) when a batch does not fit.
    pub optimize: bool,
    /// At most this much system RAM for offloaded training state. `None`: half the RAM.
    pub ram_limit_bytes: Option<u64>,
    /// System RAM always left free for the operating system and other programs.
    pub ram_reserve_bytes: u64,
    /// Warn when offloading is expected to slow training noticeably.
    pub warn_slowdown: bool,
    /// Manual mode: rows per micro-batch (`None`: the whole batch) and the optimizer state in RAM.
    pub micro_rows: Option<usize>,
    pub optimizer_in_ram: bool,
}

impl Default for MemorySettings {
    fn default() -> Self {
        MemorySettings {
            mode: MemoryMode::Automatic,
            gpu_memory_bytes: None,
            gpu_margin_percent: 10,
            optimize: true,
            ram_limit_bytes: None,
            ram_reserve_bytes: 4 << 30,
            warn_slowdown: true,
            micro_rows: None,
            optimizer_in_ram: false,
        }
    }
}

impl MemorySettings {
    /// GPU memory training may use, in bytes (`None`: no known size).
    pub fn gpu_budget(&self) -> Option<u64> {
        self.gpu_memory_bytes.map(|b| b / 100 * (100 - self.gpu_margin_percent.min(90) as u64))
    }
    /// System RAM offloading may use now: within the limit and leaving the reserve free.
    pub fn ram_allowance(&self) -> u64 {
        let (total, available) = system_ram();
        let limit = self.ram_limit_bytes.unwrap_or(total / 2);
        limit.min(available.saturating_sub(self.ram_reserve_bytes))
    }
}

/// What training is doing with memory, measured.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryReport {
    /// In words: "all on the GPU", "micro-batches of 4 rows", "optimizer state in system RAM, ...".
    pub strategy: String,
    pub device: String,
    /// Rows per micro-batch (`None`: whole batches).
    pub micro_rows: Option<usize>,
    pub optimizer_in_ram: bool,
    /// GPU memory training holds (model buffers and working buffers), counted by Yendory.
    pub gpu_bytes: u64,
    /// The GPU allocator's own count (DirectX 12 and Vulkan; `None` elsewhere).
    pub gpu_allocated_bytes: Option<u64>,
    /// The GPU budget in force (bytes), if known.
    pub gpu_budget_bytes: Option<u64>,
    /// System RAM holding offloaded training state.
    pub ram_offload_bytes: u64,
    /// GPU <-> system RAM transfers for offloading, total bytes and seconds.
    pub transfer_bytes: u64,
    pub transfer_seconds: f64,
    /// Training steps, the tokens in them and their total time (seconds).
    pub steps: u64,
    pub tokens: u64,
    pub step_seconds: f64,
    /// Steps that did not fit and were retried with less memory.
    pub refits: u32,
    /// The latest change of strategy and why.
    pub note: Option<String>,
}

impl MemoryReport {
    /// Share of training time spent on offloading transfers.
    pub fn transfer_share(&self) -> f64 {
        if self.step_seconds > 0. {
            self.transfer_seconds / self.step_seconds
        } else {
            0.
        }
    }
}

static SETTINGS: Mutex<Option<MemorySettings>> = Mutex::new(None);
static REPORT: Mutex<Option<MemoryReport>> = Mutex::new(None);

pub fn set_memory_settings(s: MemorySettings) {
    *SETTINGS.lock().unwrap() = Some(s);
}
pub fn memory_settings() -> MemorySettings {
    SETTINGS.lock().unwrap().clone().unwrap_or_default()
}
/// The latest training memory report (updated every step of GPU training).
pub fn memory_report() -> Option<MemoryReport> {
    REPORT.lock().unwrap().clone()
}
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn publish_report(r: &MemoryReport) {
    *REPORT.lock().unwrap() = Some(r.clone());
}

/// System RAM: (total, available) in bytes.
pub fn system_ram() -> (u64, u64) {
    #[cfg(feature = "gpu")]
    {
        let mut s = sysinfo::System::new();
        s.refresh_memory();
        (s.total_memory(), s.available_memory())
    }
    #[cfg(not(feature = "gpu"))]
    {
        (0, 0)
    }
}

/// The sizes a step's memory plan is made from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepShape {
    pub parameters: usize,
    /// Values the forward pass keeps per token (`Bicameral::activation_floats_for`).
    pub activation_floats_per_token: usize,
    pub rows: usize,
    /// The longest row (rows are padded to it).
    pub tokens: usize,
    /// Output words (the output layer's working values grow with it).
    pub vocab: usize,
}

impl StepShape {
    /// Estimated GPU working memory for one row: the activations kept for the backward pass,
    /// plus half again for the working buffers' size rounding, the backward pass's temporary
    /// values and the inputs (measured in `tests/gpu_memory.rs`).
    pub fn row_bytes(&self) -> u64 {
        (self.activation_floats_per_token as u64 * self.tokens.max(1) as u64 * 4) * 3 / 2
    }
    /// Estimated GPU working memory of a step in parts of `rows` rows: the rows, and the output
    /// layer's scores (computed for at most 16M values at a time).
    pub fn working_bytes(&self, rows: usize) -> u64 {
        let scores = (rows as u64 * self.tokens.max(1) as u64 * self.vocab as u64).min(16 << 20);
        rows as u64 * self.row_bytes() + scores * 4
    }
    /// System RAM offloading the optimizer state relies on: both AdamW moments and the
    /// gradient copied down each step.
    pub fn ram_offload_bytes(&self) -> u64 {
        3 * self.parameters as u64 * 4
    }
}

/// How a step runs: rows per micro-batch and where the optimizer state is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub micro_rows: usize,
    pub optimizer_in_ram: bool,
}

impl Plan {
    pub fn describe(&self, rows: usize) -> String {
        let place = if self.optimizer_in_ram {
            "Optimizer state in system RAM, weights and gradients on the GPU"
        } else {
            "All training state on the GPU"
        };
        if self.micro_rows < rows {
            format!("{place}; micro-batches of {} of the batch's {rows} rows", self.micro_rows)
        } else {
            format!("{place}; whole batches")
        }
    }
}

/// Chooses each step's plan and adapts it when a step does not fit: first smaller micro-batches
/// (exact), then the optimizer state in system RAM (Automatic mode, within the RAM limits).
#[derive(Clone, Debug, Default)]
pub struct Fitter {
    /// Rows x tokens that fitted at most, learned from steps that did not fit.
    cap_row_tokens: Option<usize>,
    /// The optimizer state was moved to system RAM.
    ram: bool,
}

impl Fitter {
    /// The plan for a step. `fixed_gpu` and `fixed_ram`: GPU memory of the model's own buffers
    /// with the optimizer state on the GPU and in RAM; `ram_ok`: the RAM limits allow
    /// offloading it. When no plan is estimated to fit, the one using least GPU memory is
    /// returned (the budget is enforced when the step runs, so trying it is safe).
    pub fn plan(
        &self,
        s: &MemorySettings,
        shape: &StepShape,
        fixed_gpu: u64,
        fixed_ram: u64,
        ram_ok: bool,
        in_ram_now: bool,
    ) -> Result<Plan, String> {
        let rows = shape.rows.max(1);
        if s.mode == MemoryMode::Manual {
            let ram = s.optimizer_in_ram || in_ram_now;
            if ram && !in_ram_now && !ram_ok {
                return Err(format!(
                    "The optimizer state ({}) does not fit in the system RAM allowed for offloading",
                    gib(shape.ram_offload_bytes())
                ));
            }
            return Ok(Plan {
                micro_rows: s.micro_rows.unwrap_or(rows).clamp(1, rows),
                optimizer_in_ram: ram,
            });
        }
        let micro_ok = s.optimize;
        let layouts: &[bool] = if in_ram_now || self.ram {
            &[true]
        } else if s.mode == MemoryMode::Automatic && ram_ok {
            &[false, true]
        } else {
            &[false]
        };
        let cap = self.cap_row_tokens.map_or(usize::MAX, |c| (c / shape.tokens.max(1)).max(1));
        let least = if micro_ok { 1 } else { rows };
        for &ram in layouts {
            let fixed = if ram { fixed_ram } else { fixed_gpu };
            let fits = |k: usize| s.gpu_budget().map_or(true, |b| fixed + shape.working_bytes(k) <= b);
            let k = (1..=cap.min(rows)).rev().find(|&k| fits(k)).unwrap_or(0);
            if k >= least {
                return Ok(Plan { micro_rows: k, optimizer_in_ram: ram });
            }
        }
        Ok(Plan { micro_rows: least.min(cap).max(1), optimizer_in_ram: *layouts.last().unwrap() })
    }

    /// After a step that did not fit (nothing was applied): the next thing to try, described,
    /// or `None` when nothing is left within the settings.
    pub fn after_out_of_memory(
        &mut self,
        s: &MemorySettings,
        plan: &Plan,
        shape: &StepShape,
        ram_ok: bool,
    ) -> Option<String> {
        if s.mode == MemoryMode::Manual {
            return None;
        }
        if s.optimize && plan.micro_rows > 1 {
            let k = plan.micro_rows / 2;
            self.cap_row_tokens = Some(k * shape.tokens.max(1));
            return Some(format!("micro-batches of {k} row{}", if k == 1 { "" } else { "s" }));
        }
        if !plan.optimizer_in_ram && s.mode == MemoryMode::Automatic && ram_ok {
            self.ram = true;
            self.cap_row_tokens = None;
            return Some("the optimizer state in system RAM".into());
        }
        None
    }
}

/// Whether a step of `shape` fits the settings on the GPU, and how it would run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FitEstimate {
    pub micro_rows: usize,
    pub optimizer_in_ram: bool,
    /// The plan in words.
    pub plan: String,
    /// Estimated GPU memory of the plan (model buffers and working memory).
    pub gpu_bytes: u64,
    /// System RAM the plan offloads to.
    pub ram_bytes: u64,
    /// Within the GPU budget and the RAM limits; `None` when the GPU's memory is not given.
    pub fits: Option<bool>,
}

/// Estimates how a step of `shape` would run under the settings (the plan the trainer would
/// choose first) and whether it stays within the limits. `fixed_gpu`/`fixed_ram`: the model's
/// own GPU buffers with the optimizer state on the GPU and in RAM.
pub fn estimate(
    s: &MemorySettings,
    shape: &StepShape,
    fixed_gpu: u64,
    fixed_ram: u64,
    ram_allowance: u64,
) -> FitEstimate {
    let ram_ok = ram_allowance >= shape.ram_offload_bytes();
    let plan = Fitter::default()
        .plan(s, shape, fixed_gpu, fixed_ram, ram_ok, false)
        .unwrap_or(Plan { micro_rows: 1, optimizer_in_ram: s.optimizer_in_ram });
    let gpu_bytes =
        if plan.optimizer_in_ram { fixed_ram } else { fixed_gpu } + shape.working_bytes(plan.micro_rows);
    let ram_bytes = if plan.optimizer_in_ram { shape.ram_offload_bytes() } else { 0 };
    let ram_fits = !plan.optimizer_in_ram || ram_ok;
    FitEstimate {
        micro_rows: plan.micro_rows,
        optimizer_in_ram: plan.optimizer_in_ram,
        plan: plan.describe(shape.rows.max(1)),
        gpu_bytes,
        ram_bytes,
        fits: s.gpu_budget().map(|b| gpu_bytes <= b && ram_fits),
    }
}

/// "1.5 GB".
pub fn gib(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64)
    } else {
        format!("{} MB", bytes.div_ceil(1 << 20))
    }
}

/// An error that means "did not fit in GPU memory" (the step may be retried with less).
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn is_out_of_memory(e: &str) -> bool {
    let e = e.to_ascii_lowercase();
    e.contains("out of memory") || e.contains("outofmemory") || e.contains("over the gpu memory budget")
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1 << 20;

    fn shape() -> StepShape {
        // About 1 MB of working memory per row.
        StepShape { parameters: 1_000_000, activation_floats_per_token: 2048, rows: 16, tokens: 85, vocab: 0 }
    }

    fn settings(budget: u64) -> MemorySettings {
        MemorySettings { gpu_memory_bytes: Some(budget * 10 / 9 + 1000), ..MemorySettings::default() }
    }

    #[test]
    fn plans_follow_the_order_of_preference() {
        let sh = shape();
        let row = sh.row_bytes();
        let (gpu, ram) = (20 * MB, 12 * MB);
        let f = Fitter::default();
        // Everything fits: whole batches, all on the GPU.
        let p = f.plan(&settings(gpu + 16 * row), &sh, gpu, ram, true, false).unwrap();
        assert_eq!(p, Plan { micro_rows: 16, optimizer_in_ram: false });
        // Room for 5 rows: micro-batches before moving anything to RAM.
        let p = f.plan(&settings(gpu + 5 * row), &sh, gpu, ram, true, false).unwrap();
        assert_eq!(p, Plan { micro_rows: 5, optimizer_in_ram: false });
        // Not even one row with the optimizer state on the GPU: it goes to RAM.
        let p = f.plan(&settings(ram + 3 * row), &sh, gpu, ram, true, false).unwrap();
        assert_eq!(p, Plan { micro_rows: 3, optimizer_in_ram: true });
        // ... unless the RAM limits refuse it, or the mode keeps everything on the GPU.
        let p = f.plan(&settings(ram + 3 * row), &sh, gpu, ram, false, false).unwrap();
        assert!(!p.optimizer_in_ram);
        let only_gpu = MemorySettings { mode: MemoryMode::GpuOnly, ..settings(ram + 3 * row) };
        assert!(!f.plan(&only_gpu, &sh, gpu, ram, true, false).unwrap().optimizer_in_ram);
        // Without micro-batches, whole batches or nothing.
        let whole = MemorySettings { optimize: false, ..settings(gpu + 5 * row) };
        assert_eq!(
            f.plan(&whole, &sh, gpu, ram, true, false).unwrap(),
            Plan { micro_rows: 16, optimizer_in_ram: true }
        );
        // Unknown GPU size: whole batches until a step does not fit.
        let unknown = MemorySettings::default();
        assert_eq!(f.plan(&unknown, &sh, gpu, ram, true, false).unwrap().micro_rows, 16);
    }

    #[test]
    fn steps_that_do_not_fit_shrink_then_offload_then_stop() {
        let sh = shape();
        let s = MemorySettings::default();
        let mut f = Fitter::default();
        let mut p = f.plan(&s, &sh, 0, 0, true, false).unwrap();
        let mut changes = Vec::new();
        while let Some(c) = f.after_out_of_memory(&s, &p, &sh, true) {
            changes.push(c);
            p = f.plan(&s, &sh, 0, 0, true, p.optimizer_in_ram).unwrap();
            assert!(changes.len() < 20);
        }
        println!("{changes:?}");
        // 16 -> 8 -> 4 -> 2 -> 1 rows, then the optimizer state to RAM (which frees GPU memory,
        // so whole batches are tried again), then 8 -> 4 -> 2 -> 1 rows: then nothing is left.
        assert_eq!(changes.len(), 9, "{changes:?}");
        assert!(changes[4].contains("RAM"));
        assert!(p.optimizer_in_ram && p.micro_rows == 1);
        // A longer batch later gets fewer rows (the cap is in rows x tokens).
        let longer = StepShape { tokens: 2 * sh.tokens, ..sh };
        let mut f = Fitter::default();
        let p = Plan { micro_rows: 8, optimizer_in_ram: false };
        f.after_out_of_memory(&s, &p, &sh, true).unwrap();
        assert_eq!(f.plan(&s, &sh, 0, 0, true, false).unwrap().micro_rows, 4);
        assert_eq!(f.plan(&s, &longer, 0, 0, true, false).unwrap().micro_rows, 2);
        // Manual settings are not changed behind your back.
        let manual =
            MemorySettings { mode: MemoryMode::Manual, micro_rows: Some(4), ..MemorySettings::default() };
        let p = f.plan(&manual, &sh, 0, 0, true, false).unwrap();
        assert_eq!(p, Plan { micro_rows: 4, optimizer_in_ram: false });
        assert!(Fitter::default().after_out_of_memory(&manual, &p, &sh, true).is_none());
        let manual_ram = MemorySettings { optimizer_in_ram: true, ..manual };
        assert!(Fitter::default().plan(&manual_ram, &sh, 0, 0, false, false).is_err(), "over the RAM limit");
    }

    #[test]
    fn ram_allowance_keeps_the_reserve_free() {
        let (total, available) = system_ram();
        let s = MemorySettings {
            ram_limit_bytes: Some(u64::MAX),
            ram_reserve_bytes: available + 1,
            ..Default::default()
        };
        assert_eq!(s.ram_allowance(), 0, "never into the reserve");
        let s = MemorySettings { ram_limit_bytes: Some(MB), ram_reserve_bytes: 0, ..Default::default() };
        assert!(s.ram_allowance() <= MB);
        if cfg!(feature = "gpu") {
            assert!(total > 0 && available <= total);
        }
        assert_eq!(
            MemorySettings { gpu_memory_bytes: Some(1000), ..Default::default() }.gpu_budget(),
            Some(900)
        );
    }
}
