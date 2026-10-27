//! GPU training (feature `gpu`): the training step (Bicameral and Nova Neuro) as compute shaders
//! through wgpu (DirectX 12 on Windows, Vulkan on Linux and Android, Metal on macOS).
//!
//! The model is defined once (`Bicameral::forward_hidden` over the [`Graph`] trait); this module
//! is a second implementation of that trait whose nodes live in GPU buffers. A training step
//! records every operation of the forward pass, the output layer and loss, the backward pass
//! and AdamW, then submits them at once: weights, gradients, optimizer state and activations
//! stay on the card, and only the per-token losses come back. A batch's rows are padded to the
//! same length and computed together (attention and positions stay within each row, and padded
//! positions have zero loss weight, so they change nothing).
//!
//! The values are those of the CPU tape; only the order in which floating-point numbers are
//! added differs (see `tests/gpu.rs`, which compares losses, gradients and steps with the CPU).
use crate::learn::Row;
use crate::model::{Ablation, Bicameral, Focus, Graph};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum K {
    Fill,
    AddInto,
    Binary,
    Unary,
    UnaryBack,
    Embed,
    EmbedBack,
    Gemm,
    AddBias,
    ColsumPart,
    ColsumFinish,
    RowInv,
    Norm,
    NormBackX,
    Rope,
    Attention,
    AttentionBackQ,
    AttentionBackKv,
    Attention32,
    AttentionBackQ32,
    AttentionBackKv32,
    Xent,
    Sumsq,
    NormFinish,
    Adam,
    SoftmaxRows,
    SoftmaxRowsBack,
    GateCol,
    GateColBack,
    Scan,
    ScanBack,
    SlotScores,
    SlotRead,
    Outer,
}
const KERNELS: [(K, &str); 34] = [
    (K::Fill, "fill"),
    (K::AddInto, "add_into"),
    (K::Binary, "binary"),
    (K::Unary, "unary"),
    (K::UnaryBack, "unary_back"),
    (K::Embed, "embed"),
    (K::EmbedBack, "embed_back"),
    (K::Gemm, "gemm"),
    (K::AddBias, "add_bias"),
    (K::ColsumPart, "colsum_part"),
    (K::ColsumFinish, "colsum_finish"),
    (K::RowInv, "row_inv"),
    (K::Norm, "norm"),
    (K::NormBackX, "norm_back_x"),
    (K::Rope, "rope"),
    (K::Attention, "attention"),
    (K::AttentionBackQ, "attention_back_q"),
    (K::AttentionBackKv, "attention_back_kv"),
    (K::Attention32, "attention32"),
    (K::AttentionBackQ32, "attention_back_q32"),
    (K::AttentionBackKv32, "attention_back_kv32"),
    (K::Xent, "xent"),
    (K::Sumsq, "sumsq"),
    (K::NormFinish, "norm_finish"),
    (K::Adam, "adam"),
    (K::SoftmaxRows, "softmax_rows"),
    (K::SoftmaxRowsBack, "softmax_rows_back"),
    (K::GateCol, "gate_col"),
    (K::GateColBack, "gate_col_back"),
    (K::Scan, "scan"),
    (K::ScanBack, "scan_back"),
    (K::SlotScores, "slot_scores"),
    (K::SlotRead, "slot_read"),
    (K::Outer, "outer"),
];
fn kernel_name(k: K) -> &'static str {
    KERNELS.iter().find(|e| e.0 == k).map_or("?", |e| e.1)
}

/// GPU time of one kind of kernel in a profiled step.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct KernelTime {
    pub kernel: String,
    /// Dispatches of this kernel in the step.
    pub count: usize,
    pub ms: f64,
}

/// Where a training step's time goes, and how much GPU memory training uses.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct StepProfile {
    pub tokens: usize,
    pub loss: f32,
    /// CPU: preparing the step (character features, input data, recording every operation).
    pub cpu_prepare_ms: f64,
    /// Sum of the kernels' GPU times.
    pub gpu_ms: f64,
    /// Wall time of the profiled run (longer than a normal step: kernels are timed one by one).
    pub timed_run_ms: f64,
    pub dispatches: usize,
    /// Kernel kinds, slowest first.
    pub kernels: Vec<KernelTime>,
    /// GPU timestamps were used (otherwise each kernel was timed from submission to completion).
    pub timestamps: bool,
    pub upload_bytes: u64,
    pub memory_bytes: u64,
}

const F32_SLOTS: usize = 7;
/// Floats read back per part by [`Gpu::read_chunked`] (16 MB of staging memory).
const READ_CHUNK: usize = 1 << 22;
/// Bytes of kernel parameters per dispatch (16 u32).
const PARAM_BYTES: u64 = 64;
/// The largest head dimension the attention kernels hold in registers.
const MAX_HEAD_DIM: usize = 64;

/// A GPU and the compiled kernels.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    layout: wgpu::BindGroupLayout,
    pipelines: HashMap<K, wgpu::ComputePipeline>,
    dummies: Vec<wgpu::Buffer>,
    align: u64,
    timestamps: bool,
    /// The first GPU error reported by the driver or wgpu (instead of a panic).
    error: Arc<Mutex<Option<String>>>,
}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

fn rank(t: wgpu::DeviceType) -> u8 {
    match t {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    }
}

/// Which graphics API the GPU is driven through. The kernels and the training logic are the
/// same for every API (wgpu translates the compute shaders); only the driver path differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Api {
    /// DirectX 12 on Windows (then Vulkan); the platform's primary API elsewhere.
    Auto,
    Dx12,
    Vulkan,
    Metal,
}
impl Api {
    pub fn name(self) -> &'static str {
        match self {
            Api::Auto => "automatic",
            Api::Dx12 => "DirectX 12",
            Api::Vulkan => "Vulkan",
            Api::Metal => "Metal",
        }
    }
    fn backends(self) -> wgpu::Backends {
        match self {
            Api::Auto if cfg!(windows) => wgpu::Backends::DX12 | wgpu::Backends::VULKAN,
            Api::Auto => wgpu::Backends::PRIMARY,
            Api::Dx12 => wgpu::Backends::DX12,
            Api::Vulkan => wgpu::Backends::VULKAN,
            Api::Metal => wgpu::Backends::METAL,
        }
    }
}

impl Gpu {
    /// Opens the best GPU: a discrete card first, then an integrated one. A software adapter
    /// (a CPU emulating a GPU) is used only when `allow_software` is set (tests); it would be
    /// slower than training on the CPU directly.
    pub fn open(allow_software: bool) -> Result<Gpu, String> {
        Gpu::open_with(Api::Auto, allow_software)
    }

    /// Opens the best GPU through one graphics API (or says why it cannot be used).
    pub fn open_with(api: Api, allow_software: bool) -> Result<Gpu, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        if cfg!(feature = "dxc") {
            desc.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::StaticDxc;
        }
        if std::env::var_os("WGPU_BACKEND").is_none() || api != Api::Auto {
            desc.backends = api.backends();
        }
        // No wgpu memory budget threshold: wgpu 30 counts every new buffer, however small, as a
        // whole memory block against the operating system's budget, so once usage is within a
        // block of the threshold even tiny buffers fail. Training's own budget (`learn::memory`,
        // from the GPU memory given in the settings) limits what training allocates instead.
        let instance = wgpu::Instance::new(desc);
        let mut adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        adapters.retain(|a| allow_software || a.get_info().device_type != wgpu::DeviceType::Cpu);
        // DirectX 12 before Vulkan for the same card on Windows (the backends are listed in
        // that order of preference).
        adapters.sort_by_key(|a| {
            let i = a.get_info();
            (rank(i.device_type), u8::from(i.backend != wgpu::Backend::Dx12))
        });
        let adapter = adapters.into_iter().next().ok_or_else(|| match api {
            Api::Auto => "No GPU found".to_string(),
            other => format!("No GPU can be used through {}", other.name()),
        })?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        if limits.max_storage_buffers_per_shader_stage < 8 {
            return Err(format!("{}: too few storage buffers per shader", info.name));
        }
        // GPU timestamps, when the driver has them, let the profiler time every kernel.
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("yendory"),
            required_limits: limits.clone(),
            required_features: if timestamps {
                wgpu::Features::TIMESTAMP_QUERY
            } else {
                wgpu::Features::empty()
            },
            ..Default::default()
        }))
        .map_err(|e| format!("{}: {e}", info.name))?;
        // GPU errors are recorded and returned by the next operation that checks, instead of
        // wgpu's default of panicking (which would end the training thread without a word).
        let error: Arc<Mutex<Option<String>>> = Arc::default();
        let sink = error.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let mut slot = sink.lock().unwrap_or_else(|p| p.into_inner());
            if slot.is_none() {
                *slot = Some(e.to_string());
            }
        }));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yendory kernels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels.wgsl").into()),
        });
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(PARAM_BYTES),
            },
            count: None,
        }];
        entries.extend((1..=8).map(storage));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yendory layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yendory pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = KERNELS
            .iter()
            .map(|&(k, entry)| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (k, pipeline)
            })
            .collect();
        let dummies = (0..=F32_SLOTS)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("unused {i}")),
                    size: 16,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let align = (limits.min_uniform_buffer_offset_alignment as u64).max(PARAM_BYTES);
        if let Some(e) = error.lock().unwrap().take() {
            return Err(format!("{}: {e}", info.name));
        }
        Ok(Gpu { device, queue, info, limits, layout, pipelines, dummies, align, timestamps, error })
    }

    /// "NVIDIA GeForce RTX 2070 SUPER (Dx12)".
    pub fn name(&self) -> String {
        format!("{} ({:?})", self.info.name, self.info.backend)
    }

    /// The graphics card's name.
    pub fn adapter_name(&self) -> String {
        self.info.name.clone()
    }

    /// The graphics API in use and, for DirectX 12, the shader compiler.
    pub fn api_name(&self) -> String {
        match self.info.backend {
            wgpu::Backend::Dx12 if cfg!(feature = "dxc") => "DirectX 12 (DXC)".into(),
            wgpu::Backend::Dx12 => "DirectX 12".into(),
            wgpu::Backend::Vulkan => "Vulkan".into(),
            wgpu::Backend::Metal => "Metal".into(),
            other => format!("{other:?}"),
        }
    }

    /// The driver, as the graphics API reports it.
    pub fn driver(&self) -> String {
        format!("{} {}", self.info.driver, self.info.driver_info).trim().to_string()
    }

    /// The largest number of floats one buffer may hold on this GPU.
    pub fn max_floats(&self) -> usize {
        (self.limits.max_storage_buffer_binding_size.min(self.limits.max_buffer_size) / 4) as usize
    }

    fn storage(&self, floats: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (floats.max(4) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn upload_f32(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        self.upload_with_usage(
            bytemuck::cast_slice(data),
            label,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )
    }

    /// A buffer holding `bytes`, created empty and then written through the queue. (Buffers
    /// filled at creation are mapped by wgpu, which panics instead of reporting an error when
    /// the buffer could not be created, e.g. out of GPU memory; this way a failure is reported
    /// by the next [`Gpu::check`].)
    fn upload_with_usage(&self, bytes: &[u8], label: &str, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let size = (bytes.len().max(16) as u64).next_multiple_of(4);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            let mut padded = bytes.to_vec();
            padded.resize(padded.len().next_multiple_of(4), 0);
            self.queue.write_buffer(&buffer, 0, &padded);
        }
        buffer
    }

    fn read(&self, buffer: &wgpu::Buffer, floats: usize) -> Result<Vec<f32>, String> {
        let bytes = self.read_bytes(buffer, (floats.max(1) * 4) as u64)?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes)[..floats].to_vec())
    }

    /// Reads a large buffer in parts of [`READ_CHUNK`] floats, so the staging memory on the
    /// GPU stays small (reading gradients when the optimizer state is in system RAM).
    fn read_chunked(&self, buffer: &wgpu::Buffer, floats: usize) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(floats);
        let mut at = 0;
        while at < floats {
            let n = READ_CHUNK.min(floats - at);
            let bytes = self.read_range(buffer, (at * 4) as u64, (n * 4) as u64)?;
            out.extend_from_slice(bytemuck::cast_slice::<u8, f32>(&bytes));
            at += n;
        }
        Ok(out)
    }

    /// Bytes the GPU's memory allocator has handed out (DirectX 12 and Vulkan), if it says.
    pub fn allocated_bytes(&self) -> Option<u64> {
        self.device.generate_allocator_report().map(|r| r.total_allocated_bytes)
    }

    /// The GPU error reported since the last check, if any.
    fn check(&self) -> Result<(), String> {
        match self.error.lock().unwrap_or_else(|p| p.into_inner()).take() {
            Some(e) => Err(format!("GPU error: {e}")),
            None => Ok(()),
        }
    }

    /// Copies the first `bytes` bytes of a buffer back to the CPU (waits for the GPU).
    fn read_bytes(&self, buffer: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>, String> {
        self.read_range(buffer, 0, bytes)
    }

    /// Copies `bytes` bytes of a buffer from `offset` back to the CPU (waits for the GPU).
    fn read_range(&self, buffer: &wgpu::Buffer, offset: u64, bytes: u64) -> Result<Vec<u8>, String> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buffer, offset, &staging, 0, bytes);
        self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU: {e}"))?;
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| format!("GPU read failed: {e}"))?;
        let out = {
            let view = staging.get_mapped_range(..).map_err(|e| format!("GPU read failed: {e:?}"))?;
            view.to_vec()
        };
        staging.unmap();
        // Every result passes through here: an error during the work it depends on fails it.
        self.check()?;
        Ok(out)
    }
}

type Opened = HashMap<Api, Result<Arc<Gpu>, String>>;
static SHARED: OnceLock<Mutex<Opened>> = OnceLock::new();
static API: Mutex<Api> = Mutex::new(Api::Auto);

/// Chooses the graphics API [`shared`] opens the GPU through.
pub fn set_api(api: Api) {
    *API.lock().unwrap() = api;
}
pub fn api() -> Api {
    *API.lock().unwrap()
}

/// The GPU through the chosen API, opened once per process and API (or why there is none). A
/// software adapter is used only when `YENDORY_GPU_SOFTWARE=1` is set (tests on machines
/// without a graphics card).
pub fn shared() -> Result<Arc<Gpu>, String> {
    shared_with(api())
}

/// The GPU through one API, opened once per process.
pub fn shared_with(api: Api) -> Result<Arc<Gpu>, String> {
    let mut opened = SHARED.get_or_init(Default::default).lock().unwrap();
    opened
        .entry(api)
        .or_insert_with(|| {
            let software = std::env::var("YENDORY_GPU_SOFTWARE").is_ok_and(|v| v == "1");
            Gpu::open_with(api, software).map(Arc::new)
        })
        .clone()
}

// ---- recording ---------------------------------------------------------------------------

/// A buffer used by a step: the model's own, or one of the step's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum B {
    Weights,
    Grads,
    Moments,
    Variance,
    Flags,
    Partial,
    Step(usize),
}

struct Dispatch {
    k: K,
    f: [Option<B>; F32_SLOTS],
    u: Option<B>,
    params: [u32; 16],
    groups: (u32, u32, u32),
}

/// Buffers are reused between steps (and within a step once nothing needs them any more):
/// the dispatches run in the order they are recorded, so a buffer released in the backward
/// pass can hold a later result. Sizes are rounded up to size classes, so batches of different
/// lengths (lessons pad each batch to its longest row) share buffers, and a size class not
/// used for [`KEEP_STEPS`] steps is released: GPU memory follows the recent batches instead of
/// growing with every length ever seen.
#[derive(Default)]
struct Pool {
    free: HashMap<u64, Vec<wgpu::Buffer>>,
    /// Buffers for input data (written before a step runs), kept apart from the working
    /// buffers so an input never shares memory with anything the step computes.
    inputs: HashMap<u64, Vec<wgpu::Buffer>>,
    /// Steps finished, and the last step each size class was asked for.
    steps: u64,
    last_use: HashMap<u64, u64>,
    /// Bytes of every buffer the pool has created and still holds (free or in use).
    owned: u64,
    /// At most this many bytes of working buffers (the GPU memory budget less the model's own
    /// buffers); `None`: no limit beyond the GPU's.
    budget: Option<u64>,
    /// A buffer was refused during this step because of the budget: the step must not run.
    over_budget: bool,
}

/// Buffers smaller than this are placeholders for refused allocations (never pooled).
const SMALLEST_CLASS: u64 = 256;

/// Steps a size class is kept after its last use.
const KEEP_STEPS: u64 = 2;

/// Bytes of the size class holding `bytes`: eight classes per power of two (at most 12.5%
/// larger than asked), at least 256 bytes.
fn size_class(bytes: u64) -> u64 {
    let b = bytes.max(SMALLEST_CLASS);
    let step = 1u64 << (63 - b.leading_zeros()).saturating_sub(3);
    b.div_ceil(step) * step
}

impl Pool {
    fn with_budget(budget: Option<u64>) -> Pool {
        Pool { budget, ..Pool::default() }
    }

    /// The end of a step: size classes unused for `KEEP_STEPS` steps are released.
    fn end_step(&mut self) {
        self.steps += 1;
        self.release_unused(KEEP_STEPS);
        let (free, inputs) = (&self.free, &self.inputs);
        self.last_use.retain(|size, _| free.contains_key(size) || inputs.contains_key(size));
    }

    /// Releases the free buffers of size classes not asked for in the last `keep` steps (with
    /// `keep` 0: not in the current step, so none of them is used by the step being recorded).
    fn release_unused(&mut self, keep: u64) {
        let (now, last) = (self.steps, &self.last_use);
        let mut freed = 0;
        for m in [&mut self.free, &mut self.inputs] {
            m.retain(|size, v| {
                let kept = last.get(size).is_some_and(|&s| s + keep >= now);
                if !kept {
                    freed += size * v.len() as u64;
                }
                kept
            });
        }
        self.owned = self.owned.saturating_sub(freed);
    }

    /// Accounts for a new buffer of `bytes`; `false` (and the step marked over budget) when it
    /// would exceed the budget even after releasing the free buffers this step does not use.
    fn reserve(&mut self, bytes: u64) -> bool {
        if let Some(budget) = self.budget {
            if self.owned + bytes > budget {
                self.release_unused(0);
            }
            if self.owned + bytes > budget {
                self.over_budget = true;
                return false;
            }
        }
        self.owned += bytes;
        true
    }

    /// Bytes of the buffers held between steps.
    fn held_bytes(&self) -> u64 {
        self.free.iter().chain(&self.inputs).map(|(size, v)| size * v.len() as u64).sum()
    }
}

/// The error of a step that would exceed the GPU memory budget (nothing was run).
pub const OVER_BUDGET: &str = "over the GPU memory budget";

fn groups(threads: usize) -> (u32, u32, u32) {
    let g = threads.div_ceil(256).max(1);
    let x = g.min(65535);
    (x as u32, g.div_ceil(x) as u32, 1)
}
/// Workgroups of the attention kernels: 64 rows of one (sequence, head) each.
fn attention_groups(count: usize, t: usize) -> (u32, u32, u32) {
    (t.div_ceil(64) as u32, (count / t.max(1)) as u32, 1)
}
fn bits(x: f32) -> u32 {
    x.to_bits()
}

#[derive(Clone)]
enum Op {
    Input,
    /// `segs`: the rows sorted by token (see the `embed_back` kernel), `nseg` tokens.
    Embed {
        table: usize,
        d: usize,
        segs: B,
        nseg: usize,
    },
    Linear {
        input: usize,
        weight: usize,
        bias: Option<usize>,
        rows: usize,
        cols: usize,
        out: usize,
    },
    Add {
        a: usize,
        b: usize,
    },
    Mul {
        a: usize,
        b: usize,
    },
    Silu {
        src: usize,
    },
    Sigmoid {
        src: usize,
    },
    Norm {
        src: usize,
        gain: usize,
        d: usize,
        inv: B,
    },
    Rope {
        src: usize,
        heads: usize,
        d: usize,
        base: f32,
    },
    Attention {
        q: usize,
        k: usize,
        v: usize,
        t: usize,
        d: usize,
        heads: usize,
        stats: B,
    },
    SoftmaxRows {
        src: usize,
        cols: usize,
    },
    GateCol {
        x: usize,
        g: usize,
        k: usize,
        gcols: usize,
        cols: usize,
    },
    Scan {
        z: usize,
        c: usize,
        seq: usize,
        cols: usize,
    },
    SlotScores {
        q: usize,
        m: usize,
        slots: usize,
        width: usize,
        scale: f32,
    },
    SlotRead {
        p: usize,
        m: usize,
        slots: usize,
        width: usize,
    },
    Outer {
        a: usize,
        b: usize,
        slots: usize,
        width: usize,
    },
}

/// Threads of the scan kernels: one per (sequence, column).
fn scan_threads(rows: usize, seq: usize, cols: usize) -> usize {
    rows.div_ceil(seq.max(1)) * cols
}

struct Node {
    buf: Option<B>,
    len: usize,
    grad: Option<B>,
    op: Op,
    /// Parameter nodes: this range of the weights (and of the gradients).
    param: Option<(usize, usize)>,
}

/// One training step being recorded.
pub(crate) struct GpuGraph<'a> {
    gpu: &'a Gpu,
    pool: &'a mut Pool,
    /// Tokens per sequence (rows are `rows / seq` sequences).
    seq: usize,
    nodes: Vec<Node>,
    params: HashMap<String, usize>,
    bufs: Vec<wgpu::Buffer>,
    released: Vec<bool>,
    uploads: Vec<(usize, Vec<u8>)>,
    ops: Vec<Dispatch>,
    ids: Option<B>,
    /// Bytes of input data uploaded for this step.
    upload_bytes: u64,
    /// Which of `bufs` are input buffers.
    inputs: Vec<usize>,
    /// Loss only: no gradients are recorded.
    forward_only: bool,
    /// Greedy check: the number of valid tokens and the margin (see `GpuModel::greedy_hits`).
    greedy: Option<(u32, f32)>,
    /// The greedy check's per-token results (1 = the target is the clear choice).
    hits: Option<B>,
}

impl<'a> GpuGraph<'a> {
    fn new(gpu: &'a Gpu, pool: &'a mut Pool, seq: usize) -> Self {
        GpuGraph {
            gpu,
            pool,
            seq,
            nodes: Vec::new(),
            params: HashMap::new(),
            bufs: Vec::new(),
            released: Vec::new(),
            uploads: Vec::new(),
            ops: Vec::new(),
            ids: None,
            upload_bytes: 0,
            inputs: Vec::new(),
            forward_only: false,
            greedy: None,
            hits: None,
        }
    }

    fn alloc(&mut self, floats: usize) -> B {
        let bytes = size_class((floats.max(4) * 4) as u64);
        self.pool.last_use.insert(bytes, self.pool.steps);
        let buf = match self.pool.free.get_mut(&bytes).and_then(Vec::pop) {
            Some(b) => b,
            None if self.pool.reserve(bytes) => self.gpu.storage((bytes / 4) as usize, "activation"),
            // Over the budget: a placeholder; the step is refused before it runs.
            None => self.gpu.dummies[0].clone(),
        };
        self.bufs.push(buf);
        self.released.push(false);
        B::Step(self.bufs.len() - 1)
    }

    /// The buffer may hold anything recorded after this point (dispatches run in order).
    fn release(&mut self, b: B) {
        if let B::Step(i) = b {
            if !self.released[i] {
                self.released[i] = true;
                let buf = self.bufs[i].clone();
                if buf.size() >= SMALLEST_CLASS {
                    self.pool.free.entry(buf.size()).or_default().push(buf);
                }
            }
        }
    }

    /// Returns every buffer of the step to the pool.
    fn finish(&mut self) {
        for i in 0..self.bufs.len() {
            self.release(B::Step(i));
        }
        for i in std::mem::take(&mut self.inputs) {
            let buf = self.bufs[i].clone();
            if buf.size() >= SMALLEST_CLASS {
                self.pool.inputs.entry(buf.size()).or_default().push(buf);
            }
        }
        self.pool.over_budget = false;
        self.pool.end_step();
    }

    /// Input data (features, token ids, targets, loss weights) in a buffer of its own: it is
    /// written before the step runs, so it must not share memory with anything the step reuses.
    /// Input buffers come from their own pool (sizes rounded up to a size class, so batches
    /// of similar size reuse them) and return to it when the step finishes.
    fn upload(&mut self, mut bytes: Vec<u8>, _floats: usize) -> B {
        bytes.resize(bytes.len().max(16), 0);
        self.upload_bytes += bytes.len() as u64;
        let size = size_class(bytes.len() as u64);
        self.pool.last_use.insert(size, self.pool.steps);
        let buf = match self.pool.inputs.get_mut(&size).and_then(Vec::pop) {
            Some(b) => b,
            None if self.pool.reserve(size) => self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("input"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            None => self.gpu.dummies[0].clone(),
        };
        self.bufs.push(buf);
        self.released.push(true); // returned to the input pool by `finish`
        self.inputs.push(self.bufs.len() - 1);
        self.uploads.push((self.bufs.len() - 1, bytes));
        B::Step(self.bufs.len() - 1)
    }

    fn dispatch(&mut self, k: K, f: &[Option<B>], u: Option<B>, params: &[u32], groups: (u32, u32, u32)) {
        let mut ff = [None; F32_SLOTS];
        ff[..f.len()].copy_from_slice(f);
        let mut pp = [0u32; 16];
        pp[..params.len()].copy_from_slice(params);
        self.ops.push(Dispatch { k, f: ff, u, params: pp, groups });
    }

    /// One of the slot kernels into a new buffer of `n` values (f0, f1 -> f2).
    #[allow(clippy::too_many_arguments)]
    fn slot_kernel(&mut self, k: K, x: B, y: B, n: usize, slots: usize, width: usize, scale: f32) -> B {
        let out = self.alloc(n);
        let p = [n as u32, slots as u32, width as u32, bits(scale)];
        self.dispatch(k, &[Some(x), Some(y), Some(out)], None, &p, groups(n));
        out
    }

    fn fill(&mut self, b: B, offset: usize, n: usize, value: f32) {
        self.dispatch(K::Fill, &[Some(b)], None, &[n as u32, offset as u32, bits(value)], groups(n));
    }

    fn node(&mut self, buf: B, len: usize, op: Op) -> usize {
        self.nodes.push(Node { buf: Some(buf), len, grad: None, op, param: None });
        self.nodes.len() - 1
    }

    /// The buffer and element offset of a node's values.
    fn at(&self, id: usize) -> (B, usize) {
        match self.nodes[id].param {
            Some((a, _)) => (B::Weights, a),
            None => (self.nodes[id].buf.expect("node values released"), 0),
        }
    }

    /// `C (=|+=) A x B` with strided views: A(i, k) = a.0[a.1 + i * a.2 + k * a.3],
    /// B(k, j) = b.0[b.1 + k * b.2 + j * b.3], C(i, j) = c.0[c.1 + i * n + j].
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        a: (B, usize, usize, usize),
        b: (B, usize, usize, usize),
        c: (B, usize),
        m: usize,
        n: usize,
        k: usize,
        acc: bool,
    ) {
        let p = [
            m as u32,
            n as u32,
            k as u32,
            u32::from(acc),
            a.1 as u32,
            a.2 as u32,
            a.3 as u32,
            b.1 as u32,
            b.2 as u32,
            b.3 as u32,
            c.1 as u32,
        ];
        let g = (n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1);
        self.dispatch(K::Gemm, &[Some(a.0), Some(b.0), Some(c.0)], None, &p, g);
    }

    /// Adds the column sums of `src` (rows x n) into the gradients at `off`: plain sums, or with
    /// `x` and `inv` an RMS norm's gain gradient (sum of src * x * inv(row)). Rows are summed in
    /// parallel chunks, then the chunks.
    fn colsum(&mut self, src: B, x: Option<B>, inv: Option<B>, n: usize, rows: usize, off: usize) {
        const CHUNK: usize = 64;
        let chunks = rows.div_ceil(CHUNK).max(1);
        let part = self.alloc(chunks * n);
        let mode = u32::from(x.is_some());
        self.dispatch(
            K::ColsumPart,
            &[Some(src), x, Some(part), inv],
            None,
            &[n as u32, rows as u32, CHUNK as u32, mode],
            (n.div_ceil(256) as u32, chunks as u32, 1),
        );
        self.dispatch(
            K::ColsumFinish,
            &[Some(part), Some(B::Grads)],
            None,
            &[n as u32, chunks as u32, off as u32],
            groups(n),
        );
        self.release(part);
    }

    /// Adds `delta` (a buffer of the node's length) into the node's gradient. `owned`: the
    /// buffer is the caller's and may become the gradient itself.
    fn accum(&mut self, id: usize, delta: B, owned: bool) {
        let n = self.nodes[id].len;
        if let Some((a, _)) = self.nodes[id].param {
            self.dispatch(
                K::AddInto,
                &[Some(delta), Some(B::Grads)],
                None,
                &[n as u32, 0, a as u32],
                groups(n),
            );
            if owned {
                self.release(delta);
            }
            return;
        }
        match self.nodes[id].grad {
            None if owned => self.nodes[id].grad = Some(delta),
            None => {
                let g = self.alloc(n);
                self.fill(g, 0, n, 0.);
                self.dispatch(K::AddInto, &[Some(delta), Some(g)], None, &[n as u32, 0, 0], groups(n));
                self.nodes[id].grad = Some(g);
            }
            Some(g) => {
                self.dispatch(K::AddInto, &[Some(delta), Some(g)], None, &[n as u32, 0, 0], groups(n));
                if owned {
                    self.release(delta);
                }
            }
        }
    }

    /// Mean cross-entropy of `h · embedᵀ + bias` over all rows (weights `mask`, divided by
    /// max(sum of weights, 1)), and its backward pass: gradients go to the model's gradient
    /// buffer (embedding and bias) and to `h`. Logits are computed a chunk of rows at a time.
    /// Returns the buffer of per-row losses.
    #[allow(clippy::too_many_arguments)]
    fn head_xent(
        &mut self,
        h: usize,
        embed: usize,
        bias: usize,
        d: usize,
        vocab: usize,
        targets: B,
        mask: B,
        denom: f32,
    ) -> B {
        let rows = self.nodes[h].len / d;
        let (hb, _) = self.at(h);
        let (ea, ba) = (self.nodes[embed].param.unwrap().0, self.nodes[bias].param.unwrap().0);
        let losses = self.alloc(rows);
        let (valid, margin) = self.greedy.unwrap_or((0, 0.));
        let hits = (valid > 0).then(|| self.alloc(rows));
        self.hits = hits;
        let dh = self.alloc(rows * d);
        let chunk = ((16usize << 20) / vocab.max(1)).clamp(1, rows.max(1));
        let logits = self.alloc(chunk * vocab);
        for r0 in (0..rows).step_by(chunk) {
            let rc = chunk.min(rows - r0);
            // logits = h · embedᵀ + bias
            self.gemm((hb, r0 * d, d, 1), (B::Weights, ea, 1, d), (logits, 0), rc, vocab, d, false);
            self.dispatch(
                K::AddBias,
                &[Some(B::Weights), Some(logits)],
                None,
                &[(rc * vocab) as u32, vocab as u32, ba as u32],
                groups(rc * vocab),
            );
            // One workgroup per row: the row's loss, and the logits become their gradient.
            let p = [vocab as u32, rc as u32, r0 as u32, valid, bits(denom), bits(1.), bits(margin)];
            let wg = groups(rc * 256);
            self.dispatch(K::Xent, &[Some(logits), Some(mask), Some(losses), hits], Some(targets), &p, wg);
            if self.forward_only {
                continue;
            }
            // d embed += gᵀ · h, d bias += column sums of g, d h = g · embed
            self.gemm((logits, 0, 1, vocab), (hb, r0 * d, d, 1), (B::Grads, ea), vocab, d, rc, true);
            self.colsum(logits, None, None, vocab, rc, ba);
            self.gemm((logits, 0, vocab, 1), (B::Weights, ea, d, 1), (dh, r0 * d), rc, d, vocab, false);
        }
        self.release(logits);
        if self.forward_only {
            self.release(dh);
        } else {
            self.accum(h, dh, true);
        }
        losses
    }

    /// Back-propagates from the node whose gradient is already set, in reverse order.
    fn backward(&mut self) {
        for id in (0..self.nodes.len()).rev() {
            let op = std::mem::replace(&mut self.nodes[id].op, Op::Input);
            if matches!(op, Op::Input) {
                continue;
            }
            let Some(up) = self.nodes[id].grad.take() else {
                if let Some(b) = self.nodes[id].buf.take() {
                    self.release(b);
                }
                continue;
            };
            let n = self.nodes[id].len;
            match op {
                Op::Input => {}
                Op::Add { a, b } => {
                    self.accum(a, up, false);
                    self.accum(b, up, false);
                }
                Op::Mul { a, b } => {
                    let (av, bv) = (self.at(a).0, self.at(b).0);
                    let ga = self.alloc(n);
                    self.dispatch(
                        K::Binary,
                        &[Some(up), Some(bv), Some(ga)],
                        None,
                        &[n as u32, 1],
                        groups(n),
                    );
                    self.accum(a, ga, true);
                    let gb = self.alloc(n);
                    self.dispatch(
                        K::Binary,
                        &[Some(up), Some(av), Some(gb)],
                        None,
                        &[n as u32, 1],
                        groups(n),
                    );
                    self.accum(b, gb, true);
                }
                Op::Silu { src } => {
                    let x = self.at(src).0;
                    let g = self.alloc(n);
                    self.dispatch(
                        K::UnaryBack,
                        &[Some(up), Some(x), Some(g)],
                        None,
                        &[n as u32, 0],
                        groups(n),
                    );
                    self.accum(src, g, true);
                }
                Op::Sigmoid { src } => {
                    let s = self.at(id).0;
                    let g = self.alloc(n);
                    self.dispatch(
                        K::UnaryBack,
                        &[Some(up), Some(s), Some(g)],
                        None,
                        &[n as u32, 1],
                        groups(n),
                    );
                    self.accum(src, g, true);
                }
                Op::Embed { table, d, segs, nseg } => {
                    let a = self.nodes[table].param.unwrap().0;
                    let rows = n / d;
                    self.dispatch(
                        K::EmbedBack,
                        &[Some(up), Some(B::Grads)],
                        Some(segs),
                        &[d as u32, nseg as u32, a as u32, rows as u32],
                        groups(nseg * d),
                    );
                }
                Op::Linear { input, weight, bias, rows, cols, out } => {
                    let wa = self.nodes[weight].param.unwrap().0;
                    let x = self.at(input).0;
                    // dW += Gᵀ · X
                    self.gemm((up, 0, 1, out), (x, 0, cols, 1), (B::Grads, wa), out, cols, rows, true);
                    if let Some(b) = bias {
                        let ba = self.nodes[b].param.unwrap().0;
                        self.colsum(up, None, None, out, rows, ba);
                    }
                    // dX = G · W
                    let dx = self.alloc(rows * cols);
                    self.gemm((up, 0, out, 1), (B::Weights, wa, cols, 1), (dx, 0), rows, cols, out, false);
                    self.accum(input, dx, true);
                }
                Op::Norm { src, gain, d, inv } => {
                    let ga = self.nodes[gain].param.unwrap().0;
                    let x = self.at(src).0;
                    let rows = n / d;
                    let dx = self.alloc(n);
                    let p = [d as u32, rows as u32, ga as u32];
                    self.dispatch(
                        K::NormBackX,
                        &[Some(up), Some(x), Some(B::Weights), Some(inv), Some(dx)],
                        None,
                        &p,
                        groups(rows),
                    );
                    self.colsum(up, Some(x), Some(inv), d, rows, ga);
                    self.release(inv);
                    self.accum(src, dx, true);
                }
                Op::Rope { src, heads, d, base } => {
                    let dx = self.alloc(n);
                    let pairs = n / 2;
                    let p = [pairs as u32, d as u32, heads as u32, self.seq as u32, bits(base), 1];
                    self.dispatch(K::Rope, &[Some(up), Some(dx)], None, &p, groups(pairs));
                    self.accum(src, dx, true);
                }
                Op::Attention { q, k, v, t, d, heads, stats } => {
                    let count = n / d * heads;
                    let (qb, kb, vb) = (self.at(q).0, self.at(k).0, self.at(v).0);
                    let sc = 1. / ((d / heads) as f32).sqrt();
                    let p = [count as u32, t as u32, d as u32, heads as u32, bits(sc)];
                    let wg = attention_groups(count, t);
                    let dq = self.alloc(n);
                    let (kq, kkv) = if d / heads == 32 {
                        (K::AttentionBackQ32, K::AttentionBackKv32)
                    } else {
                        (K::AttentionBackQ, K::AttentionBackKv)
                    };
                    self.dispatch(
                        kq,
                        &[Some(up), Some(qb), Some(kb), Some(vb), Some(stats), Some(dq)],
                        None,
                        &p,
                        wg,
                    );
                    let (dk, dv) = (self.alloc(n), self.alloc(n));
                    self.dispatch(
                        kkv,
                        &[Some(up), Some(qb), Some(kb), Some(vb), Some(stats), Some(dk), Some(dv)],
                        None,
                        &p,
                        wg,
                    );
                    self.release(stats);
                    self.accum(q, dq, true);
                    self.accum(k, dk, true);
                    self.accum(v, dv, true);
                }
                Op::SoftmaxRows { src, cols } => {
                    let y = self.at(id).0;
                    let rows = n / cols;
                    let dx = self.alloc(n);
                    self.dispatch(
                        K::SoftmaxRowsBack,
                        &[Some(up), Some(y), Some(dx)],
                        None,
                        &[cols as u32, rows as u32],
                        groups(rows * 256),
                    );
                    self.accum(src, dx, true);
                }
                Op::GateCol { x, g, k, gcols, cols } => {
                    let (xv, gv) = (self.at(x).0, self.at(g).0);
                    let rows = n / cols;
                    let dx = self.alloc(n);
                    let p = [n as u32, cols as u32, gcols as u32, k as u32];
                    self.dispatch(K::GateCol, &[Some(up), Some(gv), Some(dx)], None, &p, groups(n));
                    self.accum(x, dx, true);
                    let glen = self.nodes[g].len;
                    let dg = self.alloc(glen);
                    self.fill(dg, 0, glen, 0.);
                    let p = [rows as u32, cols as u32, gcols as u32, k as u32];
                    self.dispatch(K::GateColBack, &[Some(up), Some(xv), Some(dg)], None, &p, groups(rows));
                    self.accum(g, dg, true);
                }
                Op::Scan { z, c, seq, cols } => {
                    let (m, zv, cv) = (self.at(id).0, self.at(z).0, self.at(c).0);
                    let rows = n / cols;
                    let threads = scan_threads(rows, seq, cols);
                    let (dz, dc) = (self.alloc(n), self.alloc(n));
                    self.dispatch(
                        K::ScanBack,
                        &[Some(up), Some(zv), Some(cv), Some(m), Some(dz), Some(dc)],
                        None,
                        &[threads as u32, cols as u32, seq as u32, rows as u32],
                        groups(threads),
                    );
                    self.accum(z, dz, true);
                    self.accum(c, dc, true);
                }
                // The three slot kernels are each other's gradients.
                Op::SlotScores { q, m, slots, width, scale } => {
                    let (qv, mv) = (self.at(q).0, self.at(m).0);
                    let rows = n / slots;
                    let dq = self.slot_kernel(K::SlotRead, up, mv, rows * width, slots, width, scale);
                    let dm = self.slot_kernel(K::Outer, up, qv, rows * slots * width, slots, width, scale);
                    self.accum(q, dq, true);
                    self.accum(m, dm, true);
                }
                Op::SlotRead { p, m, slots, width } => {
                    let (pv, mv) = (self.at(p).0, self.at(m).0);
                    let rows = n / width;
                    let dp = self.slot_kernel(K::SlotScores, up, mv, rows * slots, slots, width, 1.);
                    let dm = self.slot_kernel(K::Outer, pv, up, rows * slots * width, slots, width, 1.);
                    self.accum(p, dp, true);
                    self.accum(m, dm, true);
                }
                Op::Outer { a, b, slots, width } => {
                    let (av, bv) = (self.at(a).0, self.at(b).0);
                    let rows = n / (slots * width);
                    let da = self.slot_kernel(K::SlotScores, bv, up, rows * slots, slots, width, 1.);
                    let db = self.slot_kernel(K::SlotRead, av, up, rows * width, slots, width, 1.);
                    self.accum(a, da, true);
                    self.accum(b, db, true);
                }
            }
            self.release(up);
            if let Some(b) = self.nodes[id].buf.take() {
                self.release(b);
            }
        }
    }
}

impl Graph for GpuGraph<'_> {
    fn parameter(&mut self, name: &str, range: std::ops::Range<usize>) -> usize {
        if let Some(&id) = self.params.get(name) {
            return id;
        }
        self.nodes.push(Node {
            buf: None,
            len: range.len(),
            grad: None,
            op: Op::Input,
            param: Some((range.start, range.end)),
        });
        let id = self.nodes.len() - 1;
        self.params.insert(name.to_owned(), id);
        id
    }
    fn input(&mut self, x: Vec<f32>) -> usize {
        let n = x.len();
        let b = self.upload(bytemuck::cast_slice(&x).to_vec(), n);
        self.node(b, n, Op::Input)
    }
    fn zeros_like(&mut self, like: usize) -> usize {
        let n = self.nodes[like].len;
        let b = self.alloc(n);
        self.fill(b, 0, n, 0.);
        self.node(b, n, Op::Input)
    }
    fn embed(&mut self, ids: &[usize], table: usize, d: usize) -> usize {
        let u: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let ib = self.upload(bytemuck::cast_slice(&u).to_vec(), u.len());
        self.ids = Some(ib);
        // Rows sorted by token, for the gradient (one thread per token and column).
        let mut order: Vec<u32> = (0..u.len() as u32).collect();
        order.sort_by_key(|&r| u[r as usize]);
        let mut starts = Vec::new();
        let mut tokens = Vec::new();
        for (k, &r) in order.iter().enumerate() {
            if k == 0 || u[r as usize] != u[order[k - 1] as usize] {
                starts.push(k as u32);
                tokens.push(u[r as usize]);
            }
        }
        starts.push(order.len() as u32);
        let nseg = tokens.len();
        let seg_data: Vec<u32> = order.into_iter().chain(starts).chain(tokens).collect();
        let segs = self.upload(bytemuck::cast_slice(&seg_data).to_vec(), seg_data.len());
        let n = ids.len() * d;
        let out = self.alloc(n);
        let a = self.nodes[table].param.unwrap().0;
        self.dispatch(
            K::Embed,
            &[Some(B::Weights), Some(out)],
            Some(ib),
            &[n as u32, d as u32, a as u32],
            groups(n),
        );
        self.node(out, n, Op::Embed { table, d, segs, nseg })
    }
    fn linear(&mut self, x: usize, w: usize, b: Option<usize>, in_dim: usize, out_dim: usize) -> usize {
        let rows = self.nodes[x].len / in_dim;
        let (xb, xo) = self.at(x);
        let wa = self.nodes[w].param.unwrap().0;
        let out = self.alloc(rows * out_dim);
        self.gemm((xb, xo, in_dim, 1), (B::Weights, wa, 1, in_dim), (out, 0), rows, out_dim, in_dim, false);
        if let Some(bias) = b {
            let ba = self.nodes[bias].param.unwrap().0;
            let n = rows * out_dim;
            self.dispatch(
                K::AddBias,
                &[Some(B::Weights), Some(out)],
                None,
                &[n as u32, out_dim as u32, ba as u32],
                groups(n),
            );
        }
        self.node(
            out,
            rows * out_dim,
            Op::Linear { input: x, weight: w, bias: b, rows, cols: in_dim, out: out_dim },
        )
    }
    fn add(&mut self, a: usize, b: usize) -> usize {
        let n = self.nodes[a].len;
        let out = self.alloc(n);
        let (ab, bb) = (self.at(a).0, self.at(b).0);
        self.dispatch(K::Binary, &[Some(ab), Some(bb), Some(out)], None, &[n as u32, 0], groups(n));
        self.node(out, n, Op::Add { a, b })
    }
    fn mul(&mut self, a: usize, b: usize) -> usize {
        let n = self.nodes[a].len;
        let out = self.alloc(n);
        let (ab, bb) = (self.at(a).0, self.at(b).0);
        self.dispatch(K::Binary, &[Some(ab), Some(bb), Some(out)], None, &[n as u32, 1], groups(n));
        self.node(out, n, Op::Mul { a, b })
    }
    fn silu(&mut self, src: usize) -> usize {
        let n = self.nodes[src].len;
        let out = self.alloc(n);
        let s = self.at(src).0;
        self.dispatch(K::Unary, &[Some(s), Some(out)], None, &[n as u32, 0], groups(n));
        self.node(out, n, Op::Silu { src })
    }
    fn sigmoid(&mut self, src: usize) -> usize {
        let n = self.nodes[src].len;
        let out = self.alloc(n);
        let s = self.at(src).0;
        self.dispatch(K::Unary, &[Some(s), Some(out)], None, &[n as u32, 1], groups(n));
        self.node(out, n, Op::Sigmoid { src })
    }
    fn norm(&mut self, src: usize, gain: usize, d: usize) -> usize {
        let n = self.nodes[src].len;
        let rows = n / d;
        let x = self.at(src).0;
        let ga = self.nodes[gain].param.unwrap().0;
        let inv = self.alloc(rows);
        self.dispatch(
            K::RowInv,
            &[Some(x), None, None, Some(inv)],
            None,
            &[d as u32, rows as u32],
            groups(rows),
        );
        let out = self.alloc(n);
        self.dispatch(
            K::Norm,
            &[Some(x), Some(B::Weights), Some(out), Some(inv)],
            None,
            &[n as u32, d as u32, ga as u32],
            groups(n),
        );
        self.node(out, n, Op::Norm { src, gain, d, inv })
    }
    fn rope_dim(&mut self, src: usize, heads: usize, d: usize, base: f32) -> usize {
        let n = self.nodes[src].len;
        let x = self.at(src).0;
        let out = self.alloc(n);
        let pairs = n / 2;
        let p = [pairs as u32, d as u32, heads as u32, self.seq as u32, bits(base), 0];
        self.dispatch(K::Rope, &[Some(x), Some(out)], None, &p, groups(pairs));
        self.node(out, n, Op::Rope { src, heads, d, base })
    }
    fn attention(&mut self, q: usize, k: usize, v: usize, t: usize, d: usize, heads: usize) -> usize {
        let n = self.nodes[q].len;
        let count = n / d * heads;
        let (qb, kb, vb) = (self.at(q).0, self.at(k).0, self.at(v).0);
        let out = self.alloc(n);
        let stats = self.alloc(3 * count);
        let sc = 1. / ((d / heads) as f32).sqrt();
        let p = [count as u32, t as u32, d as u32, heads as u32, bits(sc)];
        let kernel = if d / heads == 32 { K::Attention32 } else { K::Attention };
        self.dispatch(
            kernel,
            &[Some(qb), Some(kb), Some(vb), Some(out), Some(stats)],
            None,
            &p,
            attention_groups(count, t),
        );
        self.node(out, n, Op::Attention { q, k, v, t, d, heads, stats })
    }
    // Nova Neuro's operations (the same values as the CPU tape's).
    fn softmax_rows(&mut self, src: usize, cols: usize) -> usize {
        let n = self.nodes[src].len;
        assert_eq!(n % cols, 0);
        let rows = n / cols;
        let x = self.at(src).0;
        let out = self.alloc(n);
        self.dispatch(
            K::SoftmaxRows,
            &[Some(x), Some(out)],
            None,
            &[cols as u32, rows as u32],
            groups(rows * 256),
        );
        self.node(out, n, Op::SoftmaxRows { src, cols })
    }
    fn gate_col(&mut self, x: usize, g: usize, k: usize, gcols: usize, cols: usize) -> usize {
        let n = self.nodes[x].len;
        assert_eq!(n % cols, 0);
        assert_eq!(self.nodes[g].len, n / cols * gcols);
        assert!(k < gcols);
        let (xv, gv) = (self.at(x).0, self.at(g).0);
        let out = self.alloc(n);
        let p = [n as u32, cols as u32, gcols as u32, k as u32];
        self.dispatch(K::GateCol, &[Some(xv), Some(gv), Some(out)], None, &p, groups(n));
        self.node(out, n, Op::GateCol { x, g, k, gcols, cols })
    }
    fn scan(&mut self, z: usize, c: usize, seq: usize, cols: usize) -> usize {
        let n = self.nodes[z].len;
        assert_eq!(self.nodes[c].len, n);
        assert_eq!(n % cols, 0);
        let seq = seq.max(1);
        let rows = n / cols;
        let (zv, cv) = (self.at(z).0, self.at(c).0);
        let out = self.alloc(n);
        let threads = scan_threads(rows, seq, cols);
        self.dispatch(
            K::Scan,
            &[Some(zv), Some(cv), Some(out)],
            None,
            &[threads as u32, cols as u32, seq as u32, rows as u32],
            groups(threads),
        );
        self.node(out, n, Op::Scan { z, c, seq, cols })
    }
    fn slot_scores(&mut self, q: usize, m: usize, slots: usize, width: usize, scale: f32) -> usize {
        let rows = self.nodes[q].len / width;
        assert_eq!(self.nodes[m].len, rows * slots * width);
        let (qv, mv) = (self.at(q).0, self.at(m).0);
        let out = self.slot_kernel(K::SlotScores, qv, mv, rows * slots, slots, width, scale);
        self.node(out, rows * slots, Op::SlotScores { q, m, slots, width, scale })
    }
    fn slot_read(&mut self, p: usize, m: usize, slots: usize, width: usize) -> usize {
        let rows = self.nodes[p].len / slots;
        assert_eq!(self.nodes[m].len, rows * slots * width);
        let (pv, mv) = (self.at(p).0, self.at(m).0);
        let out = self.slot_kernel(K::SlotRead, pv, mv, rows * width, slots, width, 1.);
        self.node(out, rows * width, Op::SlotRead { p, m, slots, width })
    }
    fn outer(&mut self, a: usize, b: usize, slots: usize, width: usize) -> usize {
        let rows = self.nodes[a].len / slots;
        assert_eq!(self.nodes[b].len, rows * width);
        let (av, bv) = (self.at(a).0, self.at(b).0);
        let out = self.slot_kernel(K::Outer, av, bv, rows * slots * width, slots, width, 1.);
        self.node(out, rows * slots * width, Op::Outer { a, b, slots, width })
    }
}

// ---- the model on the GPU ----------------------------------------------------------------

/// A model's weights, gradients and AdamW state on the GPU. With the optimizer state in system
/// RAM ([`GpuModel::upload_with`]) the GPU holds only the weights and gradients: the GPU
/// computes each gradient, the CPU applies AdamW to the model's own moments, and the new
/// weights go back to the GPU.
pub struct GpuModel {
    gpu: Arc<Gpu>,
    n: usize,
    weights: wgpu::Buffer,
    grads: wgpu::Buffer,
    moments: wgpu::Buffer,
    variance: wgpu::Buffer,
    flags: wgpu::Buffer,
    partial: wgpu::Buffer,
    /// AdamW steps taken (the model's `step`).
    pub step: u64,
    pool: Mutex<Pool>,
    last_upload_bytes: std::sync::atomic::AtomicU64,
    /// The AdamW moments are the CPU model's (`moments`/`variance` are placeholders).
    optimizer_in_ram: bool,
    /// The GPU memory training may use in all (model buffers and working buffers).
    budget: Option<u64>,
}

/// What a step fitted to memory did ([`GpuModel::fit_step`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FitStats {
    pub loss: f32,
    /// Bytes moved between the GPU and system RAM for offloading (gradient down, weights up).
    pub transfer_bytes: u64,
    pub transfer_seconds: f64,
}

fn focus_mask(f: Focus) -> u32 {
    match f {
        Focus::Joint => 7,
        Focus::Left => 1,
        Focus::Right => 2,
        Focus::Interconnect => 4,
        Focus::DenseAndInterconnect => 6,
    }
}

impl GpuModel {
    /// Copies a model (weights, AdamW state and step) to the GPU.
    pub fn upload(gpu: &Arc<Gpu>, model: &Bicameral) -> Result<GpuModel, String> {
        GpuModel::upload_with(gpu, model, false)
    }

    /// Copies a model to the GPU; with `optimizer_in_ram` its AdamW state stays in system RAM
    /// (in `model`) and only the weights and gradients take GPU memory.
    pub fn upload_with(
        gpu: &Arc<Gpu>,
        model: &Bicameral,
        optimizer_in_ram: bool,
    ) -> Result<GpuModel, String> {
        let n = model.weights.len();
        if n > gpu.max_floats() {
            return Err(format!(
                "The model ({n} parameters) does not fit in one buffer on {} (at most {})",
                gpu.name(),
                gpu.max_floats()
            ));
        }
        let c = &model.config;
        if c.heads == 0 || c.d % c.heads != 0 || c.d / c.heads > MAX_HEAD_DIM || (c.d / c.heads) % 2 != 0 {
            return Err(format!("GPU attention needs an even head size of at most {MAX_HEAD_DIM}"));
        }
        let mut flags = vec![0u32; n];
        for s in &model.slots {
            let group = match s.group {
                Focus::Left => 1,
                Focus::Right => 2,
                Focus::Interconnect => 4,
                _ => 0,
            };
            let f = group | if s.decay { 8 } else { 0 };
            flags[s.range.clone()].iter_mut().for_each(|x| *x = f);
        }
        let flags = gpu.upload_with_usage(bytemuck::cast_slice(&flags), "flags", wgpu::BufferUsages::STORAGE);
        let (gx, gy, _) = groups(n);
        let (moments, variance) = if optimizer_in_ram {
            (gpu.storage(4, "momentum (in system RAM)"), gpu.storage(4, "variance (in system RAM)"))
        } else {
            (gpu.upload_f32(&model.momentum, "momentum"), gpu.upload_f32(&model.variance, "variance"))
        };
        let m = GpuModel {
            n,
            weights: gpu.upload_f32(&model.weights, "weights"),
            grads: gpu.storage(n, "gradients"),
            moments,
            variance,
            flags,
            partial: gpu.storage((gx * gy) as usize + 1, "gradient norm"),
            step: model.step,
            gpu: gpu.clone(),
            pool: Mutex::new(Pool::default()),
            last_upload_bytes: Default::default(),
            optimizer_in_ram,
            budget: None,
        };
        // Creating the model's buffers may have failed (out of GPU memory).
        gpu.check()?;
        Ok(m)
    }

    /// The AdamW state is in system RAM (the CPU applies each step).
    pub fn optimizer_in_ram(&self) -> bool {
        self.optimizer_in_ram
    }

    /// GPU memory of the model's own buffers: weights, gradients, parameter flags and, unless
    /// it is in system RAM, the AdamW state; with it in RAM, the staging buffer for reading
    /// gradients back.
    pub fn fixed_bytes(&self) -> u64 {
        GpuModel::fixed_bytes_for(self.n, self.optimizer_in_ram)
    }

    /// [`GpuModel::fixed_bytes`] for a model of `n` parameters.
    pub fn fixed_bytes_for(n: usize, optimizer_in_ram: bool) -> u64 {
        let n = n as u64 * 4;
        let partial = (n / 4).div_ceil(256) * 4 + 4;
        if optimizer_in_ram {
            3 * n + partial + (READ_CHUNK as u64 * 4).min(n)
        } else {
            5 * n + partial
        }
    }

    /// Limits the GPU memory this model's training uses in all (`None`: no limit beyond the
    /// GPU's own). A step that would need more fails with [`OVER_BUDGET`] before anything runs.
    pub fn set_budget(&mut self, total: Option<u64>) {
        self.budget = total;
        let working = total.map(|t| t.saturating_sub(self.fixed_bytes()));
        self.pool.lock().unwrap().budget = working;
    }

    pub fn budget(&self) -> Option<u64> {
        self.budget
    }

    /// Releases every working buffer kept for reuse.
    pub fn clear_pool(&self) {
        let mut p = self.pool.lock().unwrap();
        *p = Pool::with_budget(p.budget);
    }

    /// Moves the AdamW state to system RAM: the model (weights, moments, step) is brought up to
    /// date from the GPU, and the GPU's copies of the moments are released.
    pub fn move_optimizer_to_ram(&mut self, model: &mut Bicameral) -> Result<(), String> {
        if self.optimizer_in_ram {
            return Ok(());
        }
        self.download(model)?;
        self.moments = self.gpu.storage(4, "momentum (in system RAM)");
        self.variance = self.gpu.storage(4, "variance (in system RAM)");
        self.optimizer_in_ram = true;
        self.clear_pool();
        self.set_budget(self.budget);
        Ok(())
    }

    /// Runs `f` with the working-buffer pool; after an error the pool is emptied (a failed
    /// step may leave buffers that are not returned, or invalid after an out-of-memory error).
    fn with_pool<T>(&self, f: impl FnOnce(&mut Pool) -> Result<T, String>) -> Result<T, String> {
        let mut pool = std::mem::take(&mut *self.pool.lock().unwrap());
        let result = f(&mut pool);
        if result.is_err() {
            pool = Pool::with_budget(pool.budget);
        }
        *self.pool.lock().unwrap() = pool;
        result
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    /// Copies the weights, AdamW state and step back into `model`.
    pub fn download(&self, model: &mut Bicameral) -> Result<(), String> {
        if model.weights.len() != self.n {
            return Err("The model's shape changed since it was copied to the GPU".into());
        }
        model.weights = self.gpu.read(&self.weights, self.n)?;
        if !self.optimizer_in_ram {
            model.momentum = self.gpu.read(&self.moments, self.n)?;
            model.variance = self.gpu.read(&self.variance, self.n)?;
        }
        model.step = self.step;
        Ok(())
    }

    fn buffer<'a>(&'a self, g: &'a GpuGraph, b: B) -> &'a wgpu::Buffer {
        match b {
            B::Weights => &self.weights,
            B::Grads => &self.grads,
            B::Moments => &self.moments,
            B::Variance => &self.variance,
            B::Flags => &self.flags,
            B::Partial => &self.partial,
            B::Step(i) => &g.bufs[i],
        }
    }

    /// Records the forward pass, the loss and the backward pass of a batch (rows padded to the
    /// longest). Returns the graph, the per-row loss buffer and the number of rows.
    #[allow(clippy::too_many_arguments)]
    fn record<'a>(
        &'a self,
        pool: &'a mut Pool,
        model: &Bicameral,
        rows: &[Row],
        abl: Ablation,
        denom: Option<f32>,
        gradients: bool,
        zero_grads: bool,
    ) -> Result<(GpuGraph<'a>, B, usize), String> {
        self.record_with(pool, model, rows, abl, denom, gradients, zero_grads, None)
    }

    /// [`GpuModel::record`] with the greedy check on (the number of valid tokens and the margin;
    /// its per-token results are in the graph's `hits`).
    #[allow(clippy::too_many_arguments)]
    fn record_with<'a>(
        &'a self,
        pool: &'a mut Pool,
        model: &Bicameral,
        rows: &[Row],
        abl: Ablation,
        denom: Option<f32>,
        gradients: bool,
        zero_grads: bool,
        greedy: Option<(u32, f32)>,
    ) -> Result<(GpuGraph<'a>, B, usize), String> {
        if rows.is_empty() {
            return Err("Empty training batch".into());
        }
        if model.weights.len() != self.n {
            return Err("The model's shape changed since it was copied to the GPU".into());
        }
        let c = &model.config;
        let t = rows.iter().map(|r| r.ids.len()).max().unwrap_or(1).max(1);
        if t > c.max_tokens
            || rows.iter().any(|r| r.ids.len() != r.targets.len() || r.ids.len() != r.mask.len())
        {
            return Err("Invalid training rows".into());
        }
        // Character features feed Bicameral's Dense hemisphere; Nova Neuro does not use them.
        let neuro = model.neuro.is_some();
        let width = if neuro { 0 } else { c.window_chars * 16 };
        let feats: Vec<Result<Vec<f32>, String>> = std::thread::scope(|s| {
            let hs: Vec<_> = rows
                .iter()
                .map(|r| s.spawn(move || if neuro { Ok(Vec::new()) } else { model.features(&r.ids) }))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("feature thread failed".into()))).collect()
        });
        let total = rows.len() * t;
        let (mut ids, mut targets, mut mask) =
            (Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total));
        let mut features = Vec::with_capacity(total * width);
        for (r, f) in rows.iter().zip(feats) {
            if r.targets.iter().any(|&x| x as usize >= c.vocab) {
                return Err("Invalid target token".into());
            }
            let pad = t - r.ids.len();
            ids.extend(r.ids.iter().copied().chain(std::iter::repeat(0).take(pad)));
            targets.extend(r.targets.iter().copied().chain(std::iter::repeat(0).take(pad)));
            mask.extend(r.mask.iter().copied().chain(std::iter::repeat(0.).take(pad)));
            features.extend(f?.into_iter().chain(std::iter::repeat(0.).take(pad * width)));
        }
        let denom = denom.unwrap_or_else(|| mask.iter().map(|&m| m as f64).sum::<f64>().max(1.) as f32);
        let mut g = GpuGraph::new(&self.gpu, pool, t);
        g.greedy = greedy;
        if gradients && zero_grads {
            g.fill(B::Grads, 0, self.n, 0.);
        }
        let h = model.forward_hidden(&mut g, &ids, features, t, abl);
        let tb = g.upload(bytemuck::cast_slice(&targets).to_vec(), targets.len());
        let mb = g.upload(bytemuck::cast_slice(&mask).to_vec(), mask.len());
        let embed = g.parameter("embed", model.slot_range("embed"));
        let bias = g.parameter("out_bias", model.slot_range("out_bias"));
        g.forward_only = !gradients;
        let losses = g.head_xent(h, embed, bias, c.d, c.vocab, tb, mb, denom);
        if gradients {
            g.backward();
        }
        if g.pool.over_budget {
            g.finish();
            return Err(format!(
                "This batch ({} rows of up to {t} tokens) is {OVER_BUDGET}; nothing was run",
                rows.len()
            ));
        }
        Ok((g, losses, total))
    }

    fn run(&self, g: &mut GpuGraph) {
        // Without profiling nothing in `run_timed` can fail.
        let _ = self.run_timed(g, None);
    }

    /// Submits the recorded step. With `times`, every dispatch's GPU time is measured (with GPU
    /// timestamps when the driver has them, otherwise by running dispatches one at a time) and
    /// appended as (kernel, milliseconds).
    fn run_timed(&self, g: &mut GpuGraph, times: Option<&mut Vec<(K, f64)>>) -> Result<(), String> {
        let gpu = &*self.gpu;
        for (i, bytes) in std::mem::take(&mut g.uploads) {
            gpu.queue.write_buffer(&g.bufs[i], 0, &bytes);
        }
        let stride = gpu.align;
        let mut params = vec![0u8; (g.ops.len().max(1) as u64 * stride) as usize];
        for (i, d) in g.ops.iter().enumerate() {
            let at = i * stride as usize;
            params[at..at + PARAM_BYTES as usize].copy_from_slice(bytemuck::cast_slice(&d.params));
        }
        let uniform = gpu.upload_with_usage(&params, "kernel parameters", wgpu::BufferUsages::UNIFORM);
        let groups: Vec<wgpu::BindGroup> = g
            .ops
            .iter()
            .map(|d| {
                let mut entries = vec![wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: 0,
                        size: wgpu::BufferSize::new(PARAM_BYTES),
                    }),
                }];
                for (slot, b) in d.f.iter().enumerate() {
                    let buf = b.map(|b| self.buffer(g, b)).unwrap_or(&gpu.dummies[slot]);
                    entries.push(wgpu::BindGroupEntry {
                        binding: slot as u32 + 1,
                        resource: buf.as_entire_binding(),
                    });
                }
                let ub = d.u.map(|b| self.buffer(g, b)).unwrap_or(&gpu.dummies[F32_SLOTS]);
                entries.push(wgpu::BindGroupEntry { binding: 8, resource: ub.as_entire_binding() });
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &gpu.layout,
                    entries: &entries,
                })
            })
            .collect();
        let result = match times {
            None => {
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    for (i, (d, bg)) in g.ops.iter().zip(&groups).enumerate() {
                        pass.set_pipeline(&gpu.pipelines[&d.k]);
                        pass.set_bind_group(0, bg, &[(i as u64 * stride) as u32]);
                        pass.dispatch_workgroups(d.groups.0, d.groups.1, d.groups.2);
                    }
                }
                gpu.queue.submit([enc.finish()]);
                Ok(())
            }
            Some(times) if gpu.timestamps => {
                // One pass per dispatch, its start and end time written by the GPU.
                let period = gpu.queue.get_timestamp_period() as f64;
                let mut result = Ok(());
                for (c, chunk) in g.ops.chunks(2048).enumerate() {
                    let count = (chunk.len() * 2) as u32;
                    let set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("kernel times"),
                        ty: wgpu::QueryType::Timestamp,
                        count,
                    });
                    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("kernel times"),
                        size: count as u64 * 8,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                    let mut enc = gpu.device.create_command_encoder(&Default::default());
                    for (j, d) in chunk.iter().enumerate() {
                        let i = c * 2048 + j;
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                                query_set: &set,
                                beginning_of_pass_write_index: Some(2 * j as u32),
                                end_of_pass_write_index: Some(2 * j as u32 + 1),
                            }),
                        });
                        pass.set_pipeline(&gpu.pipelines[&d.k]);
                        pass.set_bind_group(0, &groups[i], &[(i as u64 * stride) as u32]);
                        pass.dispatch_workgroups(d.groups.0, d.groups.1, d.groups.2);
                    }
                    enc.resolve_query_set(&set, 0..count, &resolve, 0);
                    gpu.queue.submit([enc.finish()]);
                    match gpu.read_bytes(&resolve, count as u64 * 8) {
                        Ok(bytes) => {
                            let ticks: &[u64] = bytemuck::cast_slice(&bytes);
                            for (j, d) in chunk.iter().enumerate() {
                                let ns = ticks[2 * j + 1].saturating_sub(ticks[2 * j]) as f64 * period;
                                times.push((d.k, ns / 1e6));
                            }
                        }
                        Err(e) => result = Err(e),
                    }
                }
                result
            }
            Some(times) => {
                // No GPU timestamps: one dispatch at a time, timed from submission to completion
                // (includes the submission cost, so small kernels look slower than they are).
                let mut result = Ok(());
                for (i, (d, bg)) in g.ops.iter().zip(&groups).enumerate() {
                    let mut enc = gpu.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&gpu.pipelines[&d.k]);
                        pass.set_bind_group(0, bg, &[(i as u64 * stride) as u32]);
                        pass.dispatch_workgroups(d.groups.0, d.groups.1, d.groups.2);
                    }
                    let t = std::time::Instant::now();
                    gpu.queue.submit([enc.finish()]);
                    if let Err(e) = gpu.device.poll(wgpu::PollType::wait_indefinitely()) {
                        result = Err(format!("GPU: {e}"));
                        break;
                    }
                    times.push((d.k, t.elapsed().as_secs_f64() * 1e3));
                }
                result
            }
        };
        g.ops.clear();
        result
    }

    /// Mean loss and gradient of a batch (no update): the same as
    /// `learn::batch_loss_and_gradients` on the CPU.
    pub fn loss_and_gradients(
        &self,
        model: &Bicameral,
        rows: &[Row],
        abl: Ablation,
    ) -> Result<(f32, Vec<f32>), String> {
        let result = self.with_pool(|pool| {
            let (mut g, losses, total) = self.record(pool, model, rows, abl, None, true, true)?;
            self.run(&mut g);
            let loss = self.gpu.read(self.buffer(&g, losses), total)?.iter().map(|&x| x as f64).sum::<f64>();
            let grads = self.gpu.read(&self.grads, self.n)?;
            g.finish();
            Ok((loss as f32, grads))
        });
        result
    }

    /// One training step: the batch's mean loss and gradient (as
    /// `learn::batch_loss_and_gradients`), then AdamW on the parameters `focus` selects (as
    /// `Bicameral::adam_step`). Returns the loss.
    pub fn train_step(
        &mut self,
        model: &Bicameral,
        rows: &[Row],
        abl: Ablation,
        lr: f32,
        focus: Focus,
    ) -> Result<f32, String> {
        self.step_with(model, rows, abl, lr, focus, None).map(|(loss, _)| loss)
    }

    /// One training step like [`GpuModel::train_step`], measured: where the time goes (CPU
    /// preparation, GPU time of every kernel) and how much GPU memory training uses. The step
    /// is a real step (the weights change).
    pub fn profile_step(
        &mut self,
        model: &Bicameral,
        rows: &[Row],
        abl: Ablation,
        lr: f32,
        focus: Focus,
    ) -> Result<StepProfile, String> {
        let mut p = StepProfile::default();
        let (loss, _) = self.step_with(model, rows, abl, lr, focus, Some(&mut p))?;
        p.loss = loss;
        p.memory_bytes = self.memory_bytes();
        Ok(p)
    }

    fn step_with(
        &mut self,
        model: &Bicameral,
        rows: &[Row],
        abl: Ablation,
        lr: f32,
        focus: Focus,
        profile: Option<&mut StepProfile>,
    ) -> Result<(f32, ()), String> {
        if !lr.is_finite() || lr <= 0. {
            return Err("Invalid learning rate".into());
        }
        if self.optimizer_in_ram {
            return Err("The optimizer state is in system RAM: use GpuModel::fit_step".into());
        }
        let step = self.step + 1;
        let result = self.with_pool(|pool| {
            let started = std::time::Instant::now();
            let (mut g, losses, total) = self.record(pool, model, rows, abl, None, true, true)?;
            self.record_adam(&mut g, lr, focus, step);
            let recorded = started.elapsed();
            let upload_bytes: u64 = g.upload_bytes;
            self.last_upload_bytes.store(upload_bytes, std::sync::atomic::Ordering::Relaxed);
            match profile {
                None => self.run(&mut g),
                Some(prof) => {
                    let mut times = Vec::new();
                    let t = std::time::Instant::now();
                    self.run_timed(&mut g, Some(&mut times))?;
                    prof.dispatches = times.len();
                    let mut by: Vec<(K, usize, f64)> = Vec::new();
                    for (k, ms) in times {
                        match by.iter_mut().find(|e| e.0 == k) {
                            Some(e) => {
                                e.1 += 1;
                                e.2 += ms;
                            }
                            None => by.push((k, 1, ms)),
                        }
                    }
                    by.sort_by(|a, b| b.2.total_cmp(&a.2));
                    prof.gpu_ms = by.iter().map(|e| e.2).sum();
                    prof.kernels = by
                        .into_iter()
                        .map(|(k, count, ms)| KernelTime { kernel: kernel_name(k).into(), count, ms })
                        .collect();
                    prof.cpu_prepare_ms = recorded.as_secs_f64() * 1e3;
                    prof.timed_run_ms = t.elapsed().as_secs_f64() * 1e3;
                    prof.timestamps = self.gpu.timestamps;
                    prof.upload_bytes = upload_bytes;
                    prof.tokens = rows.iter().map(|r| r.ids.len()).sum();
                }
            }
            let loss = self.gpu.read(self.buffer(&g, losses), total)?.iter().map(|&x| x as f64).sum::<f64>();
            g.finish();
            Ok(loss as f32)
        });
        let result = result.map(|l| (l, ()));
        if result.is_ok() {
            self.step = step;
        }
        result
    }

    /// Records AdamW (with clipping by the norm of the focused groups' gradient) as step `step`.
    fn record_adam(&self, g: &mut GpuGraph, lr: f32, focus: Focus, step: u64) {
        let (gx, gy, _) = groups(self.n);
        let parts = (gx * gy) as usize;
        // Clipped by the norm of the gradients this step applies (the focused groups).
        g.dispatch(
            K::Sumsq,
            &[Some(B::Grads), Some(B::Partial)],
            Some(B::Flags),
            &[self.n as u32, focus_mask(focus)],
            (gx, gy, 1),
        );
        g.dispatch(K::NormFinish, &[None, Some(B::Partial)], None, &[parts as u32], (1, 1, 1));
        let (beta1, beta2) = (0.9f32, 0.95f32);
        let p = [
            self.n as u32,
            focus_mask(focus),
            parts as u32,
            0,
            bits(lr),
            bits(1. - beta1.powf(step as f32)),
            bits(1. - beta2.powf(step as f32)),
            bits(beta1),
            bits(beta2),
        ];
        g.dispatch(
            K::Adam,
            &[Some(B::Weights), Some(B::Grads), Some(B::Moments), Some(B::Variance), Some(B::Partial)],
            Some(B::Flags),
            &p,
            (gx, gy, 1),
        );
    }

    /// The batch's gradient into the GPU's gradient buffer, `micro` rows at a time (each part
    /// divided by the whole batch's weight, so the parts add up to the batch's mean, as
    /// `learn::batch_loss_and_gradients`). Returns the batch's loss. Nothing is applied: if
    /// any part fails (out of memory, over the budget), the weights are unchanged.
    fn accumulate(
        &self,
        model: &Bicameral,
        rows: &[Row],
        abl: Ablation,
        micro: usize,
    ) -> Result<f32, String> {
        let denom = crate::learn::batch_weight(rows) as f32;
        let micro = micro.max(1);
        self.with_pool(|pool| {
            let mut loss = 0f64;
            let mut uploaded = 0;
            for (i, part) in rows.chunks(micro).enumerate() {
                // Dropout masks follow the row's place in the whole batch.
                let (mut g, losses, total) =
                    self.record(pool, model, part, abl.at_row(i * micro), Some(denom), true, i == 0)?;
                uploaded += g.upload_bytes;
                self.run(&mut g);
                loss += self.gpu.read(self.buffer(&g, losses), total)?.iter().map(|&x| x as f64).sum::<f64>();
                g.finish();
            }
            self.last_upload_bytes.store(uploaded, std::sync::atomic::Ordering::Relaxed);
            Ok(loss as f32)
        })
    }

    /// One training step fitted to memory: the gradient in parts of `micro_rows` rows
    /// (gradient accumulation: the same step as the whole batch at once), then AdamW on the
    /// GPU, or, with the optimizer state in system RAM, on the CPU (the gradient comes down,
    /// the new weights go up; `model` is then up to date). An error before the update leaves
    /// everything unchanged (`self.step` and `model.step` tell whether a step was applied).
    pub fn fit_step(
        &mut self,
        model: &mut Bicameral,
        rows: &[Row],
        abl: Ablation,
        lr: f32,
        focus: Focus,
        micro_rows: usize,
    ) -> Result<FitStats, String> {
        if !lr.is_finite() || lr <= 0. {
            return Err("Invalid learning rate".into());
        }
        if rows.is_empty() {
            return Err("Empty training batch".into());
        }
        let micro = micro_rows.clamp(1, rows.len());
        if !self.optimizer_in_ram && micro == rows.len() {
            // All at once: one submission with AdamW included (the fastest).
            let loss = self.train_step(model, rows, abl, lr, focus)?;
            return Ok(FitStats { loss, ..FitStats::default() });
        }
        let loss = self.accumulate(model, rows, abl, micro)?;
        if !self.optimizer_in_ram {
            let step = self.step + 1;
            self.with_pool(|pool| {
                let mut g = GpuGraph::new(&self.gpu, pool, 1);
                self.record_adam(&mut g, lr, focus, step);
                self.run(&mut g);
                g.finish();
                self.gpu.check()
            })?;
            self.step = step;
            return Ok(FitStats { loss, ..FitStats::default() });
        }
        let t = std::time::Instant::now();
        let grads = self.gpu.read_chunked(&self.grads, self.n)?;
        let down = t.elapsed().as_secs_f64();
        model.adam_step(&grads, lr, focus)?;
        let t = std::time::Instant::now();
        self.set_weights(model)?;
        // Wait for the upload, so its time is measured and an error is seen now.
        self.gpu.queue.submit([]);
        self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU: {e}"))?;
        self.gpu.check()?;
        Ok(FitStats {
            loss,
            transfer_bytes: 2 * self.n as u64 * 4,
            transfer_seconds: down + t.elapsed().as_secs_f64(),
        })
    }

    /// The loss and gradient of part of a batch, divided by `denom` (the whole batch's total
    /// weight), so the parts of a batch add up to its mean (hybrid CPU + GPU training). No
    /// update is made.
    pub fn part_loss_and_gradients(
        &self,
        model: &Bicameral,
        rows: &[Row],
        abl: Ablation,
        denom: f32,
    ) -> Result<(f32, Vec<f32>), String> {
        let result = self.with_pool(|pool| {
            let (mut g, losses, total) = self.record(pool, model, rows, abl, Some(denom), true, true)?;
            self.run(&mut g);
            let loss = self.gpu.read(self.buffer(&g, losses), total)?.iter().map(|&x| x as f64).sum::<f64>();
            let grads = self.gpu.read(&self.grads, self.n)?;
            g.finish();
            Ok((loss as f32, grads))
        });
        result
    }

    /// The mean loss of rows with the weights on the GPU, exactly as
    /// `learn::batch_loss` computes it, without gradients (held-out checks during GPU training).
    /// Rows are computed `chunk` at a time.
    pub fn mean_loss(
        &self,
        model: &Bicameral,
        rows: &[Row],
        abl: Ablation,
        chunk: usize,
    ) -> Result<f32, String> {
        if rows.is_empty() {
            return Ok(0.);
        }
        // Each row's loss enters with its weight, divided by the total weight of all rows.
        let denom = rows.iter().map(|r| r.mask.iter().map(|&m| m as f64).sum::<f64>()).sum::<f64>().max(1.);
        let result = self.with_pool(|pool| {
            let mut total = 0f64;
            for part in rows.chunks(chunk.max(1)) {
                let (mut g, losses, n) =
                    self.record(pool, model, part, abl, Some(denom as f32), false, false)?;
                self.run(&mut g);
                total += self.gpu.read(self.buffer(&g, losses), n)?.iter().map(|&x| x as f64).sum::<f64>();
                g.finish();
            }
            Ok(total as f32)
        });
        result
    }

    /// Each row's own mean loss (its weighted loss divided by max(its total weight, 1), as
    /// `Bicameral::loss` gives it), computed `chunk` rows at a time.
    pub fn row_losses(
        &self,
        model: &Bicameral,
        rows: &[Row],
        abl: Ablation,
        chunk: usize,
    ) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(rows.len());
        self.with_pool(|pool| {
            for part in rows.chunks(chunk.max(1)) {
                let t = part.iter().map(|r| r.ids.len()).max().unwrap_or(1);
                let (mut g, losses, n) = self.record(pool, model, part, abl, Some(1.), false, false)?;
                self.run(&mut g);
                let per_token = self.gpu.read(self.buffer(&g, losses), n)?;
                g.finish();
                for (k, r) in part.iter().enumerate() {
                    let sum: f64 = per_token[k * t..(k + 1) * t].iter().map(|&x| x as f64).sum();
                    let weight: f64 = r.mask.iter().map(|&m| m as f64).sum();
                    out.push((sum / weight.max(1.)) as f32);
                }
            }
            Ok(())
        })?;
        Ok(out)
    }

    /// Greedy check, `chunk` rows at a time: for each row, whether at every weighted position
    /// its target is the clear choice of greedy writing (the most likely of the model's valid
    /// tokens, every other one below it by more than `margin` in logits, so that the small
    /// differences between GPU and CPU arithmetic cannot change the choice). A row whose
    /// weighted targets are an answer followed by a line break is then exactly what greedy
    /// writing gives after the row's prompt.
    pub fn greedy_hits(
        &self,
        model: &Bicameral,
        rows: &[Row],
        margin: f32,
        chunk: usize,
    ) -> Result<Vec<bool>, String> {
        let valid = (128 + model.tokenizer.words().len()).min(model.config.vocab) as u32;
        let mut out = Vec::with_capacity(rows.len());
        self.with_pool(|pool| {
            for part in rows.chunks(chunk.max(1)) {
                let t = part.iter().map(|r| r.ids.len()).max().unwrap_or(1);
                let (mut g, _, n) = self.record_with(
                    pool,
                    model,
                    part,
                    Ablation::default(),
                    Some(1.),
                    false,
                    false,
                    Some((valid, margin)),
                )?;
                self.run(&mut g);
                let hits = g.hits.ok_or("The greedy check was not recorded")?;
                let hits = self.gpu.read(self.buffer(&g, hits), n)?;
                g.finish();
                for (k, r) in part.iter().enumerate() {
                    out.push(r.mask.iter().enumerate().all(|(i, &m)| m <= 0. || hits[k * t + i] > 0.5));
                }
            }
            Ok(())
        })?;
        Ok(out)
    }

    /// Replaces the weights on the GPU with the model's (hybrid training: the CPU takes the
    /// optimizer step).
    pub fn set_weights(&mut self, model: &Bicameral) -> Result<(), String> {
        if model.weights.len() != self.n {
            return Err("The model's shape changed since it was copied to the GPU".into());
        }
        self.gpu.queue.write_buffer(&self.weights, 0, bytemuck::cast_slice(&model.weights));
        self.step = model.step;
        Ok(())
    }

    /// GPU memory this model's training uses: weights, gradients, AdamW state, and the step's
    /// working buffers (kept between steps for reuse), plus the last step's input data.
    pub fn memory_bytes(&self) -> u64 {
        self.fixed_bytes() + self.pool.lock().unwrap().held_bytes()
    }

    /// The working buffers the pool accounts for (equal to those it holds between steps).
    pub fn working_bytes(&self) -> u64 {
        self.pool.lock().unwrap().owned
    }
}
