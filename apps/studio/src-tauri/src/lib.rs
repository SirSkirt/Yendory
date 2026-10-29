//! Yendory Studio backend: thin Tauri commands over the Yendory Engine. All neural work
//! (teaching, replies, exams, measurements) is done by `yendory_engine`; this file only manages
//! files, the models library and the teaching thread.
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::{DialogExt, FilePath};
use tauri_plugin_fs::{FsExt, OpenOptions};
use yendory_engine::fingerprint::sha256_hex;
use yendory_engine::learn::course::{CourseEvent, CourseFile, FileOutcome, FileRecord};
use yendory_engine::learn::eval::{evaluate, reply, Report};
use yendory_engine::learn::session::{EducationStage, Event, RetryPolicy, SessionConfig};
use yendory_engine::learn::{batch_loss_and_gradients, Row};
use yendory_engine::learn::{parse_lesson, Item, Kind};
use yendory_engine::{
    size_presets, Ablation, Bicameral, Config, Focus, Measurements, ThinkOptions, ThinkingTree, TokenId,
};

mod chat;
mod conversation;
mod courses;
mod designer;
mod filter;
mod growth;
mod memory;
mod neuro;
mod practice;
mod reading;
mod reflection;
mod review;
mod rewards;
mod sources;
mod study;
mod update;

const MAX_LESSON_BYTES: usize = 8 << 20;
/// Large enough for the largest model the Engine accepts (1 billion parameters × 12 bytes,
/// plus the header and word list).
const MAX_CHECKPOINT_BYTES: usize = if usize::BITS >= 64 { 12_200_000_000u64 as usize } else { usize::MAX };

type Res<T> = Result<T, String>;

/// A token count as people read it: 160M, 2.5M, 400K.
pub(crate) fn tokens_text(n: u64) -> String {
    let short = |x: f64, unit: &str| {
        if (x - x.round()).abs() < 0.05 {
            format!("{:.0}{unit}", x.round())
        } else {
            format!("{x:.1}{unit}")
        }
    };
    match n {
        n if n >= 1_000_000 => short(n as f64 / 1e6, "M"),
        n if n >= 1_000 => short(n as f64 / 1e3, "K"),
        n => n.to_string(),
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------
// Models library: <app data>/models/<id>/{model.nnova2, meta.json, archive.json, history.jsonl}.
// model.nnova2 holds either format: its header says Bicameral (.nnova2) or Nova Neuro (.nneuro).
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaughtFile {
    name: String,
    sha256: String,
    outcome: FileOutcome,
    end_step: u64,
    time: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Meta {
    id: String,
    name: String,
    created: u64,
    updated: u64,
    step: u64,
    parameters: usize,
    /// Transformer (left), dense (right), interconnect.
    groups: [usize; 3],
    words: usize,
    /// Word slots the model has allocated (vocab − 128).
    #[serde(default)]
    slots: usize,
    /// Vocabulary limit: slots are added up to this many words as they are learned. `None`:
    /// the allocated slots are the limit (Nova-2's fixed 256).
    #[serde(default)]
    word_limit: Option<usize>,
    /// Context length in tokens.
    #[serde(default)]
    context: usize,
    #[serde(default)]
    taught: Vec<TaughtFile>,
    /// Where the model came from: "new", or the imported file's name.
    #[serde(default)]
    origin: String,
    /// Corpora this model has read (pre-training), oldest first.
    #[serde(default)]
    read: Vec<reading::ReadRecord>,
    /// Conversation stage runs, oldest first.
    #[serde(default)]
    conversation: Vec<conversation::ConversationRecord>,
    /// "bicameral" or "neuro" (Nova Neuro, experimental).
    #[serde(default = "bicameral")]
    architecture: String,
    /// Nova Neuro: the modules and their sizes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    design: Option<yendory_engine::model::NeuroSpec>,
    /// Transformer blocks (0: a Nova Neuro brain of other networks only; unknown in older files).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    blocks: Option<usize>,
    /// A restored checkpoint's interrupted Classroom run, which the next run can continue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resume: Option<yendory_engine::learn::session::CheckpointInfo>,
    /// Network growth: the setting, the original size and every growth (docs/GROWTH.md).
    #[serde(default)]
    growth: growth::GrowthInfo,
}
fn bicameral() -> String {
    "bicameral".into()
}

struct Active {
    meta: Meta,
    model: Bicameral,
    archive: Vec<Item>,
}

#[derive(Default)]
struct Studio {
    active: Mutex<Option<Active>>,
    lessons: Mutex<Vec<CourseFile>>,
    /// Teaching lessons or reading a corpus (one training run at a time).
    teaching: AtomicBool,
    stop: Arc<AtomicBool>,
    /// The download running and the ones waiting their turn (one runs at a time).
    downloads: Mutex<reading::DownloadQueue>,
    cancel_download: Arc<AtomicBool>,
    /// A chat reply is being written (one at a time), and the request to stop it.
    chatting: AtomicBool,
    chat_stop: Arc<AtomicBool>,
}

fn models_dir(app: &AppHandle) -> Res<PathBuf> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join("models");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Accepted checkpoints of a training session that has not finished (see
/// `yendory_engine::learn::session::save_checkpoint`).
fn checkpoints_dir(app: &AppHandle, id: &str) -> Res<PathBuf> {
    Ok(model_dir(app, id)?.join("checkpoints"))
}

/// The model was saved: its checkpoints are older, so they go.
fn clear_checkpoints(app: &AppHandle, id: &str) {
    if let Ok(dir) = checkpoints_dir(app, id) {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Training must not start while checkpoints of an unfinished session wait to be restored or
/// discarded (a new session's checkpoints would mix with them).
fn no_waiting_checkpoints(app: &AppHandle, id: &str) -> Res<()> {
    let dir = checkpoints_dir(app, id)?;
    if yendory_engine::learn::session::checkpoint_files(&dir).is_empty() {
        Ok(())
    } else {
        Err("This model has checkpoints from a training session that did not finish. Restore the newest or discard them first (§ 1 Models).".into())
    }
}

fn model_dir(app: &AppHandle, id: &str) -> Res<PathBuf> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err("Invalid model id".into());
    }
    Ok(models_dir(app)?.join(id))
}

fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> Res<()> {
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    drop(f);
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

fn refresh_meta(meta: &mut Meta, model: &Bicameral) {
    meta.step = model.step;
    meta.parameters = model.parameter_count();
    meta.groups = [
        model.group_parameters(Focus::Left),
        model.group_parameters(Focus::Right),
        model.group_parameters(Focus::Interconnect),
    ];
    meta.words = model.tokenizer.words().len();
    meta.slots = model.word_capacity();
    meta.context = model.config.max_tokens;
    meta.architecture = model.architecture().into();
    meta.design = model.neuro.clone();
    meta.blocks = Some(model.config.blocks);
    meta.updated = now();
}

fn save_all(app: &AppHandle, meta: &Meta, model: &Bicameral, archive: &[Item]) -> Res<()> {
    save_in(&model_dir(app, &meta.id)?, meta, model, archive)
}

/// Saves a model, its lesson archive and meta.json into its folder (each file atomically).
fn save_in(dir: &std::path::Path, meta: &Meta, model: &Bicameral, archive: &[Item]) -> Res<()> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    write_atomic(&dir.join("model.nnova2"), &model.to_bytes()?)?;
    write_atomic(&dir.join("archive.json"), &serde_json::to_vec(archive).map_err(|e| e.to_string())?)?;
    write_atomic(&dir.join("meta.json"), &serde_json::to_vec_pretty(meta).map_err(|e| e.to_string())?)
}

fn load_meta(app: &AppHandle, id: &str) -> Res<Meta> {
    let bytes = std::fs::read(model_dir(app, id)?.join("meta.json")).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

fn append_history(app: &AppHandle, id: &str, value: serde_json::Value) {
    if let Ok(dir) = model_dir(app, id) {
        let _ = yendory_engine::learn::append_jsonl(&dir.join("history.jsonl"), &value);
    }
}

fn check_name(name: &str) -> Res<String> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 60 || name.chars().any(char::is_control) {
        return Err("Give the model a name of 1 to 60 characters".into());
    }
    Ok(name.to_string())
}

fn new_id() -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("m{nanos:x}")
}

fn install(app: &AppHandle, studio: &Studio, name: &str, model: Bicameral, origin: &str) -> Res<Meta> {
    let mut meta = Meta {
        id: new_id(),
        name: check_name(name)?,
        created: now(),
        updated: now(),
        step: 0,
        parameters: 0,
        groups: [0; 3],
        words: 0,
        taught: Vec::new(),
        origin: origin.into(),
        read: Vec::new(),
        conversation: Vec::new(),
        slots: 0,
        word_limit: None,
        context: 0,
        architecture: bicameral(),
        design: None,
        blocks: None,
        resume: None,
        growth: Default::default(),
    };
    refresh_meta(&mut meta, &model);
    save_all(app, &meta, &model, &[])?;
    *studio.active.lock().unwrap() = Some(Active { meta: meta.clone(), model, archive: Vec::new() });
    Ok(meta)
}

/// Runs training work so that a crash inside it (a panic) becomes an error the app reports,
/// instead of a thread that ends silently and leaves the app waiting.
fn guarded<T>(work: impl FnOnce() -> Res<T>) -> Res<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_else(|p| {
        let text = p
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown error".into());
        Err(format!("Training stopped unexpectedly: {text}. The last saved model is unchanged."))
    })
}

fn not_teaching(studio: &Studio) -> Res<()> {
    if studio.teaching.load(Ordering::Acquire) {
        return Err("Wait until teaching has finished (or stop it) first".into());
    }
    Ok(())
}

fn with_active<T>(studio: &Studio, f: impl FnOnce(&Active) -> Res<T>) -> Res<T> {
    let guard = studio.active.lock().unwrap();
    f(guard.as_ref().ok_or("Open or create a model first")?)
}

#[tauri::command]
fn list_models(app: AppHandle) -> Res<Vec<Meta>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(models_dir(&app)?).map_err(|e| e.to_string())? {
        let id = entry.map_err(|e| e.to_string())?.file_name().to_string_lossy().to_string();
        if let Ok(meta) = load_meta(&app, &id) {
            out.push(meta);
        }
    }
    out.sort_by_key(|m| std::cmp::Reverse(m.updated));
    Ok(out)
}

#[tauri::command]
fn create_model(
    app: AppHandle,
    studio: State<Studio>,
    name: String,
    size: Option<String>,
    word_limit: Option<usize>,
    context: Option<usize>,
) -> Res<Meta> {
    not_teaching(&studio)?;
    let mut config = match size.as_deref() {
        None => Config::default(),
        Some(id) => size_presets().into_iter().find(|p| p.id == id).ok_or("Unknown model size")?.config,
    };
    if let Some(c) = context {
        config.max_tokens = c;
        Bicameral::check_config(&config)
            .map_err(|_| format!("Choose a context of 1 to {} tokens", yendory_engine::model::MAX_CONTEXT))?;
    }
    if let Some(limit) = word_limit {
        check_word_limit(&config, limit)?;
        // No slots yet: one is added for each word the model learns, up to the limit.
        config.vocab = yendory_engine::tokenizer::BASE;
    }
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1) | 1;
    let meta = install(&app, &studio, &name, Bicameral::with_config(config, seed), "new")?;
    if word_limit.is_none() {
        return Ok(meta);
    }
    set_limit(&app, &studio, &meta.id, word_limit)
}

/// A vocabulary limit must leave the model within the Engine's limits when every slot is used.
fn check_word_limit(config: &Config, limit: usize) -> Res<()> {
    let full = Config { vocab: yendory_engine::tokenizer::BASE + limit, ..config.clone() };
    Bicameral::check_config(&full).map_err(|_| {
        let base =
            Bicameral::parameters_for(&Config { vocab: yendory_engine::tokenizer::BASE, ..config.clone() });
        let most = (yendory_engine::model::MAX_PARAMETERS.saturating_sub(base)) / (config.d + 1);
        format!("At this size the vocabulary can hold up to {most} words (1 billion parameters)")
    })
}

fn set_limit(app: &AppHandle, studio: &Studio, id: &str, limit: Option<usize>) -> Res<Meta> {
    let mut guard = studio.active.lock().unwrap();
    let active = guard.as_mut().filter(|a| a.meta.id == id).ok_or("Open the model first")?;
    if let Some(l) = limit {
        if l < active.model.tokenizer.words().len() {
            return Err(format!("The model already knows {} words", active.model.tokenizer.words().len()));
        }
        check_word_limit(&active.model.config, l)?;
    }
    active.meta.word_limit = limit;
    let dir = model_dir(app, id)?;
    write_atomic(
        &dir.join("meta.json"),
        &serde_json::to_vec_pretty(&active.meta).map_err(|e| e.to_string())?,
    )?;
    Ok(active.meta.clone())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelCost {
    /// Parameters, file size and memory once every word up to the vocabulary limit is learned
    /// (training rows as long as the context).
    parameters: usize,
    file_bytes: u64,
    chat_bytes: u64,
    reading_bytes: u64,
    teaching_bytes: u64,
    /// Parameters each learned word adds (one embedding row and one output bias).
    per_word: usize,
    /// The largest vocabulary limit this size allows.
    max_words: usize,
    /// Activation memory of one training row as long as the context, and threads training.
    row_bytes: u64,
    threads: usize,
}

/// What a size, a vocabulary limit (`None`: Nova-2's fixed 256 slots) and a context cost.
#[tauri::command]
fn model_cost(size: String, limit: Option<usize>, context: usize) -> Res<ModelCost> {
    let mut config = size_presets().into_iter().find(|p| p.id == size).ok_or("Unknown model size")?.config;
    config.max_tokens = context.clamp(1, yendory_engine::model::MAX_CONTEXT);
    let base =
        Bicameral::parameters_for(&Config { vocab: yendory_engine::tokenizer::BASE, ..config.clone() });
    let per_word = config.d + 1;
    let max_words = yendory_engine::model::MAX_PARAMETERS.saturating_sub(base) / per_word;
    let words = limit.unwrap_or(config.vocab - yendory_engine::tokenizer::BASE);
    let parameters = base + words.min(max_words) * per_word;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let rb = row_bytes(&config);
    let (file_bytes, chat_bytes, reading_bytes, teaching_bytes) = memory(parameters, threads, rb);
    Ok(ModelCost {
        parameters,
        file_bytes,
        chat_bytes,
        reading_bytes,
        teaching_bytes,
        per_word,
        max_words,
        row_bytes: rb,
        threads,
    })
}

/// Changes the open model's context length. No weight changes (positions are encoded by
/// rotation); the model uses a longer context well once it has trained on text that long.
#[tauri::command]
fn set_context(app: AppHandle, studio: State<Studio>, id: String, context: usize) -> Res<Meta> {
    not_teaching(&studio)?;
    let mut guard = studio.active.lock().unwrap();
    let active = guard.as_mut().filter(|a| a.meta.id == id).ok_or("Open the model first")?;
    let model = active
        .model
        .with_context(context)
        .map_err(|_| format!("Choose a context of 1 to {} tokens", yendory_engine::model::MAX_CONTEXT))?;
    let mut meta = active.meta.clone();
    refresh_meta(&mut meta, &model);
    save_all(&app, &meta, &model, &active.archive)?;
    active.model = model;
    active.meta = meta.clone();
    Ok(meta)
}

/// Training speed of one size at one context length on this device: one row per thread, each
/// as long as the context (attention compares every token with every earlier one, so the cost
/// per token grows with the context).
#[tauri::command]
async fn measure_context(studio: State<'_, Studio>, size: String, context: usize) -> Res<f64> {
    not_teaching(&studio)?;
    let mut config = size_presets().into_iter().find(|p| p.id == size).ok_or("Unknown model size")?.config;
    config.max_tokens = context;
    Bicameral::check_config(&config)?;
    tauri::async_runtime::spawn_blocking(move || -> Res<f64> {
        let m = Bicameral::with_config(config, 1);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let mut text = String::new();
        let mut i = 0;
        while text.len() < context + 1 {
            text.push_str(&format!("U: Where is key {i}?\nN: Key {i} is in box {}.\n", i % 7));
            i += 1;
        }
        let ids = m.tokenizer.encode(&text)?;
        let rows: Vec<Row> = (0..threads)
            .map(|r| Row {
                ids: ids[r..r + context].to_vec(),
                targets: ids[r + 1..r + context + 1].to_vec(),
                mask: vec![1.; context],
            })
            .collect();
        let t = std::time::Instant::now();
        batch_loss_and_gradients(&m, &rows, Ablation::default())?;
        Ok((threads * context) as f64 / t.elapsed().as_secs_f64())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Changes the open model's vocabulary limit (`None`: back to its allocated slots).
#[tauri::command]
fn set_word_limit(app: AppHandle, studio: State<Studio>, id: String, limit: Option<usize>) -> Res<Meta> {
    not_teaching(&studio)?;
    set_limit(&app, &studio, &id, limit)
}

// ---------------------------------------------------------------------------------------------
// Model sizes: memory computed from the size, speed measured on this device
// ---------------------------------------------------------------------------------------------

/// Training and generation speed measured on this device for one size.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Speed {
    /// Training tokens per second (one batch of 8 × 64 tokens, all threads).
    training: f64,
    /// Generated tokens per second (KV-cached greedy replies).
    generation: f64,
    threads: usize,
    time: u64,
    /// Training tokens per second on the GPU (batches of 16 rows of 128 tokens, as reading
    /// uses), when there is one.
    #[serde(default)]
    gpu_training: Option<f64>,
    #[serde(default)]
    gpu_name: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// Training device: the GPU (when there is one) or the CPU
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DeviceOption {
    /// "cpu", "gpu-dx12", "gpu-vulkan", "hybrid-vulkan", ...
    id: String,
    label: String,
    /// The hardware it uses, or why it cannot be used.
    detail: String,
    available: bool,
    experimental: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DeviceInfo {
    /// The setting: "auto" or one of the options' ids.
    choice: String,
    options: Vec<DeviceOption>,
    /// Where training runs now (an option's id) and its description.
    active: String,
    active_name: String,
    /// Why the choice could not be followed (e.g. Vulkan unavailable), if so.
    note: Option<String>,
    /// The last training benchmark, if any.
    benchmark: Option<serde_json::Value>,
    /// Kept for older frontends: "gpu" or "cpu", and the GPU's name or why there is none.
    training_on: String,
    gpu: Option<String>,
    gpu_error: Option<String>,
    /// CPU threads chosen (0: all) and the processor's total.
    cpu_threads: usize,
    cpu_total: usize,
    /// GPU checks (`yendory_engine::learn::set_gpu_checks`): on by default.
    gpu_checks: bool,
    /// Reply limits fitted to each answer (`yendory_engine::learn::eval::set_fitted_replies`).
    fitted_replies: bool,
}

fn settings_file(app: &AppHandle) -> Res<PathBuf> {
    Ok(app.path().app_data_dir().map_err(|e| e.to_string())?.join("settings.json"))
}

fn read_settings(app: &AppHandle) -> serde_json::Map<String, serde_json::Value> {
    settings_file(app)
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

fn write_setting(app: &AppHandle, key: &str, value: serde_json::Value) -> Res<()> {
    // One writer at a time, so two settings saved together both stay.
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut settings = read_settings(app);
    settings.insert(key.into(), value);
    write_atomic(&settings_file(app)?, &serde_json::to_vec_pretty(&settings).map_err(|e| e.to_string())?)
}

/// Form settings the pages remember between runs (only these keys; other settings have their
/// own commands).
const PAGE_SETTINGS: &[&str] = &["reading", "classroom", "conversation"];

#[tauri::command]
fn page_settings(app: AppHandle, page: String) -> Option<serde_json::Value> {
    PAGE_SETTINGS.contains(&page.as_str()).then(|| read_settings(&app).get(&page).cloned()).flatten()
}

#[tauri::command]
fn save_page_settings(app: AppHandle, page: String, settings: serde_json::Value) -> Res<()> {
    if !PAGE_SETTINGS.contains(&page.as_str()) || !settings.is_object() {
        return Err("Unknown page settings".into());
    }
    write_setting(&app, &page, settings)
}

fn device_choice(app: &AppHandle) -> String {
    read_settings(app)
        .get("trainingDevice")
        .and_then(|d| d.as_str())
        .map(String::from)
        .unwrap_or_else(|| "auto".into())
}

fn thread_choice(app: &AppHandle) -> usize {
    read_settings(app).get("cpuThreads").and_then(|d| d.as_u64()).unwrap_or(0) as usize
}

fn checks_choice(app: &AppHandle) -> bool {
    read_settings(app).get("gpuChecks").and_then(|d| d.as_bool()).unwrap_or(true)
}

fn fitted_choice(app: &AppHandle) -> bool {
    read_settings(app).get("fittedReplies").and_then(|d| d.as_bool()).unwrap_or(true)
}

fn stored_benchmark(app: &AppHandle) -> Option<serde_json::Value> {
    read_settings(app).get("benchmark").cloned()
}

use yendory_engine::gpu::Api;
use yendory_engine::learn::Device;

fn device_id(d: Device) -> String {
    let api = |a: Api| match a {
        Api::Auto => "",
        Api::Dx12 => "-dx12",
        Api::Vulkan => "-vulkan",
        Api::Metal => "-metal",
    };
    match d {
        Device::Cpu => "cpu".into(),
        Device::Gpu(a) => format!("gpu{}", api(a)),
        Device::Hybrid(a) => format!("hybrid{}", api(a)),
    }
}

fn parse_device(id: &str) -> Option<Device> {
    let api = |rest: &str| match rest {
        "" => Some(Api::Auto),
        "-dx12" => Some(Api::Dx12),
        "-vulkan" => Some(Api::Vulkan),
        "-metal" => Some(Api::Metal),
        _ => None,
    };
    if id == "cpu" {
        Some(Device::Cpu)
    } else if let Some(r) = id.strip_prefix("gpu") {
        api(r).map(Device::Gpu)
    } else if let Some(r) = id.strip_prefix("hybrid") {
        api(r).map(Device::Hybrid)
    } else {
        None
    }
}

fn device_label(d: Device) -> String {
    match d {
        Device::Cpu => "CPU".into(),
        Device::Gpu(Api::Auto) => "GPU".into(),
        Device::Gpu(a) => format!("GPU — {}", a.name()),
        Device::Hybrid(Api::Auto) => "Hybrid — CPU + GPU (experimental)".into(),
        Device::Hybrid(a) => format!("Hybrid — CPU + GPU, {} (experimental)", a.name()),
    }
}

/// Applies a device choice ("auto": the last benchmark's fastest, else the GPU when there is
/// one, else the CPU) and reports every option and where training runs.
fn apply_device(
    choice: &str,
    threads: usize,
    checks: bool,
    benchmark: Option<serde_json::Value>,
) -> DeviceInfo {
    use yendory_engine::learn::bench::{candidates, device_name};
    yendory_engine::learn::set_cpu_threads(threads);
    yendory_engine::learn::set_gpu_checks(checks);
    let fitted = yendory_engine::learn::eval::fitted_replies();
    let options: Vec<DeviceOption> = candidates()
        .into_iter()
        .map(|d| {
            let name = device_name(d);
            DeviceOption {
                id: device_id(d),
                label: device_label(d),
                available: name.is_ok(),
                detail: name.unwrap_or_else(|e| e),
                experimental: matches!(d, Device::Hybrid(_)),
            }
        })
        .collect();
    let available = |d: Device| options.iter().any(|o| o.id == device_id(d) && o.available);
    let gpu = yendory_engine::learn::gpu_name();
    let mut note = None;
    let wanted = if choice == "auto" {
        let fastest = benchmark
            .as_ref()
            .and_then(|b| b.get("result")?.get("fastest").cloned())
            .and_then(|f| serde_json::from_value::<Device>(f).ok())
            .filter(|&d| available(d));
        fastest.unwrap_or(if gpu.is_ok() { Device::Gpu(Api::Auto) } else { Device::Cpu })
    } else {
        parse_device(choice).unwrap_or(Device::Cpu)
    };
    let active = match wanted {
        Device::Cpu => Device::Cpu,
        d if available(d) || (d == Device::Gpu(Api::Auto) && gpu.is_ok()) => d,
        d => {
            let why = yendory_engine::learn::bench::device_name(d).err().unwrap_or_default();
            note = Some(format!("{} cannot be used ({why}); training runs on the CPU.", device_label(d)));
            Device::Cpu
        }
    };
    yendory_engine::learn::set_device(active);
    let active_name = yendory_engine::learn::bench::device_name(active).unwrap_or_else(|_| "CPU".into());
    DeviceInfo {
        choice: choice.into(),
        options,
        active: device_id(active),
        active_name,
        note,
        benchmark,
        training_on: if active == Device::Cpu { "cpu".into() } else { "gpu".into() },
        gpu: gpu.as_ref().ok().cloned(),
        gpu_error: gpu.err(),
        cpu_threads: threads,
        cpu_total: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        gpu_checks: checks,
        fitted_replies: fitted,
    }
}

#[tauri::command]
async fn training_device(app: AppHandle) -> Res<DeviceInfo> {
    yendory_engine::learn::eval::set_fitted_replies(fitted_choice(&app));
    let (choice, threads, bench) = (device_choice(&app), thread_choice(&app), stored_benchmark(&app));
    let checks = checks_choice(&app);
    tauri::async_runtime::spawn_blocking(move || apply_device(&choice, threads, checks, bench))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn set_training_device(app: AppHandle, studio: State<'_, Studio>, choice: String) -> Res<DeviceInfo> {
    not_teaching(&studio)?;
    if choice != "auto" && parse_device(&choice).is_none() {
        return Err("Unknown training device".into());
    }
    write_setting(&app, "trainingDevice", choice.clone().into())?;
    let (threads, bench, checks) = (thread_choice(&app), stored_benchmark(&app), checks_choice(&app));
    tauri::async_runtime::spawn_blocking(move || apply_device(&choice, threads, checks, bench))
        .await
        .map_err(|e| e.to_string())
}

/// Limits the CPU threads Yendory uses (0: all), e.g. to keep some free for other work.
#[tauri::command]
async fn set_cpu_threads(app: AppHandle, studio: State<'_, Studio>, threads: usize) -> Res<DeviceInfo> {
    not_teaching(&studio)?;
    write_setting(&app, "cpuThreads", threads.into())?;
    let (choice, bench, checks) = (device_choice(&app), stored_benchmark(&app), checks_choice(&app));
    tauri::async_runtime::spawn_blocking(move || apply_device(&choice, threads, checks, bench))
        .await
        .map_err(|e| e.to_string())
}

/// Reply limits fitted to each answer on or off (see `yendory_engine::learn::eval::reply_budget`).
#[tauri::command]
async fn set_fitted_replies(app: AppHandle, studio: State<'_, Studio>, on: bool) -> Res<DeviceInfo> {
    not_teaching(&studio)?;
    write_setting(&app, "fittedReplies", on.into())?;
    yendory_engine::learn::eval::set_fitted_replies(on);
    let (choice, threads, bench) = (device_choice(&app), thread_choice(&app), stored_benchmark(&app));
    let checks = checks_choice(&app);
    tauri::async_runtime::spawn_blocking(move || apply_device(&choice, threads, checks, bench))
        .await
        .map_err(|e| e.to_string())
}

/// GPU checks on or off (see `yendory_engine::learn::set_gpu_checks`).
#[tauri::command]
async fn set_gpu_checks(app: AppHandle, studio: State<'_, Studio>, on: bool) -> Res<DeviceInfo> {
    not_teaching(&studio)?;
    write_setting(&app, "gpuChecks", on.into())?;
    let (choice, threads, bench) = (device_choice(&app), thread_choice(&app), stored_benchmark(&app));
    tauri::async_runtime::spawn_blocking(move || apply_device(&choice, threads, on, bench))
        .await
        .map_err(|e| e.to_string())
}

/// Benchmarks training of the open model on every device with identical batches (the model is
/// not changed), saves the result and returns the device information (Auto then follows the
/// fastest device that computes the same loss as the CPU).
#[tauri::command]
async fn benchmark_training(
    app: AppHandle,
    studio: State<'_, Studio>,
    rows: usize,
    tokens: usize,
    steps: usize,
) -> Res<DeviceInfo> {
    if !(1..=128).contains(&rows)
        || !(16..=yendory_engine::model::MAX_CONTEXT).contains(&tokens)
        || !(1..=20).contains(&steps)
    {
        return Err("Choose 1-128 rows, at least 16 tokens per row and 1-20 steps".into());
    }
    let (meta, model) = with_active(&studio, |a| Ok((a.meta.clone(), a.model.clone())))?;
    if studio.teaching.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("Wait until teaching or reading has finished".into());
    }
    let tokens = tokens.min(model.config.max_tokens);
    let app2 = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let batch = yendory_engine::learn::bench::sample_rows(&model, rows, tokens);
        let devices = yendory_engine::learn::bench::candidates();
        yendory_engine::learn::bench::run(&model, &batch, steps, &devices, &mut |d| {
            let _ = app2.emit("benchmark-progress", d.to_string());
        })
    })
    .await;
    studio.teaching.store(false, Ordering::Release);
    let result = result.map_err(|e| e.to_string())?;
    let record = serde_json::json!({
        "time": now(),
        "model": meta.name,
        "parameters": model_parameters(&studio),
        "rows": rows,
        "tokens": tokens,
        "result": result,
    });
    write_setting(&app, "benchmark", record.clone())?;
    let (choice, threads, checks) = (device_choice(&app), thread_choice(&app), checks_choice(&app));
    tauri::async_runtime::spawn_blocking(move || apply_device(&choice, threads, checks, Some(record)))
        .await
        .map_err(|e| e.to_string())
}

fn model_parameters(studio: &Studio) -> usize {
    with_active(studio, |a| Ok(a.model.parameter_count())).unwrap_or(0)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SizeInfo {
    id: &'static str,
    name: &'static str,
    parameters: usize,
    /// Transformer (left), dense (right), interconnect.
    groups: [usize; 3],
    /// `.nnova2` file: weights and both Adam moments (12 bytes per parameter).
    file_bytes: u64,
    /// Estimated memory while talking, reading a corpus, and teaching lessons.
    chat_bytes: u64,
    reading_bytes: u64,
    teaching_bytes: u64,
    measured: Option<Speed>,
}

fn speed_file(app: &AppHandle) -> Res<PathBuf> {
    Ok(app.path().app_data_dir().map_err(|e| e.to_string())?.join("speed.json"))
}

fn load_speeds(app: &AppHandle) -> std::collections::BTreeMap<String, Speed> {
    speed_file(app)
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

/// Memory estimates from the parameter count `p`: a model copy (weights + Adam moments) is 12p
/// bytes and a gradient buffer 4p. Counted from the code paths, including short-lived copies:
/// talking keeps 2 copies (the open model and the one answering); reading keeps up to 6 (open,
/// reading thread, working, best, the serialised checkpoint being saved, the open model being
/// replaced); teaching up to 10 (those plus the course, lesson-start, accepted and candidate
/// models). Training adds one gradient buffer per thread and two for the batch, and each
/// thread's activations for one training row (`row_bytes`, from
/// `Bicameral::activation_floats_per_token` × the row length). Checked: a 15M model reading
/// peaked at 1.57 GB with 4 threads.
fn memory(p: usize, threads: usize, row_bytes: u64) -> (u64, u64, u64, u64) {
    let (copy, grad) = (12 * p as u64, 4 * p as u64);
    let training = grad * (threads as u64 + 2) + row_bytes * threads as u64;
    (copy, 2 * copy, 6 * copy + training, 10 * copy + training)
}

/// Activation bytes of one training row as long as the context.
fn row_bytes(config: &Config) -> u64 {
    4 * Bicameral::activation_floats_per_token(config) as u64 * config.max_tokens as u64
}

#[tauri::command]
fn model_sizes(app: AppHandle) -> Vec<SizeInfo> {
    let speeds = load_speeds(&app);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    size_presets()
        .into_iter()
        .map(|p| {
            let m = Bicameral::with_config(p.config, 1);
            let n = m.parameter_count();
            let (file_bytes, chat_bytes, reading_bytes, teaching_bytes) =
                memory(n, threads, row_bytes(&Config::default()));
            SizeInfo {
                id: p.id,
                name: p.name,
                parameters: n,
                groups: [
                    m.group_parameters(Focus::Left),
                    m.group_parameters(Focus::Right),
                    m.group_parameters(Focus::Interconnect),
                ],
                file_bytes,
                chat_bytes,
                reading_bytes,
                teaching_bytes,
                measured: speeds.get(p.id).cloned(),
            }
        })
        .collect()
}

/// Measures training and generation speed of one size on this device with a temporary model.
#[tauri::command]
async fn measure_speed(app: AppHandle, studio: State<'_, Studio>, size: String) -> Res<Speed> {
    not_teaching(&studio)?;
    let preset = size_presets().into_iter().find(|p| p.id == size).ok_or("Unknown model size")?;
    let speed = tauri::async_runtime::spawn_blocking(move || -> Res<Speed> {
        let m = Bicameral::with_config(preset.config, 1);
        let text = "U: Where is the key?\nN: The key is in the box under the old table.\n".repeat(3);
        let ids = m.tokenizer.encode(&text)?;
        let n = 64.min(ids.len() - 1);
        let rows: Vec<Row> = (0..8)
            .map(|r| Row {
                ids: ids[r..r + n].to_vec(),
                targets: ids[r + 1..r + n + 1].to_vec(),
                mask: vec![1.; n],
            })
            .collect();
        batch_loss_and_gradients(&m, &rows[..1], Ablation::default())?;
        let t = std::time::Instant::now();
        batch_loss_and_gradients(&m, &rows, Ablation::default())?;
        let training = (rows.len() * n) as f64 / t.elapsed().as_secs_f64();
        let prompt = m.tokenizer.encode("U: Where is the key?\nN: ")?;
        let t = std::time::Instant::now();
        let mut generated = 0;
        while t.elapsed().as_secs_f64() < 0.5 {
            generated += m.generate_cached_ids(&prompt, 24, Ablation::default())?.len();
        }
        let generation = generated as f64 / t.elapsed().as_secs_f64();
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        // GPU: batches of 16 rows of 128 tokens (reading's default), after one warm-up step.
        let (gpu_training, gpu_name) = match yendory_engine::gpu::shared() {
            Ok(gpu) => {
                let long = m.tokenizer.encode(&text.repeat(8))?;
                let n = 128.min(long.len() - 17);
                let rows: Vec<Row> = (0..16)
                    .map(|r| Row {
                        ids: long[r..r + n].to_vec(),
                        targets: long[r + 1..r + n + 1].to_vec(),
                        mask: vec![1.; n],
                    })
                    .collect();
                let mut g = yendory_engine::gpu::GpuModel::upload(&gpu, &m)?;
                g.train_step(&m, &rows, Ablation::default(), 1e-4, Focus::Joint)?;
                let t = std::time::Instant::now();
                let mut steps = 0;
                while steps < 3 || t.elapsed().as_secs_f64() < 1. {
                    g.train_step(&m, &rows, Ablation::default(), 1e-4, Focus::Joint)?;
                    steps += 1;
                }
                (Some((steps * rows.len() * n) as f64 / t.elapsed().as_secs_f64()), Some(gpu.name()))
            }
            Err(_) => (None, None),
        };
        Ok(Speed { training, generation, threads, time: now(), gpu_training, gpu_name })
    })
    .await
    .map_err(|e| e.to_string())??;
    let mut speeds = load_speeds(&app);
    speeds.insert(size, speed.clone());
    write_atomic(&speed_file(&app)?, &serde_json::to_vec_pretty(&speeds).map_err(|e| e.to_string())?)?;
    Ok(speed)
}

#[tauri::command]
fn open_model(app: AppHandle, studio: State<Studio>, id: String) -> Res<Meta> {
    not_teaching(&studio)?;
    let dir = model_dir(&app, &id)?;
    let mut meta = load_meta(&app, &id)?;
    let model = Bicameral::load(dir.join("model.nnova2"))?;
    meta.slots = model.word_capacity();
    meta.context = model.config.max_tokens;
    // The file decides the size (it can differ from meta.json if saving was interrupted).
    meta.parameters = model.parameter_count();
    meta.groups = [
        model.group_parameters(Focus::Left),
        model.group_parameters(Focus::Right),
        model.group_parameters(Focus::Interconnect),
    ];
    meta.design = model.neuro.clone();
    meta.blocks = Some(model.config.blocks);
    let archive: Vec<Item> = match std::fs::read(dir.join("archive.json")) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| e.to_string())?,
        Err(_) => Vec::new(),
    };
    *studio.active.lock().unwrap() = Some(Active { meta: meta.clone(), model, archive });
    Ok(meta)
}

#[tauri::command]
fn rename_model(app: AppHandle, studio: State<Studio>, id: String, name: String) -> Res<Meta> {
    let mut meta = load_meta(&app, &id)?;
    meta.name = check_name(&name)?;
    let dir = model_dir(&app, &id)?;
    write_atomic(&dir.join("meta.json"), &serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?)?;
    if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == id) {
        a.meta.name = meta.name.clone();
    }
    Ok(meta)
}

/// An accepted checkpoint waiting in the open model's checkpoint folder.
#[derive(Serialize)]
struct CheckpointView {
    name: String,
    step: u64,
    epoch: u32,
    /// Where the session was (epochs, learning rate), when recorded.
    info: Option<yendory_engine::learn::session::CheckpointInfo>,
    bytes: u64,
    /// Seconds since 1970 when it was written.
    saved: u64,
}

fn checkpoint_views(app: &AppHandle, id: &str) -> Res<Vec<CheckpointView>> {
    let dir = checkpoints_dir(app, id)?;
    let mut out = Vec::new();
    for path in yendory_engine::learn::session::checkpoint_files(&dir).into_iter().rev() {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
        // accepted-step-<step>-epoch-<epoch>.nnova2
        let parts: Vec<&str> = name.trim_end_matches(".nnova2").split('-').collect();
        let number = |key: &str| {
            parts
                .iter()
                .position(|p| *p == key)
                .and_then(|i| parts.get(i + 1))
                .and_then(|v| v.parse::<u64>().ok())
        };
        let meta = std::fs::metadata(&path).ok();
        let info = yendory_engine::learn::session::checkpoint_info(&path);
        out.push(CheckpointView {
            info,
            step: number("step").unwrap_or(0),
            epoch: number("epoch").unwrap_or(0) as u32,
            bytes: meta.as_ref().map_or(0, |m| m.len()),
            saved: meta
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs()),
            name,
        });
    }
    Ok(out)
}

/// Checkpoints of the open model's unfinished training session, newest first (empty when the
/// last session ended normally).
#[tauri::command]
fn model_checkpoints(app: AppHandle, studio: State<Studio>) -> Res<Vec<CheckpointView>> {
    let id = with_active(&studio, |a| Ok(a.meta.id.clone()))?;
    checkpoint_views(&app, &id)
}

/// Makes a waiting checkpoint the open model: the current model file is kept as
/// `model-before-restore.nnova2`, the checkpoint becomes `model.nnova2`, and the checkpoint
/// folder is cleared.
#[tauri::command]
fn restore_checkpoint(app: AppHandle, studio: State<Studio>, name: String) -> Res<Meta> {
    not_teaching(&studio)?;
    let (mut meta, archive, current) =
        with_active(&studio, |a| Ok((a.meta.clone(), a.archive.clone(), a.model.clone())))?;
    let dir = checkpoints_dir(&app, &meta.id)?;
    let path = yendory_engine::learn::session::checkpoint_files(&dir)
        .into_iter()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(name.as_str()))
        .ok_or("That checkpoint is no longer there")?;
    let model = Bicameral::load(&path).map_err(|e| format!("The checkpoint cannot be read: {e}"))?;
    let info = yendory_engine::learn::session::checkpoint_info(&path);
    if model.architecture() != current.architecture() {
        return Err("That checkpoint belongs to a model of another architecture".into());
    }
    let model_path = model_dir(&app, &meta.id)?;
    write_atomic(&model_path.join("model-before-restore.nnova2"), &current.to_bytes()?)?;
    refresh_meta(&mut meta, &model);
    meta.resume = info.filter(|i| i.epoch < i.epochs);
    save_all(&app, &meta, &model, &archive)?;
    clear_checkpoints(&app, &meta.id);
    append_history(
        &app,
        &meta.id,
        serde_json::json!({"type": "checkpoint_restored", "time": now(), "file": name, "step": model.step}),
    );
    if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == meta.id) {
        a.model = model;
        a.meta = meta.clone();
    }
    Ok(meta)
}

/// Removes the open model's waiting checkpoints (the model keeps its last saved state).
#[tauri::command]
fn discard_checkpoints(app: AppHandle, studio: State<Studio>) -> Res<()> {
    not_teaching(&studio)?;
    let id = with_active(&studio, |a| Ok(a.meta.id.clone()))?;
    clear_checkpoints(&app, &id);
    append_history(&app, &id, serde_json::json!({"type": "checkpoints_discarded", "time": now()}));
    Ok(())
}

#[tauri::command]
fn delete_model(app: AppHandle, studio: State<Studio>, id: String) -> Res<()> {
    not_teaching(&studio)?;
    let dir = model_dir(&app, &id)?;
    let mut active = studio.active.lock().unwrap();
    if active.as_ref().is_some_and(|a| a.meta.id == id) {
        *active = None;
    }
    std::fs::remove_dir_all(dir).map_err(|e| e.to_string())
}

fn display_name(path: &FilePath) -> String {
    let raw = path.to_string();
    let last = raw.rsplit(['/', '\\']).next().unwrap_or(&raw).to_string();
    // Android content URIs encode the document path ("primary%3ALessons%2F01.txt").
    let decoded = last.replace("%2F", "/").replace("%3A", ":").replace("%20", " ");
    decoded.rsplit(['/', ':']).next().unwrap_or(&decoded).to_string()
}

fn read_picked(app: &AppHandle, path: &FilePath, limit: usize) -> Res<Vec<u8>> {
    let bytes = app.fs().read(path.clone()).map_err(|e| format!("{}: {e}", display_name(path)))?;
    if bytes.len() > limit {
        return Err(format!("{} is larger than {} MB", display_name(path), limit >> 20));
    }
    Ok(bytes)
}

#[tauri::command]
async fn import_model(app: AppHandle, studio: State<'_, Studio>) -> Res<Option<Meta>> {
    not_teaching(&studio)?;
    let Some(path) = app
        .dialog()
        .file()
        .add_filter("Nova checkpoint (Bicameral .nnova2, Neuro .nneuro)", &["nnova2", "nneuro"])
        .blocking_pick_file()
    else {
        return Ok(None);
    };
    let bytes = read_picked(&app, &path, MAX_CHECKPOINT_BYTES)?;
    let model = Bicameral::from_bytes(&bytes)?;
    let file = display_name(&path);
    let name = file.trim_end_matches(".nnova2").trim_end_matches(".nneuro").to_string();
    install(&app, &studio, if name.trim().is_empty() { "Imported model" } else { &name }, model, &file)
        .map(Some)
}

#[tauri::command]
async fn export_model(app: AppHandle, studio: State<'_, Studio>) -> Res<Option<String>> {
    let (name, step, bytes, neuro) = with_active(&studio, |a| {
        Ok((a.meta.name.clone(), a.model.step, a.model.to_bytes()?, a.model.neuro.is_some()))
    })?;
    let (kind, ext) =
        if neuro { ("Nova Neuro checkpoint", "nneuro") } else { ("Bicameral checkpoint", "nnova2") };
    let safe: String =
        name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
    let Some(path) = app
        .dialog()
        .file()
        .add_filter(kind, &[ext])
        .set_file_name(format!("{safe}-step{step}.{ext}"))
        .blocking_save_file()
    else {
        return Ok(None);
    };
    let mut opts = OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    let mut file = app.fs().open(path.clone(), opts).map_err(|e| e.to_string())?;
    file.write_all(&bytes).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    // Verify what was written can be read back as the same checkpoint.
    let written = app.fs().read(path.clone()).map_err(|e| e.to_string())?;
    if written != bytes {
        return Err("The exported file does not match the model; try another location".into());
    }
    Ok(Some(display_name(&path)))
}

// ---------------------------------------------------------------------------------------------
// Lessons and teaching
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LessonPreview {
    name: String,
    sha256: String,
    bytes: usize,
    train: usize,
    held_out: usize,
    kinds: Vec<(String, usize)>,
    error: Option<String>,
    already_taught: bool,
}

fn preview(file: &CourseFile, taught: &[TaughtFile]) -> LessonPreview {
    let sha256 = sha256_hex(file.text.as_bytes());
    // Only a pass counts: older "learned" records meant "a change was accepted", not mastery.
    let already_taught = taught.iter().any(|t| t.sha256 == sha256 && t.outcome == FileOutcome::Passed);
    let mut p = LessonPreview {
        name: file.name.clone(),
        sha256,
        bytes: file.text.len(),
        train: 0,
        held_out: 0,
        kinds: Vec::new(),
        error: None,
        already_taught,
    };
    match parse_lesson(&file.text, &file.name) {
        Err(e) => p.error = Some(e),
        Ok(items) => {
            p.train = items.iter().filter(|i| !i.held_out).count();
            p.held_out = items.iter().filter(|i| i.held_out).count();
            for kind in
                [Kind::Dialogue, Kind::Understanding, Kind::Conversation, Kind::Reasoning, Kind::Prose]
            {
                let n = items.iter().filter(|i| i.kind == kind).count();
                if n > 0 {
                    p.kinds.push((kind.name().into(), n));
                }
            }
            if items.iter().all(|i| i.kind == Kind::Prose) {
                p.error = Some("No lesson blocks with examples were found".into());
            }
        }
    }
    p
}

fn text_of(name: &str, bytes: Vec<u8>) -> Res<String> {
    String::from_utf8(bytes).map_err(|_| format!("{name}: not a text file (lessons must be ASCII text)"))
}

fn files_from(name: &str, bytes: Vec<u8>) -> Res<Vec<CourseFile>> {
    if !name.to_ascii_lowercase().ends_with(".zip") {
        return Ok(vec![CourseFile { name: name.into(), text: text_of(name, bytes)? }]);
    }
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| format!("{name}: {e}"))?;
    let mut out = Vec::new();
    let mut total = 0usize;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| format!("{name}: {e}"))?;
        let path = entry.name().to_string();
        let base = path.rsplit('/').next().unwrap_or(&path).to_string();
        if entry.is_dir()
            || path.starts_with("__MACOSX/")
            || base.starts_with('.')
            || !base.to_ascii_lowercase().ends_with(".txt")
        {
            continue;
        }
        if entry.size() as usize > MAX_LESSON_BYTES {
            return Err(format!("{base} inside {name} is too large"));
        }
        let mut data = Vec::new();
        entry.by_ref().take(MAX_LESSON_BYTES as u64 + 1).read_to_end(&mut data).map_err(|e| e.to_string())?;
        total += data.len();
        if total > 64 << 20 {
            return Err(format!("{name} unpacks to more than 64 MB of lessons"));
        }
        out.push(CourseFile { name: base.clone(), text: text_of(&base, data)? });
    }
    if out.is_empty() {
        return Err(format!("{name} contains no .txt lesson files"));
    }
    Ok(out)
}

fn lesson_previews(studio: &Studio) -> Vec<LessonPreview> {
    let taught = studio.active.lock().unwrap().as_ref().map(|a| a.meta.taught.clone()).unwrap_or_default();
    studio.lessons.lock().unwrap().iter().map(|f| preview(f, &taught)).collect()
}

#[tauri::command]
async fn add_lessons(app: AppHandle, studio: State<'_, Studio>) -> Res<Vec<LessonPreview>> {
    not_teaching(&studio)?;
    let Some(paths) = app
        .dialog()
        .file()
        .add_filter("Lessons (.txt) or a course (.zip)", &["txt", "zip"])
        .blocking_pick_files()
    else {
        return Ok(lesson_previews(&studio));
    };
    let mut added = Vec::new();
    for path in &paths {
        let name = display_name(path);
        let limit = if name.to_ascii_lowercase().ends_with(".zip") { 64 << 20 } else { MAX_LESSON_BYTES };
        added.extend(files_from(&name, read_picked(&app, path, limit)?)?);
    }
    let mut lessons = studio.lessons.lock().unwrap();
    lessons.extend(added);
    yendory_engine::learn::course::sort_files(&mut lessons);
    drop(lessons);
    Ok(lesson_previews(&studio))
}

/// Adds a built-in course's lessons to the Classroom list.
#[tauri::command]
fn add_builtin_lessons(studio: State<Studio>, id: String) -> Res<Vec<LessonPreview>> {
    not_teaching(&studio)?;
    let (_, files) = courses::builtin_files(&id)?;
    let mut lessons = studio.lessons.lock().unwrap();
    for f in files {
        // The same file twice would be taught twice: an existing copy is replaced.
        lessons.retain(|l| l.name != f.name || l.text != f.text);
        lessons.push(f);
    }
    yendory_engine::learn::course::sort_files(&mut lessons);
    drop(lessons);
    Ok(lesson_previews(&studio))
}

#[tauri::command]
fn lessons(studio: State<Studio>) -> Vec<LessonPreview> {
    lesson_previews(&studio)
}

#[tauri::command]
fn remove_lesson(studio: State<Studio>, index: Option<usize>) -> Res<Vec<LessonPreview>> {
    not_teaching(&studio)?;
    let mut lessons = studio.lessons.lock().unwrap();
    match index {
        Some(i) if i < lessons.len() => {
            lessons.remove(i);
        }
        Some(_) => return Err("No such lesson".into()),
        None => lessons.clear(),
    }
    drop(lessons);
    Ok(lesson_previews(&studio))
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TeachOptions {
    epochs: u32,
    batch_size: usize,
    learning_rate: f32,
    foundation: bool,
    rehearsal: bool,
    protect_answers: bool,
    /// Pass mark in percent of held-out questions answered exactly right.
    #[serde(default = "default_pass_percent")]
    pass_percent: f32,
    #[serde(default = "yes")]
    retries: bool,
    #[serde(default = "yes")]
    requeue: bool,
    /// Epochs in a row without an accepted checkpoint before a lesson attempt ends.
    #[serde(default = "default_flat_epochs")]
    flat_epochs: u32,
    /// Experimental bicameral education stage ("dense", "transformer", "cooperative").
    #[serde(default)]
    education: Option<String>,
    /// For each lesson in the list: taught together with the other chosen lessons in one mixed
    /// session (the default for every lesson), or on its own afterwards.
    #[serde(default)]
    together: Option<Vec<bool>>,
    /// Lessons taught together also train the model's saved earlier lessons in the same mix.
    #[serde(default = "yes")]
    train_earlier: bool,
    /// Keep training through epochs rejected only for earlier lessons, without saving them.
    #[serde(default = "yes")]
    recover_earlier: bool,
    /// Halve the learning rate when the first epochs only make the model worse.
    #[serde(default = "yes")]
    backoff: bool,
    /// Accept the first epoch even if the safeguards reject it.
    #[serde(default)]
    override_first_epoch: bool,
    /// Experimental: epochs of probation (0 = off; see `SessionConfig::probation_epochs`).
    #[serde(default)]
    probation_epochs: u32,
    /// New word units only from the lessons being taught (`SessionConfig::words_from_new_lessons`).
    #[serde(default = "yes")]
    words_from_new_lessons: bool,
    /// Experimental: `SessionConfig::guard_before_growth`.
    #[serde(default)]
    guard_before_growth: bool,
    /// Experimental: 4 earlier-lesson answers of every earlier lesson, judged by net change
    /// (`retention_per_lesson`, `earlier_panel_net`).
    #[serde(default)]
    earlier_every_lesson: bool,
    /// How much the earlier lessons' held-out loss may rise in a session
    /// (`SessionConfig::retention_loss_tolerance`; 0.02 unless changed).
    #[serde(default = "default_earlier_loss_rise")]
    earlier_loss_rise: f32,
    /// Experimental: with the larger earlier-lesson check, judge the earlier lessons by their
    /// answers; their loss only by the 0.25 cap (`SessionConfig::earlier_by_answers`).
    #[serde(default)]
    earlier_by_answers: bool,
    /// Continue the interrupted run a restored checkpoint came from (its epochs, learning rate
    /// and schedule), when there is one.
    #[serde(default = "yes")]
    continue_run: bool,
}

/// One saved earlier lesson in the model's rehearsal archive.
#[derive(Serialize)]
struct SavedLesson {
    name: String,
    units: usize,
    held_out: usize,
}

/// The saved earlier lessons of the open model, in the order they were saved.
#[tauri::command]
fn saved_lessons(studio: State<'_, Studio>) -> Res<Vec<SavedLesson>> {
    with_active(&studio, |a| {
        let mut out: Vec<SavedLesson> = Vec::new();
        for it in &a.archive {
            let i = match out.iter().position(|l| l.name == it.source) {
                Some(i) => i,
                None => {
                    out.push(SavedLesson { name: it.source.clone(), units: 0, held_out: 0 });
                    out.len() - 1
                }
            };
            out[i].units += 1;
            if it.held_out && it.kind != Kind::Prose {
                out[i].held_out += 1;
            }
        }
        Ok(out)
    })
}

/// Removes one saved earlier lesson from the open model's archive (e.g. a lesson renamed or
/// replaced since): it is no longer rehearsed, trained alongside or guarded.
#[tauri::command]
fn forget_saved_lesson(app: AppHandle, studio: State<'_, Studio>, name: String) -> Res<Vec<SavedLesson>> {
    if studio.teaching.load(Ordering::Acquire) {
        return Err("Wait until teaching has finished".into());
    }
    {
        let mut guard = studio.active.lock().unwrap();
        let a = guard.as_mut().ok_or("Open or create a model first")?;
        let before = a.archive.len();
        a.archive.retain(|i| i.source != name);
        if a.archive.len() == before {
            return Err(format!("{name} is not among the saved lessons"));
        }
        save_all(&app, &a.meta, &a.model, &a.archive)?;
        append_history(
            &app,
            &a.meta.id,
            serde_json::json!({"type": "saved_lesson_removed", "time": now(), "name": name}),
        );
    }
    saved_lessons(studio)
}

/// The bicameral education stage named by the frontend.
fn education_stage(name: Option<&str>) -> Res<Option<EducationStage>> {
    Ok(match name {
        None | Some("") | Some("off") => None,
        Some("dense") => Some(EducationStage::DenseKnowledge),
        Some("transformer") => Some(EducationStage::TransformerLanguage),
        Some("cooperative") => Some(EducationStage::Cooperative),
        Some(other) => return Err(format!("Unknown education stage {other}")),
    })
}
/// Batch progress sent to the page at most a few times a second (training runs several batches
/// a second; a page redrawing a group's lessons for each one fell behind and stopped answering).
/// The first and last batch of an epoch always pass, so progress never looks stuck.
#[derive(Default)]
pub(crate) struct BatchThrottle(Option<std::time::Instant>);
impl BatchThrottle {
    pub(crate) fn pass(&mut self, batch: usize, batches: usize) -> bool {
        let now = std::time::Instant::now();
        let due = self.0.is_none_or(|t| now.duration_since(t) >= std::time::Duration::from_millis(250));
        if due || batch <= 1 || batch >= batches {
            self.0 = Some(now);
            true
        } else {
            false
        }
    }
}

fn yes() -> bool {
    true
}
fn default_flat_epochs() -> u32 {
    4
}
fn default_pass_percent() -> f32 {
    50.
}
fn default_earlier_loss_rise() -> f32 {
    0.02
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TeachFinished {
    files: Vec<FileRecord>,
    stopped: bool,
    error: Option<String>,
    model: Option<Meta>,
}

/// The Classroom's session settings from the page's options (checkpoints aside).
fn teach_config(options: &TeachOptions, meta: &Meta) -> Res<SessionConfig> {
    if !(0. ..=1.).contains(&options.earlier_loss_rise) {
        return Err("The earlier lessons' held-out loss rise allowed must be 0 to 1".into());
    }
    Ok(SessionConfig {
        epochs: options.epochs,
        batch_size: options.batch_size,
        learning_rate: options.learning_rate,
        foundation: options.foundation,
        rehearsal_share: if options.rehearsal { 0.10 } else { 0.0 },
        protect_correct_answers: options.protect_answers,
        pass_mark: (options.pass_percent / 100.).clamp(0., 1.),
        patience: options.flat_epochs,
        retry: RetryPolicy {
            retries: if options.retries { 3 } else { 0 },
            requeue: options.requeue,
            ..RetryPolicy::default()
        },
        seed: now(),
        word_limit: meta.word_limit,
        education: education_stage(options.education.as_deref())?,
        train_earlier: options.train_earlier,
        recover_earlier: options.recover_earlier,
        backoff: options.backoff,
        override_first_epoch: options.override_first_epoch,
        probation_epochs: options.probation_epochs.min(10),
        words_from_new_lessons: options.words_from_new_lessons,
        guard_before_growth: options.guard_before_growth,
        retention_per_lesson: options.earlier_every_lesson.then_some(4),
        earlier_panel_net: options.earlier_every_lesson,
        retention_loss_tolerance: options.earlier_loss_rise,
        earlier_by_answers: options.earlier_by_answers,
        ..SessionConfig::default()
    })
}

#[tauri::command]
fn teach(app: AppHandle, studio: State<Studio>, options: TeachOptions) -> Res<()> {
    if !(1..=50).contains(&options.epochs)
        || !(1..=128).contains(&options.batch_size)
        || !(options.learning_rate > 0. && options.learning_rate <= 0.01)
        || !(1..=20).contains(&options.flat_epochs)
    {
        return Err(
            "Choose 1-50 epochs, a batch size of 1-128, a learning rate up to 0.01 and 1-20 flat epochs"
                .into(),
        );
    }
    let files = studio.lessons.lock().unwrap().clone();
    if files.is_empty() {
        return Err("Add lesson files first".into());
    }
    let together = options.together.clone().unwrap_or_else(|| vec![true; files.len()]);
    if together.len() != files.len() {
        return Err("The lesson list changed; choose the lessons to teach together again".into());
    }
    let (meta, model, archive) =
        with_active(&studio, |a| Ok((a.meta.clone(), a.model.clone(), a.archive.clone())))?;
    no_waiting_checkpoints(&app, &meta.id)?;
    if studio.teaching.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("Already teaching".into());
    }
    studio.stop.store(false, Ordering::Release);
    let cfg = SessionConfig {
        checkpoint_dir: Some(checkpoints_dir(&app, &meta.id)?),
        ..teach_config(&options, &meta)?
    };
    // Continuing a restored checkpoint's interrupted run: its schedule, not a new one.
    let mut cfg = cfg;
    if let (true, Some(r)) = (options.continue_run, meta.resume.as_ref()) {
        if r.epoch < r.epochs {
            cfg.epochs = r.epochs;
            cfg.learning_rate = r.learning_rate;
            cfg.batch_size = r.batch_size;
            cfg.resume_epochs = r.epoch;
            cfg.resume_scale = Some(r.lr_scale);
        }
    }
    // The resume information is used (or passed over) by this run either way.
    let meta = if meta.resume.is_some() {
        let mut m = meta;
        m.resume = None;
        write_atomic(
            &model_dir(&app, &m.id)?.join("meta.json"),
            &serde_json::to_vec_pretty(&m).map_err(|e| e.to_string())?,
        )?;
        if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == m.id) {
            a.meta.resume = None;
        }
        m
    } else {
        meta
    };
    append_history(
        &app,
        &meta.id,
        serde_json::json!({"type": "course_started", "time": now(), "files": files.len(), "options": cfg}),
    );
    let stop = studio.stop.clone();
    std::thread::spawn(move || {
        let studio = app.state::<Studio>();
        let id = meta.id.clone();
        let mut meta_now = meta.clone();
        let mut throttle = BatchThrottle::default();
        let mut observer = |e: &CourseEvent| {
            if let CourseEvent::Session { event: Event::Batch { batch, batches, .. }, .. } = e {
                if !throttle.pass(*batch, *batches) {
                    return;
                }
            }
            let _ = app.emit("teach-event", e);
            match e {
                CourseEvent::Session { event: Event::Batch { .. }, .. } => {}
                CourseEvent::FileFinished(r) => {
                    meta_now.taught.push(TaughtFile {
                        name: r.name.clone(),
                        sha256: r.sha256.clone(),
                        outcome: r.outcome,
                        end_step: r.end_step,
                        time: now(),
                    });
                    append_history(&app, &id, serde_json::json!({"time": now(), "event": e}));
                }
                _ => append_history(&app, &id, serde_json::json!({"time": now(), "event": e})),
            }
        };
        let saved_meta = Arc::new(Mutex::new(meta.clone()));
        let saved_for_hook = saved_meta.clone();
        let mut on_learned = |model: &Bicameral, archive: &[Item]| -> Res<()> {
            let mut m = saved_for_hook.lock().unwrap();
            refresh_meta(&mut m, model);
            save_all(&app, &m, model, archive)?;
            clear_checkpoints(&app, &m.id);
            if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == m.id) {
                a.model = model.clone();
                a.archive = archive.to_vec();
                a.meta = m.clone();
            }
            Ok(())
        };
        // Study Mode: the model's memory is studied into and used while teaching, then saved.
        let mut memory = study::study_mode(&app)
            .then(|| yendory_engine::memory::Studying::new(study::load_memory(&app, &id), now()));
        // Rewards: answers are graded into the model's ledger; with reward-guided training on,
        // sessions schedule their units by emphasis.
        let mut rewards = rewards::rewards_for(&app, &id);
        let result = guarded(|| {
            yendory_engine::learn::course::teach_course_together(
                &model,
                &files,
                &together,
                &archive,
                &cfg,
                memory.as_mut().map(|s| s as &mut dyn yendory_engine::memory::StudyHook),
                rewards.as_mut(),
                &mut observer,
                &mut on_learned,
                &stop,
            )
        });
        if let Some(r) = &rewards {
            rewards::keep(&app, r);
        }
        if let Some(studying) = &memory {
            if let Err(e) = study::save_memory(&app, &id, &studying.store) {
                let _ = app.emit("device-notice", format!("Study Mode memory could not be saved: {e}"));
            }
        }
        // Persist the per-file records (learned or not) and the final metadata.
        let finished = match result {
            Ok(r) => {
                let mut m = saved_meta.lock().unwrap().clone();
                m.taught = meta_now.taught.clone();
                refresh_meta(&mut m, &r.model);
                let saved = write_atomic(
                    &model_dir(&app, &m.id).map(|d| d.join("meta.json")).unwrap_or_default(),
                    &serde_json::to_vec_pretty(&m).unwrap_or_default(),
                );
                if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == m.id) {
                    a.meta = m.clone();
                }
                TeachFinished { files: r.files, stopped: r.stopped, error: saved.err(), model: Some(m) }
            }
            Err(e) => TeachFinished { files: Vec::new(), stopped: false, error: Some(e), model: None },
        };
        append_history(
            &app,
            &id,
            serde_json::json!({"type": "course_finished", "time": now(), "stopped": finished.stopped, "error": finished.error}),
        );
        studio.teaching.store(false, Ordering::Release);
        if let Some(n) = yendory_engine::learn::take_device_notice() {
            let _ = app.emit("device-notice", n);
        }
        let _ = app.emit("teach-finished", &finished);
    });
    Ok(())
}

/// The experiment: the Classroom's lessons taught from the open model three times, A with the
/// page's settings (probation off), B with probation for 3 epochs, C with probation and the
/// larger earlier-lesson check (questions from every earlier lesson, judged by net change). Half
/// of every lesson's held-out questions are kept back to test each result. The model is never
/// changed; results go to the page and the model's history.
#[tauri::command]
fn compare_settings(
    app: AppHandle,
    studio: State<Studio>,
    options: TeachOptions,
    variants: Option<Vec<String>>,
) -> Res<()> {
    if !(1..=50).contains(&options.epochs)
        || !(1..=128).contains(&options.batch_size)
        || !(options.learning_rate > 0. && options.learning_rate <= 0.01)
        || !(1..=20).contains(&options.flat_epochs)
    {
        return Err(
            "Choose 1-50 epochs, a batch size of 1-128, a learning rate up to 0.01 and 1-20 flat epochs"
                .into(),
        );
    }
    let files = studio.lessons.lock().unwrap().clone();
    if files.is_empty() {
        return Err("Add the lessons to compare on first".into());
    }
    let (meta, model, archive) =
        with_active(&studio, |a| Ok((a.meta.clone(), a.model.clone(), a.archive.clone())))?;
    if studio.teaching.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("Wait until teaching or reading has finished".into());
    }
    studio.stop.store(false, Ordering::Release);
    use yendory_engine::learn::compare::{compare, CompareEvent, Variant};
    // One seed for all three, so they differ only in the settings compared.
    let base =
        SessionConfig { probation_epochs: 0, override_first_epoch: false, ..teach_config(&options, &meta)? };
    let all = vec![
        Variant { name: "A: your settings".into(), cfg: base.clone() },
        Variant {
            name: "B: + probation (3 epochs)".into(),
            cfg: SessionConfig { probation_epochs: 3, ..base.clone() },
        },
        Variant {
            name: "C: + probation and the larger earlier-lesson check".into(),
            cfg: SessionConfig {
                probation_epochs: 3,
                retention_per_lesson: Some(4),
                earlier_panel_net: true,
                ..base.clone()
            },
        },
        Variant {
            name: "D: earlier lessons' loss may rise 0.08".into(),
            cfg: SessionConfig { retention_loss_tolerance: 0.08, ..base.clone() },
        },
        Variant {
            name: "E: larger earlier-lesson check, judged by answers".into(),
            cfg: SessionConfig {
                retention_per_lesson: Some(4),
                earlier_panel_net: true,
                earlier_by_answers: true,
                ..base.clone()
            },
        },
    ];
    // Which to run (by letter); all when none are chosen.
    let chosen: Vec<String> = variants.unwrap_or_default();
    let variants: Vec<Variant> = all
        .into_iter()
        .filter(|v| chosen.is_empty() || chosen.iter().any(|c| v.name.starts_with(&format!("{c}:"))))
        .collect();
    let stop = studio.stop.clone();
    append_history(
        &app,
        &meta.id,
        serde_json::json!({"type": "compare_started", "time": now(), "files": files.len(), "options": base}),
    );
    std::thread::spawn(move || {
        let studio = app.state::<Studio>();
        let mut throttle = BatchThrottle::default();
        let mut observer = |e: &CompareEvent| {
            if let CompareEvent::Session { event: Event::Batch { batch, batches, .. }, .. } = e {
                if !throttle.pass(*batch, *batches) {
                    return;
                }
            }
            let _ = app.emit("compare-event", e);
        };
        let course_files: Vec<CourseFile> = files.clone();
        let result = guarded(|| compare(&model, &course_files, &archive, &variants, 6, &mut observer, &stop));
        let summary = match &result {
            Ok(r) => serde_json::json!({"type": "compare_finished", "time": now(), "results": r}),
            Err(e) => serde_json::json!({"type": "compare_finished", "time": now(), "error": e}),
        };
        append_history(&app, &meta.id, summary);
        studio.teaching.store(false, Ordering::Release);
        if let Some(n) = yendory_engine::learn::take_device_notice() {
            let _ = app.emit("device-notice", n);
        }
        let _ = app.emit("compare-finished", result.err());
    });
    Ok(())
}

#[tauri::command]
fn stop_teaching(studio: State<Studio>) -> bool {
    let active = studio.teaching.load(Ordering::Acquire);
    if active {
        studio.stop.store(true, Ordering::Release);
    }
    active
}

// ---------------------------------------------------------------------------------------------
// Talking, exams, brain measurements, history
// ---------------------------------------------------------------------------------------------

fn clean_message(s: &str) -> Res<String> {
    // Typed text can never carry Nexus markers (only the model may request an action).
    let s = yendory_engine::nexus::sanitize(s);
    let s = s.trim();
    if s.is_empty() || s.len() > 300 || !s.is_ascii() || s.contains(['\n', '\r']) {
        return Err("Write a message of 1-300 ASCII characters on one line".into());
    }
    Ok(s.to_string())
}

/// The model's greedy reply. `history` holds earlier (user, model) turns of this conversation.
#[tauri::command]
async fn ask(
    app: AppHandle,
    studio: State<'_, Studio>,
    history: Vec<(String, String)>,
    message: String,
) -> Res<Answer> {
    let message = clean_message(&message)?;
    let (model_id, model) = with_active(&studio, |a| Ok((a.meta.id.clone(), a.model.clone())))?;
    let context = model.config.max_tokens;
    let reply_tokens = reply_length(context);
    // Study Mode: memories related to the message go first, marked as memory.
    let (memory, memories) = study::conversation_memories(&app, &model_id, &model, &message);
    // Keep as many recent turns as fit in the context, leaving room for the reply.
    let question = format!("U: {message}\nN: ");
    let mut used = model.tokenizer.encode(&question)?.len() + model.tokenizer.encode(&memory)?.len();
    let mut turns = Vec::new();
    for (user, said) in history.iter().rev() {
        let turn = format!(
            "U: {}\nN: {}\n",
            clean_message(user)?,
            yendory_engine::nexus::sanitize(said).replace(['\n', '\r'], " ")
        );
        let n = model.tokenizer.encode(&turn)?.len();
        if used + n + reply_tokens > context {
            break;
        }
        used += n;
        turns.push(turn);
    }
    let prompt: String =
        std::iter::once(memory).chain(turns.into_iter().rev()).chain(std::iter::once(question)).collect();
    // Shown readably: a marker the model writes appears as "Action:" and so on.
    let reply = yendory_engine::nexus::readable(&reply(&model, &prompt, reply_tokens, Ablation::default())?);
    Ok(Answer { reply, memories })
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Answer {
    reply: String,
    /// Study Mode memories put in front of the message (empty when Study Mode is off).
    memories: Vec<String>,
}

/// Longest reply in tokens: a quarter of the context, at least 48 (Nova-2's) and at most 512.
fn reply_length(context: usize) -> usize {
    (context / 4).clamp(48, 512)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExamResult {
    name: String,
    sha256: String,
    report: Report,
    /// The same exam with Dense's input to the transformer cut: how much the answers depend
    /// on the dense hemisphere.
    report_without_dense: Report,
    /// Exam questions whose prompt also appears in material this model was taught.
    seen_before: usize,
    file_was_taught: bool,
}

/// Grades every unit of a lesson file (training and held-out alike) as an exam. The file should
/// be material the model was NOT taught; overlap with taught material is reported.
#[tauri::command]
async fn exam(app: AppHandle, studio: State<'_, Studio>) -> Res<Option<ExamResult>> {
    let Some(path) = app.dialog().file().add_filter("Exam lesson file", &["txt"]).blocking_pick_file() else {
        return Ok(None);
    };
    let name = display_name(&path);
    let text = text_of(&name, read_picked(&app, &path, MAX_LESSON_BYTES)?)?;
    let items = parse_lesson(&text, &name)?;
    let (model, archive, taught) =
        with_active(&studio, |a| Ok((a.model.clone(), a.archive.clone(), a.meta.taught.clone())))?;
    let questions: Vec<&Item> = items.iter().filter(|i| i.kind != Kind::Prose).collect();
    if questions.is_empty() {
        return Err("The exam file has no questions".into());
    }
    let known: std::collections::HashSet<&str> = archive.iter().filter_map(|i| i.prompt.as_deref()).collect();
    let seen_before =
        questions.iter().filter(|q| q.prompt.as_deref().is_some_and(|p| known.contains(p))).count();
    let sha256 = sha256_hex(text.as_bytes());
    let file_was_taught = taught.iter().any(|t| t.sha256 == sha256);
    let (report, report_without_dense) =
        tauri::async_runtime::spawn_blocking(move || -> Res<(Report, Report)> {
            let refs: Vec<&Item> = items.iter().filter(|i| i.kind != Kind::Prose).collect();
            let cut = Ablation { no_afferent: true, ..Ablation::default() };
            Ok((evaluate(&model, &refs, Ablation::default())?, evaluate(&model, &refs, cut)?))
        })
        .await
        .map_err(|e| e.to_string())??;
    Ok(Some(ExamResult { name, sha256, report, report_without_dense, seen_before, file_was_taught }))
}

#[tauri::command]
async fn measure(studio: State<'_, Studio>, prompt: String) -> Res<Measurements> {
    let message = clean_message(&prompt)?;
    let model = with_active(&studio, |a| Ok(a.model.clone()))?;
    model.measure(&format!("U: {message}\nN: "))
}

/// Generates the model's reply to `prompt` and records how each token was decided.
#[tauri::command]
async fn think(studio: State<'_, Studio>, prompt: String) -> Res<ThinkingTree> {
    let message = clean_message(&prompt)?;
    let model = with_active(&studio, |a| Ok(a.model.clone()))?;
    tauri::async_runtime::spawn_blocking(move || {
        model.think(&format!("U: {message}\nN: "), &ThinkOptions::default())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Brain-scan measurements of one step of a thinking tree: the forward pass that read the
/// prompt and the first `prefix` generated tokens.
#[tauri::command]
async fn measure_step(studio: State<'_, Studio>, prompt: String, prefix: Vec<TokenId>) -> Res<Measurements> {
    let message = clean_message(&prompt)?;
    if prefix.len() > 256 {
        return Err("Too many generated tokens".into());
    }
    let model = with_active(&studio, |a| Ok(a.model.clone()))?;
    let mut ids = model.tokenizer.encode(&format!("U: {message}\nN: "))?;
    ids.extend(prefix);
    model.measure_ids(&ids)
}

#[tauri::command]
fn history(app: AppHandle, id: String) -> Res<Vec<serde_json::Value>> {
    let path = model_dir(&app, &id)?.join("history.jsonl");
    let Ok(text) = std::fs::read_to_string(path) else { return Ok(Vec::new()) };
    let lines: Vec<&str> = text.lines().collect();
    Ok(lines[lines.len().saturating_sub(2000)..]
        .iter()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Status {
    active: Option<Meta>,
    teaching: bool,
    engine: &'static str,
}

#[tauri::command]
fn status(studio: State<Studio>) -> Status {
    Status {
        active: studio.active.lock().unwrap().as_ref().map(|a| a.meta.clone()),
        teaching: studio.teaching.load(Ordering::Acquire),
        engine: env!("CARGO_PKG_VERSION"),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(Studio::default())
        .setup(|app| {
            // The saved training memory settings, before any training starts.
            memory::apply_stored(app.handle());
            // A Hugging Face token given earlier (for gated datasets).
            reading::load_hf_token(app.handle());
            // Updates on the desktop (Android installs APKs).
            #[cfg(desktop)]
            app.handle().plugin(tauri_plugin_updater::Builder::new().build())?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            status,
            chat::chats,
            chat::chat_open,
            chat::chat_new,
            chat::chat_rename,
            chat::chat_pin,
            chat::chat_delete,
            chat::chat_settings,
            chat::chat_attach,
            chat::chat_attach_paths,
            chat::chat_measure,
            chat::chat_send,
            chat::chat_regenerate,
            chat::chat_stop,
            chat::chat_export,
            chat::chat_mark,
            conversation::conversation_sources,
            conversation::conversation_material,
            conversation::converse,
            courses::add_plan_step,
            saved_lessons,
            forget_saved_lesson,
            model_checkpoints,
            restore_checkpoint,
            discard_checkpoints,
            growth::growth_view,
            growth::set_growth_mode,
            growth::expand_model,
            growth::roll_back_growth,
            growth::set_growth_rules,
            list_models,
            create_model,
            model_sizes,
            set_word_limit,
            set_context,
            measure_context,
            model_cost,
            measure_speed,
            open_model,
            rename_model,
            delete_model,
            import_model,
            export_model,
            add_lessons,
            lessons,
            remove_lesson,
            teach,
            stop_teaching,
            ask,
            exam,
            measure,
            think,
            measure_step,
            history,
            reading::corpus_sources,
            reading::practice_families,
            reading::generate_corpus,
            reading::corpora,
            reading::delete_corpus,
            reading::import_corpus,
            reading::import_conversations,
            reading::save_conversations,
            reading::download_corpus,
            reading::hf_token_set,
            update::check_update,
            update::install_update,
            reading::set_hf_token,
            reading::cancel_download,
            reading::download_queue,
            reading::remove_queued_download,
            reading::read_corpus,
            training_device,
            set_training_device,
            set_cpu_threads,
            set_gpu_checks,
            set_fitted_replies,
            compare_settings,
            add_builtin_lessons,
            courses::builtin_courses,
            courses::add_builtin_course,
            memory::training_memory_view,
            memory::set_training_memory,
            benchmark_training,
            page_settings,
            save_page_settings,
            review::review_topics,
            review::generate_review,
            review::list_reviews,
            review::review_detail,
            review::delete_review,
            review::review_assess,
            review::train_review,
            reflection::reflection_sources,
            reflection::list_reflections,
            reflection::reflection_detail,
            reflection::delete_reflection,
            reflection::start_reflection,
            reflection::save_reflection_marks,
            reflection::train_reflection,
            practice::practice_sources,
            practice::list_practice,
            practice::practice_detail,
            practice::start_practice,
            practice::save_practice_marks,
            practice::delete_practice,
            practice::train_practice,
            practice::accept_practice_auto,
            neuro::neuro_presets,
            neuro::neuro_plan,
            neuro::create_neuro_model,
            rewards::reward_settings,
            rewards::save_reward_settings,
            rewards::reset_reward_settings,
            rewards::reward_view,
            rewards::reward_questions,
            rewards::reward_audit,
            rewards::reward_reset_cycle,
            rewards::reward_clear,
            study::study_state,
            study::set_study_mode,
            study::memory_view,
            study::memory_decide,
            study::memory_add,
            study::memory_try,
            study::study_text,
            designer::list_designs,
            designer::create_design,
            designer::import_design,
            designer::design_detail,
            designer::save_design,
            designer::delete_design,
            designer::design_to_classroom,
            designer::design_to_course,
            courses::list_courses,
            courses::create_course,
            courses::course_detail,
            courses::delete_course,
            courses::add_course_files,
            courses::attach_practice,
            courses::choose_exam,
            courses::edit_course,
            courses::list_plans,
            courses::create_plan,
            courses::plan_detail,
            courses::edit_plan,
            courses::delete_plan,
            courses::restart_course,
            courses::start_plan
        ])
        .run(tauri::generate_context!())
        .expect("error while running Yendory Studio");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_counts_read_as_people_say_them() {
        assert_eq!(tokens_text(160_000_000), "160M");
        assert_eq!(tokens_text(2_500_000), "2.5M");
        assert_eq!(tokens_text(2_000_000), "2M");
        assert_eq!(tokens_text(400_000), "400K");
        assert_eq!(tokens_text(999), "999");
    }

    #[test]
    fn zip_courses_unpack_txt_files_only() {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let opts: zip::write::SimpleFileOptions = Default::default();
            w.start_file("course/01_a.txt", opts).unwrap();
            w.write_all(b"[Dialogue Lesson]\n[examples]\nhi  Hello!\n[/examples]\n[/Dialogue Lesson]\n")
                .unwrap();
            w.start_file("__MACOSX/course/._01_a.txt", opts).unwrap();
            w.write_all(b"junk").unwrap();
            w.start_file("course/readme.md", opts).unwrap();
            w.write_all(b"not a lesson").unwrap();
            w.finish().unwrap();
        }
        let files = files_from("course.zip", buf.into_inner()).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name, "01_a.txt");
        let p = preview(&files[0], &[]);
        assert_eq!(p.kinds, vec![("dialogue".to_string(), 1)]);
    }

    #[test]
    fn names_and_messages_are_checked() {
        assert!(check_name("  ").is_err());
        assert_eq!(check_name(" Echo ").unwrap(), "Echo");
        assert!(clean_message("two\nlines").is_err());
        assert!(clean_message("caf\u{e9}").is_err());
        assert_eq!(clean_message("hi\x05switch on\x06").unwrap(), "hiswitch on");
        assert_eq!(display_name(&FilePath::Path(PathBuf::from("/x/y/01_Basics.txt"))), "01_Basics.txt");
    }
}
