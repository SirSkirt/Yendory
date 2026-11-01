//! The models library: the folder Studio keeps its models in, read and written in Studio's own
//! format (`meta.json` field names and meanings as Studio's `Meta`). The Hub creates, imports,
//! renames, deletes and exports models; it never trains them.
use super::*;
use serde::Deserialize;
use serde_json::{json, Value};
use tauri_plugin_dialog::DialogExt;
use yendory_engine::model::neuro as engine_neuro;
use yendory_engine::model::{MAX_CONTEXT, MAX_PARAMETERS};
use yendory_engine::tokenizer::BASE;
use yendory_engine::{size_presets, Bicameral, Config, Focus};

/// Large enough for the largest model the Engine accepts (as in Studio).
const MAX_CHECKPOINT_BYTES: u64 = 12_200_000_000;

pub(crate) fn models_dir(app: &AppHandle) -> Res<PathBuf> {
    let dir = studio_data_dir(app)?.join("models");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

pub(crate) fn model_dir(app: &AppHandle, id: &str) -> Res<PathBuf> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err("Invalid model id".into());
    }
    Ok(models_dir(app)?.join(id))
}

fn check_name(name: &str) -> Res<String> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 60 || name.chars().any(char::is_control) {
        return Err("Give the model a name of 1 to 60 characters".into());
    }
    Ok(name.to_string())
}

/// A new model's folder name, made the way Studio makes them.
fn new_id() -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("m{nanos:x}")
}

/// Studio keeps the open model in memory and writes it back when it saves, so a change the Hub
/// makes to that model's files while Studio runs would be undone.
fn studio_closed(change: &str) -> Res<()> {
    if programs::running("studio") {
        return Err(format!(
            "Close Yendory Studio first. It may have this model open, and would undo the {change} the next time it saves."
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Reading the library
// ---------------------------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Recent {
    time: u64,
    program: String,
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanView {
    name: String,
    /// Studio's plan status: idle, running, stopped, finished, failed, needs_attention.
    status: String,
    /// Courses finished or being taught, and how many the plan has.
    position: u64,
    courses: u64,
    /// Teaching has begun and the plan has not finished: the spine wears a ribbon.
    in_progress: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelView {
    id: String,
    name: String,
    architecture: String,
    parameters: u64,
    step: u64,
    words: u64,
    slots: u64,
    word_limit: Option<u64>,
    context: u64,
    created: u64,
    updated: u64,
    origin: String,
    file_bytes: u64,
    /// Lesson files taught (Classroom, courses).
    lessons: usize,
    /// Corpora read, oldest first, each named once.
    read: Vec<String>,
    recent: Vec<Recent>,
    plan: Option<PlanView>,
}

fn num(v: &Value, key: &str) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(0)
}
fn text(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
}
fn list<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}
fn loss(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
}

/// What a lesson file's outcome means, in the words Studio's Library uses.
fn outcome_text(outcome: &str) -> &'static str {
    match outcome {
        "passed" => "passed",
        "progressed" => "improved, pass mark not reached",
        "deferred" => "not passed yet, to be tried again",
        "kept_previous" => "nothing kept",
        "duplicate" => "already taught",
        "failed" => "could not be taught",
        _ => "taught",
    }
}

fn training_text(what: &str, r: &Value) -> String {
    let accepted = r.get("accepted").and_then(Value::as_bool).unwrap_or(false);
    let stopped = r.get("stopped").and_then(Value::as_bool).unwrap_or(false);
    let mut s = match (accepted, loss(r, "baselineLoss"), loss(r, "bestLoss")) {
        (true, Some(a), Some(b)) => format!("{what}: held-out loss {a:.2} → {b:.2}"),
        (true, _, _) => format!("{what}: a better checkpoint was kept"),
        (false, _, _) => format!("{what}: held-out loss did not improve, nothing kept"),
    };
    if stopped {
        s.push_str(" (stopped early)");
    }
    s
}

/// The newest events of a model: its lessons, reading and conversation runs from meta.json,
/// and what the Hub did with it.
fn recent(meta: &Value, log: &[Value]) -> Vec<Recent> {
    let studio = |time, text| Recent { time, program: "Studio".into(), text };
    let mut out: Vec<Recent> = Vec::new();
    for t in list(meta, "taught") {
        out.push(studio(
            num(t, "time"),
            format!("Lesson {}: {}", text(t, "name"), outcome_text(&text(t, "outcome"))),
        ));
    }
    for r in list(meta, "read") {
        out.push(studio(num(r, "time"), training_text(&format!("Read {}", text(r, "corpus")), r)));
    }
    for c in list(meta, "conversation") {
        out.push(studio(num(c, "time"), training_text("Conversation practice", c)));
    }
    for e in log {
        out.push(Recent { time: num(e, "time"), program: text(e, "program"), text: text(e, "text") });
    }
    out.sort_by_key(|r| std::cmp::Reverse(r.time));
    out.truncate(8);
    out
}

/// Studio's training plans (`plans/<id>/plan.json`), read once per listing.
fn plans(app: &AppHandle) -> Vec<Value> {
    let Ok(dir) = studio_data_dir(app).map(|d| d.join("plans")) else { return Vec::new() };
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    entries.flatten().map(|e| read_json(&e.path().join("plan.json"))).filter(Value::is_object).collect()
}

/// The plan a model is following: one in progress first, else the newest.
fn plan_of(id: &str, plans: &[Value]) -> Option<PlanView> {
    let view = |p: &Value| {
        let status = text(p, "status");
        let started = num(p, "position") > 0
            || !list(p, "results").is_empty()
            || p.get("run").is_some_and(|r| !r.is_null());
        PlanView {
            name: text(p, "name"),
            in_progress: status != "finished" && (status != "idle" || started),
            status,
            position: num(p, "position"),
            courses: list(p, "courses").len() as u64,
        }
    };
    let mut mine: Vec<&Value> = plans.iter().filter(|p| text(p, "modelId") == id).collect();
    mine.sort_by_key(|p| std::cmp::Reverse(num(p, "created")));
    let views: Vec<PlanView> = mine.into_iter().map(view).collect();
    let i = views.iter().position(|v| v.in_progress).unwrap_or(0);
    views.into_iter().nth(i)
}

fn view_of(dir: &Path, meta: &Value, plans: &[Value], log: &[Value]) -> ModelView {
    let id = text(meta, "id");
    let mut read: Vec<String> = Vec::new();
    for r in list(meta, "read") {
        let corpus = text(r, "corpus");
        if !corpus.is_empty() && !read.contains(&corpus) {
            read.push(corpus);
        }
    }
    let mine: Vec<Value> = log.iter().filter(|e| text(e, "model") == id).cloned().collect();
    ModelView {
        name: text(meta, "name"),
        architecture: meta.get("architecture").and_then(Value::as_str).unwrap_or("bicameral").to_string(),
        parameters: num(meta, "parameters"),
        step: num(meta, "step"),
        words: num(meta, "words"),
        slots: num(meta, "slots"),
        word_limit: meta.get("wordLimit").and_then(Value::as_u64),
        context: num(meta, "context"),
        created: num(meta, "created"),
        updated: num(meta, "updated"),
        origin: text(meta, "origin"),
        file_bytes: std::fs::metadata(dir.join("model.nnova2")).map(|m| m.len()).unwrap_or(0),
        lessons: list(meta, "taught").len(),
        read,
        recent: recent(meta, &mine),
        plan: plan_of(&id, plans),
        id,
    }
}

/// Every model in the folder, most recently changed first.
#[tauri::command]
pub fn list_models(app: AppHandle) -> Res<Vec<ModelView>> {
    let plans = plans(&app);
    let log = read_log(&app);
    let mut out = Vec::new();
    for entry in std::fs::read_dir(models_dir(&app)?).map_err(|e| e.to_string())?.flatten() {
        let dir = entry.path();
        let meta = read_json(&dir.join("meta.json"));
        // A folder whose meta.json names another id is not a model of this library.
        if meta.is_object() && entry.file_name().to_string_lossy() == text(&meta, "id") {
            out.push(view_of(&dir, &meta, &plans, &log));
        }
    }
    out.sort_by_key(|m| std::cmp::Reverse(m.updated));
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The Hub's log: what it did with each model (Studio's history.jsonl stays Studio's)
// ---------------------------------------------------------------------------------------------

fn log_file(app: &AppHandle) -> Res<PathBuf> {
    Ok(hub_dir(app)?.join("log.jsonl"))
}

fn read_log(app: &AppHandle) -> Vec<Value> {
    let Ok(text) = log_file(app).and_then(|p| std::fs::read_to_string(p).map_err(|e| e.to_string())) else {
        return Vec::new();
    };
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(5000)..].iter().filter_map(|l| serde_json::from_str(l).ok()).collect()
}

pub(crate) fn log(app: &AppHandle, model: &str, program: &str, text: &str) {
    if let Ok(path) = log_file(app) {
        let line = json!({"time": now(), "model": model, "program": program, "text": text});
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{line}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// New models
// ---------------------------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SizeOption {
    id: &'static str,
    name: &'static str,
    /// Parameters with Nova-2's fixed 256 word slots.
    parameters: usize,
    /// Parameters each learned word adds when the vocabulary has a limit.
    per_word: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelOptions {
    bicameral: Vec<SizeOption>,
    neuro: Vec<SizeOption>,
    max_context: usize,
}

fn parameters(config: &Config, design: Option<&engine_neuro::NeuroSpec>) -> usize {
    Bicameral::tensors_for(config, design).iter().map(|(_, r, c)| r * c).sum()
}

/// The sizes and designs a new model can have (the same presets Studio offers).
#[tauri::command]
pub fn model_options() -> ModelOptions {
    ModelOptions {
        bicameral: size_presets()
            .into_iter()
            .map(|p| SizeOption {
                id: p.id,
                name: p.name,
                parameters: parameters(&p.config, None),
                per_word: p.config.d + 1,
            })
            .collect(),
        neuro: engine_neuro::presets()
            .into_iter()
            .map(|p| {
                let config = Config { vocab: BASE + 256, ..p.config.clone() };
                SizeOption {
                    id: p.id,
                    name: p.name,
                    parameters: parameters(&config, Some(&p.design)),
                    per_word: p.config.d + 1,
                }
            })
            .collect(),
        max_context: MAX_CONTEXT,
    }
}

/// A vocabulary limit must leave the model within the Engine's limits when every slot is used
/// (Studio's check).
fn check_word_limit(config: &Config, limit: usize) -> Res<()> {
    if limit == 0 {
        return Err("Choose a vocabulary limit of at least 1 word".into());
    }
    let full = Config { vocab: BASE + limit, ..config.clone() };
    Bicameral::check_config(&full).map_err(|_| {
        let base = Bicameral::parameters_for(&Config { vocab: BASE, ..config.clone() });
        let most = MAX_PARAMETERS.saturating_sub(base) / (config.d + 1);
        format!("At this size the vocabulary can hold up to {most} words (1 billion parameters)")
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewModel {
    name: String,
    /// "bicameral" or "neuro".
    architecture: String,
    /// A size preset (Bicameral) or a design preset (Neuro).
    preset: String,
    /// `None`: Nova-2's fixed 256 word slots.
    word_limit: Option<usize>,
    context: usize,
    shelf: Option<String>,
}

/// The untrained model a choice describes, built as Studio builds it.
fn build(choice: &NewModel) -> Res<Bicameral> {
    let (mut config, design) = match choice.architecture.as_str() {
        "bicameral" => {
            let p = size_presets().into_iter().find(|p| p.id == choice.preset).ok_or("Unknown model size")?;
            (p.config, None)
        }
        "neuro" => {
            let p = engine_neuro::presets()
                .into_iter()
                .find(|p| p.id == choice.preset)
                .ok_or("Unknown design")?;
            (Config { vocab: BASE + 256, ..p.config }, Some(p.design))
        }
        _ => return Err("Unknown architecture".into()),
    };
    config.max_tokens = choice.context;
    Bicameral::check_config(&config).map_err(|_| format!("Choose a context of 1 to {MAX_CONTEXT} tokens"))?;
    if let Some(limit) = choice.word_limit {
        check_word_limit(&config, limit)?;
        // No slots yet: one is added for each word the model learns, up to the limit.
        config.vocab = BASE;
    }
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1) | 1;
    Ok(Bicameral::with_design(config, design, seed))
}

/// meta.json for a model just made or imported, with the fields Studio writes (`Meta`, camelCase).
fn new_meta(id: &str, name: &str, model: &Bicameral, word_limit: Option<usize>, origin: &str) -> Value {
    let t = now();
    let mut meta = json!({
        "id": id,
        "name": name,
        "created": t,
        "updated": t,
        "step": model.step,
        "parameters": model.parameter_count(),
        "groups": [
            model.group_parameters(Focus::Left),
            model.group_parameters(Focus::Right),
            model.group_parameters(Focus::Interconnect),
        ],
        "words": model.tokenizer.words().len(),
        "slots": model.word_capacity(),
        "wordLimit": word_limit,
        "context": model.config.max_tokens,
        "taught": [],
        "origin": origin,
        "read": [],
        "conversation": [],
        "architecture": model.architecture(),
        "blocks": model.config.blocks,
    });
    if let Some(design) = &model.neuro {
        meta["design"] = serde_json::to_value(design).unwrap_or(Value::Null);
    }
    meta
}

/// Writes a new model's folder (each file atomically) and puts it on a shelf.
fn install(
    app: &AppHandle,
    name: &str,
    model: &Bicameral,
    word_limit: Option<usize>,
    origin: &str,
    shelf: Option<&str>,
) -> Res<String> {
    let name = check_name(name)?;
    let id = new_id();
    let dir = model_dir(app, &id)?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let meta = new_meta(&id, &name, model, word_limit, origin);
    let written = (|| {
        write_atomic(&dir.join("model.nnova2"), &model.to_bytes()?)?;
        write_atomic(&dir.join("archive.json"), b"[]")?;
        write_atomic(&dir.join("meta.json"), &serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?)
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(e);
    }
    shelves::place(app, &id, shelf);
    Ok(id)
}

#[tauri::command]
pub async fn create_model(app: AppHandle, choice: NewModel) -> Res<String> {
    let name = check_name(&choice.name)?;
    let model = build(&choice)?;
    let id = install(&app, &name, &model, choice.word_limit, "new", choice.shelf.as_deref())?;
    log(&app, &id, "Hub", "Created, untrained");
    Ok(id)
}

/// Imports a checkpoint (Bicameral `.nnova2` or Nova Neuro `.nneuro`) as a new model.
#[tauri::command]
pub async fn import_model(app: AppHandle, shelf: Option<String>) -> Res<Option<String>> {
    let Some(picked) = app
        .dialog()
        .file()
        .add_filter("Nova checkpoint (Bicameral .nnova2, Neuro .nneuro)", &["nnova2", "nneuro"])
        .blocking_pick_file()
    else {
        return Ok(None);
    };
    let path = picked.into_path().map_err(|e| e.to_string())?;
    let size = std::fs::metadata(&path).map_err(|e| e.to_string())?.len();
    if size > MAX_CHECKPOINT_BYTES {
        return Err("The file is larger than the largest model the Engine accepts".into());
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let model = Bicameral::from_bytes(&bytes)?;
    let file = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    let stem = file.trim_end_matches(".nnova2").trim_end_matches(".nneuro").trim();
    let name: String =
        if stem.is_empty() { "Imported model".into() } else { stem.chars().take(60).collect() };
    let id = install(&app, &name, &model, None, &file, shelf.as_deref())?;
    log(&app, &id, "Hub", &format!("Imported {file}"));
    Ok(Some(id))
}

// ---------------------------------------------------------------------------------------------
// Changing models
// ---------------------------------------------------------------------------------------------

/// Renames a model. Only `name` changes in meta.json; every other field is kept as written.
#[tauri::command]
pub fn rename_model(app: AppHandle, id: String, name: String) -> Res<()> {
    let name = check_name(&name)?;
    studio_closed("new name")?;
    let path = model_dir(&app, &id)?.join("meta.json");
    let mut meta = read_json(&path);
    if !meta.is_object() {
        return Err("This model's meta.json could not be read".into());
    }
    let old = text(&meta, "name");
    meta["name"] = Value::String(name.clone());
    write_atomic(&path, &serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?)?;
    log(&app, &id, "Hub", &format!("Renamed from {old}"));
    Ok(())
}

/// Deletes a model's folder (its checkpoint, lessons archive, history and recovery points).
#[tauri::command]
pub fn delete_model(app: AppHandle, id: String) -> Res<()> {
    studio_closed("deletion")?;
    let dir = model_dir(&app, &id)?;
    if !dir.join("meta.json").exists() {
        return Err("This model is no longer in the models folder".into());
    }
    std::fs::remove_dir_all(dir).map_err(|e| e.to_string())?;
    shelves::forget(&app, &id);
    Ok(())
}

/// Saves a copy of the model's checkpoint where the user chooses, and checks the copy.
#[tauri::command]
pub async fn export_model(app: AppHandle, id: String) -> Res<Option<String>> {
    let dir = model_dir(&app, &id)?;
    let meta = read_json(&dir.join("meta.json"));
    let neuro = text(&meta, "architecture") == "neuro";
    let (kind, ext) =
        if neuro { ("Nova Neuro checkpoint", "nneuro") } else { ("Bicameral checkpoint", "nnova2") };
    let safe: String = text(&meta, "name")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    let Some(picked) = app
        .dialog()
        .file()
        .add_filter(kind, &[ext])
        .set_file_name(format!("{safe}-step{}.{ext}", num(&meta, "step")))
        .blocking_save_file()
    else {
        return Ok(None);
    };
    let path = picked.into_path().map_err(|e| e.to_string())?;
    let bytes = std::fs::read(dir.join("model.nnova2")).map_err(|e| e.to_string())?;
    // A checkpoint the Engine cannot read is not exported.
    Bicameral::from_bytes(&bytes)?;
    let mut f = std::fs::File::create(&path).map_err(|e| e.to_string())?;
    f.write_all(&bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    if std::fs::read(&path).map_err(|e| e.to_string())? != bytes {
        return Err("The exported file does not match the model; try another location".into());
    }
    let file = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    log(&app, &id, "Hub", &format!("Exported {file}"));
    Ok(Some(file))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(architecture: &str, preset: &str, word_limit: Option<usize>, context: usize) -> NewModel {
        NewModel {
            name: "Test".into(),
            architecture: architecture.into(),
            preset: preset.into(),
            word_limit,
            context,
            shelf: None,
        }
    }

    #[test]
    fn listed_parameters_match_the_models_built() {
        let o = model_options();
        for s in &o.bicameral {
            let m = build(&choice("bicameral", s.id, None, 256)).unwrap();
            assert_eq!(m.parameter_count(), s.parameters, "{}", s.id);
        }
        for s in &o.neuro {
            let m = build(&choice("neuro", s.id, None, 256)).unwrap();
            assert_eq!(m.parameter_count(), s.parameters, "{}", s.id);
            assert_eq!(m.architecture(), "neuro");
        }
    }

    #[test]
    fn new_models_match_studios_defaults() {
        // Nova-2's size and layout: the Engine's default configuration.
        let m = build(&choice("bicameral", "nova2", None, 256)).unwrap();
        assert_eq!(m.config, Config::default());
        // A vocabulary limit starts with no word slots; the context is kept.
        let m = build(&choice("bicameral", "4m", Some(1000), 1024)).unwrap();
        assert_eq!((m.word_capacity(), m.config.max_tokens), (0, 1024));
        let meta = new_meta("m1", "Test", &m, Some(1000), "new");
        assert_eq!((num(&meta, "slots"), num(&meta, "context"), num(&meta, "wordLimit")), (0, 1024, 1000));
        assert_eq!(text(&meta, "architecture"), "bicameral");
        // A Neuro model records its design.
        let m = build(&choice("neuro", "neuro-core-1m", None, 256)).unwrap();
        let meta = new_meta("m2", "Test", &m, None, "new");
        assert!(meta["design"].is_object());
        assert_eq!(m.word_capacity(), 256);
    }

    #[test]
    fn bad_choices_are_refused_with_a_reason() {
        assert!(build(&choice("bicameral", "nope", None, 256)).is_err());
        assert!(build(&choice("bicameral", "nova2", None, 0)).is_err());
        assert!(build(&choice("bicameral", "nova2", None, MAX_CONTEXT + 1)).is_err());
        assert!(build(&choice("bicameral", "nova2", Some(0), 256)).is_err());
        assert!(build(&choice("bicameral", "nova2", Some(usize::MAX / 4), 256)).is_err());
        assert!(check_name("  ").is_err());
        assert!(check_name(&"x".repeat(61)).is_err());
    }

    #[test]
    fn events_read_from_studios_records() {
        let meta = json!({
            "id": "m1",
            "taught": [{"name": "01 Greetings.txt", "outcome": "passed", "time": 30}],
            "read": [
                {"corpus": "TinyStories", "accepted": true, "stopped": false, "baselineLoss": 1.42, "bestLoss": 1.31, "time": 20},
                {"corpus": "TinyStories", "accepted": false, "stopped": true, "baselineLoss": 1.3, "bestLoss": 1.3, "time": 10}
            ],
        });
        let log = [json!({"time": 40, "model": "m1", "program": "Hub", "text": "Exported a.nnova2"})];
        let r = recent(&meta, &log);
        let texts: Vec<&str> = r.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "Exported a.nnova2",
                "Lesson 01 Greetings.txt: passed",
                "Read TinyStories: held-out loss 1.42 → 1.31",
                "Read TinyStories: held-out loss did not improve, nothing kept (stopped early)",
            ]
        );
    }

    #[test]
    fn a_plan_in_progress_wears_the_ribbon() {
        let plans = [
            json!({"modelId": "m1", "name": "Old", "status": "finished", "position": 2, "courses": ["a", "b"], "created": 5}),
            json!({"modelId": "m1", "name": "English", "status": "stopped", "position": 1, "courses": ["a", "b", "c"], "created": 1}),
            json!({"modelId": "m2", "name": "Fresh", "status": "idle", "position": 0, "courses": ["a"], "created": 9}),
        ];
        let p = plan_of("m1", &plans).unwrap();
        assert!(p.in_progress);
        assert_eq!((p.name.as_str(), p.position, p.courses), ("English", 1, 3));
        assert!(!plan_of("m2", &plans).unwrap().in_progress);
        assert!(plan_of("m3", &plans).is_none());
    }
}
