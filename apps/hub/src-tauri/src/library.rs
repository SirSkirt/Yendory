//! The models library as the Hub shows it. Models are made, imported, renamed, deleted and
//! exported by the Yendory Network; the Hub adds what it shows beside them: Studio's training
//! plans and records, and its own log of what it did with each model.
use super::*;
use serde_json::{json, Value};
use tauri_plugin_dialog::DialogExt;

/// Studio keeps the open model in memory and writes it back when it saves, so a change made to
/// that model's files while Studio runs would be undone. (Once Studio works through the Network,
/// the Network refuses such changes itself.)
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
    lessons: usize,
    read: Vec<String>,
    recent: Vec<Recent>,
    plan: Option<PlanView>,
    /// Programs that have the model open through the Network.
    open_by: Vec<String>,
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
fn plans() -> Vec<Value> {
    let Some(dir) =
        yendory_client::paths::data_dir().map(|d| d.join(yendory_client::paths::STUDIO_ID).join("plans"))
    else {
        return Vec::new();
    };
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

/// A listing entry from the Network (`{meta, fileBytes, openBy}`) as the Hub shows it.
fn view_of(entry: &Value, plans: &[Value], log: &[Value]) -> ModelView {
    let meta = &entry["meta"];
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
        file_bytes: num(entry, "fileBytes"),
        lessons: list(meta, "taught").len(),
        read,
        recent: recent(meta, &mine),
        plan: plan_of(&id, plans),
        open_by: list(entry, "openBy").iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        id,
    }
}

/// Every model in the folder, most recently changed first.
#[tauri::command]
pub async fn list_models(app: AppHandle, hub: tauri::State<'_, Hub>) -> Res<Vec<ModelView>> {
    let entries = hub.call(&app, "models.list", json!({}))?;
    let plans = plans();
    let log = read_log(&app);
    Ok(entries.as_array().map(|a| a.iter().map(|e| view_of(e, &plans, &log)).collect()).unwrap_or_default())
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
// Making and changing models (through the Network)
// ---------------------------------------------------------------------------------------------

/// The sizes and designs a new model can have.
#[tauri::command]
pub async fn model_options(app: AppHandle, hub: tauri::State<'_, Hub>) -> Res<Value> {
    hub.call(&app, "models.options", json!({}))
}

#[tauri::command]
pub async fn create_model(
    app: AppHandle,
    hub: tauri::State<'_, Hub>,
    choice: Value,
    shelf: Option<String>,
) -> Res<String> {
    let made = hub.call(&app, "models.create", choice)?;
    let id = text(&made["meta"], "id");
    shelves::place(&app, &id, shelf.as_deref());
    log(&app, &id, "Hub", "Created, untrained");
    Ok(id)
}

/// Imports a checkpoint (Bicameral `.nnova2` or Nova Neuro `.nneuro`) as a new model.
#[tauri::command]
pub async fn import_model(
    app: AppHandle,
    hub: tauri::State<'_, Hub>,
    shelf: Option<String>,
) -> Res<Option<String>> {
    let Some(picked) = app
        .dialog()
        .file()
        .add_filter("Nova checkpoint (Bicameral .nnova2, Neuro .nneuro)", &["nnova2", "nneuro"])
        .blocking_pick_file()
    else {
        return Ok(None);
    };
    let path = picked.into_path().map_err(|e| e.to_string())?;
    let made = hub.call(&app, "models.import", json!({"path": path.display().to_string()}))?;
    let id = text(&made["meta"], "id");
    shelves::place(&app, &id, shelf.as_deref());
    log(&app, &id, "Hub", &format!("Imported {}", text(&made["meta"], "origin")));
    Ok(Some(id))
}

#[tauri::command]
pub async fn rename_model(app: AppHandle, hub: tauri::State<'_, Hub>, id: String, name: String) -> Res<()> {
    studio_closed("new name")?;
    let before = hub.call(&app, "models.list", json!({}))?;
    let old = before
        .as_array()
        .and_then(|a| a.iter().find(|e| e["meta"]["id"] == id.as_str()))
        .map(|e| text(&e["meta"], "name"));
    hub.call(&app, "models.rename", json!({"id": id, "name": name}))?;
    if let Some(old) = old {
        log(&app, &id, "Hub", &format!("Renamed from {old}"));
    }
    Ok(())
}

#[tauri::command]
pub async fn delete_model(app: AppHandle, hub: tauri::State<'_, Hub>, id: String) -> Res<()> {
    studio_closed("deletion")?;
    hub.call(&app, "models.delete", json!({"id": id}))?;
    shelves::forget(&app, &id);
    Ok(())
}

/// Saves a copy of the model's checkpoint where the user chooses (the Network checks the copy).
#[tauri::command]
pub async fn export_model(app: AppHandle, hub: tauri::State<'_, Hub>, id: String) -> Res<Option<String>> {
    let entries = hub.call(&app, "models.list", json!({}))?;
    let meta = entries
        .as_array()
        .and_then(|a| a.iter().find(|e| e["meta"]["id"] == id.as_str()))
        .map(|e| e["meta"].clone())
        .ok_or("This model is no longer in the models folder")?;
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
    hub.call(&app, "models.export", json!({"id": id, "path": path.display().to_string()}))?;
    let file = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    log(&app, &id, "Hub", &format!("Exported {file}"));
    Ok(Some(file))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let texts: Vec<String> = recent(&meta, &log).into_iter().map(|r| r.text).collect();
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

    #[test]
    fn a_listing_entry_becomes_the_card() {
        let entry = json!({"meta": {"id": "m1", "name": "Nova", "parameters": 1182144, "step": 15000,
            "read": [{"corpus": "A"}, {"corpus": "A"}, {"corpus": "B"}], "taught": [{}]}, "fileBytes": 99, "openBy": ["Yendory Studio"]});
        let v = view_of(&entry, &[], &[]);
        assert_eq!((v.id.as_str(), v.name.as_str(), v.architecture.as_str()), ("m1", "Nova", "bicameral"));
        assert_eq!(
            (v.file_bytes, v.lessons, v.read.clone()),
            (99, 1, vec!["A".to_string(), "B".to_string()])
        );
        assert_eq!(v.open_by, ["Yendory Studio"]);
    }
}
