//! Study Mode: the on/off setting, each model's memory (<model dir>/memory.json) and the
//! commands that inspect and edit it. Studying, retrieval and the teaching hooks are in
//! `yendory_engine::memory`; turning Study Mode off keeps the memory and stops every read and
//! write of it.
use super::*;
use yendory_engine::memory::{MemoryStore, Relation, RetrievalRecord, StudyReport};

pub(crate) fn study_mode(app: &AppHandle) -> bool {
    read_settings(app).get("studyMode").and_then(|v| v.as_bool()).unwrap_or(false)
}

fn memory_path(app: &AppHandle, model_id: &str) -> Res<PathBuf> {
    Ok(model_dir(app, model_id)?.join("memory.json"))
}

pub(crate) fn load_memory(app: &AppHandle, model_id: &str) -> MemoryStore {
    memory_path(app, model_id)
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

pub(crate) fn save_memory(app: &AppHandle, model_id: &str, store: &MemoryStore) -> Res<()> {
    write_atomic(&memory_path(app, model_id)?, &serde_json::to_vec(store).map_err(|e| e.to_string())?)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StudyState {
    on: bool,
    model: Option<String>,
    memories: usize,
}

fn state(app: &AppHandle, studio: &Studio) -> StudyState {
    let model = studio.active.lock().unwrap().as_ref().map(|a| (a.meta.id.clone(), a.meta.name.clone()));
    StudyState {
        on: study_mode(app),
        memories: model.as_ref().map_or(0, |(id, _)| load_memory(app, id).entries.len()),
        model: model.map(|(_, n)| n),
    }
}

#[tauri::command]
pub fn study_state(app: AppHandle, studio: State<Studio>) -> StudyState {
    state(&app, &studio)
}

#[tauri::command]
pub fn set_study_mode(app: AppHandle, studio: State<Studio>, on: bool) -> Res<StudyState> {
    not_teaching(&studio)?;
    write_setting(&app, "studyMode", on.into())?;
    Ok(state(&app, &studio))
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryView {
    on: bool,
    store: MemoryStore,
    relations: Vec<Relation>,
    /// Concept words shared by two or more memories, with how many.
    concepts: Vec<(String, usize)>,
    /// The latest retrievals, newest first.
    log: Vec<RetrievalRecord>,
    step: u64,
}

fn active_id(studio: &Studio) -> Res<(String, u64)> {
    with_active(studio, |a| Ok((a.meta.id.clone(), a.model.step)))
}

#[tauri::command]
pub fn memory_view(app: AppHandle, studio: State<Studio>) -> Res<MemoryView> {
    let (id, step) = active_id(&studio)?;
    let mut store = load_memory(&app, &id);
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for m in &store.entries {
        for t in &m.terms {
            *counts.entry(t.clone()).or_default() += 1;
        }
    }
    let mut concepts: Vec<(String, usize)> = counts.into_iter().filter(|(_, n)| *n >= 2).collect();
    concepts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let log: Vec<RetrievalRecord> = store.log.iter().rev().take(100).cloned().collect();
    store.log.clear();
    Ok(MemoryView { on: study_mode(&app), relations: store.relations(), concepts, log, store, step })
}

/// Edits, switches off or on, verifies or deletes a memory (the decision is recorded).
#[tauri::command]
pub fn memory_decide(
    app: AppHandle,
    studio: State<Studio>,
    id: u64,
    action: String,
    text: Option<String>,
) -> Res<()> {
    not_teaching(&studio)?;
    let (model_id, _) = active_id(&studio)?;
    let mut store = load_memory(&app, &model_id);
    let text = text.map(|t| clean_message(&t)).transpose()?;
    store.decide(id, &action, text.as_deref(), now())?;
    save_memory(&app, &model_id, &store)
}

/// A memory written by the teacher.
#[tauri::command]
pub fn memory_add(app: AppHandle, studio: State<Studio>, text: String) -> Res<u64> {
    not_teaching(&studio)?;
    if !study_mode(&app) {
        return Err("Study Mode is off: turn it on to add memories".into());
    }
    let (model_id, step) = active_id(&studio)?;
    let text = clean_message(&text)?;
    let mut store = load_memory(&app, &model_id);
    let id = store.add_teacher(&text, step, now());
    save_memory(&app, &model_id, &store)?;
    Ok(id)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Retrieved {
    id: u64,
    text: String,
    score: f32,
}

/// What would be retrieved for a question (nothing is recorded).
#[tauri::command]
pub fn memory_try(app: AppHandle, studio: State<Studio>, query: String) -> Res<Vec<Retrieved>> {
    let (model_id, _) = active_id(&studio)?;
    let store = load_memory(&app, &model_id);
    Ok(store
        .retrieve(&query, yendory_engine::memory::RETRIEVE)
        .into_iter()
        .filter_map(|(id, score)| store.get(id).map(|m| Retrieved { id, text: m.text.clone(), score }))
        .collect())
}

/// Studies a text now (memory only: the weights do not change). Needs Study Mode on.
#[tauri::command]
pub async fn study_text(
    app: AppHandle,
    studio: State<'_, Studio>,
    text: String,
    label: String,
) -> Res<StudyReport> {
    not_teaching(&studio)?;
    if !study_mode(&app) {
        return Err("Study Mode is off: memory-only reading needs it on".into());
    }
    let (model_id, model) = with_active(&studio, |a| Ok((a.meta.id.clone(), a.model.clone())))?;
    let (text, _) = yendory_engine::learn::corpus::normalize_ascii(&text);
    let plain = yendory_engine::learn::designer::markdown_to_plain(&text);
    let label = if label.trim().is_empty() { "studied text".to_string() } else { label };
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut store = load_memory(&app2, &model_id);
        let sha = sha256_hex(plain.as_bytes());
        let report = store.study(&model, &plain, &label, &sha, &[], now())?;
        save_memory(&app2, &model_id, &store)?;
        Ok(report)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// The memories retrieved for a conversation turn and their context lines (Study Mode on), or
/// nothing. The retrieval is recorded.
pub(crate) fn conversation_memories(
    app: &AppHandle,
    model_id: &str,
    model: &Bicameral,
    message: &str,
) -> (String, Vec<String>) {
    if !study_mode(app) {
        return (String::new(), Vec::new());
    }
    let mut store = load_memory(app, model_id);
    let ids: Vec<u64> =
        store.retrieve(message, yendory_engine::memory::RETRIEVE).into_iter().map(|(id, _)| id).collect();
    if ids.is_empty() {
        return (String::new(), Vec::new());
    }
    let ctx = store.context(&ids, model.config.max_tokens / 3);
    let texts = ids.iter().filter_map(|id| store.get(*id).map(|m| m.text.clone())).collect();
    store.record("conversation", message, &ids, now());
    let _ = save_memory(app, model_id, &store);
    (ctx, texts)
}
