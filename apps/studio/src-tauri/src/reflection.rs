//! Reflection sessions: stored as <app data>/reflections/<id>.json. Yendory proposes and
//! measures connections, Nova writes her own observations (`learn::reflect`), the teacher marks
//! them and may approve a training pass on the open model.
use super::courses::{check_id, data_dir, fresh_id};
use super::*;
use yendory_engine::learn::designer::{markdown_to_plain, BlockKind};
use yendory_engine::learn::practice::Mark;
use yendory_engine::learn::reflect::{
    connections, reflect, reflect_again, train, Observation, ReflectionTraining, Statement,
};
use yendory_engine::memory::candidates;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReflectionSession {
    id: String,
    model_id: String,
    model_name: String,
    /// What was revisited: "Study Mode memories" or a lesson's reading.
    material: String,
    /// The statements are Study Mode memories (shown to Nova as memory lines).
    memory_mode: bool,
    statements: usize,
    created: u64,
    step: u64,
    observations: Vec<Observation>,
    #[serde(default)]
    result: Option<ReflectionTraining>,
    #[serde(default)]
    step_after: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

fn path(app: &AppHandle, id: &str) -> Res<PathBuf> {
    check_id(id)?;
    Ok(data_dir(app, "reflections")?.join(format!("{id}.json")))
}
fn load(app: &AppHandle, id: &str) -> Res<ReflectionSession> {
    let b = std::fs::read(path(app, id)?).map_err(|_| "This reflection no longer exists".to_string())?;
    serde_json::from_slice(&b).map_err(|e| e.to_string())
}
fn store(app: &AppHandle, s: &ReflectionSession) -> Res<()> {
    write_atomic(&path(app, &s.id)?, &serde_json::to_vec(s).map_err(|e| e.to_string())?)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReflectionSource {
    kind: String,
    id: String,
    name: String,
    statements: usize,
}

fn sentences_of(texts: impl Iterator<Item = String>, source: &str) -> Vec<Statement> {
    let mut out: Vec<Statement> = Vec::new();
    for t in texts {
        for s in candidates(&t) {
            if !out.iter().any(|x| x.text == s) {
                out.push(Statement { text: s, memory: None, source: source.to_string() });
            }
        }
    }
    out
}

fn statements(app: &AppHandle, studio: &Studio, kind: &str, id: &str) -> Res<(String, bool, Vec<Statement>)> {
    match kind {
        "memory" => {
            if !super::study::study_mode(app) {
                return Err("Study Mode is off: reflect on a lesson's reading instead".into());
            }
            let model_id = with_active(studio, |a| Ok(a.meta.id.clone()))?;
            let st = super::study::load_memory(app, &model_id);
            let list = st
                .entries
                .iter()
                .filter(|m| m.enabled)
                .map(|m| Statement {
                    text: m.text.clone(),
                    memory: Some(m.id),
                    source: m.sources.first().map(|s| s.lesson.clone()).unwrap_or_else(|| "teacher".into()),
                })
                .collect();
            Ok(("Study Mode memories".into(), true, list))
        }
        "design" => {
            let v = super::designer::design_detail(app.clone(), id.to_string())?;
            let d = v.draft_ref().clone();
            let texts = d
                .blocks
                .iter()
                .filter(|b| b.kind == BlockKind::Reading && b.use_in_teaching)
                .map(|b| markdown_to_plain(&b.text));
            Ok((format!("Reading of {}", d.name), false, sentences_of(texts, &d.name)))
        }
        "classroom" => {
            let i: usize = id.parse().map_err(|_| "No such lesson")?;
            let f = studio.lessons.lock().unwrap().get(i).cloned().ok_or("No such lesson")?;
            let items = parse_lesson(&f.text, &f.name)?;
            let texts = items
                .iter()
                .filter(|i| i.reading.is_some())
                .flat_map(|i| i.segments.iter().map(|s| s.text.clone()));
            Ok((format!("Reading of {}", f.name), false, sentences_of(texts, &f.name)))
        }
        _ => Err("Unknown material".into()),
    }
}

#[tauri::command]
pub fn reflection_sources(app: AppHandle, studio: State<Studio>) -> Res<Vec<ReflectionSource>> {
    let mut out = Vec::new();
    if super::study::study_mode(&app) {
        if let Ok((name, _, s)) = statements(&app, &studio, "memory", "") {
            if s.len() >= 2 {
                out.push(ReflectionSource {
                    kind: "memory".into(),
                    id: String::new(),
                    name,
                    statements: s.len(),
                });
            }
        }
    }
    for d in super::designer::list_designs(app.clone())? {
        if let Ok((name, _, s)) = statements(&app, &studio, "design", &d.id) {
            if s.len() >= 2 {
                out.push(ReflectionSource {
                    kind: "design".into(),
                    id: d.id.clone(),
                    name,
                    statements: s.len(),
                });
            }
        }
    }
    let n = studio.lessons.lock().unwrap().len();
    for i in 0..n {
        if let Ok((name, _, s)) = statements(&app, &studio, "classroom", &i.to_string()) {
            if s.len() >= 2 {
                out.push(ReflectionSource {
                    kind: "classroom".into(),
                    id: i.to_string(),
                    name,
                    statements: s.len(),
                });
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReflectionSummary {
    id: String,
    material: String,
    model_name: String,
    created: u64,
    observations: usize,
    marked: usize,
    trained: bool,
}

#[tauri::command]
pub fn list_reflections(app: AppHandle) -> Res<Vec<ReflectionSummary>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(data_dir(&app, "reflections")?).map_err(|e| e.to_string())?.flatten() {
        if let Some(s) =
            std::fs::read(e.path()).ok().and_then(|b| serde_json::from_slice::<ReflectionSession>(&b).ok())
        {
            out.push(ReflectionSummary {
                id: s.id,
                material: s.material,
                model_name: s.model_name,
                created: s.created,
                observations: s.observations.len(),
                marked: s.observations.iter().filter(|o| o.mark.is_some()).count(),
                trained: s.result.is_some(),
            });
        }
    }
    out.sort_by_key(|s| std::cmp::Reverse(s.created));
    Ok(out)
}

#[tauri::command]
pub fn reflection_detail(app: AppHandle, id: String) -> Res<ReflectionSession> {
    load(&app, &id)
}

#[tauri::command]
pub fn delete_reflection(app: AppHandle, id: String) -> Res<()> {
    std::fs::remove_file(path(&app, &id)?).map_err(|e| e.to_string())
}

/// Revisits the material: connections are proposed and measured, Nova writes her observations.
#[tauri::command]
pub async fn start_reflection(
    app: AppHandle,
    studio: State<'_, Studio>,
    kind: String,
    id: String,
    count: usize,
) -> Res<ReflectionSession> {
    not_teaching(&studio)?;
    let (model_id, model_name, model) =
        with_active(&studio, |a| Ok((a.meta.id.clone(), a.meta.name.clone(), a.model.clone())))?;
    let (material, memory_mode, list) = statements(&app, &studio, &kind, &id)?;
    if list.len() < 2 {
        return Err("Not enough to reflect on: at least two statements are needed".into());
    }
    let count = count.clamp(1, 24);
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let conns = connections(&model, &list, count)?;
        if conns.is_empty() {
            return Err("No two statements share a concept word: nothing to connect yet".to_string());
        }
        // The measurement is kept on the memories (how much each helped predict another).
        if memory_mode {
            let mut st = super::study::load_memory(&app2, &model_id);
            for c in &conns {
                let helper = if c.a_helps_b { &c.a } else { &c.b };
                if let Some(mid) = helper.memory {
                    if let Some(m) = st.entries.iter_mut().find(|m| m.id == mid) {
                        m.usefulness = Some(m.usefulness.map_or(c.help, |u| u.max(c.help)));
                    }
                }
                let ids: Vec<u64> = [c.a.memory, c.b.memory].into_iter().flatten().collect();
                st.record("reflection", &format!("{} / {}", c.a.text, c.b.text), &ids, now());
            }
            super::study::save_memory(&app2, &model_id, &st)?;
        }
        let observations = reflect(&model, conns, memory_mode)?;
        let s = ReflectionSession {
            id: fresh_id('r'),
            model_id,
            model_name,
            material,
            memory_mode,
            statements: list.len(),
            created: now(),
            step: model.step,
            observations,
            result: None,
            step_after: None,
            error: None,
        };
        store(&app2, &s)?;
        Ok(s)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservationEdit {
    mark: Option<Mark>,
    correction: String,
    note: String,
}

#[tauri::command]
pub fn save_reflection_marks(
    app: AppHandle,
    id: String,
    marks: Vec<ObservationEdit>,
) -> Res<ReflectionSession> {
    let mut s = load(&app, &id)?;
    if s.result.is_some() {
        return Err("This reflection was already trained".into());
    }
    if marks.len() != s.observations.len() {
        return Err("The marks do not match the observations".into());
    }
    for (o, m) in s.observations.iter_mut().zip(marks) {
        o.mark = m.mark;
        o.correction =
            m.correction.chars().filter(|c| !c.is_control()).collect::<String>().trim().to_string();
        o.note = m.note;
    }
    store(&app, &s)?;
    Ok(s)
}

/// The approved training pass ("reflection-event" progress, "reflection-finished" at the end).
#[tauri::command]
pub fn train_reflection(
    app: AppHandle,
    studio: State<Studio>,
    id: String,
    epochs: u32,
    learning_rate: f32,
) -> Res<()> {
    if !(1..=30).contains(&epochs) || !(learning_rate > 0. && learning_rate <= 0.01) {
        return Err("Choose 1-30 epochs and a learning rate up to 0.01".into());
    }
    let s = load(&app, &id)?;
    if s.result.is_some() {
        return Err("This reflection was already trained".into());
    }
    let (meta, model, archive) =
        with_active(&studio, |a| Ok((a.meta.clone(), a.model.clone(), a.archive.clone())))?;
    if meta.id != s.model_id {
        return Err(format!("Open {} first: these are its observations", s.model_name));
    }
    recovery::no_unfinished_runs(&app, &meta.id, None)?;
    if studio.teaching.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("Wait until teaching or reading has finished".into());
    }
    studio.stop.store(false, Ordering::Release);
    let cfg = SessionConfig {
        epochs,
        batch_size: 8,
        learning_rate,
        foundation: false,
        rehearsal_share: if archive.is_empty() { 0. } else { 0.1 },
        retry: RetryPolicy { retries: 0, requeue: false, ..RetryPolicy::default() },
        grow_vocabulary: false,
        word_limit: meta.word_limit,
        seed: now(),
        ..SessionConfig::default()
    };
    let stop = studio.stop.clone();
    std::thread::spawn(move || {
        let studio = app.state::<Studio>();
        let mut s = s;
        let sid = s.id.clone();
        let mut observer = |e: &Event| {
            let _ = app.emit("reflection-event", serde_json::json!({"id": sid, "event": e}));
        };
        let mut obs = s.observations.clone();
        let outcome = guarded(|| {
            let (r, rec) = train(&model, &mut obs, s.memory_mode, &archive, &cfg, &mut observer, &stop)?;
            if r.learned {
                let mut m = meta.clone();
                refresh_meta(&mut m, &r.model);
                save_all(&app, &m, &r.model, &archive)?;
                if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == m.id) {
                    a.model = r.model.clone();
                    a.meta = m.clone();
                }
            }
            reflect_again(&r.model, &mut obs, s.memory_mode)?;
            Ok((rec, r.model.step))
        });
        s.observations = obs;
        match outcome {
            Ok((rec, step)) => {
                s.result = Some(rec);
                s.step_after = Some(step);
            }
            Err(e) => s.error = Some(e),
        }
        let _ = store(&app, &s);
        append_history(
            &app,
            &s.model_id,
            serde_json::json!({"type": "reflection", "time": now(), "material": s.material, "result": s.result, "error": s.error}),
        );
        studio.teaching.store(false, Ordering::Release);
        let _ = app.emit("reflection-finished", &s);
    });
    Ok(())
}
