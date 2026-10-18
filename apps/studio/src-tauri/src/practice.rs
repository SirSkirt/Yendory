//! Practice sessions: stored as <app data>/practice/<id>.json. Nova answers a lesson's practice
//! questions (`learn::practice`), the teacher marks them, and an approved training pass runs on
//! the open model through the ordinary teaching session.
use super::courses::{check_id, data_dir, fresh_id};
use super::*;
use yendory_engine::learn::designer::{export, BlockKind};
use yendory_engine::learn::practice::{answer_all, train, Mark, PracticeItem};
use yendory_engine::learn::session::Score;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PracticeResult {
    learned: bool,
    baseline: Score,
    final_score: Score,
    epochs: usize,
    step_before: u64,
    step_after: u64,
    /// Held-out questions right with memory after training (Study Mode), of how many.
    #[serde(default)]
    with_memory: Option<(usize, usize)>,
    time: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PracticeSession {
    id: String,
    model_id: String,
    model_name: String,
    /// The lesson the questions come from.
    lesson: String,
    created: u64,
    /// Nova answered with Study Mode memories in front.
    study_mode: bool,
    items: Vec<PracticeItem>,
    /// The lesson's units (its held-out questions judge the training pass).
    lesson_items: Vec<Item>,
    #[serde(default)]
    result: Option<PracticeResult>,
    #[serde(default)]
    error: Option<String>,
}

fn practice_path(app: &AppHandle, id: &str) -> Res<PathBuf> {
    check_id(id)?;
    Ok(data_dir(app, "practice")?.join(format!("{id}.json")))
}
fn load(app: &AppHandle, id: &str) -> Res<PracticeSession> {
    let b = std::fs::read(practice_path(app, id)?)
        .map_err(|_| "This practice session no longer exists".to_string())?;
    serde_json::from_slice(&b).map_err(|e| e.to_string())
}
fn store(app: &AppHandle, s: &PracticeSession) -> Res<()> {
    write_atomic(&practice_path(app, &s.id)?, &serde_json::to_vec(s).map_err(|e| e.to_string())?)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PracticeSource {
    kind: String,
    id: String,
    name: String,
    questions: usize,
}

/// Lessons with questions to practise: Lesson Designer lessons (their Practice blocks) and the
/// Classroom list (their taught questions).
#[tauri::command]
pub fn practice_sources(app: AppHandle, studio: State<Studio>) -> Res<Vec<PracticeSource>> {
    let mut out = Vec::new();
    for d in super::designer::list_designs(app.clone())? {
        let view = super::designer::design_detail(app.clone(), d.id.clone())?;
        let draft = view_draft(&view);
        let n = draft
            .blocks
            .iter()
            .filter(|b| b.kind == BlockKind::Practice && b.use_in_teaching)
            .map(|b| b.items.iter().filter(|i| !i.question.trim().is_empty()).count())
            .sum();
        if n > 0 {
            out.push(PracticeSource { kind: "design".into(), id: d.id, name: d.name, questions: n });
        }
    }
    for (i, f) in studio.lessons.lock().unwrap().iter().enumerate() {
        if let Ok(items) = parse_lesson(&f.text, &f.name) {
            let n =
                items.iter().filter(|it| it.trainable() && it.prompt.is_some()).count().min(MAX_QUESTIONS);
            if n > 0 {
                out.push(PracticeSource {
                    kind: "classroom".into(),
                    id: i.to_string(),
                    name: f.name.clone(),
                    questions: n,
                });
            }
        }
    }
    Ok(out)
}

/// At most this many questions per practice session.
const MAX_QUESTIONS: usize = 40;

fn view_draft(v: &super::designer::DesignView) -> yendory_engine::learn::designer::LessonDraft {
    v.draft_ref().clone()
}

/// The questions and the lesson units of a source.
fn source_items(
    app: &AppHandle,
    studio: &Studio,
    kind: &str,
    id: &str,
) -> Res<(String, Vec<PracticeItem>, Vec<Item>)> {
    let item = |q: &str, a: Option<&str>| PracticeItem {
        question: q.split_whitespace().collect::<Vec<_>>().join(" "),
        expected: a.filter(|a| !a.trim().is_empty()).map(String::from),
        answer: String::new(),
        memories: Vec::new(),
        mark: None,
        correction: String::new(),
        note: String::new(),
        target: None,
        after: None,
    };
    match kind {
        "design" => {
            let view = super::designer::design_detail(app.clone(), id.to_string())?;
            let draft = view_draft(&view);
            let ex = export(&draft)?;
            let lesson = parse_lesson(&ex.text, &draft.name)?;
            let qs = draft
                .blocks
                .iter()
                .filter(|b| b.kind == BlockKind::Practice && b.use_in_teaching)
                .flat_map(|b| b.items.iter())
                .filter(|i| !i.question.trim().is_empty())
                .take(MAX_QUESTIONS)
                .map(|i| {
                    item(&yendory_engine::learn::designer::markdown_to_plain(&i.question), Some(&i.answer))
                })
                .collect();
            Ok((draft.name, qs, lesson))
        }
        "classroom" => {
            let i: usize = id.parse().map_err(|_| "No such lesson")?;
            let f = studio.lessons.lock().unwrap().get(i).cloned().ok_or("No such lesson")?;
            let lesson = parse_lesson(&f.text, &f.name)?;
            let qs = lesson
                .iter()
                .filter(|it| it.trainable() && it.prompt.is_some())
                .take(MAX_QUESTIONS)
                .map(|it| item(it.prompt.as_deref().unwrap_or(""), it.answer.as_deref()))
                .collect();
            Ok((f.name, qs, lesson))
        }
        _ => Err("Unknown practice source".into()),
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PracticeSummary {
    id: String,
    lesson: String,
    model_name: String,
    created: u64,
    questions: usize,
    marked: usize,
    trained: bool,
}

#[tauri::command]
pub fn list_practice(app: AppHandle) -> Res<Vec<PracticeSummary>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(data_dir(&app, "practice")?).map_err(|e| e.to_string())?.flatten() {
        if let Some(s) =
            std::fs::read(e.path()).ok().and_then(|b| serde_json::from_slice::<PracticeSession>(&b).ok())
        {
            out.push(PracticeSummary {
                id: s.id,
                lesson: s.lesson,
                model_name: s.model_name,
                created: s.created,
                questions: s.items.len(),
                marked: s.items.iter().filter(|i| i.mark.is_some()).count(),
                trained: s.result.is_some(),
            });
        }
    }
    out.sort_by_key(|s| std::cmp::Reverse(s.created));
    Ok(out)
}

#[tauri::command]
pub fn practice_detail(app: AppHandle, id: String) -> Res<PracticeSession> {
    load(&app, &id)
}

/// Nova answers the lesson's practice questions with the open model.
#[tauri::command]
pub async fn start_practice(
    app: AppHandle,
    studio: State<'_, Studio>,
    kind: String,
    id: String,
) -> Res<PracticeSession> {
    not_teaching(&studio)?;
    let (model_id, model_name, model) =
        with_active(&studio, |a| Ok((a.meta.id.clone(), a.meta.name.clone(), a.model.clone())))?;
    let (lesson, mut items, lesson_items) = source_items(&app, &studio, &kind, &id)?;
    if items.is_empty() {
        return Err("This lesson has no practice questions".into());
    }
    let study_mode = super::study::study_mode(&app);
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut memory = study_mode.then(|| super::study::load_memory(&app2, &model_id));
        answer_all(&model, &mut items, memory.as_mut(), false, now())?;
        if let Some(m) = &memory {
            super::study::save_memory(&app2, &model_id, m)?;
        }
        let s = PracticeSession {
            id: fresh_id('p'),
            model_id,
            model_name,
            lesson,
            created: now(),
            study_mode,
            items,
            lesson_items,
            result: None,
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
pub struct MarkEdit {
    mark: Option<Mark>,
    correction: String,
    note: String,
}

/// Saves the teacher's marks, corrections and notes.
#[tauri::command]
pub fn save_practice_marks(app: AppHandle, id: String, marks: Vec<MarkEdit>) -> Res<PracticeSession> {
    let mut s = load(&app, &id)?;
    if s.result.is_some() {
        return Err("This session was already trained; start a new practice session".into());
    }
    if marks.len() != s.items.len() {
        return Err("The marks do not match the questions".into());
    }
    for (it, m) in s.items.iter_mut().zip(marks) {
        it.mark = m.mark;
        it.correction =
            m.correction.chars().filter(|c| !c.is_control()).collect::<String>().trim().to_string();
        it.note = m.note;
    }
    store(&app, &s)?;
    Ok(s)
}

#[tauri::command]
pub fn delete_practice(app: AppHandle, id: String) -> Res<()> {
    std::fs::remove_file(practice_path(&app, &id)?).map_err(|e| e.to_string())
}

/// The training pass on the marked answers (the open model must be the one that answered).
/// Progress arrives as "practice-event" (session events), the end as "practice-finished".
#[tauri::command]
pub fn train_practice(
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
        return Err("This session was already trained".into());
    }
    if !s.items.iter().any(|i| yendory_engine::learn::practice::target(i).is_some()) {
        return Err("Mark Nova's answers first (right, or wrong with a correction)".into());
    }
    if !s.lesson_items.iter().any(|i| i.held_out) {
        return Err("The lesson has no held-out questions to judge the training by".into());
    }
    let (meta, model, archive) =
        with_active(&studio, |a| Ok((a.meta.clone(), a.model.clone(), a.archive.clone())))?;
    if meta.id != s.model_id {
        return Err(format!("Open {} first: it gave these answers", s.model_name));
    }
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
        let mut observer = |e: &Event| {
            let _ = app.emit("practice-event", serde_json::json!({"id": s.id, "event": e}));
        };
        let step_before = model.step;
        let mut items = s.items.clone();
        let outcome = guarded(|| {
            let r = train(&model, &mut items, &s.lesson_items, &archive, &cfg, &mut observer, &stop)?;
            let Some(r) = r else { return Err("Nothing was marked for training".to_string()) };
            let mut m = meta.clone();
            if r.learned {
                refresh_meta(&mut m, &r.model);
                save_all(&app, &m, &r.model, &archive)?;
                if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == m.id) {
                    a.model = r.model.clone();
                    a.meta = m.clone();
                }
            }
            // Nova answers the same questions again, and (Study Mode) the held-out ones with memory.
            let mut memory = s.study_mode.then(|| super::study::load_memory(&app, &m.id));
            answer_all(&r.model, &mut items, memory.as_mut(), true, now())?;
            let with_memory = memory.as_mut().and_then(|store| {
                let eval = yendory_engine::memory::evaluation_items(&r.model, store, &s.lesson_items, now());
                let refs: Vec<&Item> = eval.iter().collect();
                evaluate(&r.model, &refs, Ablation::default())
                    .ok()
                    .map(|rep| (rep.overall().correct, rep.overall().total))
            });
            if let Some(store) = &memory {
                super::study::save_memory(&app, &m.id, store)?;
            }
            Ok(PracticeResult {
                learned: r.learned,
                baseline: r.baseline,
                final_score: r.final_score,
                epochs: r.epochs.len(),
                step_before,
                step_after: r.model.step,
                with_memory,
                time: now(),
            })
        });
        s.items = items;
        match outcome {
            Ok(r) => s.result = Some(r),
            Err(e) => s.error = Some(e),
        }
        let _ = store(&app, &s);
        append_history(
            &app,
            &s.model_id,
            serde_json::json!({"type": "practice", "time": now(), "lesson": s.lesson, "result": s.result, "error": s.error,
                "targets": s.items.iter().map(|i| &i.target).collect::<Vec<_>>()}),
        );
        studio.teaching.store(false, Ordering::Release);
        let _ = app.emit("practice-finished", &s);
    });
    Ok(())
}
