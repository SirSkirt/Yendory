//! Review Training sessions: stored as <app data>/reviews/<id>.json, with their material as a
//! Lesson Designer draft. Generation, assessment and comparison are `learn::review`; training is
//! the ordinary teaching session on the open model.
use super::courses::{check_id, data_dir, fresh_id};
use super::*;
use yendory_engine::learn::designer::{validate, Severity};
use yendory_engine::learn::review::{
    assess, compare, generate, lesson_items, topics, Assessment, Comparison, ReviewOptions,
};
use yendory_engine::learn::session::Score;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewTraining {
    learned: bool,
    baseline: Score,
    final_score: Score,
    step_before: u64,
    step_after: u64,
    epochs: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewSession {
    id: String,
    design_id: String,
    name: String,
    model_id: String,
    model_name: String,
    created: u64,
    memories: Vec<u64>,
    difficulty: u8,
    #[serde(default)]
    before: Option<Assessment>,
    #[serde(default)]
    after: Option<Assessment>,
    #[serde(default)]
    training: Option<ReviewTraining>,
    #[serde(default)]
    comparison: Option<Comparison>,
    #[serde(default)]
    error: Option<String>,
}

fn path(app: &AppHandle, id: &str) -> Res<PathBuf> {
    check_id(id)?;
    Ok(data_dir(app, "reviews")?.join(format!("{id}.json")))
}
fn load(app: &AppHandle, id: &str) -> Res<ReviewSession> {
    let b = std::fs::read(path(app, id)?).map_err(|_| "This review no longer exists".to_string())?;
    serde_json::from_slice(&b).map_err(|e| e.to_string())
}
fn store(app: &AppHandle, s: &ReviewSession) -> Res<()> {
    write_atomic(&path(app, &s.id)?, &serde_json::to_vec(s).map_err(|e| e.to_string())?)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewTopics {
    study_mode: bool,
    memories: usize,
    concepts: Vec<(String, Vec<u64>)>,
    lessons: Vec<(String, Vec<u64>)>,
}

#[tauri::command]
pub fn review_topics(app: AppHandle, studio: State<Studio>) -> Res<ReviewTopics> {
    let model_id = with_active(&studio, |a| Ok(a.meta.id.clone()))?;
    let on = super::study::study_mode(&app);
    if !on {
        return Ok(ReviewTopics { study_mode: false, memories: 0, concepts: vec![], lessons: vec![] });
    }
    let st = super::study::load_memory(&app, &model_id);
    let (concepts, lessons) = topics(&st);
    Ok(ReviewTopics {
        study_mode: true,
        memories: st.entries.iter().filter(|m| m.enabled).count(),
        concepts,
        lessons,
    })
}

/// Generates review material from the memories of the chosen concepts and lessons (all when
/// none is chosen) as a new Designer lesson, and opens a review session for it.
#[tauri::command]
pub fn generate_review(
    app: AppHandle,
    studio: State<Studio>,
    concepts: Vec<String>,
    lessons: Vec<String>,
    difficulty: u8,
) -> Res<ReviewSession> {
    if !super::study::study_mode(&app) {
        return Err("Review Training uses Nova's memory: turn Study Mode on first".into());
    }
    let (model_id, model_name) = with_active(&studio, |a| Ok((a.meta.id.clone(), a.meta.name.clone())))?;
    let st = super::study::load_memory(&app, &model_id);
    let (by_term, by_lesson) = topics(&st);
    let mut ids: Vec<u64> = Vec::new();
    for (_, v) in by_term.iter().filter(|(t, _)| concepts.contains(t)) {
        ids.extend(v);
    }
    for (_, v) in by_lesson.iter().filter(|(l, _)| lessons.contains(l)) {
        ids.extend(v);
    }
    if concepts.is_empty() && lessons.is_empty() {
        ids = st.entries.iter().filter(|m| m.enabled).map(|m| m.id).collect();
    }
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() {
        return Err("No memories to review: study some reading first".into());
    }
    let what = if concepts.is_empty() && lessons.is_empty() {
        "all memories".to_string()
    } else {
        concepts.iter().chain(lessons.iter()).cloned().collect::<Vec<_>>().join(", ")
    };
    let name = format!("Review - {what}");
    let draft = generate(&st, &ids, ReviewOptions { difficulty: difficulty.clamp(1, 3) }, &name);
    if draft.blocks.iter().all(|b| b.items.is_empty()) {
        return Err("None of these memories fits a pattern that gives questions; write review questions in the Designer instead".into());
    }
    let design_id = super::designer::create_from(&app, draft)?;
    let s = ReviewSession {
        id: fresh_id('v'),
        design_id,
        name,
        model_id,
        model_name,
        created: now(),
        memories: ids,
        difficulty,
        before: None,
        after: None,
        training: None,
        comparison: None,
        error: None,
    };
    store(&app, &s)?;
    Ok(s)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewView {
    session: ReviewSession,
    /// Problems in the material that block assessment and training (unverified answers...).
    errors: Vec<String>,
    examples: usize,
    held_out: usize,
}

fn view(app: &AppHandle, s: ReviewSession) -> ReviewView {
    let (errors, examples, held_out) = match super::designer::load_draft(app, &s.design_id) {
        Ok(d) => {
            let errors = validate(&d)
                .into_iter()
                .filter(|i| i.severity == Severity::Error)
                .map(|i| i.message)
                .collect();
            let count = |k: yendory_engine::learn::designer::BlockKind| {
                d.blocks
                    .iter()
                    .filter(|b| b.kind == k && b.use_in_teaching)
                    .map(|b| b.items.len())
                    .sum::<usize>()
            };
            let (ex, ho) = (
                count(yendory_engine::learn::designer::BlockKind::Examples),
                count(yendory_engine::learn::designer::BlockKind::Evaluation),
            );
            (errors, ex, ho)
        }
        Err(e) => (vec![e], 0, 0),
    };
    ReviewView { session: s, errors, examples, held_out }
}

#[tauri::command]
pub fn list_reviews(app: AppHandle) -> Res<Vec<ReviewSession>> {
    let mut out: Vec<ReviewSession> = std::fs::read_dir(data_dir(&app, "reviews")?)
        .map_err(|e| e.to_string())?
        .flatten()
        .filter_map(|e| std::fs::read(e.path()).ok().and_then(|b| serde_json::from_slice(&b).ok()))
        .collect();
    out.sort_by_key(|s| std::cmp::Reverse(s.created));
    Ok(out)
}

#[tauri::command]
pub fn review_detail(app: AppHandle, id: String) -> Res<ReviewView> {
    Ok(view(&app, load(&app, &id)?))
}

#[tauri::command]
pub fn delete_review(app: AppHandle, id: String) -> Res<()> {
    std::fs::remove_file(path(&app, &id)?).map_err(|e| e.to_string())
}

/// The approved material's units (errors refused) and the model that must be open.
fn ready(
    app: &AppHandle,
    studio: &Studio,
    s: &ReviewSession,
) -> Res<(Vec<Item>, Meta, Bicameral, Vec<Item>)> {
    let d = super::designer::load_draft(app, &s.design_id)?;
    let errors: Vec<String> =
        validate(&d).into_iter().filter(|i| i.severity == Severity::Error).map(|i| i.message).collect();
    if !errors.is_empty() {
        return Err(format!("Review the material first ({} to fix, e.g. {})", errors.len(), errors[0]));
    }
    let items = lesson_items(&d)?;
    if !items.iter().any(|i| i.held_out) {
        return Err("The material has no held-out review questions".into());
    }
    let (meta, model, archive) =
        with_active(studio, |a| Ok((a.meta.clone(), a.model.clone(), a.archive.clone())))?;
    if meta.id != s.model_id {
        return Err(format!("Open {} first: this review is about its memories", s.model_name));
    }
    Ok((items, meta, model, archive))
}

/// The baseline: the held-out review questions without memory (and with memory, reported apart).
#[tauri::command]
pub async fn review_assess(app: AppHandle, studio: State<'_, Studio>, id: String) -> Res<ReviewView> {
    not_teaching(&studio)?;
    let mut s = load(&app, &id)?;
    if s.training.is_some() {
        return Err("This review was already trained; start a new review to assess again".into());
    }
    let (items, meta, model, _) = ready(&app, &studio, &s)?;
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut memory = super::study::study_mode(&app2).then(|| super::study::load_memory(&app2, &meta.id));
        let before = assess(&model, &items, memory.as_mut(), now())?;
        if let Some(mut r) = super::rewards::rewards_for(&app2, &meta.id) {
            yendory_engine::learn::review::record_assessment(&mut r, &items, &before);
            super::rewards::keep(&app2, &r);
        }
        s.before = Some(before);
        if let Some(m) = &memory {
            super::study::save_memory(&app2, &meta.id, m)?;
        }
        store(&app2, &s)?;
        Ok(view(&app2, s))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Review training: baseline (if not yet taken), neural training on the examples (no memory),
/// assessment after, comparison. "review-event" progress, "review-finished" at the end.
#[tauri::command]
pub fn train_review(
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
    if s.training.is_some() {
        return Err("This review was already trained".into());
    }
    let (items, meta, model, archive) = ready(&app, &studio, &s)?;
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
        let outcome = guarded(|| {
            let mut memory =
                super::study::study_mode(&app).then(|| super::study::load_memory(&app, &meta.id));
            let mut rewards = super::rewards::rewards_for(&app, &meta.id);
            if s.before.is_none() {
                let _ = app.emit("review-event", serde_json::json!({"id": sid, "phase": "baseline"}));
                let before = assess(&model, &items, memory.as_mut(), now())?;
                if let Some(r) = rewards.as_mut() {
                    yendory_engine::learn::review::record_assessment(r, &items, &before);
                }
                s.before = Some(before);
            }
            let mut observer = |e: &Event| {
                let _ = app.emit("review-event", serde_json::json!({"id": sid, "event": e}));
            };
            let r = yendory_engine::learn::session::teach_rewarded(
                &model,
                &items,
                None,
                &archive,
                &cfg,
                rewards.as_mut(),
                &mut observer,
                &stop,
            );
            if let Some(rw) = &rewards {
                super::rewards::keep(&app, rw);
            }
            let r = r?;
            if r.learned {
                let mut m = meta.clone();
                refresh_meta(&mut m, &r.model);
                save_all(&app, &m, &r.model, &archive)?;
                if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == m.id) {
                    a.model = r.model.clone();
                    a.meta = m.clone();
                }
            }
            let _ = app.emit("review-event", serde_json::json!({"id": sid, "phase": "after"}));
            let after = assess(&r.model, &items, memory.as_mut(), now())?;
            if let Some(rw) = rewards.as_mut() {
                yendory_engine::learn::review::record_assessment(rw, &items, &after);
                super::rewards::keep(&app, rw);
            }
            if let Some(m) = &memory {
                super::study::save_memory(&app, &meta.id, m)?;
            }
            s.comparison = s.before.as_ref().map(|b| compare(b, &after));
            s.after = Some(after);
            s.training = Some(ReviewTraining {
                learned: r.learned,
                baseline: r.baseline,
                final_score: r.final_score,
                step_before: model.step,
                step_after: r.model.step,
                epochs: r.epochs.len(),
            });
            Ok(())
        });
        if let Err(e) = outcome {
            s.error = Some(e);
        }
        let _ = store(&app, &s);
        append_history(
            &app,
            &s.model_id,
            serde_json::json!({"type": "review_training", "time": now(), "name": s.name, "memories": s.memories,
                "before": s.before.as_ref().map(|a| (a.correct, a.total)), "after": s.after.as_ref().map(|a| (a.correct, a.total)), "error": s.error}),
        );
        studio.teaching.store(false, Ordering::Release);
        let _ = app.emit("review-finished", view(&app, s));
    });
    Ok(())
}
