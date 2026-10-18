//! Rewards: the reward settings (Stage 1 recording, Stage 2 reward-guided training) and each
//! model's reward ledger (<model dir>/rewards.json), with the commands that inspect it. Grading,
//! the ledgers and the training emphasis are in `yendory_engine::learn::reward`.
use super::*;
use yendory_engine::learn::reward::{
    AuditEntry, Concepts, LedgerSummary, RewardConfig, RewardLedger, Rewards,
};

pub(crate) fn reward_config(app: &AppHandle) -> RewardConfig {
    read_settings(app).get("rewards").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default()
}

fn ledger_path(app: &AppHandle, model_id: &str) -> Res<PathBuf> {
    Ok(model_dir(app, model_id)?.join("rewards.json"))
}

pub(crate) fn load_ledger(app: &AppHandle, model_id: &str) -> RewardLedger {
    ledger_path(app, model_id)
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

fn save_ledger(app: &AppHandle, model_id: &str, ledger: &RewardLedger) -> Res<()> {
    write_atomic(&ledger_path(app, model_id)?, &serde_json::to_vec(ledger).map_err(|e| e.to_string())?)
}

/// The open model's rewards when recording is on.
pub(crate) fn rewards_for(app: &AppHandle, model_id: &str) -> Option<Rewards> {
    let config = reward_config(app);
    config.record.then(|| Rewards::new(config, load_ledger(app, model_id), model_id))
}

/// Saves a ledger after training (a failure is reported, it does not undo the training).
pub(crate) fn keep(app: &AppHandle, r: &Rewards) {
    if let Err(e) = save_ledger(app, &r.model, &r.ledger) {
        let _ = app.emit("device-notice", format!("The reward ledger could not be saved: {e}"));
    }
}

#[tauri::command]
pub fn reward_settings(app: AppHandle) -> RewardConfig {
    reward_config(&app)
}

fn check(c: &RewardConfig) -> Res<()> {
    let within = |v: f32, lo: f32, hi: f32| v.is_finite() && (lo..=hi).contains(&v);
    let ok = within(c.progress_budget, 0., 10.)
        && within(c.first_letter, 0., 1.)
        && within(c.per_letter, 0., 1.)
        && within(c.partial_cap, 0., 1.)
        && (1..=1000).contains(&c.discovery_attempts)
        && within(c.relevance_word, 0., 0.1)
        && within(c.relevance_cap, 0., 1.)
        && within(c.incorrect_reward, -1., 0.)
        && within(c.retention_first, 0., 10.)
        && within(c.retention_repeat, 0., 10.)
        && within(c.retention_floor, 0., 1.)
        && within(c.generalization, 0., 10.)
        && within(c.memory_share, 0., 1.)
        && c.quiz_limit <= 500
        && within(c.min_emphasis, 0.05, 1.)
        && within(c.max_emphasis, 1., 4.);
    if !ok {
        return Err("A reward setting is out of range (see the limits beside each value)".into());
    }
    if c.weighted_training && !c.record {
        return Err("Reward-guided training needs the ledger: turn on recording too".into());
    }
    if c.weighted_training && c.quiz_limit == 0 {
        return Err("Reward-guided training needs at least one question asked per session".into());
    }
    Ok(())
}

#[tauri::command]
pub fn save_reward_settings(
    app: AppHandle,
    studio: State<Studio>,
    config: RewardConfig,
) -> Res<RewardConfig> {
    not_teaching(&studio)?;
    check(&config)?;
    write_setting(&app, "rewards", serde_json::to_value(&config).map_err(|e| e.to_string())?)?;
    Ok(config)
}

#[tauri::command]
pub fn reset_reward_settings(app: AppHandle, studio: State<Studio>) -> Res<RewardConfig> {
    save_reward_settings(app, studio, RewardConfig::default())
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RewardView {
    config: RewardConfig,
    model: Option<String>,
    summary: Option<LedgerSummary>,
}

#[tauri::command]
pub fn reward_view(app: AppHandle, studio: State<Studio>) -> RewardView {
    let open = studio.active.lock().unwrap().as_ref().map(|a| (a.meta.id.clone(), a.meta.name.clone()));
    RewardView {
        config: reward_config(&app),
        summary: open.as_ref().map(|(id, _)| load_ledger(&app, id).summary()),
        model: open.map(|(_, n)| n),
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionRow {
    id: String,
    lesson: String,
    question: String,
    expected: String,
    template: String,
    trained: bool,
    unseen: bool,
    last_percent: Option<f32>,
    last_memory_percent: Option<f32>,
    first_percent: Option<f32>,
    failures: u32,
    attempts: u32,
    retained: usize,
    forgotten: bool,
    retention: f32,
    generalization: f32,
    milestones: Vec<String>,
    complete: bool,
    /// Training emphasis now (Stage 2), with its reasons; taught questions only.
    emphasis: Option<f32>,
    reasons: Vec<String>,
    notes: Vec<String>,
    checkpoints: Vec<yendory_engine::learn::reward::CheckpointResult>,
}

/// The open model's recorded questions: `filter` = all, gaps, stuck, forgotten, retained,
/// unseen or memory (right only with memories); `search` matches question, answer or lesson.
#[tauri::command]
pub fn reward_questions(
    app: AppHandle,
    studio: State<Studio>,
    filter: String,
    search: String,
) -> Res<Vec<QuestionRow>> {
    let id = with_active(&studio, |a| Ok(a.meta.id.clone()))?;
    let cfg = reward_config(&app);
    let ledger = load_ledger(&app, &id);
    let concepts = Concepts::all(&ledger);
    let needle = search.trim().to_ascii_lowercase();
    let mut rows: Vec<QuestionRow> = ledger
        .questions
        .values()
        .filter(|q| match filter.as_str() {
            "gaps" => q.trained && q.last_percent.is_some_and(|p| p < 100.),
            "stuck" => q.failures >= cfg.weights.stuck_after,
            "forgotten" => q.forgotten,
            "retained" => q.retained_streak() >= 2,
            "unseen" => q.unseen,
            "memory" => q.last_memory_percent == Some(100.) && q.last_percent.is_some_and(|p| p < 100.),
            _ => true,
        })
        .filter(|q| {
            needle.is_empty()
                || [&q.question, &q.expected, &q.lesson]
                    .iter()
                    .any(|s| s.to_ascii_lowercase().contains(&needle))
        })
        .map(|q| {
            let cycle = ledger.cycles.get(&q.cycle);
            let e = q.trained.then(|| ledger.emphasis_of(&cfg, &q.id, &concepts));
            QuestionRow {
                id: q.id.clone(),
                lesson: q.lesson.clone(),
                question: q.question.clone(),
                expected: q.expected.clone(),
                template: q.template.clone(),
                trained: q.trained,
                unseen: q.unseen,
                last_percent: q.last_percent,
                last_memory_percent: q.last_memory_percent,
                first_percent: q.first_percent,
                failures: q.failures,
                attempts: q.attempts,
                retained: q.retained_streak(),
                forgotten: q.forgotten,
                retention: q.retention,
                generalization: q.generalization,
                milestones: cycle.map(|c| c.neural.milestones.clone()).unwrap_or_default(),
                complete: cycle.is_some_and(|c| c.neural.complete),
                emphasis: e.as_ref().map(|e| e.value),
                reasons: e.map(|e| e.reasons).unwrap_or_default(),
                notes: q.notes.clone(),
                checkpoints: q.checkpoints.clone(),
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        b.emphasis.unwrap_or(0.).total_cmp(&a.emphasis.unwrap_or(0.)).then(a.question.cmp(&b.question))
    });
    rows.truncate(400);
    Ok(rows)
}

/// The audit log, newest first (of one question when `question` is given).
#[tauri::command]
pub fn reward_audit(
    app: AppHandle,
    studio: State<Studio>,
    question: Option<String>,
    limit: usize,
) -> Res<Vec<AuditEntry>> {
    let id = with_active(&studio, |a| Ok(a.meta.id.clone()))?;
    let ledger = load_ledger(&app, &id);
    Ok(ledger
        .audit
        .iter()
        .rev()
        .filter(|e| match &question {
            Some(q) => &e.question_id == q,
            None => true,
        })
        .take(limit.clamp(1, 1000))
        .cloned()
        .collect())
}

/// Clears one question's reward cycle (its milestones may be earned again).
#[tauri::command]
pub fn reward_reset_cycle(app: AppHandle, studio: State<Studio>, id: String) -> Res<()> {
    not_teaching(&studio)?;
    let model = with_active(&studio, |a| Ok(a.meta.id.clone()))?;
    let mut ledger = load_ledger(&app, &model);
    if !ledger.reset_cycle(&id) {
        return Err("This question is not in the ledger".into());
    }
    save_ledger(&app, &model, &ledger)
}

/// Clears the open model's whole ledger (the model is not affected).
#[tauri::command]
pub fn reward_clear(app: AppHandle, studio: State<Studio>) -> Res<()> {
    not_teaching(&studio)?;
    let model = with_active(&studio, |a| Ok(a.meta.id.clone()))?;
    save_ledger(&app, &model, &RewardLedger::default())
}
