//! Run records and recovery points for Reading and Conversation (0.9.0).
//!
//! Every reading or conversation run has a record, `models/<id>/runs/<run>/run.json`: what kind of
//! run it is, the settings it was started with, the datasets (with their SHA-256), its progress and
//! its status. While it trains, the Engine hands out recovery snapshots (the working model with
//! its optimizer state, and the run's position: step, random generator, judge, curriculum) on an
//! interval and when the run is stopped. Each snapshot goes into one of two slots, `slot-a` and
//! `slot-b`, in turn, so writing one never touches the other; its files' SHA-256 are written into
//! the record last (atomically), so the record always points at a complete snapshot. Loading
//! checks the sums and falls back to the other slot if the newest is damaged.
//!
//! The accepted model (`model.nnova2`) is saved separately when a check accepts it; a recovery
//! snapshot never replaces it.
//!
//! A record still marked `running` when Studio starts belongs to a run that was interrupted (the
//! app closed, crashed, or the computer restarted). Studio asks what to do with it; it never
//! resumes or discards one by itself.
use super::*;
use serde::de::DeserializeOwned;
use std::path::Path;
use yendory_engine::learn::recovery::{Resume, Snapshot};

pub(crate) const RUN_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunStatus {
    /// Training (or, found at start-up, interrupted).
    Running,
    /// Stopped by the user; it can be continued.
    Stopped,
    /// Ended with an error; it can be continued from its last recovery point.
    Failed,
    Finished,
    /// The user discarded it (its recovery point is deleted).
    Discarded,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FileSum {
    pub name: String,
    pub sha256: String,
    pub bytes: u64,
}

/// One recovery snapshot on disk.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SlotInfo {
    pub slot: String,
    pub step: u64,
    pub time: u64,
    /// Taken because the run was stopped.
    pub stopped: bool,
    pub files: Vec<FileSum>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetRef {
    pub name: String,
    pub sha256: String,
}

/// The plan step a run belongs to (runs of a plan are continued from the Planner).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PlanRef {
    pub plan_id: String,
    pub step_id: String,
    pub plan_name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RunRecord {
    pub version: u32,
    pub id: String,
    /// "reading" or "conversation".
    pub kind: String,
    pub model_id: String,
    pub model_name: String,
    #[serde(default)]
    pub plan: Option<PlanRef>,
    pub label: String,
    pub started: u64,
    pub updated: u64,
    pub status: RunStatus,
    /// Studio's version when the run started.
    pub app_version: String,
    /// The page's settings the run was started with (ReadOptions or ConverseOptions): a
    /// resumed run uses these, not the page's current ones.
    pub options: serde_json::Value,
    pub datasets: Vec<DatasetRef>,
    pub step: u64,
    pub steps: u64,
    pub tokens: u64,
    /// The newest recovery snapshot and the one before it.
    pub snapshot: Option<SlotInfo>,
    pub previous: Option<SlotInfo>,
    #[serde(default)]
    pub error: Option<String>,
    /// How many times it was continued.
    #[serde(default)]
    pub resumed: u32,
    /// Recovery points that could not be written (e.g. the disk was full), and the last reason.
    #[serde(default)]
    pub failed_saves: u32,
    #[serde(default)]
    pub save_error: Option<String>,
}

impl RunRecord {
    /// It can be continued from a recovery point.
    pub fn resumable(&self) -> bool {
        matches!(self.status, RunStatus::Running | RunStatus::Stopped | RunStatus::Failed)
            && self.snapshot.is_some()
    }
    /// Still needs a decision (continue or discard).
    pub fn unfinished(&self) -> bool {
        matches!(self.status, RunStatus::Running | RunStatus::Stopped | RunStatus::Failed)
    }
}

pub(crate) fn runs_dir(model_dir: &Path) -> std::path::PathBuf {
    model_dir.join("runs")
}

fn run_dir(model_dir: &Path, run_id: &str) -> Res<std::path::PathBuf> {
    if run_id.is_empty() || !run_id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err("Invalid run id".into());
    }
    Ok(runs_dir(model_dir).join(run_id))
}

pub(crate) fn save_record(model_dir: &Path, r: &RunRecord) -> Res<()> {
    let dir = run_dir(model_dir, &r.id)?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_atomic(&dir.join("run.json"), &serde_json::to_vec_pretty(r).map_err(|e| e.to_string())?)
}

pub(crate) fn load_record(model_dir: &Path, run_id: &str) -> Res<RunRecord> {
    let bytes = std::fs::read(run_dir(model_dir, run_id)?.join("run.json")).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| format!("The run record is damaged: {e}"))
}

/// Every run record of a model (records that cannot be read are skipped).
pub(crate) fn records(model_dir: &Path) -> Vec<RunRecord> {
    let Ok(entries) = std::fs::read_dir(runs_dir(model_dir)) else { return Vec::new() };
    let mut out: Vec<RunRecord> = entries
        .flatten()
        .filter_map(|e| std::fs::read(e.path().join("run.json")).ok())
        .filter_map(|b| serde_json::from_slice(&b).ok())
        .collect();
    out.sort_by_key(|r| r.started);
    out
}

/// A new run record, saved as running.
pub(crate) fn begin(
    model_dir: &Path,
    kind: &str,
    meta: &Meta,
    label: String,
    options: serde_json::Value,
    datasets: Vec<DatasetRef>,
    plan: Option<PlanRef>,
) -> Res<RunRecord> {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let r = RunRecord {
        version: RUN_VERSION,
        id: format!("{kind}-{nanos:x}"),
        kind: kind.into(),
        model_id: meta.id.clone(),
        model_name: meta.name.clone(),
        plan,
        label,
        started: now(),
        updated: now(),
        status: RunStatus::Running,
        app_version: env!("CARGO_PKG_VERSION").into(),
        options,
        datasets,
        step: 0,
        steps: 0,
        tokens: 0,
        snapshot: None,
        previous: None,
        error: None,
        resumed: 0,
        failed_saves: 0,
        save_error: None,
    };
    save_record(model_dir, &r)?;
    Ok(r)
}

/// Writes a snapshot into the slot not holding the newest one, then points the record at it.
/// `progress` is (step, steps, tokens).
pub(crate) fn write_snapshot<S: Serialize>(
    model_dir: &Path,
    r: &mut RunRecord,
    snap: &Snapshot<'_, S>,
    progress: (u64, u64, u64),
) -> Res<()> {
    let slot = if r.snapshot.as_ref().is_some_and(|s| s.slot == "a") { "b" } else { "a" };
    let dir = run_dir(model_dir, &r.id)?.join(format!("slot-{slot}"));
    // The slot's old files go first, so a half-written slot never looks complete.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut files = Vec::new();
    let mut put = |name: &str, bytes: Vec<u8>| -> Res<()> {
        write_atomic(&dir.join(name), &bytes)?;
        files.push(FileSum { name: name.into(), sha256: sha256_hex(&bytes), bytes: bytes.len() as u64 });
        Ok(())
    };
    let working = snap.model.to_bytes()?;
    let extra = snap.extra.map(|m| m.to_bytes()).transpose()?;
    // Never fill the disk the accepted model is saved on: a recovery point is skipped (and the
    // previous one kept) unless it leaves room to save the model again, plus 256 MB.
    let need = (working.len() + extra.as_ref().map_or(0, Vec::len)) as u64;
    if let Some(free) = free_space(model_dir) {
        let keep = working.len() as u64 + (256 << 20);
        if free < need + keep {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(format!(
                "not enough free disk space ({} MB free; this recovery point needs {} MB and {} MB must stay free)",
                free >> 20,
                need >> 20,
                keep >> 20
            ));
        }
    }
    put("working.bin", working)?;
    if let Some(extra) = extra {
        put("extra.bin", extra)?;
    }
    put("state.json", serde_json::to_vec(snap.state).map_err(|e| e.to_string())?)?;
    let info = SlotInfo { slot: slot.into(), step: progress.0, time: now(), stopped: snap.stopped, files };
    // The record keeps the newest snapshot and the one before (in the other slot).
    let mut next = r.clone();
    next.previous = next.snapshot.take();
    next.snapshot = Some(info);
    (next.step, next.steps, next.tokens) = progress;
    next.updated = now();
    save_record(model_dir, &next)?;
    *r = next;
    Ok(())
}

/// Free space on the disk holding `path` (None when it cannot be told).
pub(crate) fn free_space(path: &Path) -> Option<u64> {
    let path = std::fs::canonicalize(path).ok()?;
    // Windows' canonical paths start with \\?\, which mount points do not.
    let text = path.to_string_lossy();
    let path = std::path::PathBuf::from(text.strip_prefix(r"\\?\").unwrap_or(&text));
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .filter(|d| path.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

/// A snapshot's files, read back and checked against their sums.
fn read_slot<S: DeserializeOwned>(model_dir: &Path, r: &RunRecord, info: &SlotInfo) -> Res<Resume<S>> {
    let dir = run_dir(model_dir, &r.id)?.join(format!("slot-{}", info.slot));
    let read = |name: &str| -> Res<Option<Vec<u8>>> {
        let Some(sum) = info.files.iter().find(|f| f.name == name) else { return Ok(None) };
        let bytes = std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
        if bytes.len() as u64 != sum.bytes || sha256_hex(&bytes) != sum.sha256 {
            return Err(format!("{name} is damaged (its checksum does not match)"));
        }
        Ok(Some(bytes))
    };
    let model = Bicameral::from_bytes(&read("working.bin")?.ok_or("the working model is missing")?)?;
    let extra = read("extra.bin")?.map(|b| Bicameral::from_bytes(&b)).transpose()?;
    let state = serde_json::from_slice(&read("state.json")?.ok_or("the run's state is missing")?)
        .map_err(|e| format!("the run's state cannot be read: {e}"))?;
    Ok(Resume { state, model, extra })
}

/// The newest snapshot that passes its checks (the one before it when the newest is damaged).
/// Returns it with the slot used and the reason the newest was skipped, if it was.
pub(crate) fn load_snapshot<S: DeserializeOwned>(
    model_dir: &Path,
    r: &RunRecord,
) -> Res<(Resume<S>, SlotInfo, Option<String>)> {
    let newest = r.snapshot.as_ref().ok_or("This run has no recovery point yet")?;
    match read_slot(model_dir, r, newest) {
        Ok(s) => Ok((s, newest.clone(), None)),
        Err(e) => match r.previous.as_ref() {
            Some(prev) => match read_slot(model_dir, r, prev) {
                Ok(s) => Ok((s, prev.clone(), Some(e))),
                Err(e2) => Err(format!(
                    "Both recovery points are damaged ({e}; {e2}). The accepted model is not affected."
                )),
            },
            None => Err(format!("The recovery point is damaged ({e}). The accepted model is not affected.")),
        },
    }
}

/// Marks a run finished, stopped, failed or discarded. Finished and discarded runs' recovery
/// points are deleted (the record stays, for the history).
pub(crate) fn end(model_dir: &Path, r: &mut RunRecord, status: RunStatus, error: Option<String>) -> Res<()> {
    r.status = status;
    r.error = error;
    r.updated = now();
    save_record(model_dir, r)?;
    if matches!(status, RunStatus::Finished | RunStatus::Discarded) {
        if let Ok(dir) = run_dir(model_dir, &r.id) {
            for slot in ["slot-a", "slot-b"] {
                let _ = std::fs::remove_dir_all(dir.join(slot));
            }
        }
    }
    Ok(())
}

/// The recovery interval (minutes; 0 turns interval snapshots off, a stop still saves one).
pub(crate) fn interval_minutes(app: &AppHandle) -> u64 {
    read_settings(app).get("recoveryMinutes").and_then(|v| v.as_u64()).unwrap_or(10).min(240)
}

/// A run on its way: the record, kept up to date as snapshots are written.
pub(crate) struct LiveRun {
    pub dir: std::path::PathBuf,
    pub record: Mutex<RunRecord>,
}

impl LiveRun {
    /// Writes a snapshot; a failure is recorded and reported but does not stop training (the
    /// previous recovery point stays valid).
    pub fn snapshot<S: Serialize>(&self, app: &AppHandle, snap: &Snapshot<'_, S>, progress: (u64, u64, u64)) {
        let mut r = self.record.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = write_snapshot(&self.dir, &mut r, snap, progress) {
            r.failed_saves += 1;
            r.save_error = Some(e.clone());
            let _ = save_record(&self.dir, &r);
            let _ = app.emit(
                "recovery-notice",
                format!("A recovery point could not be saved ({e}); training continues and the previous recovery point is kept."),
            );
        }
    }
    pub fn end(&self, status: RunStatus, error: Option<String>) {
        let mut r = self.record.lock().unwrap_or_else(|e| e.into_inner());
        let _ = end(&self.dir, &mut r, status, error);
    }
    pub fn id(&self) -> String {
        self.record.lock().unwrap_or_else(|e| e.into_inner()).id.clone()
    }
}

/// How a reading or conversation run starts: as a new run (of a plan's step, or not), or
/// continuing a run from its recovery point.
pub(crate) struct RunContext<S> {
    pub plan: Option<PlanRef>,
    pub resume: Option<(RunRecord, Resume<S>)>,
}

impl<S> RunContext<S> {
    pub fn new() -> Self {
        RunContext { plan: None, resume: None }
    }
}

/// Starts the record of a run, or takes over the record of the run being continued.
pub(crate) fn live_run<S>(
    app: &AppHandle,
    kind: &str,
    meta: &Meta,
    label: &str,
    options: &serde_json::Value,
    datasets: Vec<DatasetRef>,
    ctx: &mut RunContext<S>,
) -> Res<(LiveRun, Option<Resume<S>>)> {
    let dir = model_dir(app, &meta.id)?;
    let (record, resume) = match ctx.resume.take() {
        Some((mut r, resume)) => {
            r.status = RunStatus::Running;
            r.resumed += 1;
            r.error = None;
            r.updated = now();
            save_record(&dir, &r)?;
            (r, Some(resume))
        }
        None => (begin(&dir, kind, meta, label.into(), options.clone(), datasets, ctx.plan.clone())?, None),
    };
    *app.state::<Studio>().run.lock().unwrap() = Some(record.id.clone());
    Ok((LiveRun { dir, record: Mutex::new(record) }, resume))
}

/// Ends a live run's record from the run's outcome (`Ok(stopped)` or the error).
pub(crate) fn close_run(app: &AppHandle, live: &LiveRun, outcome: Result<bool, &str>) {
    match outcome {
        Ok(false) => live.end(RunStatus::Finished, None),
        Ok(true) => live.end(RunStatus::Stopped, None),
        Err(e) => live.end(RunStatus::Failed, Some(e.into())),
    }
    let studio = app.state::<Studio>();
    let mut run = studio.run.lock().unwrap();
    if run.as_deref() == Some(live.id().as_str()) {
        *run = None;
    }
}

/// What a page needs to know about an unfinished run.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UnfinishedRun {
    #[serde(flatten)]
    pub record: RunRecord,
    /// It was still marked as training when Studio started: the app closed or crashed, or the
    /// computer restarted.
    pub interrupted: bool,
    pub resumable: bool,
}

/// Unfinished runs of every model (not the one training now: `live` is its record, or the model
/// in training, whose only running record is its own since training cannot start on a model with
/// an unfinished run).
pub(crate) fn unfinished(
    app: &AppHandle,
    live: Option<&str>,
    training: Option<&str>,
) -> Res<Vec<UnfinishedRun>> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(models_dir(app)?) else { return Ok(out) };
    for e in entries.flatten() {
        for r in records(&e.path()) {
            let current = r.status == RunStatus::Running && Some(r.model_id.as_str()) == training;
            if !r.unfinished() || Some(r.id.as_str()) == live || current {
                continue;
            }
            out.push(UnfinishedRun {
                interrupted: r.status == RunStatus::Running,
                resumable: r.resumable(),
                record: r,
            });
        }
    }
    out.sort_by_key(|u| std::cmp::Reverse(u.record.updated));
    Ok(out)
}

/// Training must not start on a model with an unfinished run (its recovery point would no
/// longer fit the model): continue it or discard it first. A plan may continue its own step.
pub(crate) fn no_unfinished_runs(app: &AppHandle, model_id: &str, plan: Option<&str>) -> Res<()> {
    let dir = model_dir(app, model_id)?;
    if let Some(r) = records(&dir)
        .into_iter()
        .find(|r| r.unfinished() && (plan.is_none() || r.plan.as_ref().map(|p| p.plan_id.as_str()) != plan))
    {
        return Err(format!(
            "This model has an unfinished {} ({}, started {}). Continue it or discard it first (§ 1 Models, Unfinished training).",
            r.kind,
            r.label,
            r.started
        ));
    }
    Ok(())
}

/// The unfinished run of a plan's step, if there is one.
pub(crate) fn plan_step_run(model_dir: &Path, plan_id: &str, step_id: &str) -> Option<RunRecord> {
    records(model_dir).into_iter().rev().find(|r| {
        r.unfinished() && r.plan.as_ref().is_some_and(|p| p.plan_id == plan_id && p.step_id == step_id)
    })
}

#[tauri::command]
pub fn unfinished_runs(app: AppHandle, studio: State<Studio>) -> Res<Vec<UnfinishedRun>> {
    let live = studio.run.lock().unwrap().clone();
    let training = if studio.teaching.load(Ordering::Acquire) {
        studio.active.lock().unwrap().as_ref().map(|a| a.meta.id.clone())
    } else {
        None
    };
    unfinished(&app, live.as_deref(), training.as_deref())
}

/// A Classroom session that did not finish: accepted epochs wait in the model's `checkpoints/`
/// to be restored (and the run continued) or discarded on § 1 Models.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ClassroomSession {
    model_id: String,
    model_name: String,
    checkpoints: usize,
    /// When the newest was saved.
    newest: u64,
}

#[tauri::command]
pub fn interrupted_classroom(app: AppHandle, studio: State<Studio>) -> Res<Vec<ClassroomSession>> {
    let training = if studio.teaching.load(Ordering::Acquire) {
        studio.active.lock().unwrap().as_ref().map(|a| a.meta.id.clone())
    } else {
        None
    };
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(models_dir(&app)?) else { return Ok(out) };
    for e in entries.flatten() {
        let id = e.file_name().to_string_lossy().to_string();
        if Some(&id) == training.as_ref() {
            continue;
        }
        let files = yendory_engine::learn::session::checkpoint_files(&e.path().join("checkpoints"));
        if files.is_empty() {
            continue;
        }
        let newest = files
            .iter()
            .filter_map(|f| f.metadata().and_then(|m| m.modified()).ok())
            .filter_map(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .max()
            .unwrap_or(0);
        let model_name = load_meta(&app, &id).map(|m| m.name).unwrap_or_else(|_| id.clone());
        out.push(ClassroomSession { model_id: id, model_name, checkpoints: files.len(), newest });
    }
    Ok(out)
}

#[tauri::command]
pub fn discard_run(app: AppHandle, studio: State<Studio>, model_id: String, run_id: String) -> Res<()> {
    if studio.run.lock().unwrap().as_deref() == Some(run_id.as_str()) {
        return Err("This run is training now; stop it first".into());
    }
    let dir = model_dir(&app, &model_id)?;
    let mut r = load_record(&dir, &run_id)?;
    let error = r.error.clone();
    end(&dir, &mut r, RunStatus::Discarded, error)?;
    append_history(
        &app,
        &model_id,
        serde_json::json!({"type": "run_discarded", "time": now(), "run": run_id}),
    );
    Ok(())
}

/// A continued run, for the page that shows it.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ResumeStarted {
    kind: String,
    meta: Meta,
    label: String,
    step: u64,
    steps: u64,
    /// The checks the run made before its recovery point (Reading's evaluations or
    /// Conversation's checks).
    previous: serde_json::Value,
    /// Set when the newest recovery point was damaged and the one before it is used.
    note: Option<String>,
}

/// Continues an unfinished run from its recovery point (opening its model), with the settings it
/// was started with. Only on the owner's request: Studio never does this by itself.
#[tauri::command]
pub fn resume_run(
    app: AppHandle,
    studio: State<Studio>,
    model_id: String,
    run_id: String,
) -> Res<ResumeStarted> {
    not_teaching(&studio)?;
    let dir = model_dir(&app, &model_id)?;
    let r = load_record(&dir, &run_id)?;
    if !r.resumable() {
        return Err(if r.unfinished() {
            "This run ended before its first recovery point, so there is nothing to continue. Discard it and start a new run; the accepted model is safe.".into()
        } else {
            "This run has finished or was discarded".into()
        });
    }
    if let Some(p) = &r.plan {
        return Err(format!(
            "This run is a step of the plan \"{}\": continue it with Resume on the Planner page",
            p.plan_name
        ));
    }
    no_waiting_checkpoints(&app, &model_id)?;
    let meta = open_model_in(&app, &studio, &model_id)?;
    let (model, archive) = with_active(&studio, |a| Ok((a.model.clone(), a.archive.clone())))?;
    let note = |skipped: Option<String>, slot: &SlotInfo| {
        skipped.map(|e| {
            format!("The newest recovery point could not be used ({e}); continuing from the one before it, at step {}", slot.step)
        })
    };
    let bad_options = |e: serde_json::Error| format!("The run's settings cannot be read: {e}");
    match r.kind.as_str() {
        "reading" => {
            let options: reading::ReadOptions =
                serde_json::from_value(r.options.clone()).map_err(bad_options)?;
            let job = reading::reading_job(&app, &options, meta.word_limit)?;
            let (resume, slot, skipped) =
                load_snapshot::<yendory_engine::learn::corpus::ReadingState>(&dir, &r)?;
            let previous = serde_json::to_value(&resume.state.evaluations).unwrap_or_default();
            let (step, steps) = (resume.state.step, resume.state.steps);
            let label = r.label.clone();
            reading::start_reading(
                app.clone(),
                &studio,
                job,
                meta.clone(),
                model,
                RunContext { plan: None, resume: Some((r, resume)) },
            )?;
            Ok(ResumeStarted {
                kind: "reading".into(),
                meta,
                label,
                step,
                steps,
                previous,
                note: note(skipped, &slot),
            })
        }
        "conversation" => {
            let options: conversation::ConverseOptions =
                serde_json::from_value(r.options.clone()).map_err(bad_options)?;
            let job = conversation::conversation_job(&app, &options)?;
            let (resume, slot, skipped) =
                load_snapshot::<yendory_engine::learn::conversation::ConversationState>(&dir, &r)?;
            let previous = serde_json::to_value(&resume.state.checks).unwrap_or_default();
            let (step, steps) = (resume.state.step, resume.state.steps);
            let label = r.label.clone();
            conversation::start_conversation(
                app.clone(),
                &studio,
                job,
                (meta.clone(), model, archive),
                RunContext { plan: None, resume: Some((r, resume)) },
            )?;
            Ok(ResumeStarted {
                kind: "conversation".into(),
                meta,
                label,
                step,
                steps,
                previous,
                note: note(skipped, &slot),
            })
        }
        other => Err(format!("Unknown kind of run: {other}")),
    }
}

#[tauri::command]
pub fn recovery_minutes(app: AppHandle) -> u64 {
    interval_minutes(&app)
}

#[tauri::command]
pub fn set_recovery_minutes(app: AppHandle, minutes: u64) -> Res<u64> {
    if minutes > 240 {
        return Err("Choose 0 to 240 minutes".into());
    }
    write_setting(&app, "recoveryMinutes", minutes.into())?;
    Ok(minutes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use yendory_engine::Config;

    fn meta() -> Meta {
        serde_json::from_value(serde_json::json!({
            "id": "m1", "name": "Nova", "created": 1, "updated": 1, "step": 0, "parameters": 0,
            "groups": [0, 0, 0], "words": 0, "taught": [], "origin": ""
        }))
        .unwrap()
    }

    fn model(seed: u64) -> Bicameral {
        let c = Config {
            d: 16,
            heads: 2,
            ff: 24,
            blocks: 2,
            split: 1,
            r1: 12,
            r2: 10,
            max_tokens: 32,
            ..Config::default()
        };
        Bicameral::with_config(c, seed)
    }

    fn temp() -> std::path::PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir().join(format!("yendory-recovery-{nanos:x}"));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Snapshots alternate between the two slots; the newest that passes its checksums is
    /// loaded, the one before it when the newest is damaged, and neither replaces anything else.
    #[test]
    fn snapshots_alternate_and_a_damaged_one_falls_back() {
        let dir = temp();
        let mut r =
            begin(&dir, "reading", &meta(), "test".into(), serde_json::json!({}), vec![], None).unwrap();
        assert_eq!(load_record(&dir, &r.id).unwrap().status, RunStatus::Running);
        let (a, b) = (model(1), model(2));
        write_snapshot(
            &dir,
            &mut r,
            &Snapshot { model: &a, extra: None, state: &1u64, stopped: false },
            (10, 100, 1000),
        )
        .unwrap();
        write_snapshot(
            &dir,
            &mut r,
            &Snapshot { model: &b, extra: Some(&a), state: &2u64, stopped: true },
            (20, 100, 2000),
        )
        .unwrap();
        let saved = load_record(&dir, &r.id).unwrap();
        assert_eq!((saved.step, saved.tokens), (20, 2000));
        assert_eq!(saved.snapshot.as_ref().unwrap().slot, "b");
        assert_eq!(saved.previous.as_ref().unwrap().slot, "a");
        let (s, slot, skipped): (Resume<u64>, _, _) = load_snapshot(&dir, &saved).unwrap();
        assert_eq!((s.state, slot.step, skipped), (2, 20, None));
        assert_eq!(s.model.weights, b.weights);
        assert_eq!(s.extra.unwrap().weights, a.weights);
        // Damage the newest: the previous one is used, and says why.
        let bad = dir.join("runs").join(&r.id).join("slot-b").join("working.bin");
        let mut bytes = std::fs::read(&bad).unwrap();
        let n = bytes.len() / 2;
        bytes[n] ^= 0x55;
        std::fs::write(&bad, bytes).unwrap();
        let (s, slot, skipped): (Resume<u64>, _, _) = load_snapshot(&dir, &saved).unwrap();
        assert_eq!((s.state, slot.step), (1, 10));
        assert!(skipped.unwrap().contains("damaged"));
        assert_eq!(s.model.weights, a.weights);
        // Both damaged: an error that says the accepted model is safe.
        let bad = dir.join("runs").join(&r.id).join("slot-a").join("state.json");
        std::fs::write(&bad, b"{").unwrap();
        let err = load_snapshot::<u64>(&dir, &saved).err().unwrap();
        assert!(err.contains("Both recovery points are damaged") && err.contains("not affected"), "{err}");
        // A third snapshot goes into slot a again (the older one), never over the newest.
        write_snapshot(
            &dir,
            &mut r,
            &Snapshot { model: &a, extra: None, state: &3u64, stopped: false },
            (30, 100, 3000),
        )
        .unwrap();
        assert_eq!(r.snapshot.as_ref().unwrap().slot, "a");
        assert_eq!(r.previous.as_ref().unwrap().slot, "b");
        // Finishing deletes the recovery points but keeps the record.
        end(&dir, &mut r, RunStatus::Finished, None).unwrap();
        assert!(!dir.join("runs").join(&r.id).join("slot-a").exists());
        assert_eq!(load_record(&dir, &r.id).unwrap().status, RunStatus::Finished);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A crash part-way through a reading (the run is killed by a panic after its third
    /// recovery point, with nothing else saved): the recovery point read back from disk
    /// continues the run, which ends with exactly the model an uninterrupted run ends with.
    #[test]
    fn a_reading_killed_mid_run_continues_from_disk_to_the_same_model() {
        use std::cell::RefCell;
        use yendory_engine::learn::corpus::{
            pretrain_mix_with, Category, Corpus, MixPart, PretrainConfig, PretrainEvent, ReadingState,
        };
        use yendory_engine::learn::recovery::{Recovery, RecoveryEvery};
        let text: String = (0..240)
            .map(|i| format!("Mia had a red kite number {i}. The wind took it up high.\n<|endoftext|>\n"))
            .collect();
        let parts = vec![MixPart {
            name: "kites".into(),
            category: Category::General,
            corpus: Corpus::prepare(&text),
            share: 1.,
            stage: None,
        }];
        let cfg = PretrainConfig {
            tokens: 6_000,
            batch_size: 4,
            window: 24,
            evaluations: 5,
            eval_windows: 8,
            ..Default::default()
        };
        let start = model(7);
        let read = |m: &Bicameral,
                    resume: Option<Resume<ReadingState>>,
                    dir: Option<(&Path, &RefCell<RunRecord>)>,
                    crash_after: Option<u64>| {
            let saved = RefCell::new(m.clone());
            let mut on_snapshot = |snap: Snapshot<'_, ReadingState>| {
                if let Some((dir, r)) = dir {
                    let p = (snap.state.step, snap.state.steps, snap.state.tokens);
                    write_snapshot(dir, &mut r.borrow_mut(), &snap, p).unwrap();
                    // The accepted model as saved at this moment (model.nnova2).
                    std::fs::write(dir.join("model.nnova2"), saved.borrow().to_bytes().unwrap()).unwrap();
                }
                Ok(())
            };
            let mut recovery = Recovery {
                every: RecoveryEvery { seconds: 0, steps: 12 },
                on_snapshot: Some(&mut on_snapshot),
                resume,
            };
            pretrain_mix_with(
                m,
                &parts,
                &cfg,
                &mut |e| {
                    if let (PretrainEvent::Batch { step, .. }, Some(at)) = (e, crash_after) {
                        assert!(*step < at, "crash");
                    }
                },
                &mut |a| {
                    *saved.borrow_mut() = a.clone();
                    Ok(())
                },
                &std::sync::atomic::AtomicBool::new(false),
                &mut recovery,
            )
            .map(|r| r.model)
        };
        let full = read(&start, None, None, None).unwrap();
        let dir = temp();
        let rec = RefCell::new(
            begin(&dir, "reading", &meta(), "kites".into(), serde_json::json!({}), vec![], None).unwrap(),
        );
        let crashed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            read(&start, None, Some((&dir, &rec)), Some(40))
        }));
        assert!(crashed.is_err(), "the run was killed");
        // After the "restart": the record on disk says running (interrupted), with a recovery point.
        let r = load_record(&dir, &rec.borrow().id).unwrap();
        assert_eq!(r.status, RunStatus::Running);
        assert!(r.resumable() && r.step >= 24 && r.step < 40, "{}", r.step);
        let (resume, ..): (Resume<ReadingState>, _, _) = load_snapshot(&dir, &r).unwrap();
        let saved = Bicameral::from_bytes(&std::fs::read(dir.join("model.nnova2")).unwrap()).unwrap();
        let resumed = read(&saved, Some(resume), None, None).unwrap();
        assert_eq!(resumed.step, full.step);
        assert!(
            resumed.weights == full.weights
                && resumed.momentum == full.momentum
                && resumed.variance == full.variance
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn free_space_is_known_for_the_data_folder() {
        let dir = temp();
        let free = free_space(&dir);
        assert!(free.is_some_and(|f| f > 0), "{free:?}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn records_say_what_is_unfinished_and_resumable() {
        let dir = temp();
        let mut r =
            begin(&dir, "conversation", &meta(), "x".into(), serde_json::json!({}), vec![], None).unwrap();
        assert!(r.unfinished() && !r.resumable(), "no recovery point yet");
        let m = model(3);
        write_snapshot(
            &dir,
            &mut r,
            &Snapshot { model: &m, extra: None, state: &0u64, stopped: true },
            (5, 9, 50),
        )
        .unwrap();
        end(&dir, &mut r, RunStatus::Stopped, None).unwrap();
        let all = records(&dir);
        assert_eq!(all.len(), 1);
        assert!(all[0].unfinished() && all[0].resumable());
        end(&dir, &mut r, RunStatus::Discarded, None).unwrap();
        assert!(!records(&dir)[0].unfinished());
        let _ = std::fs::remove_dir_all(dir);
    }
}
