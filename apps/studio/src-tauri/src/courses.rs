//! Course Builder and Lesson Planner: courses stored as <app data>/courses/<id>.json, plans as
//! <app data>/plans/<id>/plan.json (with the best checkpoint of the running course next to
//! it), and the planner thread. Grading, remediation and resuming are done by
//! `yendory_engine::learn::program`; this file only stores, loads and reports.
use super::*;
use yendory_engine::learn::curriculum::{
    prepare, read_name, CourseCheck, CourseDef, CourseSettings, SectionWeights, StageFrom,
};
use yendory_engine::learn::program::{
    restart_policy, run_course, CourseGrade, CourseHost, CourseRun, ExamGrade, Phase, ProgramEvent,
    RemediationRecord, Restart, RunOptions, RunSummary, Snapshot,
};

pub(crate) fn data_dir(app: &AppHandle, name: &str) -> Res<PathBuf> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join(name);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

pub(crate) fn check_id(id: &str) -> Res<()> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err("Invalid id".into());
    }
    Ok(())
}

pub(crate) fn fresh_id(prefix: char) -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{prefix}{nanos:x}")
}

// ---------------------------------------------------------------------------------------------
// Courses
// ---------------------------------------------------------------------------------------------

fn course_path(app: &AppHandle, id: &str) -> Res<PathBuf> {
    check_id(id)?;
    Ok(data_dir(app, "courses")?.join(format!("{id}.json")))
}

pub(crate) fn load_course(app: &AppHandle, id: &str) -> Res<CourseDef> {
    let bytes =
        std::fs::read(course_path(app, id)?).map_err(|_| "This course no longer exists".to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

pub(crate) fn save_course(app: &AppHandle, def: &CourseDef) -> Res<()> {
    write_atomic(&course_path(app, &def.id)?, &serde_json::to_vec(def).map_err(|e| e.to_string())?)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LessonView {
    name: String,
    number: Option<u32>,
    stage: Option<EducationStage>,
    stage_from: StageFrom,
    /// The stage the lesson is taught in (the course default for `stage_from` Default).
    effective_stage: Option<EducationStage>,
    pass_mark: f32,
    practice: Vec<String>,
    name_problem: Option<String>,
    has_extra_practice_block: bool,
    /// Taught together with the other "together" lessons of its stage (one mixed session).
    together: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CourseView {
    id: String,
    name: String,
    lessons: Vec<LessonView>,
    exam: Option<String>,
    unmatched_practice: Vec<String>,
    default_stage: Option<EducationStage>,
    default_pass_mark: f32,
    section_weights: SectionWeights,
    exam_minimum: Option<f32>,
    pretest: bool,
    settings: CourseSettings,
    check: CourseCheck,
    notes: Vec<String>,
}

fn view(def: &CourseDef, notes: Vec<String>) -> CourseView {
    let prep = prepare(def);
    CourseView {
        id: def.id.clone(),
        name: def.name.clone(),
        lessons: def
            .lessons
            .iter()
            .map(|l| LessonView {
                name: l.file.name.clone(),
                number: l.number,
                stage: l.stage,
                stage_from: l.stage_from,
                effective_stage: def.stage_of(l),
                pass_mark: l.pass_mark,
                practice: l.practice.iter().map(|p| p.name.clone()).collect(),
                name_problem: l.name_problem.clone(),
                has_extra_practice_block: l.file.text.contains("[Extra Practice]"),
                together: l.together,
            })
            .collect(),
        exam: def.exam.as_ref().map(|e| e.name.clone()),
        unmatched_practice: def.unmatched_practice.iter().map(|p| p.name.clone()).collect(),
        default_stage: def.default_stage,
        default_pass_mark: def.default_pass_mark,
        section_weights: def.section_weights,
        exam_minimum: def.exam_minimum,
        pretest: def.pretest,
        settings: def.settings.clone(),
        check: prep.check,
        notes,
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CourseSummary {
    id: String,
    name: String,
    lessons: usize,
    required_grade: f32,
    ready: bool,
}

#[tauri::command]
pub fn list_courses(app: AppHandle) -> Res<Vec<CourseSummary>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(data_dir(&app, "courses")?).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let Ok(bytes) = std::fs::read(&path) else { continue };
        let Ok(def) = serde_json::from_slice::<CourseDef>(&bytes) else { continue };
        out.push(CourseSummary {
            ready: prepare(&def).check.problems.is_empty(),
            id: def.id.clone(),
            name: def.name.clone(),
            lessons: def.lessons.len(),
            required_grade: def.required_grade(),
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

#[tauri::command]
pub fn create_course(app: AppHandle, name: String) -> Res<CourseView> {
    let def = CourseDef::new(&fresh_id('c'), &check_name(&name)?);
    save_course(&app, &def)?;
    Ok(view(&def, Vec::new()))
}

/// A built-in course (Basic English, Reasoning, Math) and its lessons.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BuiltinView {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    lessons: Vec<BuiltinLesson>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BuiltinLesson {
    name: String,
    taught: usize,
    held_out: usize,
}

#[tauri::command]
pub fn builtin_courses() -> Vec<BuiltinView> {
    yendory_engine::learn::builtin::builtin_courses()
        .into_iter()
        .map(|c| BuiltinView {
            id: c.id,
            name: c.name,
            description: c.description,
            lessons: c
                .lessons
                .iter()
                .map(|f| {
                    let items = parse_lesson(&f.text, &f.name).unwrap_or_default();
                    BuiltinLesson {
                        name: f.name.clone(),
                        taught: items.iter().filter(|i| !i.held_out && i.answer.is_some()).count(),
                        held_out: items.iter().filter(|i| i.held_out).count(),
                    }
                })
                .collect(),
        })
        .collect()
}

/// The lesson files of a built-in course.
pub(crate) fn builtin_files(id: &str) -> Res<(String, Vec<CourseFile>)> {
    yendory_engine::learn::builtin::builtin_courses()
        .into_iter()
        .find(|c| c.id == id)
        .map(|c| (c.name.to_string(), c.lessons))
        .ok_or_else(|| "Unknown built-in course".into())
}

/// Adds a built-in course to the course list (a new course each time; edit it like any other).
#[tauri::command]
pub fn add_builtin_course(app: AppHandle, id: String) -> Res<CourseView> {
    let (name, files) = builtin_files(&id)?;
    let mut def = CourseDef::new(&fresh_id('c'), &name);
    let notes = def.add_files(files);
    save_course(&app, &def)?;
    Ok(view(&def, notes))
}

#[tauri::command]
pub async fn course_detail(app: AppHandle, id: String) -> Res<CourseView> {
    Ok(view(&load_course(&app, &id)?, Vec::new()))
}

#[tauri::command]
pub fn delete_course(app: AppHandle, id: String) -> Res<()> {
    std::fs::remove_file(course_path(&app, &id)?).map_err(|e| e.to_string())
}

fn pick_texts(app: &AppHandle, zip: bool) -> Res<Vec<CourseFile>> {
    let mut dialog = app.dialog().file();
    dialog = if zip {
        dialog.add_filter("Lessons (.txt) or a course (.zip)", &["txt", "zip"])
    } else {
        dialog.add_filter("Lesson text (.txt)", &["txt"])
    };
    let Some(paths) = dialog.blocking_pick_files() else { return Ok(Vec::new()) };
    let mut files = Vec::new();
    for path in &paths {
        let name = display_name(path);
        let limit = if name.to_ascii_lowercase().ends_with(".zip") { 64 << 20 } else { MAX_LESSON_BYTES };
        files.extend(files_from(&name, read_picked(app, path, limit)?)?);
    }
    Ok(files)
}

/// Adds lesson, practice and exam files; names decide what each file is.
#[tauri::command]
pub async fn add_course_files(app: AppHandle, id: String) -> Res<CourseView> {
    let mut def = load_course(&app, &id)?;
    let files = pick_texts(&app, true)?;
    let notes = def.add_files(files);
    save_course(&app, &def)?;
    Ok(view(&def, notes))
}

/// Attaches practice files to one lesson, whatever their names.
#[tauri::command]
pub async fn attach_practice(app: AppHandle, id: String, index: usize) -> Res<CourseView> {
    let mut def = load_course(&app, &id)?;
    let files = pick_texts(&app, false)?;
    let lesson = def.lessons.get_mut(index).ok_or("No such lesson")?;
    let notes =
        files.iter().map(|f| format!("{} is extra practice for {}", f.name, lesson.file.name)).collect();
    lesson.practice.extend(files);
    save_course(&app, &def)?;
    Ok(view(&def, notes))
}

/// Chooses the course's teacher-written exam file, whatever its name.
#[tauri::command]
pub async fn choose_exam(app: AppHandle, id: String) -> Res<CourseView> {
    let mut def = load_course(&app, &id)?;
    let Some(path) = app.dialog().file().add_filter("Exam (.txt)", &["txt"]).blocking_pick_file() else {
        return Ok(view(&def, Vec::new()));
    };
    let name = display_name(&path);
    def.exam = Some(CourseFile { text: text_of(&name, read_picked(&app, &path, MAX_LESSON_BYTES)?)?, name });
    save_course(&app, &def)?;
    Ok(view(&def, Vec::new()))
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum CourseEdit {
    Rename {
        name: String,
    },
    DefaultStage {
        stage: Option<EducationStage>,
    },
    DefaultPassMark {
        percent: f32,
    },
    /// Sets every lesson's pass mark (and the default for new lessons).
    AllPassMarks {
        percent: f32,
    },
    Weights {
        weights: SectionWeights,
    },
    ExamMinimum {
        percent: Option<f32>,
    },
    Pretest {
        on: bool,
    },
    Settings {
        settings: CourseSettings,
    },
    LessonPassMark {
        index: usize,
        percent: f32,
    },
    /// Teach this lesson together with the others of its stage, or on its own.
    LessonTogether {
        index: usize,
        on: bool,
    },
    /// Every lesson together, or every lesson on its own.
    AllTogether {
        on: bool,
    },
    /// "name" (from the file name), "default" (the course default), or a stage / "whole".
    LessonStage {
        index: usize,
        stage: String,
    },
    RemoveLesson {
        index: usize,
    },
    RemovePractice {
        index: usize,
        practice: usize,
    },
    RemoveUnmatched {
        index: usize,
    },
    RemoveExam,
}

fn percent(p: f32) -> Res<f32> {
    if !(0.0..=100.0).contains(&p) {
        return Err("Choose a percentage from 0 to 100".into());
    }
    Ok(p / 100.)
}

#[tauri::command]
pub fn edit_course(app: AppHandle, id: String, edit: CourseEdit) -> Res<CourseView> {
    let mut def = load_course(&app, &id)?;
    let lesson = |def: &mut CourseDef, i: usize| -> Res<usize> {
        if i < def.lessons.len() {
            Ok(i)
        } else {
            Err("No such lesson".into())
        }
    };
    match edit {
        CourseEdit::Rename { name } => def.name = check_name(&name)?,
        CourseEdit::DefaultStage { stage } => def.default_stage = stage,
        CourseEdit::DefaultPassMark { percent: p } => def.default_pass_mark = percent(p)?,
        CourseEdit::AllPassMarks { percent: p } => {
            let m = percent(p)?;
            def.default_pass_mark = m;
            for l in &mut def.lessons {
                l.pass_mark = m;
            }
        }
        CourseEdit::Weights { weights } => {
            let w = [weights.dense, weights.transformer, weights.coop, weights.whole_brain];
            if w.iter().any(|x| !(0.0..=100.0).contains(x)) {
                return Err("Section weights are 0-100".into());
            }
            def.section_weights = weights;
        }
        CourseEdit::ExamMinimum { percent: p } => def.exam_minimum = p.map(percent).transpose()?,
        CourseEdit::Pretest { on } => def.pretest = on,
        CourseEdit::Settings { settings } => {
            if settings.time_limit_hours.is_some_and(|h| !(h > 0. && h <= 24. * 30.)) {
                return Err("Choose a time limit of up to 720 hours".into());
            }
            if !(1..=20).contains(&settings.flat_epochs) {
                return Err("Choose 1 to 20 flat epochs".into());
            }
            def.settings = settings;
        }
        CourseEdit::LessonPassMark { index, percent: p } => {
            let i = lesson(&mut def, index)?;
            def.lessons[i].pass_mark = percent(p)?;
        }
        CourseEdit::LessonTogether { index, on } => {
            let i = lesson(&mut def, index)?;
            def.lessons[i].together = on;
        }
        CourseEdit::AllTogether { on } => {
            for l in &mut def.lessons {
                l.together = on;
            }
        }
        CourseEdit::LessonStage { index, stage } => {
            let i = lesson(&mut def, index)?;
            let l = &mut def.lessons[i];
            match stage.as_str() {
                "name" => {
                    let info = read_name(&l.file.name);
                    l.stage = info.stage;
                    l.stage_from = if info.stage.is_some() { StageFrom::Name } else { StageFrom::Default };
                }
                "default" => {
                    l.stage = None;
                    l.stage_from = StageFrom::Default;
                }
                other => {
                    l.stage = match other {
                        "whole" => None,
                        s => Some(education_stage(Some(s))?.ok_or("Unknown stage")?),
                    };
                    l.stage_from = StageFrom::Manual;
                }
            }
            def.sort();
        }
        CourseEdit::RemoveLesson { index } => {
            let i = lesson(&mut def, index)?;
            let l = def.lessons.remove(i);
            def.unmatched_practice.extend(l.practice);
        }
        CourseEdit::RemovePractice { index, practice } => {
            let i = lesson(&mut def, index)?;
            if practice >= def.lessons[i].practice.len() {
                return Err("No such practice file".into());
            }
            def.lessons[i].practice.remove(practice);
        }
        CourseEdit::RemoveUnmatched { index } => {
            if index >= def.unmatched_practice.len() {
                return Err("No such practice file".into());
            }
            def.unmatched_practice.remove(index);
        }
        CourseEdit::RemoveExam => def.exam = None,
    }
    save_course(&app, &def)?;
    Ok(view(&def, Vec::new()))
}

// ---------------------------------------------------------------------------------------------
// Plans
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Idle,
    Running,
    Stopped,
    /// Every course passed.
    Finished,
    /// A course failed (or ran out of time); the plan does not go on to the next course.
    Failed,
    /// Something needs a decision before the plan can continue (see `message`).
    NeedsAttention,
}

/// A finished run of a course.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunEntry {
    course_id: String,
    name: String,
    fingerprint: String,
    phase: Phase,
    overridden: bool,
    pretest: Option<ExamGrade>,
    grades: Vec<CourseGrade>,
    remediation: Vec<RemediationRecord>,
    /// Each lesson's teaching outcome.
    lessons: Vec<(String, Option<FileOutcome>)>,
    started_step: u64,
    end_step: u64,
    finished: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Plan {
    id: String,
    name: String,
    model_id: String,
    model_name: String,
    created: u64,
    courses: Vec<String>,
    /// Index of the course being taught (or next).
    position: usize,
    status: PlanStatus,
    message: Option<String>,
    /// The version of the current course its run uses (frozen when the run started).
    #[serde(default)]
    frozen: Option<CourseDef>,
    #[serde(default)]
    run: Option<CourseRun>,
    #[serde(default)]
    run_started_step: u64,
    /// The next run of the current course was granted by the one override.
    #[serde(default)]
    next_overridden: bool,
    #[serde(default)]
    results: Vec<RunEntry>,
}

fn plan_dir(app: &AppHandle, id: &str) -> Res<PathBuf> {
    check_id(id)?;
    Ok(data_dir(app, "plans")?.join(id))
}

fn load_plan(app: &AppHandle, id: &str) -> Res<Plan> {
    let bytes =
        std::fs::read(plan_dir(app, id)?.join("plan.json")).map_err(|_| "No such plan".to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

fn save_plan(app: &AppHandle, plan: &Plan) -> Res<()> {
    let dir = plan_dir(app, &plan.id)?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_atomic(&dir.join("plan.json"), &serde_json::to_vec(plan).map_err(|e| e.to_string())?)
}

/// Course runs of a model, for the restart rules (kept with the model, so a new plan does not
/// reset them).
fn model_runs(app: &AppHandle, model_id: &str) -> Vec<RunSummary> {
    model_dir(app, model_id)
        .ok()
        .and_then(|d| std::fs::read(d.join("course_runs.json")).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

fn add_model_run(app: &AppHandle, model_id: &str, run: RunSummary) -> Res<()> {
    let mut runs = model_runs(app, model_id);
    runs.push(run);
    write_atomic(
        &model_dir(app, model_id)?.join("course_runs.json"),
        &serde_json::to_vec(&runs).map_err(|e| e.to_string())?,
    )
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanCourse {
    id: String,
    name: String,
    /// "passed", "failed", "current", "pending" or "missing".
    state: String,
    grade: Option<f32>,
    required: Option<f32>,
    /// Lessons, and how many of them are taught together.
    lessons: usize,
    together: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanView {
    id: String,
    name: String,
    model_id: String,
    model_name: String,
    courses: Vec<PlanCourse>,
    position: usize,
    status: PlanStatus,
    message: Option<String>,
    run: Option<CourseRun>,
    results: Vec<RunEntry>,
    /// When the current course failed: what a restart would be (fresh, override or refused).
    restart: Option<Restart>,
    next_overridden: bool,
}

fn plan_view(app: &AppHandle, plan: &Plan) -> PlanView {
    let courses = plan
        .courses
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let def = load_course(app, id).ok();
            let last = plan.results.iter().rev().find(|r| r.course_id == *id);
            let state = if def.is_none() {
                "missing"
            } else if i < plan.position {
                "passed"
            } else if i == plan.position && plan.status == PlanStatus::Failed {
                "failed"
            } else if i == plan.position && plan.status != PlanStatus::Finished {
                "current"
            } else {
                "pending"
            };
            PlanCourse {
                id: id.clone(),
                name: def.as_ref().map(|d| d.name.clone()).unwrap_or_else(|| "(deleted course)".into()),
                state: state.into(),
                grade: last.and_then(|r| r.grades.last().map(|g| g.grade)),
                lessons: def.as_ref().map_or(0, |d| d.lessons.len()),
                together: def.as_ref().map_or(0, |d| d.lessons.iter().filter(|l| l.together).count()),
                required: def.map(|d| d.required_grade()),
            }
        })
        .collect();
    let restart = matches!(plan.status, PlanStatus::Failed | PlanStatus::NeedsAttention)
        .then(|| plan.courses.get(plan.position))
        .flatten()
        .and_then(|id| load_course(app, id).ok())
        .map(|def| restart_policy(&model_runs(app, &plan.model_id), &def.fingerprint()));
    PlanView {
        id: plan.id.clone(),
        name: plan.name.clone(),
        model_id: plan.model_id.clone(),
        model_name: plan.model_name.clone(),
        courses,
        position: plan.position,
        status: plan.status,
        message: plan.message.clone(),
        run: plan.run.clone(),
        results: plan.results.clone(),
        restart,
        next_overridden: plan.next_overridden,
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanSummary {
    id: String,
    name: String,
    model_name: String,
    courses: usize,
    position: usize,
    status: PlanStatus,
}

#[tauri::command]
pub fn list_plans(app: AppHandle) -> Res<Vec<PlanSummary>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(data_dir(&app, "plans")?).map_err(|e| e.to_string())? {
        let id = entry.map_err(|e| e.to_string())?.file_name().to_string_lossy().to_string();
        if let Ok(p) = load_plan(&app, &id) {
            out.push(PlanSummary {
                id: p.id,
                name: p.name,
                model_name: p.model_name,
                courses: p.courses.len(),
                position: p.position,
                status: p.status,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// A new plan for the open model.
#[tauri::command]
pub fn create_plan(app: AppHandle, studio: State<Studio>, name: String) -> Res<PlanView> {
    let (model_id, model_name) = with_active(&studio, |a| Ok((a.meta.id.clone(), a.meta.name.clone())))?;
    let plan = Plan {
        id: fresh_id('p'),
        name: check_name(&name)?,
        model_id,
        model_name,
        created: now(),
        courses: Vec::new(),
        position: 0,
        status: PlanStatus::Idle,
        message: None,
        frozen: None,
        run: None,
        run_started_step: 0,
        next_overridden: false,
        results: Vec::new(),
    };
    save_plan(&app, &plan)?;
    Ok(plan_view(&app, &plan))
}

#[tauri::command]
pub fn plan_detail(app: AppHandle, id: String) -> Res<PlanView> {
    Ok(plan_view(&app, &load_plan(&app, &id)?))
}

fn running(studio: &Studio, plan: &Plan) -> Res<()> {
    if plan.status == PlanStatus::Running && studio.teaching.load(Ordering::Acquire) {
        return Err("Stop the plan first".into());
    }
    Ok(())
}

/// Renames the plan or sets its course order. Courses already passed and the course in progress
/// stay where they are; only the courses after them can be changed.
#[tauri::command]
pub fn edit_plan(
    app: AppHandle,
    studio: State<Studio>,
    id: String,
    name: Option<String>,
    courses: Option<Vec<String>>,
) -> Res<PlanView> {
    let mut plan = load_plan(&app, &id)?;
    running(&studio, &plan)?;
    if let Some(n) = name {
        plan.name = check_name(&n)?;
    }
    if let Some(c) = courses {
        // A failed course stays where it is: the plan never goes past it.
        let failed_before = plan.courses.get(plan.position).is_some_and(|id| {
            plan.results.iter().rev().find(|r| r.course_id == *id).is_some_and(|r| r.phase != Phase::Passed)
        });
        let current = plan.run.is_some()
            || failed_before
            || matches!(plan.status, PlanStatus::Failed | PlanStatus::NeedsAttention);
        let fixed = (plan.position + usize::from(current)).min(plan.courses.len());
        if c.len() < fixed || c[..fixed] != plan.courses[..fixed] {
            return Err("Courses already passed, in progress or failed cannot be moved or removed".into());
        }
        for id in &c {
            check_id(id)?;
        }
        plan.courses = c;
        if plan.status == PlanStatus::Finished && plan.position < plan.courses.len() {
            plan.status = PlanStatus::Idle;
        }
    }
    save_plan(&app, &plan)?;
    Ok(plan_view(&app, &plan))
}

#[tauri::command]
pub fn delete_plan(app: AppHandle, studio: State<Studio>, id: String) -> Res<()> {
    let plan = load_plan(&app, &id)?;
    running(&studio, &plan)?;
    std::fs::remove_dir_all(plan_dir(&app, &id)?).map_err(|e| e.to_string())
}

/// After a failed course: start a new run of it (revised: a fresh run; unchanged: only with the
/// one override per version, which is recorded). The model keeps everything it learned.
#[tauri::command]
pub fn restart_course(
    app: AppHandle,
    studio: State<Studio>,
    id: String,
    confirm_override: bool,
) -> Res<PlanView> {
    let mut plan = load_plan(&app, &id)?;
    running(&studio, &plan)?;
    if !matches!(plan.status, PlanStatus::Failed | PlanStatus::NeedsAttention | PlanStatus::Stopped) {
        return Err("Only a failed, stopped or blocked course can be restarted".into());
    }
    let course_id = plan.courses.get(plan.position).ok_or("No course to restart")?.clone();
    let def = load_course(&app, &course_id)?;
    match restart_policy(&model_runs(&app, &plan.model_id), &def.fingerprint()) {
        Restart::Fresh => plan.next_overridden = false,
        Restart::NeedsOverride if confirm_override => {
            plan.next_overridden = true;
            append_history(
                &app,
                &plan.model_id,
                serde_json::json!({"type": "course_override", "time": now(), "course": def.name, "fingerprint": def.fingerprint()}),
            );
        }
        Restart::NeedsOverride => {
            return Err("This version of the course already failed; confirm the one override to retry it unchanged".into())
        }
        Restart::Refused => {
            return Err("This version already failed after its override: revise the course (practice material, lessons, pass marks or exam) first".into())
        }
    }
    plan.run = None;
    plan.frozen = None;
    plan.status = PlanStatus::Idle;
    plan.message = Some(format!(
        "{} will start again from the model's current (best) checkpoint{}",
        def.name,
        if plan.next_overridden { ", with the one override for this version" } else { "" }
    ));
    let _ = std::fs::remove_file(plan_dir(&app, &id)?.join("best.nnova2"));
    save_plan(&app, &plan)?;
    Ok(plan_view(&app, &plan))
}

struct StudioHost<'a> {
    app: &'a AppHandle,
    plan: Plan,
    meta: Meta,
    /// Study Mode: the model's memory, studied into and used while the course is taught.
    study: Option<yendory_engine::memory::Studying>,
    /// The model's reward ledger (taken by the run, given back when it ends).
    rewards: Option<yendory_engine::learn::reward::Rewards>,
}
impl StudioHost<'_> {
    fn emit(&self, event: &ProgramEvent) {
        let _ = self.app.emit("plan-event", serde_json::json!({"planId": self.plan.id, "event": event}));
    }
}
impl CourseHost for StudioHost<'_> {
    fn event(&mut self, e: &ProgramEvent) {
        self.emit(e);
        let batch = matches!(
            e,
            ProgramEvent::Lesson { event: CourseEvent::Session { event: Event::Batch { .. }, .. }, .. }
                | ProgramEvent::Remediation { event: Event::Batch { .. }, .. }
        );
        if let ProgramEvent::Lesson { event: CourseEvent::FileFinished(r), .. } = e {
            self.meta.taught.push(TaughtFile {
                name: r.name.clone(),
                sha256: r.sha256.clone(),
                outcome: r.outcome,
                end_step: r.end_step,
                time: now(),
            });
        }
        if !batch {
            append_history(
                self.app,
                &self.meta.id,
                serde_json::json!({"time": now(), "plan": self.plan.id, "event": e}),
            );
        }
    }
    fn study(&mut self) -> Option<&mut dyn yendory_engine::memory::StudyHook> {
        self.study.as_mut().map(|s| s as &mut dyn yendory_engine::memory::StudyHook)
    }
    fn take_rewards(&mut self) -> Option<yendory_engine::learn::reward::Rewards> {
        self.rewards.take()
    }
    fn return_rewards(&mut self, r: yendory_engine::learn::reward::Rewards) {
        super::rewards::keep(self.app, &r);
        self.rewards = Some(r);
    }
    fn save(&mut self, model: &Bicameral, archive: &[Item], run: &CourseRun) -> Res<()> {
        refresh_meta(&mut self.meta, model);
        save_all(self.app, &self.meta, model, archive)?;
        if let Some(s) = &self.study {
            super::study::save_memory(self.app, &self.meta.id, &s.store)?;
        }
        let studio = self.app.state::<Studio>();
        if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == self.meta.id) {
            a.model = model.clone();
            a.archive = archive.to_vec();
            a.meta = self.meta.clone();
        }
        self.plan.run = Some(run.clone());
        save_plan(self.app, &self.plan)
    }
    fn save_best(&mut self, best: &Snapshot) -> Res<()> {
        let dir = plan_dir(self.app, &self.plan.id)?;
        write_atomic(
            &dir.join("best_archive.json"),
            &serde_json::to_vec(&best.archive).map_err(|e| e.to_string())?,
        )?;
        write_atomic(&dir.join("best.nnova2"), &best.model.to_bytes()?)
    }
}

fn load_best(app: &AppHandle, plan_id: &str) -> Option<Snapshot> {
    let dir = plan_dir(app, plan_id).ok()?;
    let model = Bicameral::load(dir.join("best.nnova2")).ok()?;
    let archive = serde_json::from_slice(&std::fs::read(dir.join("best_archive.json")).ok()?).ok()?;
    Some(Snapshot { model, archive })
}

/// Starts or resumes a plan on its model (which must be open).
#[tauri::command]
pub fn start_plan(app: AppHandle, studio: State<Studio>, id: String) -> Res<PlanView> {
    let mut plan = load_plan(&app, &id)?;
    if plan.courses.is_empty() {
        return Err("Add courses to the plan first".into());
    }
    if plan.position >= plan.courses.len() {
        return Err("Every course of this plan has passed; add courses to continue".into());
    }
    if plan.status == PlanStatus::Failed {
        return Err("The course failed: revise it and use Restart course, or give it its one override".into());
    }
    let (meta, model, archive) = with_active(&studio, |a| {
        if a.meta.id != plan.model_id {
            return Err(format!("Open the model \"{}\" first: this plan teaches it", plan.model_name));
        }
        Ok((a.meta.clone(), a.model.clone(), a.archive.clone()))
    })?;
    if studio.teaching.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("Already teaching".into());
    }
    studio.stop.store(false, Ordering::Release);
    plan.status = PlanStatus::Running;
    plan.message = None;
    if let Err(e) = save_plan(&app, &plan) {
        studio.teaching.store(false, Ordering::Release);
        return Err(e);
    }
    let view = plan_view(&app, &plan);
    let stop = studio.stop.clone();
    std::thread::spawn(move || {
        let id = plan.id.clone();
        let plan = match guarded(|| Ok(run_plan(&app, plan, meta, model, archive, &stop))) {
            Ok(p) => p,
            // A crash: the plan as last saved, marked for attention with the reason.
            Err(e) => match load_plan(&app, &id) {
                Ok(mut p) => {
                    p.status = PlanStatus::NeedsAttention;
                    p.message = Some(e);
                    let _ = save_plan(&app, &p);
                    p
                }
                Err(_) => {
                    let studio = app.state::<Studio>();
                    studio.teaching.store(false, Ordering::Release);
                    let _ = app.emit("device-notice", e);
                    return;
                }
            },
        };
        let studio = app.state::<Studio>();
        studio.teaching.store(false, Ordering::Release);
        if let Some(n) = yendory_engine::learn::take_device_notice() {
            let _ = app.emit("device-notice", n);
        }
        let _ = app.emit("plan-finished", plan_view(&app, &plan));
    });
    Ok(view)
}

fn run_plan(
    app: &AppHandle,
    mut plan: Plan,
    meta: Meta,
    mut model: Bicameral,
    mut archive: Vec<Item>,
    stop: &AtomicBool,
) -> Plan {
    let mut meta = meta;
    let attention = |plan: &mut Plan, msg: String| {
        plan.status = PlanStatus::NeedsAttention;
        plan.message = Some(msg);
    };
    loop {
        if plan.position >= plan.courses.len() {
            plan.status = PlanStatus::Finished;
            plan.message = Some("Every course in the plan passed".into());
            break;
        }
        if plan.run.is_none() {
            let course_id = plan.courses[plan.position].clone();
            let def = match load_course(app, &course_id) {
                Ok(d) => d,
                Err(e) => {
                    attention(&mut plan, e);
                    break;
                }
            };
            let prep = prepare(&def);
            if !prep.check.problems.is_empty() {
                attention(
                    &mut plan,
                    format!("{} cannot start: {}", def.name, prep.check.problems.join("; ")),
                );
                break;
            }
            let overridden = match restart_policy(&model_runs(app, &plan.model_id), &prep.fingerprint) {
                Restart::Fresh => false,
                Restart::NeedsOverride if plan.next_overridden => true,
                Restart::NeedsOverride => {
                    plan.status = PlanStatus::Failed;
                    plan.message = Some(format!(
                        "This version of {} failed before; revise it or use its one override",
                        def.name
                    ));
                    break;
                }
                Restart::Refused => {
                    plan.status = PlanStatus::Failed;
                    plan.message = Some(format!(
                        "This version of {} failed after its override; revise it first",
                        def.name
                    ));
                    break;
                }
            };
            plan.next_overridden = false;
            plan.run = Some(CourseRun::new(&prep, overridden, now() | 1));
            plan.run_started_step = model.step;
            plan.frozen = Some(def);
            if let Ok(dir) = plan_dir(app, &plan.id) {
                let _ = std::fs::remove_file(dir.join("best.nnova2"));
            }
            let _ = save_plan(app, &plan);
        }
        let Some(def) = plan.frozen.clone() else {
            attention(&mut plan, "The course version of this run is missing; restart the course".into());
            break;
        };
        let prep = prepare(&def);
        let mut run = plan.run.take().expect("run set above");
        if run.phase != Phase::Pretest && run.saved_step != model.step {
            attention(
                &mut plan,
                format!(
                    "The model changed (step {} → {}) since this course stopped; restart the course to continue",
                    run.saved_step, model.step
                ),
            );
            plan.run = Some(run);
            break;
        }
        let best = load_best(app, &plan.id);
        let opts = RunOptions { word_limit: meta.word_limit, ..RunOptions::default() };
        let study = super::study::study_mode(app)
            .then(|| yendory_engine::memory::Studying::new(super::study::load_memory(app, &meta.id), now()));
        let rewards = super::rewards::rewards_for(app, &meta.id);
        let mut host = StudioHost { app, plan: plan.clone(), meta: meta.clone(), study, rewards };
        let result = run_course(&model, &archive, &prep, &mut run, best, &opts, &mut host, stop);
        meta = host.meta;
        match result {
            Err(e) => {
                attention(&mut plan, e);
                plan.run = Some(run);
                break;
            }
            Ok(r) => {
                model = r.model;
                archive = r.archive;
                if r.stopped {
                    plan.status = PlanStatus::Stopped;
                    plan.message =
                        Some(format!("Stopped during {}; Resume continues where it stopped", run.name));
                    plan.run = Some(run);
                    break;
                }
            }
        }
        // The run finished: the model returned is the run's best when it failed.
        refresh_meta(&mut meta, &model);
        let _ = save_all(app, &meta, &model, &archive);
        let studio = app.state::<Studio>();
        if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == meta.id) {
            a.model = model.clone();
            a.archive = archive.clone();
            a.meta = meta.clone();
        }
        let _ = add_model_run(
            app,
            &plan.model_id,
            RunSummary { fingerprint: run.fingerprint.clone(), phase: run.phase, overridden: run.overridden },
        );
        plan.results.push(RunEntry {
            course_id: run.course_id.clone(),
            name: run.name.clone(),
            fingerprint: run.fingerprint.clone(),
            phase: run.phase,
            overridden: run.overridden,
            pretest: run.pretest.clone(),
            grades: run.grades.clone(),
            remediation: run.remediation.clone(),
            lessons: prep
                .lessons
                .iter()
                .zip(&run.records)
                .map(|(l, r)| (l.name.clone(), r.as_ref().map(|r| r.outcome)))
                .collect(),
            started_step: plan.run_started_step,
            end_step: model.step,
            finished: now(),
        });
        if run.phase == Phase::Passed {
            plan.position += 1;
            plan.frozen = None;
            plan.run = None;
            let _ = save_plan(app, &plan);
            continue;
        }
        plan.status = PlanStatus::Failed;
        plan.message = Some(format!(
            "{} {} (best grade {:.1}%); the plan stops here. The model keeps its best checkpoint of the course",
            run.name,
            if run.phase == Phase::TimedOut { "ran out of time" } else { "failed after three remediation attempts" },
            100. * run.best_grade.unwrap_or(0.)
        ));
        plan.run = Some(run);
        break;
    }
    let _ = save_plan(app, &plan);
    plan
}
