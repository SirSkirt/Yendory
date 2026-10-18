//! Lesson Designer: drafts stored as <app data>/designs/<id>.json. Analysis, validation and
//! the lesson file come from `yendory_engine::learn::designer`; this file stores drafts and
//! hands finished lessons to the Classroom list or a course.
use super::courses::{check_id, data_dir, fresh_id, load_course, save_course};
use super::*;
use yendory_engine::learn::curriculum::CourseDef;
use yendory_engine::learn::designer::{
    analyze, export, file_name, validate, AnalyzeOptions, Exported, Issue, LessonDraft, Severity,
};

/// Imported datasets are limited to this size.
const MAX_DATASET_BYTES: usize = 16 << 20;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredDesign {
    draft: LessonDraft,
    created: u64,
    updated: u64,
}

fn design_path(app: &AppHandle, id: &str) -> Res<PathBuf> {
    check_id(id)?;
    Ok(data_dir(app, "designs")?.join(format!("{id}.json")))
}

fn load(app: &AppHandle, id: &str) -> Res<StoredDesign> {
    let bytes =
        std::fs::read(design_path(app, id)?).map_err(|_| "This lesson draft no longer exists".to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

fn store(app: &AppHandle, d: &StoredDesign) -> Res<()> {
    write_atomic(&design_path(app, &d.draft.id)?, &serde_json::to_vec_pretty(d).map_err(|e| e.to_string())?)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DesignSummary {
    pub(crate) id: String,
    pub(crate) name: String,
    source_file: String,
    finished: bool,
    blocks: usize,
    items: usize,
    errors: usize,
    updated: u64,
}

fn summary(d: &StoredDesign) -> DesignSummary {
    let issues = validate(&d.draft);
    DesignSummary {
        id: d.draft.id.clone(),
        name: d.draft.name.clone(),
        source_file: d.draft.source_file.clone(),
        finished: d.draft.finished,
        blocks: d.draft.blocks.len(),
        items: d.draft.blocks.iter().map(|b| b.items.len()).sum(),
        errors: issues.iter().filter(|i| i.severity == Severity::Error).count(),
        updated: d.updated,
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DesignView {
    draft: LessonDraft,
    issues: Vec<Issue>,
    /// The lesson file as it would be taught, or why it cannot be written.
    lesson: Result<Exported, String>,
    file_name: String,
    updated: u64,
}

impl DesignView {
    pub(crate) fn draft_ref(&self) -> &LessonDraft {
        &self.draft
    }
}

fn view(d: &StoredDesign) -> DesignView {
    DesignView {
        issues: validate(&d.draft),
        lesson: export(&d.draft),
        file_name: file_name(&d.draft),
        draft: d.draft.clone(),
        updated: d.updated,
    }
}

/// Saves a draft made elsewhere (Review Training) as a new Designer lesson; returns its id.
pub(crate) fn create_from(app: &AppHandle, mut draft: LessonDraft) -> Res<String> {
    draft.id = fresh_id('d');
    let t = now();
    let id = draft.id.clone();
    store(app, &StoredDesign { draft, created: t, updated: t })?;
    Ok(id)
}

pub(crate) fn load_draft(app: &AppHandle, id: &str) -> Res<LessonDraft> {
    Ok(load(app, id)?.draft)
}

#[tauri::command]
pub fn list_designs(app: AppHandle) -> Res<Vec<DesignSummary>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(data_dir(&app, "designs")?).map_err(|e| e.to_string())?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Some(d) =
            std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<StoredDesign>(&b).ok())
        {
            out.push(summary(&d));
        }
    }
    out.sort_by_key(|d| std::cmp::Reverse(d.updated));
    Ok(out)
}

/// A new, empty lesson.
#[tauri::command]
pub fn create_design(app: AppHandle, name: String) -> Res<DesignView> {
    let t = now();
    let d = StoredDesign {
        draft: LessonDraft {
            id: fresh_id('d'),
            name: if name.trim().is_empty() { "New lesson".into() } else { name.trim().into() },
            source_file: String::new(),
            source_sha256: String::new(),
            stage: None,
            finished: false,
            notes: String::new(),
            blocks: Vec::new(),
        },
        created: t,
        updated: t,
    };
    store(&app, &d)?;
    Ok(view(&d))
}

/// Imports a `.txt` dataset: the analysis proposes one draft, or several related lessons for
/// a large file (`split`). Returns the new drafts.
#[tauri::command]
pub async fn import_design(app: AppHandle, split: bool) -> Res<Vec<DesignSummary>> {
    let Some(path) = app.dialog().file().add_filter("Text dataset (.txt)", &["txt"]).blocking_pick_file()
    else {
        return Ok(Vec::new());
    };
    let name = display_name(&path);
    let bytes = read_picked(&app, &path, MAX_DATASET_BYTES)?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let drafts = analyze(&text, &name, AnalyzeOptions { split });
    if drafts.iter().all(|d| d.blocks.is_empty()) {
        return Err(format!("{name}: nothing could be read from this file"));
    }
    let mut out = Vec::new();
    let t = now();
    for (k, mut draft) in drafts.into_iter().enumerate() {
        draft.id = format!("{}-{k}", fresh_id('d'));
        let d = StoredDesign { draft, created: t, updated: t };
        store(&app, &d)?;
        out.push(summary(&d));
    }
    Ok(out)
}

#[tauri::command]
pub fn design_detail(app: AppHandle, id: String) -> Res<DesignView> {
    Ok(view(&load(&app, &id)?))
}

/// Saves the edited draft (unfinished drafts are saved as they are) and returns its checks.
#[tauri::command]
pub fn save_design(app: AppHandle, draft: LessonDraft) -> Res<DesignView> {
    let old = load(&app, &draft.id)?;
    let d = StoredDesign { draft, created: old.created, updated: now() };
    store(&app, &d)?;
    Ok(view(&d))
}

#[tauri::command]
pub fn delete_design(app: AppHandle, id: String) -> Res<()> {
    std::fs::remove_file(design_path(&app, &id)?).map_err(|e| e.to_string())
}

/// The lesson file of a draft that has no errors.
fn lesson_file(app: &AppHandle, id: &str) -> Res<(CourseFile, Exported)> {
    let d = load(app, id)?;
    let errors: Vec<String> =
        validate(&d.draft).into_iter().filter(|i| i.severity == Severity::Error).map(|i| i.message).collect();
    if !errors.is_empty() {
        return Err(format!("Fix these first: {}", errors.join("; ")));
    }
    let ex = export(&d.draft)?;
    Ok((CourseFile { name: file_name(&d.draft), text: ex.text.clone() }, ex))
}

/// Adds the finished lesson to the Classroom list (replacing a lesson of the same name).
#[tauri::command]
pub fn design_to_classroom(app: AppHandle, studio: State<Studio>, id: String) -> Res<String> {
    not_teaching(&studio)?;
    let (file, ex) = lesson_file(&app, &id)?;
    let mut lessons = studio.lessons.lock().unwrap();
    lessons.retain(|l| l.name != file.name);
    let name = file.name.clone();
    lessons.push(file);
    yendory_engine::learn::course::sort_files(&mut lessons);
    Ok(format!("{name} is in the Classroom list: {} units taught, {} held out", ex.taught, ex.held_out))
}

/// Saves the finished lesson into a course (an existing one, or a new one with `new_course`
/// as its name). Its file name orders it among the course's lessons.
#[tauri::command]
pub fn design_to_course(
    app: AppHandle,
    id: String,
    course: Option<String>,
    new_course: Option<String>,
) -> Res<String> {
    let (file, ex) = lesson_file(&app, &id)?;
    let mut def = match (course, new_course) {
        (Some(c), _) if !c.is_empty() => load_course(&app, &c)?,
        (_, Some(n)) if !n.trim().is_empty() => CourseDef::new(&fresh_id('c'), n.trim()),
        _ => return Err("Choose a course or name a new one".into()),
    };
    let name = file.name.clone();
    let notes = def.add_files(vec![file]);
    save_course(&app, &def)?;
    let mut msg = format!(
        "{name} saved in the course {}: {} units taught, {} held out, {} extra practice",
        def.name, ex.taught, ex.held_out, ex.extra_practice
    );
    if !notes.is_empty() {
        msg.push_str(&format!(" ({})", notes.join("; ")));
    }
    Ok(msg)
}
