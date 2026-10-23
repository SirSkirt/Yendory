//! Course Builder: courses made of lesson files, their stages, pass marks, extra practice
//! material and final exam, and the preparation that turns them into training and evaluation
//! items. Running a course is `program.rs`; the rules are in docs/LEARNING.md ("Courses").
//!
//! File names decide order and stage: the first number in a name orders the lessons; the words
//! "Dense", "Transformer" and "Co-op" choose Stage A, B or C. "Practice" marks extra practice
//! material for the lesson with the same number and stage; "Exam" marks the course's final exam.
//!
//! Evaluation questions are never trained on: about 10% of each lesson's questions are held out
//! for the lesson's grade (the usual hash), and without a teacher-written exam about another 10%
//! are held out, by a second independent hash, for the final exam. Practice material and exam
//! questions are checked against both, and matches are excluded.
use super::course::CourseFile;
use super::grading::{derive_conflicts, AnswerKey, GradeMode};
use super::lessons::{blocks, canonical, crc32, parse_lesson, qa, Item, Kind};
use super::session::EducationStage;
use crate::fingerprint::sha256_hex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Fewer evaluation questions than this in a lesson or an exam section gives a warning: scores
/// become coarse (with 3 questions the only scores are 0, 33, 67 and 100%).
pub const FEW_QUESTIONS: usize = 8;
/// Share (%) of a lesson's remaining questions held out for the automatically assembled exam.
pub const EXAM_PERCENT: u32 = 11;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileRole {
    Lesson,
    Practice,
    Exam,
}

/// What a file name says about the file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NameInfo {
    pub role: FileRole,
    pub number: Option<u32>,
    pub stage: Option<EducationStage>,
    /// The name is ambiguous (e.g. names two stages); the file needs attention.
    pub problem: Option<String>,
}

fn base_name(name: &str) -> &str {
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    match file.rfind('.') {
        Some(i) if i > 0 => &file[..i],
        _ => file,
    }
}

/// Reads the role, number and stage from a file name ("03 Dense - colours.txt",
/// "Lesson 3 Co-op Practice.txt", "Final Exam.txt"). Keywords are whole words, any case.
pub fn read_name(name: &str) -> NameInfo {
    let base = base_name(name);
    let lower = base.to_ascii_lowercase();
    let words: Vec<&str> =
        lower.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let mut stages = BTreeSet::new();
    let (mut practice, mut exam) = (false, false);
    let mut i = 0;
    while i < words.len() {
        match words[i] {
            "dense" => {
                stages.insert(0);
            }
            "transformer" => {
                stages.insert(1);
            }
            "coop" | "cooperative" => {
                stages.insert(2);
            }
            "co" if words.get(i + 1) == Some(&"op") => {
                stages.insert(2);
                i += 1;
            }
            "practice" => practice = true,
            "exam" | "exams" => exam = true,
            _ => {}
        }
        i += 1;
    }
    let number = {
        let digits: String =
            base.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    };
    let stage_of = |s: u8| match s {
        0 => EducationStage::DenseKnowledge,
        1 => EducationStage::TransformerLanguage,
        _ => EducationStage::Cooperative,
    };
    let mut problem = None;
    if stages.len() > 1 {
        problem = Some("the name contains more than one stage keyword (Dense, Transformer, Co-op)".into());
    }
    if practice && exam {
        problem = Some("the name says both \"Practice\" and \"Exam\"".into());
    }
    NameInfo {
        role: if exam {
            FileRole::Exam
        } else if practice {
            FileRole::Practice
        } else {
            FileRole::Lesson
        },
        number,
        stage: if stages.len() == 1 { stages.first().map(|&s| stage_of(s)) } else { None },
        problem,
    }
}

/// Where a lesson's stage came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageFrom {
    /// A keyword in the file name.
    Name,
    /// No keyword: the course's default stage.
    Default,
    /// Chosen by hand in the Course Builder.
    Manual,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LessonDef {
    pub file: CourseFile,
    pub number: Option<u32>,
    /// The lesson's stage (None: whole brain). Ignored when `stage_from` is Default.
    pub stage: Option<EducationStage>,
    pub stage_from: StageFrom,
    /// Share of the lesson's evaluation questions to answer correctly (0..=1).
    pub pass_mark: f32,
    /// Extra practice files: held in reserve, used only when the course is failed.
    #[serde(default)]
    pub practice: Vec<CourseFile>,
    /// The file name is ambiguous; the course cannot start until the stage is chosen by hand.
    #[serde(default)]
    pub name_problem: Option<String>,
    /// Taught together with the course's other "together" lessons of the same stage, in one
    /// mixed session (the default), instead of on its own.
    #[serde(default = "yes")]
    pub together: bool,
}

fn yes() -> bool {
    true
}

/// Final-exam section weights (relative; sections without questions are left out and the
/// others share their weight proportionally).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SectionWeights {
    pub dense: f32,
    pub transformer: f32,
    pub coop: f32,
    pub whole_brain: f32,
}
impl Default for SectionWeights {
    fn default() -> Self {
        SectionWeights { dense: 33.33, transformer: 33.33, coop: 33.34, whole_brain: 33.33 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Section {
    Dense,
    Transformer,
    Coop,
    WholeBrain,
}
impl Section {
    pub fn of(stage: Option<EducationStage>) -> Section {
        match stage {
            Some(EducationStage::DenseKnowledge) => Section::Dense,
            Some(EducationStage::TransformerLanguage) => Section::Transformer,
            Some(EducationStage::Cooperative) => Section::Coop,
            None => Section::WholeBrain,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Section::Dense => "Dense",
            Section::Transformer => "Transformer",
            Section::Coop => "Co-op",
            Section::WholeBrain => "Whole brain",
        }
    }
    fn mode(self) -> GradeMode {
        if self == Section::Dense {
            GradeMode::Keyword
        } else {
            GradeMode::Sentence
        }
    }
    fn weight(self, w: &SectionWeights) -> f32 {
        match self {
            Section::Dense => w.dense,
            Section::Transformer => w.transformer,
            Section::Coop => w.coop,
            Section::WholeBrain => w.whole_brain,
        }
    }
}

/// Training settings of a course. Changing them does not count as a revision of the course.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CourseSettings {
    pub epochs: u32,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub foundation: bool,
    pub rehearsal: bool,
    pub protect_answers: bool,
    /// Lesson retries after the first attempt (0..=3).
    pub retries: u32,
    /// A lesson not passed after its retries moves once to the end of the course's queue.
    pub requeue: bool,
    /// Stop the plan when the course has taken longer than this.
    pub time_limit_hours: Option<f32>,
    /// Epochs in a row without an accepted checkpoint before a lesson attempt ends; training
    /// continues through them from the best safe candidate.
    pub flat_epochs: u32,
    /// Judge this lesson's held-out answers by their net change (see
    /// `SessionConfig::protect_answers_net`). Off: no answer may be lost.
    pub protect_answers_net: bool,
    /// Earlier lessons' held-out loss may rise by this share instead of a fixed 0.02
    /// (see `SessionConfig::retention_relative`).
    pub retention_relative: Option<f32>,
    /// Earlier-lesson held-out units per kind for that loss.
    pub retention_units: usize,
    /// Dropout while training lessons (0 = off; see `SessionConfig::dropout`).
    pub dropout: f32,
    /// Share of each epoch's rows taken from general reading text (0 = off; the text comes from
    /// `RunOptions::reading`; see `SessionConfig::reading_share`).
    pub reading_share: f32,
    /// Weight blending: one more candidate per epoch (0 = off; see `SessionConfig::blend`).
    pub blend: f32,
}
impl Default for CourseSettings {
    fn default() -> Self {
        CourseSettings {
            epochs: 4,
            batch_size: 16,
            learning_rate: 1e-3,
            foundation: true,
            rehearsal: true,
            protect_answers: true,
            retries: 3,
            requeue: true,
            time_limit_hours: None,
            flat_epochs: 4,
            protect_answers_net: true,
            retention_relative: None,
            retention_units: 16,
            dropout: 0.,
            reading_share: 0.,
            blend: 0.,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CourseDef {
    pub id: String,
    pub name: String,
    pub lessons: Vec<LessonDef>,
    /// Teacher-written final exam; without one the exam is assembled from the lessons.
    #[serde(default)]
    pub exam: Option<CourseFile>,
    /// Practice files whose lesson is not in the course (yet).
    #[serde(default)]
    pub unmatched_practice: Vec<CourseFile>,
    /// Stage of lessons whose name has no stage keyword (None: whole brain).
    #[serde(default)]
    pub default_stage: Option<EducationStage>,
    /// Pass mark given to newly added lessons.
    pub default_pass_mark: f32,
    #[serde(default)]
    pub section_weights: SectionWeights,
    /// Optional minimum final-exam score (0..=1), in addition to the course grade.
    #[serde(default)]
    pub exam_minimum: Option<f32>,
    /// Give the final exam before the course as an ungraded baseline.
    #[serde(default)]
    pub pretest: bool,
    #[serde(default)]
    pub settings: CourseSettings,
}

fn stage_rank(s: Option<EducationStage>) -> u8 {
    match s {
        Some(EducationStage::DenseKnowledge) => 0,
        Some(EducationStage::TransformerLanguage) => 1,
        Some(EducationStage::Cooperative) => 2,
        None => 3,
    }
}

impl CourseDef {
    pub fn new(id: &str, name: &str) -> Self {
        CourseDef {
            id: id.into(),
            name: name.into(),
            lessons: Vec::new(),
            exam: None,
            unmatched_practice: Vec::new(),
            default_stage: None,
            default_pass_mark: 0.7,
            section_weights: SectionWeights::default(),
            exam_minimum: None,
            pretest: false,
            settings: CourseSettings::default(),
        }
    }

    /// The stage a lesson is taught in.
    pub fn stage_of(&self, l: &LessonDef) -> Option<EducationStage> {
        match l.stage_from {
            StageFrom::Default => self.default_stage,
            _ => l.stage,
        }
    }

    /// Adds files: lessons are placed by number and stage, practice files attached to their
    /// lesson, an exam file becomes the course's exam. A file with the same name replaces the
    /// earlier one. Returns notes about what was done.
    pub fn add_files(&mut self, files: Vec<CourseFile>) -> Vec<String> {
        let mut notes = Vec::new();
        for file in files {
            let info = read_name(&file.name);
            match info.role {
                FileRole::Exam => {
                    if let Some(old) = &self.exam {
                        notes.push(format!("{} replaces the exam {}", file.name, old.name));
                    }
                    self.exam = Some(file);
                }
                FileRole::Practice => {
                    self.unmatched_practice.retain(|p| p.name != file.name);
                    for l in &mut self.lessons {
                        l.practice.retain(|p| p.name != file.name);
                    }
                    self.unmatched_practice.push(file);
                }
                FileRole::Lesson => {
                    let lesson = LessonDef {
                        number: info.number,
                        stage: info.stage,
                        stage_from: if info.stage.is_some() { StageFrom::Name } else { StageFrom::Default },
                        pass_mark: self.default_pass_mark,
                        practice: Vec::new(),
                        name_problem: info.problem,
                        file,
                        together: true,
                    };
                    match self.lessons.iter_mut().find(|l| l.file.name == lesson.file.name) {
                        Some(old) => {
                            notes
                                .push(format!("{} replaced (pass mark and practice kept)", lesson.file.name));
                            old.file = lesson.file;
                        }
                        None => self.lessons.push(lesson),
                    }
                }
            }
        }
        self.match_practice(&mut notes);
        self.sort();
        notes
    }

    /// Attaches waiting practice files to the lesson with the same number and stage.
    fn match_practice(&mut self, notes: &mut Vec<String>) {
        let waiting = std::mem::take(&mut self.unmatched_practice);
        for p in waiting {
            let info = read_name(&p.name);
            let matches: Vec<usize> = self
                .lessons
                .iter()
                .enumerate()
                .filter(|(_, l)| l.number.is_some() && l.number == info.number && l.stage == info.stage)
                .map(|(i, _)| i)
                .collect();
            match matches[..] {
                [i] => {
                    notes.push(format!("{} is extra practice for {}", p.name, self.lessons[i].file.name));
                    self.lessons[i].practice.push(p);
                }
                [] => {
                    notes.push(format!("{}: no lesson with the same number and stage yet", p.name));
                    self.unmatched_practice.push(p);
                }
                _ => {
                    notes.push(format!("{}: several lessons have the same number and stage", p.name));
                    self.unmatched_practice.push(p);
                }
            }
        }
    }

    /// Lesson order: by number (numbered first), then Dense, Transformer, Co-op, whole brain,
    /// then by name.
    pub fn sort(&mut self) {
        self.lessons.sort_by(|a, b| {
            (a.number.is_none(), a.number, stage_rank(a.stage), &a.file.name).cmp(&(
                b.number.is_none(),
                b.number,
                stage_rank(b.stage),
                &b.file.name,
            ))
        });
    }

    /// The course's content fingerprint: lesson files, stages, pass marks, practice material,
    /// exam and grading settings. Training settings are not part of it, so changing only them
    /// is not a revision.
    pub fn fingerprint(&self) -> String {
        let lessons: Vec<_> = self
            .lessons
            .iter()
            .map(|l| {
                let mut practice: Vec<String> =
                    l.practice.iter().map(|p| sha256_hex(p.text.as_bytes())).collect();
                practice.sort();
                let mut lesson = serde_json::json!({
                    "sha256": sha256_hex(l.file.text.as_bytes()),
                    "stage": self.stage_of(l),
                    "pass_mark": l.pass_mark,
                    "practice": practice,
                });
                // Taught on its own instead of together: a different course (recorded only
                // then, so courses made before this setting keep their fingerprint).
                if !l.together {
                    lesson["together"] = false.into();
                }
                lesson
            })
            .collect();
        let doc = serde_json::json!({
            "lessons": lessons,
            "exam": self.exam.as_ref().map(|e| sha256_hex(e.text.as_bytes())),
            "weights": self.section_weights,
            "exam_minimum": self.exam_minimum,
        });
        sha256_hex(doc.to_string().as_bytes())
    }

    /// The course grade needed to pass: the average of the lesson pass marks.
    pub fn required_grade(&self) -> f32 {
        if self.lessons.is_empty() {
            0.
        } else {
            self.lessons.iter().map(|l| l.pass_mark).sum::<f32>() / self.lessons.len() as f32
        }
    }
}

/// A lesson ready to teach and grade.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedLesson {
    pub name: String,
    pub sha256: String,
    pub stage: Option<EducationStage>,
    pub pass_mark: f32,
    /// Training units and the lesson's evaluation questions (`held_out`).
    pub items: Vec<Item>,
    /// Extra practice units (training only), used in remediation.
    pub practice: Vec<Item>,
    /// Taught together with the other "together" lessons of its stage (see `LessonDef`).
    #[serde(default = "yes")]
    pub together: bool,
}
impl PreparedLesson {
    pub fn eval_items(&self) -> Vec<&Item> {
        self.items.iter().filter(|i| i.held_out && i.kind != Kind::Prose).collect()
    }
    pub fn train_items(&self) -> Vec<&Item> {
        self.items.iter().filter(|i| !i.held_out).collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExamSection {
    pub section: Section,
    /// Share of the exam (sections' weights add up to 1).
    pub weight: f32,
    pub items: Vec<Item>,
}

/// What the Course Builder shows for each lesson.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LessonCheck {
    pub name: String,
    pub train: usize,
    pub eval: usize,
    pub exam: usize,
    pub practice: usize,
    /// Practice questions excluded because they match an evaluation or exam question.
    pub practice_excluded: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CourseCheck {
    /// The course cannot start until these are fixed.
    pub problems: Vec<String>,
    pub warnings: Vec<String>,
    pub lessons: Vec<LessonCheck>,
    /// Questions per exam section, with the section's share of the exam.
    pub sections: Vec<(Section, usize, f32)>,
    pub teacher_exam: bool,
    pub required_grade: f32,
    pub fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedCourse {
    pub id: String,
    pub name: String,
    pub fingerprint: String,
    pub lessons: Vec<PreparedLesson>,
    pub exam: Vec<ExamSection>,
    pub required_grade: f32,
    pub exam_minimum: Option<f32>,
    pub pretest: bool,
    pub settings: CourseSettings,
    pub check: CourseCheck,
}

/// The text a question is compared by (leak checks, exam selection).
fn question_key(it: &Item) -> String {
    match it.kind {
        Kind::Conversation => {
            canonical(&it.users.iter().chain(&it.replies).cloned().collect::<Vec<_>>().join(" "))
        }
        _ => canonical(it.prompt.as_deref().unwrap_or("")),
    }
}

/// The second, independent held-out decision: questions reserved for the course's final exam.
fn in_exam_pool(it: &Item) -> bool {
    crc32(format!("exam:{}", question_key(it)).as_bytes()) % 100 < EXAM_PERCENT
}

/// Gives every question an answer key (accepted forms and rules) and trains on the first form.
pub(crate) fn apply_keys(items: &mut [Item], mode: GradeMode) -> Result<(), String> {
    for it in items.iter_mut() {
        if !matches!(it.kind, Kind::Dialogue | Kind::Understanding | Kind::Reasoning) {
            continue;
        }
        let (Some(q), Some(a)) = (it.prompt.clone(), it.answer.clone()) else { continue };
        let key =
            AnswerKey::parse(&a, mode).ok_or_else(|| format!("\"{q}\": the answer has no accepted form"))?;
        it.answer = Some(key.target().to_string());
        it.segments = qa(&q, key.target());
        it.key = Some(key);
    }
    let mut keyed: Vec<(String, AnswerKey)> =
        items.iter().filter_map(|it| Some((it.prompt.clone()?, it.key.clone()?))).collect();
    {
        let mut refs: Vec<(&str, &mut AnswerKey)> = keyed.iter_mut().map(|(q, k)| (q.as_str(), k)).collect();
        derive_conflicts(&mut refs);
    }
    let mut next = keyed.into_iter();
    for it in items.iter_mut().filter(|it| it.key.is_some() && it.prompt.is_some()) {
        if let Some((_, k)) = next.next() {
            it.key = Some(k);
        }
    }
    Ok(())
}

/// Practice items from a text: everything is training material.
fn practice_items(text: &str, source: &str, mode: GradeMode) -> Result<Vec<Item>, String> {
    let mut items = parse_lesson(text, source)?;
    for it in &mut items {
        it.held_out = false;
    }
    apply_keys(&mut items, mode)?;
    Ok(items)
}

/// Turns a course into training, practice and evaluation items and checks it.
pub fn prepare(def: &CourseDef) -> PreparedCourse {
    let mut check = CourseCheck {
        teacher_exam: def.exam.is_some(),
        required_grade: def.required_grade(),
        fingerprint: def.fingerprint(),
        ..CourseCheck::default()
    };
    if def.lessons.is_empty() {
        check.problems.push("The course has no lessons".into());
    }
    for p in &def.unmatched_practice {
        check.warnings.push(format!("{}: practice file without a matching lesson (not used)", p.name));
    }
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut lessons = Vec::new();
    let mut pool: Vec<(Section, Item)> = Vec::new();
    for l in &def.lessons {
        let stage = def.stage_of(l);
        let mode = Section::of(stage).mode();
        let name = l.file.name.clone();
        let sha256 = sha256_hex(l.file.text.as_bytes());
        let mut lc = LessonCheck { name: name.clone(), ..LessonCheck::default() };
        if let Some(p) = &l.name_problem {
            if l.stage_from != StageFrom::Manual {
                check.problems.push(format!("{name}: {p}; choose its stage"));
            }
        }
        if !(0.0..=1.0).contains(&l.pass_mark) {
            check.problems.push(format!("{name}: the pass mark must be 0-100%"));
        }
        if let Some((_, first)) = seen.iter().find(|(h, _)| *h == sha256) {
            check.problems.push(format!("{name}: same content as {first}"));
        }
        seen.push((sha256.clone(), name.clone()));
        let mut items = match parse_lesson(&l.file.text, &name).and_then(|mut items| {
            apply_keys(&mut items, mode)?;
            Ok(items)
        }) {
            Ok(items) => items,
            Err(e) => {
                check.problems.push(format!("{name}: {e}"));
                check.lessons.push(lc);
                continue;
            }
        };
        if def.exam.is_none() {
            let (exam, keep): (Vec<Item>, Vec<Item>) =
                items.into_iter().partition(|it| !it.held_out && it.kind != Kind::Prose && in_exam_pool(it));
            items = keep;
            lc.exam = exam.len();
            pool.extend(exam.into_iter().map(|mut it| {
                it.held_out = true;
                (Section::of(stage), it)
            }));
        }
        lc.train = items.iter().filter(|i| !i.held_out && i.kind != Kind::Prose).count();
        lc.eval = items.iter().filter(|i| i.held_out && i.kind != Kind::Prose).count();
        if lc.train == 0 {
            check.problems.push(format!("{name}: no training questions"));
        }
        if lc.eval == 0 {
            check.problems.push(format!(
                "{name}: no evaluation questions (about 10% of the questions are held out by hash; add more examples)"
            ));
        } else if lc.eval < FEW_QUESTIONS {
            let need = (l.pass_mark * lc.eval as f32).ceil() as usize;
            check.warnings.push(format!(
                "{name}: only {} evaluation question(s); a {:.0}% mark means {need} of {}",
                lc.eval,
                l.pass_mark * 100.,
                lc.eval
            ));
        }
        // Extra practice: attached files and an [Extra Practice] block in the lesson itself.
        let mut practice = Vec::new();
        for p in &l.practice {
            match practice_items(&p.text, &name, mode) {
                Ok(items) => practice.extend(items),
                Err(e) => check.problems.push(format!("{}: {e}", p.name)),
            }
        }
        for (block, body) in blocks(&l.file.text.replace("\r\n", "\n")) {
            if block == "Extra Practice" {
                match practice_items(body, &name, mode) {
                    Ok(items) => practice.extend(items),
                    Err(e) => check.problems.push(format!("{name} (Extra Practice): {e}")),
                }
            }
        }
        lessons.push((
            PreparedLesson {
                name,
                sha256,
                stage,
                pass_mark: l.pass_mark,
                items,
                practice,
                together: l.together,
            },
            check.lessons.len(),
        ));
        check.lessons.push(lc);
    }

    // The final exam: the teacher's file, or the reserved pool.
    let mut exam_items: Vec<(Section, Item)> = Vec::new();
    if let Some(exam) = &def.exam {
        let text = exam.text.replace("\r\n", "\n");
        let mut found = false;
        for (block, body) in blocks(&text) {
            let section = match block {
                "Dense" => Section::Dense,
                "Transformer" => Section::Transformer,
                "Co-op" | "Coop" | "Cooperative" => Section::Coop,
                "Whole Brain" | "Whole brain" => Section::WholeBrain,
                _ => continue,
            };
            found = true;
            match parse_lesson(body, &exam.name).and_then(|mut items| {
                apply_keys(&mut items, section.mode())?;
                Ok(items)
            }) {
                Ok(items) => exam_items.extend(items.into_iter().map(|it| (section, it))),
                Err(e) => check.problems.push(format!("{} ({}): {e}", exam.name, section.name())),
            }
        }
        if !found {
            match parse_lesson(&text, &exam.name).and_then(|mut items| {
                apply_keys(&mut items, GradeMode::Sentence)?;
                Ok(items)
            }) {
                Ok(items) => exam_items.extend(items.into_iter().map(|it| (Section::WholeBrain, it))),
                Err(e) => check.problems.push(format!("{}: {e}", exam.name)),
            }
        }
        exam_items.retain(|(_, it)| it.kind != Kind::Prose);
        for (_, it) in &mut exam_items {
            it.held_out = true;
        }
        // Exam questions that appear in any lesson or practice material are not independent.
        let taught: BTreeSet<String> = lessons
            .iter()
            .flat_map(|(l, _)| l.items.iter().chain(&l.practice))
            .filter(|i| i.kind != Kind::Prose)
            .map(question_key)
            .collect();
        let before = exam_items.len();
        exam_items.retain(|(_, it)| !taught.contains(&question_key(it)));
        if exam_items.len() < before {
            check.warnings.push(format!(
                "{} exam question(s) also appear in the lessons or practice material and were left out of the exam",
                before - exam_items.len()
            ));
        }
    } else {
        exam_items = pool;
    }

    // Practice must never contain an evaluation or exam question.
    let protected: BTreeSet<String> = lessons
        .iter()
        .flat_map(|(l, _)| l.eval_items())
        .map(question_key)
        .chain(exam_items.iter().map(|(_, it)| question_key(it)))
        .collect();
    for (l, at) in lessons.iter_mut() {
        let lc = &mut check.lessons[*at];
        let before = l.practice.len();
        l.practice.retain(|it| it.kind == Kind::Prose || !protected.contains(&question_key(it)));
        lc.practice_excluded = before - l.practice.len();
        lc.practice = l.practice.iter().filter(|i| i.kind != Kind::Prose).count();
        if lc.practice_excluded > 0 {
            check.warnings.push(format!(
                "{}: {} practice question(s) match evaluation or exam questions and were excluded",
                l.name, lc.practice_excluded
            ));
        }
    }

    let mut sections: Vec<ExamSection> = Vec::new();
    for s in [Section::Dense, Section::Transformer, Section::Coop, Section::WholeBrain] {
        let items: Vec<Item> = exam_items.iter().filter(|(x, _)| *x == s).map(|(_, it)| it.clone()).collect();
        if !items.is_empty() {
            sections.push(ExamSection { section: s, weight: s.weight(&def.section_weights).max(0.), items });
        }
    }
    let total: f32 = sections.iter().map(|s| s.weight).sum();
    if sections.is_empty() {
        check.problems.push(if def.exam.is_some() {
            "The exam has no questions that are independent of the lessons".into()
        } else {
            "Not enough questions for a final exam: add examples to the lessons or a teacher-written exam file".into()
        });
    } else if total <= 0. {
        check.problems.push("The exam's section weights add up to 0".into());
    }
    for s in &mut sections {
        s.weight = if total > 0. { s.weight / total } else { 0. };
        if s.items.len() < FEW_QUESTIONS {
            check.warnings.push(format!(
                "The exam's {} section has only {} question(s): {}",
                s.section.name(),
                s.items.len(),
                if def.exam.is_some() {
                    "add questions to the exam file"
                } else {
                    "add examples to the lessons or write an exam file (questions are never reused)"
                }
            ));
        }
    }
    check.sections = sections.iter().map(|s| (s.section, s.items.len(), s.weight)).collect();
    if let Some(m) = def.exam_minimum {
        if !(0.0..=1.0).contains(&m) {
            check.problems.push("The minimum exam score must be 0-100%".into());
        }
    }
    let s = &def.settings;
    if !(1..=50).contains(&s.epochs)
        || !(1..=128).contains(&s.batch_size)
        || !(s.learning_rate > 0. && s.learning_rate <= 0.01)
        || s.retries > 3
    {
        check.problems.push(
            "Choose 1-50 epochs, a batch size of 1-128, a learning rate up to 0.01 and 0-3 retries".into(),
        );
    }
    PreparedCourse {
        id: def.id.clone(),
        name: def.name.clone(),
        fingerprint: check.fingerprint.clone(),
        lessons: lessons.into_iter().map(|(l, _)| l).collect(),
        exam: sections,
        required_grade: def.required_grade(),
        exam_minimum: def.exam_minimum,
        pretest: def.pretest,
        settings: def.settings.clone(),
        check,
    }
}
