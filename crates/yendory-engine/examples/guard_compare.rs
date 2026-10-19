//! Controlled comparison of the teaching safeguards. Every configuration starts from the same
//! checkpoint and teaches the same courses (through the Course Builder pipeline: pre-test,
//! lessons, course grade) with the same settings and seeds; only the safeguards differ.
//!
//! After every lesson, every lesson's evaluation questions (held out, never trained) are asked
//! again, so gains, losses, forgetting of earlier lessons and survival of improvements can be
//! counted. Course grades are the first grade after teaching (before any remediation).
//!
//! GC_SEED=<n> cargo run --release --example guard_compare -- <strict|net|relative|both> <start.nnova2>
//!     <out.json> <threads> <relative share, e.g. 0.03> <course: lesson files separated by ,>...
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use yendory_engine::learn::course::CourseFile;
use yendory_engine::learn::curriculum::{prepare, CourseDef, PreparedCourse};
use yendory_engine::learn::eval::evaluate;
use yendory_engine::learn::lessons::Item;
use yendory_engine::learn::program::{run_course, CourseHost, CourseRun, ProgramEvent, RunOptions, Snapshot};
use yendory_engine::learn::{batch_loss_and_gradients, encode_item, Rng};
use yendory_engine::{Ablation, Bicameral};

struct Host<'a> {
    stop: &'a AtomicBool,
    latest: &'a RefCell<Option<Bicameral>>,
    events: &'a RefCell<Vec<ProgramEvent>>,
}
impl CourseHost for Host<'_> {
    fn event(&mut self, e: &ProgramEvent) {
        if let ProgramEvent::Graded { .. } = e {
            // The first course grade is what this comparison measures; no remediation.
            self.stop.store(true, Ordering::Release);
        }
        self.events.borrow_mut().push(e.clone());
    }
    fn save(&mut self, model: &Bicameral, _: &[Item], _: &CourseRun) -> Result<(), String> {
        *self.latest.borrow_mut() = Some(model.clone());
        Ok(())
    }
    fn save_best(&mut self, _: &Snapshot) -> Result<(), String> {
        Ok(())
    }
}

/// Which evaluation questions of each lesson the model answers right.
fn answers(model: &Bicameral, lessons: &[(String, Vec<Item>)]) -> Vec<Vec<bool>> {
    lessons
        .iter()
        .map(|(_, items)| {
            let refs: Vec<&Item> = items.iter().collect();
            evaluate(model, &refs, Ablation::default()).unwrap().outcomes.iter().map(|o| o.correct).collect()
        })
        .collect()
}

/// Teacher-forced loss on each lesson's evaluation questions.
fn losses(model: &Bicameral, lessons: &[(String, Vec<Item>)]) -> Vec<f32> {
    lessons
        .iter()
        .map(|(_, items)| {
            let mut rng = Rng::new(1);
            let rows: Vec<_> = items
                .iter()
                .filter_map(|i| {
                    encode_item(&model.tokenizer, &i.segments, model.config.max_tokens, &mut rng).unwrap()
                })
                .collect();
            batch_loss_and_gradients(model, &rows, Ablation::default()).unwrap().0
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args[1].as_str();
    let mut model = Bicameral::load(&args[2]).unwrap();
    let out = &args[3];
    yendory_engine::learn::set_cpu_threads(args[4].parse().unwrap());
    let share: f32 = args[5].parse().unwrap();
    let courses: Vec<Vec<String>> =
        args[6..].iter().map(|c| c.split(',').map(String::from).collect()).collect();

    let preps: Vec<PreparedCourse> = courses
        .iter()
        .enumerate()
        .map(|(ci, files)| {
            let mut def = CourseDef::new(&format!("c{ci}"), &format!("Course {}", ci + 1));
            def.add_files(
                files
                    .iter()
                    .map(|p| CourseFile {
                        name: std::path::Path::new(p).file_name().unwrap().to_string_lossy().into(),
                        text: std::fs::read_to_string(p).unwrap(),
                    })
                    .collect(),
            );
            def.settings.protect_answers_net = matches!(mode, "net" | "both");
            if matches!(mode, "relative" | "both") {
                def.settings.retention_relative = Some(share);
                def.settings.retention_units = 48;
            }
            let prep = prepare(&def);
            assert!(prep.check.problems.is_empty(), "{:?}", prep.check.problems);
            prep
        })
        .collect();
    // Every lesson's evaluation questions, in teaching order.
    let lessons: Vec<(String, Vec<Item>)> = preps
        .iter()
        .flat_map(|p| {
            p.lessons.iter().map(|l| (l.name.clone(), l.eval_items().into_iter().cloned().collect()))
        })
        .collect();

    let t = std::time::Instant::now();
    let start_answers = answers(&model, &lessons);
    let mut timeline = vec![("start".to_string(), start_answers.clone())];
    let mut taught_at: Vec<Option<usize>> = vec![None; lessons.len()];
    let mut lesson_reports = Vec::new();
    let mut course_reports = Vec::new();
    let mut archive: Vec<Item> = Vec::new();
    let mut offset = 0;
    for (ci, prep) in preps.iter().enumerate() {
        let stop = AtomicBool::new(false);
        let latest = RefCell::new(None);
        let events = RefCell::new(Vec::new());
        // GC_SEED chooses another seed for a repeat of the comparison (default 7).
        let seed: u64 = std::env::var("GC_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(7);
        let mut run = CourseRun::new(prep, false, seed + ci as u64);
        let opts = RunOptions { word_limit: Some(7500), ..RunOptions::default() };
        // Teach lesson by lesson; after each, ask every lesson's questions again.
        let mut host = Host { stop: &stop, latest: &latest, events: &events };
        let r = run_course(&model, &archive, prep, &mut run, None, &opts, &mut host, &stop).unwrap();
        model = r.model;
        archive = r.archive;
        for e in events.borrow().iter() {
            match e {
                ProgramEvent::LessonTaught { index, name, outcome, record, .. } => {
                    let li = offset + index;
                    lesson_reports.push(serde_json::json!({
                        "course": ci + 1, "lesson": name, "index": li, "outcome": outcome,
                        "heldout_loss": record.as_ref().and_then(|r| r.baseline.as_ref().zip(r.final_score.as_ref()))
                            .map(|(b, f)| (b.heldout_loss, f.heldout_loss)),
                        "own_questions": record.as_ref().map(|r| (r.own_before, r.own_after)),
                    }));
                }
                ProgramEvent::Graded { after_attempt: 0, grade } => course_reports.push(serde_json::json!({
                    "course": ci + 1, "grade": grade.grade, "required": grade.required, "passed": grade.passed,
                    "lessons": grade.lessons.iter().map(|l| (l.name.clone(), l.course_end, l.pass_mark, l.passed)).collect::<Vec<_>>(),
                    "exam": grade.exam.score,
                })),
                _ => {}
            }
        }
        let now = timeline.len();
        for t in &mut taught_at[offset..offset + prep.lessons.len()] {
            *t = Some(now);
        }
        timeline.push((format!("after course {}", ci + 1), answers(&model, &lessons)));
        offset += prep.lessons.len();
        println!("[{:.0?}] {mode}: course {} done", t.elapsed(), ci + 1);
    }
    let end = &timeline.last().unwrap().1;
    let start_loss = losses(&Bicameral::load(&args[2]).unwrap(), &lessons);
    let end_loss = losses(&model, &lessons);
    let count = |v: &[bool]| v.iter().filter(|&&x| x).count();
    let mut per_lesson = Vec::new();
    for (li, (name, _)) in lessons.iter().enumerate() {
        let s = &start_answers[li];
        let e = &end[li];
        let after_course = &timeline[taught_at[li].unwrap()].1[li];
        let gained = s.iter().zip(e).filter(|(a, b)| !**a && **b).count();
        let lost = s.iter().zip(e).filter(|(a, b)| **a && !**b).count();
        per_lesson.push(serde_json::json!({
            "lesson": name, "questions": s.len(),
            "correct_start": count(s), "correct_after_its_course": count(after_course), "correct_end": count(e),
            "gained": gained, "lost": lost, "net": gained as i64 - lost as i64,
            "loss_start": start_loss[li], "loss_end": end_loss[li],
        }));
    }
    let total_q: usize = start_answers.iter().map(|v| v.len()).sum();
    let summary = serde_json::json!({
        "mode": mode, "share": share, "seconds": t.elapsed().as_secs(),
        "correct_start": start_answers.iter().map(|v| count(v)).sum::<usize>(),
        "correct_end": end.iter().map(|v| count(v)).sum::<usize>(), "questions": total_q,
        "lesson_passes": lesson_reports.iter().filter(|l| l["outcome"] == "passed").count(),
        "per_lesson": per_lesson, "lesson_runs": lesson_reports, "courses": course_reports,
        "timeline": timeline.iter().map(|(n, a)| (n.clone(), a.iter().map(|v| count(v)).collect::<Vec<_>>())).collect::<Vec<_>>(),
    });
    std::fs::write(out, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("{}", serde_json::to_string(&summary["timeline"]).unwrap());
}
