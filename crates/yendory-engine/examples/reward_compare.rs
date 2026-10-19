//! Controlled comparison of reward-guided training (Stage 2) against ordinary supervised
//! training. Both arms start from the same checkpoint, teach the same courses through the Course
//! Builder pipeline with the same settings and seeds, and both keep the reward ledger (recording
//! does not change training; tests/reward.rs checks the weights are identical), so reward
//! metrics can be set against real performance. Only `rewarded` schedules units by emphasis.
//!
//! Measured: every lesson's evaluation questions (held out, never trained) at the start and after
//! every course (unseen accuracy, gains, losses, retention of earlier courses), the courses'
//! final exams (protected from the ledger), held-out loss, candidates rejected as unstable, and
//! the ledger's totals and flags.
//!
//! RC_VARIANT=gentle (optional) RC_SEED=<n> cargo run --release --example reward_compare -- <standard|rewarded> <start.nnova2>
//!     <out.json> <threads> <course: lesson files separated by ,>...
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use yendory_engine::learn::course::{CourseEvent, CourseFile};
use yendory_engine::learn::curriculum::{prepare, CourseDef, PreparedCourse};
use yendory_engine::learn::eval::evaluate;
use yendory_engine::learn::lessons::Item;
use yendory_engine::learn::program::{run_course, CourseHost, CourseRun, ProgramEvent, RunOptions, Snapshot};
use yendory_engine::learn::reward::{RewardConfig, RewardLedger, Rewards};
use yendory_engine::learn::session::{Event, Rejection};
use yendory_engine::learn::{batch_loss_and_gradients, encode_item, Rng};
use yendory_engine::{Ablation, Bicameral};

struct Host<'a> {
    stop: &'a AtomicBool,
    events: &'a RefCell<Vec<ProgramEvent>>,
    rewards: Option<Rewards>,
}
impl CourseHost for Host<'_> {
    fn event(&mut self, e: &ProgramEvent) {
        if let ProgramEvent::Graded { .. } = e {
            // The first course grade is what this comparison measures; no remediation.
            self.stop.store(true, Ordering::Release);
        }
        if let ProgramEvent::Lesson {
            event: CourseEvent::Session { event: Event::Batch { .. }, .. }, ..
        } = e
        {
            return;
        }
        self.events.borrow_mut().push(e.clone());
    }
    fn save(&mut self, _: &Bicameral, _: &[Item], _: &CourseRun) -> Result<(), String> {
        Ok(())
    }
    fn save_best(&mut self, _: &Snapshot) -> Result<(), String> {
        Ok(())
    }
    fn take_rewards(&mut self) -> Option<Rewards> {
        self.rewards.take()
    }
    fn return_rewards(&mut self, r: Rewards) {
        self.rewards = Some(r);
    }
}

fn answers(model: &Bicameral, lessons: &[(String, Vec<Item>)]) -> Vec<Vec<bool>> {
    lessons
        .iter()
        .map(|(_, items)| {
            let refs: Vec<&Item> = items.iter().collect();
            evaluate(model, &refs, Ablation::default()).unwrap().outcomes.iter().map(|o| o.correct).collect()
        })
        .collect()
}

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
    assert!(matches!(mode, "standard" | "rewarded"), "mode: standard or rewarded");
    let mut model = Bicameral::load(&args[2]).unwrap();
    let out = &args[3];
    yendory_engine::learn::set_cpu_threads(args[4].parse().unwrap());
    let courses: Vec<Vec<String>> =
        args[5..].iter().map(|c| c.split(',').map(String::from).collect()).collect();
    let seed: u64 = std::env::var("RC_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(7);

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
            let prep = prepare(&def);
            assert!(prep.check.problems.is_empty(), "{:?}", prep.check.problems);
            prep
        })
        .collect();
    let lessons: Vec<(String, Vec<Item>)> = preps
        .iter()
        .flat_map(|p| {
            p.lessons.iter().map(|l| (l.name.clone(), l.eval_items().into_iter().cloned().collect()))
        })
        .collect();

    let mut config = RewardConfig { weighted_training: mode == "rewarded", ..RewardConfig::default() };
    // RC_VARIANT=gentle: the gentler schedule, fixed before any of its results were seen (known
    // answers keep regular practice, a narrower range, no concept or variation boosts).
    let variant = std::env::var("RC_VARIANT").unwrap_or_default();
    if variant == "gentle" {
        config.min_emphasis = 0.6;
        config.max_emphasis = 2.0;
        let w = &mut config.weights;
        w.gap = 0.75;
        w.failures = 0.25;
        w.forgotten = 0.5;
        w.progress = 0.25;
        w.concept = 0.;
        w.variation = 0.;
        w.difficulty = 0.;
        w.correct_factor = 0.8;
        w.retained_factor = 0.85;
        w.stuck_cap = 1.25;
    }
    let mut rewards = Some(Rewards::new(config, RewardLedger::default(), "compare"));
    let t = std::time::Instant::now();
    let start_answers = answers(&model, &lessons);
    let mut timeline = vec![("start".to_string(), start_answers.clone())];
    let mut taught_at: Vec<Option<usize>> = vec![None; lessons.len()];
    let (mut lesson_reports, mut course_reports) = (Vec::new(), Vec::new());
    let (mut candidates, mut unstable, mut accepted) = (0usize, 0usize, 0usize);
    let mut emphasis_epochs = 0usize;
    let mut archive: Vec<Item> = Vec::new();
    let mut offset = 0;
    for (ci, prep) in preps.iter().enumerate() {
        let stop = AtomicBool::new(false);
        let events = RefCell::new(Vec::new());
        let mut run = CourseRun::new(prep, false, seed + ci as u64);
        let opts = RunOptions { word_limit: Some(7500) };
        let mut host = Host { stop: &stop, events: &events, rewards: rewards.take() };
        let r = run_course(&model, &archive, prep, &mut run, None, &opts, &mut host, &stop).unwrap();
        rewards = host.rewards.take();
        model = r.model;
        archive = r.archive;
        for e in events.borrow().iter() {
            match e {
                ProgramEvent::Lesson { event: CourseEvent::Session { event, .. }, .. } => match event {
                    Event::Candidate(c) => {
                        candidates += 1;
                        unstable += usize::from(c.rejections.contains(&Rejection::HeldoutLossRose));
                    }
                    Event::Epoch(e) => accepted += usize::from(e.selected.is_some()),
                    Event::Emphasis { .. } => emphasis_epochs += 1,
                    _ => {}
                },
                ProgramEvent::LessonTaught { index, name, outcome, record, .. } => {
                    lesson_reports.push(serde_json::json!({
                        "course": ci + 1, "lesson": name, "index": offset + index, "outcome": outcome,
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
        for x in &mut taught_at[offset..offset + prep.lessons.len()] {
            *x = Some(now);
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
        let (s, e) = (&start_answers[li], &end[li]);
        let after_course = &timeline[taught_at[li].unwrap()].1[li];
        let gained = s.iter().zip(e).filter(|(a, b)| !**a && **b).count();
        let lost = s.iter().zip(e).filter(|(a, b)| **a && !**b).count();
        let lost_after_course = after_course.iter().zip(e).filter(|(a, b)| **a && !**b).count();
        per_lesson.push(serde_json::json!({
            "lesson": name, "questions": s.len(),
            "correct_start": count(s), "correct_after_its_course": count(after_course), "correct_end": count(e),
            "gained": gained, "lost": lost, "lost_after_its_course": lost_after_course,
            "loss_start": start_loss[li], "loss_end": end_loss[li],
        }));
    }
    let ledger = rewards.as_ref().map(|r| r.ledger.summary());
    let summary = serde_json::json!({
        "mode": mode, "variant": variant, "seed": seed, "seconds": t.elapsed().as_secs(),
        "correct_start": start_answers.iter().map(|v| count(v)).sum::<usize>(),
        "correct_end": end.iter().map(|v| count(v)).sum::<usize>(),
        "questions": start_answers.iter().map(|v| v.len()).sum::<usize>(),
        "lesson_passes": lesson_reports.iter().filter(|l| l["outcome"] == "passed").count(),
        "candidates": candidates, "unstable_candidates": unstable, "accepted_epochs": accepted,
        "emphasis_epochs": emphasis_epochs,
        "per_lesson": per_lesson, "lesson_runs": lesson_reports, "courses": course_reports,
        "timeline": timeline.iter().map(|(n, a)| (n.clone(), a.iter().map(|v| count(v)).collect::<Vec<_>>())).collect::<Vec<_>>(),
        "ledger": ledger,
    });
    std::fs::write(out, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("{}", serde_json::to_string(&summary["timeline"]).unwrap());
}
