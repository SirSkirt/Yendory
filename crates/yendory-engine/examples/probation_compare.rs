//! Controlled comparison: do rejected updates, given room to recover (probation), end in better
//! generalization than restarting from the accepted model?
//!
//! One Classroom session the way the Studio teaches a course together with saved earlier lessons
//! (`train_earlier`): the group's lessons and the earlier lessons are trained in one mix and
//! judged together. Every lesson's held-out questions are split in two by position: the JUDGE
//! half is what the session sees and chooses checkpoints by; the TEST half is never trained on
//! and never used to choose anything, so it measures generalization. After every epoch the
//! accepted model and the working model (where training continues from) are both asked the
//! test half.
//!
//! cargo run --release --example probation_compare -- <arm name> <probation epochs> <start model>
//!     <out.json> <epochs> <seed> <threads> <earlier lesson files dir> [group course id]
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use yendory_engine::learn::builtin::builtin_courses;
use yendory_engine::learn::eval::evaluate;
use yendory_engine::learn::lessons::{parse_lesson, Item, Kind};
use yendory_engine::learn::session::{teach_rewarded, EpochHook, Event, SessionConfig};
use yendory_engine::learn::{batch_loss, encode_item, Rng};
use yendory_engine::{Ablation, Bicameral};

#[derive(Clone, Debug, serde::Serialize)]
struct Test {
    /// Right / asked on the new lessons' and the earlier lessons' test halves.
    new_right: usize,
    new_total: usize,
    earlier_right: usize,
    earlier_total: usize,
    /// Teacher-forced loss on the test halves.
    new_loss: f32,
    earlier_loss: f32,
    /// Which test questions were right (new, then earlier), for gained / lost counts.
    right: Vec<bool>,
}

fn test(model: &Bicameral, new: &[Item], earlier: &[Item]) -> Test {
    let ask = |items: &[Item]| -> (Vec<bool>, f32) {
        let refs: Vec<&Item> = items.iter().collect();
        let r = evaluate(model, &refs, Ablation::default()).unwrap();
        let mut rng = Rng::new(1);
        let rows: Vec<_> = items
            .iter()
            .filter_map(|it| {
                encode_item(&model.tokenizer, &it.segments, model.config.max_tokens, &mut rng).unwrap()
            })
            .collect();
        (
            r.outcomes.iter().map(|o| o.correct).collect(),
            batch_loss(model, &rows, Ablation::default()).unwrap(),
        )
    };
    let (n, nl) = ask(new);
    let (e, el) = ask(earlier);
    Test {
        new_right: n.iter().filter(|x| **x).count(),
        new_total: n.len(),
        earlier_right: e.iter().filter(|x| **x).count(),
        earlier_total: e.len(),
        new_loss: nl,
        earlier_loss: el,
        right: n.into_iter().chain(e).collect(),
    }
}

/// Splits each lesson's held-out questions by position: even ones are judged, odd ones tested.
fn split(items: Vec<Item>) -> (Vec<Item>, Vec<Item>) {
    let mut seen: std::collections::HashMap<String, usize> = Default::default();
    let (mut keep, mut test) = (Vec::new(), Vec::new());
    for it in items {
        if it.held_out && it.kind != Kind::Prose {
            let k = seen.entry(it.source.clone()).or_default();
            *k += 1;
            if *k % 2 == 0 {
                test.push(it);
                continue;
            }
        }
        keep.push(it);
    }
    (keep, test)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (arm, probation, start, out, epochs, seed, threads, dir) = (
        a[1].clone(),
        a[2].parse::<u32>().unwrap(),
        &a[3],
        &a[4],
        a[5].parse::<u32>().unwrap(),
        a[6].parse::<u64>().unwrap(),
        a[7].parse::<usize>().unwrap(),
        &a[8],
    );
    let course = a.get(9).cloned().unwrap_or_else(|| "deciding".into());
    yendory_engine::learn::set_cpu_threads(threads);
    let model = Bicameral::load(start).unwrap();
    let mut earlier = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    files.sort();
    for f in files.iter().filter(|f| f.extension().is_some_and(|x| x == "txt")) {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        earlier.extend(parse_lesson(&std::fs::read_to_string(f).unwrap(), &name).unwrap());
    }
    let c = builtin_courses().into_iter().find(|c| c.id == course).expect("course");
    let mut group = Vec::new();
    for f in &c.lessons {
        group.extend(parse_lesson(&f.text, &f.name).unwrap());
    }
    let (group, group_test) = split(group);
    let (earlier, earlier_test) = split(earlier);
    let all: Vec<Item> = group.iter().chain(&earlier).cloned().collect();
    println!(
        "{arm}: {} items ({} new, {} earlier); test halves {} new, {} earlier",
        all.len(),
        group.len(),
        earlier.len(),
        group_test.len(),
        earlier_test.len()
    );

    let log: Arc<Mutex<Vec<serde_json::Value>>> = Default::default();
    let start_test = test(&model, &group_test, &earlier_test);
    println!(
        "start: new {}/{} earlier {}/{} (loss {:.4} / {:.4})",
        start_test.new_right,
        start_test.new_total,
        start_test.earlier_right,
        start_test.earlier_total,
        start_test.new_loss,
        start_test.earlier_loss
    );
    let hook = {
        let (log, gt, et) = (log.clone(), group_test.clone(), earlier_test.clone());
        EpochHook(Arc::new(move |epoch: u32, accepted: &Bicameral, working: &Bicameral| {
            let acc = test(accepted, &gt, &et);
            let work = if working.step == accepted.step { acc.clone() } else { test(working, &gt, &et) };
            println!(
                "epoch {epoch}: accepted new {}/{} earlier {}/{} | working new {}/{} earlier {}/{} (loss {:.4} / {:.4})",
                acc.new_right,
                acc.new_total,
                acc.earlier_right,
                acc.earlier_total,
                work.new_right,
                work.new_total,
                work.earlier_right,
                work.earlier_total,
                work.new_loss,
                work.earlier_loss
            );
            log.lock().unwrap().push(serde_json::json!({
                "epoch": epoch, "accepted_step": accepted.step, "working_step": working.step,
                "accepted": acc, "working": work,
            }));
        }))
    };
    let cfg = SessionConfig {
        epochs,
        batch_size: 8,
        learning_rate: 5e-5,
        foundation: false,
        rehearsal_share: 0.10,
        protect_correct_answers: true,
        protect_answers_net: true,
        pass_mark: 0.5,
        patience: 10,
        seed,
        train_earlier: true,
        recover_earlier: true,
        backoff: true,
        eval_limit: None,
        probation_epochs: probation,
        epoch_hook: Some(hook),
        ..SessionConfig::default()
    };
    let others: Vec<Item> = earlier.clone();
    let epochs_log: Arc<Mutex<Vec<serde_json::Value>>> = Default::default();
    let t0 = std::time::Instant::now();
    let r = {
        let el = epochs_log.clone();
        let mut observer = move |e: &Event| match e {
            Event::Started { baseline, .. } => {
                println!(
                    "judge baseline: {}/{} loss {:.4}",
                    baseline.exact_correct, baseline.exact_total, baseline.heldout_loss
                )
            }
            Event::Epoch(rec) => {
                for c in &rec.candidates {
                    println!(
                        "  candidate {}: {}/{} loss {:.4} earlier loss {:?} panel {}/{} {:?}",
                        c.focus,
                        c.score.exact_correct,
                        c.score.exact_total,
                        c.score.heldout_loss,
                        c.score.retention_loss,
                        c.score.panel_correct,
                        c.score.panel_total,
                        c.reasons
                    );
                }
                println!(
                    "  epoch {}: selected {:?} continued {:?} restored {} probation {:?}",
                    rec.epoch, rec.selected, rec.continued, rec.restored, rec.probation
                );
                el.lock().unwrap().push(serde_json::to_value(rec).unwrap());
            }
            Event::LearningRateLowered { epoch, scale, .. } => {
                println!("  epoch {epoch}: learning rate x{scale}")
            }
            _ => {}
        };
        teach_rewarded(&model, &all, None, &others, &cfg, None, &mut observer, &AtomicBool::new(false))
            .unwrap()
    };
    let end = test(&r.model, &group_test, &earlier_test);
    let gained = start_test.right.iter().zip(&end.right).filter(|(a, b)| !**a && **b).count();
    let lost = start_test.right.iter().zip(&end.right).filter(|(a, b)| **a && !**b).count();
    println!(
        "{arm} end ({:.0} min): accepted new {}/{} earlier {}/{} loss {:.4} / {:.4}; test answers gained {gained}, lost {lost}",
        t0.elapsed().as_secs_f64() / 60.,
        end.new_right,
        end.new_total,
        end.earlier_right,
        end.earlier_total,
        end.new_loss,
        end.earlier_loss
    );
    let json = serde_json::json!({
        "arm": arm, "probation_epochs": probation, "seed": seed, "epochs": epochs,
        "start": start_test, "end": end, "gained": gained, "lost": lost,
        "learned": r.learned, "judge_baseline": r.baseline, "judge_final": r.final_score,
        "trajectory": *log.lock().unwrap(), "session": *epochs_log.lock().unwrap(),
        "minutes": t0.elapsed().as_secs_f64() / 60.,
    });
    std::fs::write(out, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
}
