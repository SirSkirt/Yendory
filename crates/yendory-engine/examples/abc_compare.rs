//! The Classroom's "Compare settings" experiment (learn::compare) from the command line, on a
//! Studio model and its saved lessons: A the owner's Classroom settings, B + probation (3
//! epochs), C + probation and the larger earlier-lesson check. To keep it affordable on a CPU,
//! the earlier lessons are a sample of `earlier lessons` of the saved lessons, spread evenly
//! over their names.
//!
//! cargo run --release --example abc_compare -- <model> <archive.json> <course id>
//!     <earlier lessons> <epochs> <out.json> [variants, e.g. AB] [learning rate]
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::builtin::builtin_courses;
use yendory_engine::learn::compare::{compare, CompareEvent, Variant};
use yendory_engine::learn::lessons::Item;
use yendory_engine::learn::session::{Event, SessionConfig};
use yendory_engine::Bicameral;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = Bicameral::load(&a[1]).unwrap();
    let archive: Vec<Item> = serde_json::from_slice(&std::fs::read(&a[2]).unwrap()).unwrap();
    let course = builtin_courses().into_iter().find(|c| c.id == a[3]).expect("course");
    let keep: usize = a[4].parse().unwrap();
    let epochs: u32 = a[5].parse().unwrap();
    let out = &a[6];
    let which = a.get(7).cloned().unwrap_or_else(|| "ABC".into());
    let learning_rate: f32 = a.get(8).map(|x| x.parse().unwrap()).unwrap_or(5e-5);

    let mut sources: Vec<&str> = archive.iter().map(|i| i.source.as_str()).collect();
    sources.sort();
    sources.dedup();
    let chosen: std::collections::HashSet<&str> =
        (0..keep.min(sources.len())).map(|k| sources[k * sources.len() / keep.min(sources.len())]).collect();
    let earlier: Vec<Item> = archive.iter().filter(|i| chosen.contains(i.source.as_str())).cloned().collect();
    println!("earlier lessons: {} of {} ({} items)", chosen.len(), sources.len(), earlier.len());

    // The owner's last Classroom settings (history.jsonl course_started options).
    let base = SessionConfig {
        epochs,
        batch_size: 8,
        learning_rate,
        foundation: false,
        keep_dense_reliance: 0.5,
        patience: 10,
        rehearsal_share: 0.10,
        heldout_loss_tolerance: 0.08,
        retention_loss_tolerance: 0.02,
        retention_panel: 8,
        retention_units: 16,
        protect_correct_answers: true,
        protect_answers_net: true,
        grow_vocabulary: true,
        word_limit: Some(20000),
        min_word_count: 12,
        seed: 1793140207,
        pass_mark: 0.5,
        min_loss_improvement: 0.01,
        backoff: true,
        override_first_epoch: false,
        recover_earlier: true,
        earlier_answers_net: false,
        train_earlier: true,
        probation_epochs: 0,
        ..SessionConfig::default()
    };
    let all = [
        Variant { name: "A: owner's settings".into(), cfg: base.clone() },
        Variant {
            name: "B: + probation (3 epochs)".into(),
            cfg: SessionConfig { probation_epochs: 3, ..base.clone() },
        },
        Variant {
            name: "C: + probation and the larger earlier-lesson check".into(),
            cfg: SessionConfig {
                probation_epochs: 3,
                retention_per_lesson: Some(4),
                earlier_panel_net: true,
                ..base.clone()
            },
        },
    ];
    let variants: Vec<Variant> =
        all.into_iter().filter(|v| which.contains(v.name.chars().next().unwrap())).collect();

    let t0 = std::time::Instant::now();
    let mut observer = |e: &CompareEvent| {
        let m = t0.elapsed().as_secs() / 60;
        match e {
            CompareEvent::Started { new_test, earlier_test, start, start_context, .. } => println!(
                "[{m} min] test halves: {new_test} new, {earlier_test} earlier; start new {}/{} earlier {}/{} loss {:.3}/{:.3}; context {:?}",
                start.new_right, start.new_total, start.earlier_right, start.earlier_total,
                start.new_loss, start.earlier_loss, start_context.as_ref().map(|c| (c.passed, c.total))
            ),
            CompareEvent::VariantStarted { name, .. } => println!("[{m} min] {name}"),
            CompareEvent::Tested { test: t, .. } => println!(
                "[{m} min]   epoch {}: accepted new {}/{} earlier {}/{} | working new {}/{} earlier {}/{} loss {:.3}/{:.3}",
                t.epoch, t.accepted.new_right, t.accepted.new_total, t.accepted.earlier_right,
                t.accepted.earlier_total, t.working.new_right, t.working.new_total, t.working.earlier_right,
                t.working.earlier_total, t.working.new_loss, t.working.earlier_loss
            ),
            CompareEvent::Session { event: Event::Epoch(rec), .. } => {
                for c in &rec.candidates {
                    println!(
                        "      candidate: judged {}/{} loss {:.3} panel {}/{} {:?}",
                        c.score.exact_correct, c.score.exact_total, c.score.heldout_loss,
                        c.score.panel_correct, c.score.panel_total, c.reasons
                    );
                }
                println!("      selected {:?} continued {:?} probation {:?}", rec.selected, rec.continued, rec.probation);
            }
            CompareEvent::VariantFinished { result: r, .. } => println!(
                "[{m} min] {} finished: new {}/{} earlier {}/{} (gained {}, lost {}), accepted {} of {} epochs, context {:?}",
                r.name, r.end.new_right, r.end.new_total, r.end.earlier_right, r.end.earlier_total, r.gained,
                r.lost, r.accepted_epochs, r.epochs, r.context.as_ref().map(|c| (c.passed, c.total))
            ),
            _ => {}
        }
    };
    let files = course.lessons.clone();
    let results =
        compare(&model, &files, &earlier, &variants, 6, &mut observer, &AtomicBool::new(false)).unwrap();
    std::fs::write(out, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}
