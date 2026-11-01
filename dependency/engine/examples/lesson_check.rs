//! Held-out answers of lesson files and of a built-in course, per lesson, with reply limits
//! fitted to each answer and with the old 24-token limit.
//! cargo run --release --example lesson_check -- <model> <lesson files dir or archive.json> [course id]
use yendory_engine::learn::builtin::builtin_courses;
use yendory_engine::learn::eval::{evaluate, set_fitted_replies};
use yendory_engine::learn::lessons::{parse_lesson, Item, Kind};
use yendory_engine::{Ablation, Bicameral};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = Bicameral::load(&a[1]).unwrap();
    let mut items: Vec<Item> = Vec::new();
    if a[2].ends_with(".json") {
        // A Studio model's saved lessons (archive.json).
        items = serde_json::from_slice(&std::fs::read(&a[2]).unwrap()).unwrap();
    } else {
        let mut files: Vec<_> = std::fs::read_dir(&a[2]).unwrap().map(|e| e.unwrap().path()).collect();
        files.sort();
        for f in files.iter().filter(|f| f.extension().is_some_and(|x| x == "txt")) {
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            items.extend(parse_lesson(&std::fs::read_to_string(f).unwrap(), &name).unwrap());
        }
    }
    if let Some(c) = a.get(3).and_then(|id| builtin_courses().into_iter().find(|c| c.id == *id)) {
        for f in &c.lessons {
            items.extend(parse_lesson(&f.text, &f.name).unwrap());
        }
    }
    let held: Vec<&Item> = items.iter().filter(|i| i.held_out && i.kind != Kind::Prose).collect();
    let t = std::time::Instant::now();
    let fitted = evaluate(&m, &held, Ablation::default()).unwrap();
    set_fitted_replies(false);
    let capped = evaluate(&m, &held, Ablation::default()).unwrap();
    set_fitted_replies(true);
    println!("{} questions in {:.0}s", held.len(), t.elapsed().as_secs_f64());
    let mut by: std::collections::BTreeMap<String, [usize; 4]> = Default::default();
    for (f, c) in fitted.outcomes.iter().zip(&capped.outcomes) {
        let e = by.entry(f.source.clone()).or_default();
        e[0] += usize::from(f.correct);
        e[1] += usize::from(c.correct);
        e[2] += 1;
        e[3] += usize::from(c.cut_off);
    }
    println!("lesson: right with fitted limits / right cut at 24 tokens / asked (cut off at 24)");
    for (s, [f, c, n, cut]) in &by {
        println!("  {s}: {f} / {c} / {n} ({cut})");
    }
    println!(
        "all: {} / {} of {} (cut off at 24: {}; with fitted limits: {})",
        fitted.overall().correct,
        capped.overall().correct,
        fitted.overall().total,
        capped.cut_off,
        fitted.cut_off
    );
    for o in fitted.outcomes.iter().filter(|o| !o.correct).take(12) {
        println!(
            "  Q: {}\n  expected: {}\n  wrote:    {}",
            o.prompt.trim_end().replace('\n', " | "),
            o.expected,
            o.reply
        );
    }
}
