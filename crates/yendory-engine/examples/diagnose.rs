//! Diagnose learning: teach one lesson file with Studio's default settings and print every
//! candidate (train loss, held-out loss, exact answers, why it was or was not accepted).
//! cargo run --release --example diagnose -- <lesson.txt> [fresh|<model.nnova2>] [gpu] [epochs]
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::session::{teach, Event, SessionConfig};
use yendory_engine::learn::{parse_lesson, set_gpu_training};
use yendory_engine::{Bicameral, Config};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let items = parse_lesson(&text, "lesson").unwrap();
    let model = match args.get(2).map(String::as_str) {
        None | Some("fresh") => Bicameral::with_config(Config::default(), 7),
        // As Studio creates a model with a vocabulary limit: no word slots yet.
        Some("studio") => Bicameral::with_config(Config { vocab: 128, ..Config::default() }, 7),
        Some("studio8m") => {
            let c = yendory_engine::size_presets().into_iter().find(|p| p.id == "8m").unwrap().config;
            Bicameral::with_config(Config { vocab: 128, max_tokens: 1024, ..c }, 7)
        }
        Some(path) => Bicameral::load(path).unwrap(),
    };
    if args.get(3).map(String::as_str) == Some("gpu") {
        set_gpu_training(true);
        println!("GPU: {:?}", yendory_engine::learn::gpu_name());
    }
    let cfg = SessionConfig {
        epochs: args.get(4).and_then(|e| e.parse().ok()).unwrap_or(4),
        word_limit: args.get(5).and_then(|e| e.parse().ok()),
        ..SessionConfig::default()
    };
    let t = std::time::Instant::now();
    let r = teach(
        &model,
        &items,
        &[],
        &cfg,
        &mut |e| match e {
            Event::Started { train_units, heldout_units, new_words, baseline, .. } => println!(
                "start: {train_units} train, {heldout_units} held out, {new_words} new words; baseline loss {:.4} exact {}/{}",
                baseline.heldout_loss, baseline.exact_correct, baseline.exact_total
            ),
            Event::Candidate(c) => println!(
                "  epoch {} {:<12} train {:.4} held-out {:.4} exact {}/{} eligible {} {:?}",
                c.epoch, c.focus, c.train_loss, c.score.heldout_loss, c.score.exact_correct, c.score.exact_total, c.eligible, c.reasons
            ),
            Event::Epoch(e) => println!("epoch {} selected {:?} -> accepted loss {:.4} exact {}", e.epoch, e.selected, e.accepted_score.heldout_loss, e.accepted_score.exact_correct),
            _ => {}
        },
        &AtomicBool::new(false),
    )
    .unwrap();
    println!(
        "learned {} final loss {:.4} exact {}/{} in {:.0?}",
        r.learned,
        r.final_score.heldout_loss,
        r.final_score.exact_correct,
        r.final_score.exact_total,
        t.elapsed()
    );
}
