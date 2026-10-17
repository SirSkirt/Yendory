//! Teach lesson files as Studio's Classroom does (default settings) and print each file's
//! attempts and outcome. cargo run --release --example course_diag -- <model|fresh|studio> <files...>
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::course::{teach_course, CourseEvent, CourseFile};
use yendory_engine::learn::session::{Event, SessionConfig};
use yendory_engine::{Bicameral, Config};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = match args[1].as_str() {
        "fresh" => Bicameral::with_config(Config::default(), 7),
        "studio" => Bicameral::with_config(Config { vocab: 128, ..Config::default() }, 7),
        path => Bicameral::load(path).unwrap(),
    };
    let files: Vec<CourseFile> = args[2..]
        .iter()
        .map(|p| CourseFile {
            name: std::path::Path::new(p).file_name().unwrap().to_string_lossy().into(),
            text: std::fs::read_to_string(p).unwrap(),
        })
        .collect();
    let cfg = SessionConfig { seed: 5, word_limit: Some(7500), ..SessionConfig::default() };
    let t = std::time::Instant::now();
    let r = teach_course(
        &model,
        &files,
        &[],
        &cfg,
        &mut |e| match e {
            CourseEvent::AttemptStarted { index, attempt, learning_rate, epochs, remedy, .. } => {
                println!("[{:.0?}] file {index} attempt {attempt} lr {learning_rate} epochs {epochs} remedy {remedy:?}", t.elapsed())
            }
            CourseEvent::Diagnosed { attempt, diagnosis, .. } => println!("   diagnosed after {attempt}: {diagnosis:?}"),
            CourseEvent::Session { event: Event::Epoch(ep), .. } => println!(
                "   epoch {} selected {:?} loss {:.3} exact {}/{}",
                ep.epoch, ep.selected, ep.accepted_score.heldout_loss, ep.accepted_score.exact_correct, ep.accepted_score.exact_total
            ),
            CourseEvent::FileFinished(f) => println!(
                "== {} pass {} -> {:?}: {:?} -> {:?}",
                f.name,
                f.pass,
                f.outcome,
                f.baseline.as_ref().map(|s| (s.heldout_loss, s.exact_correct, s.exact_total)),
                f.final_score.as_ref().map(|s| (s.heldout_loss, s.exact_correct, s.exact_total))
            ),
            _ => {}
        },
        &mut |_, _| Ok(()),
        &AtomicBool::new(false),
    )
    .unwrap();
    println!("done in {:.0?}; model step {}", t.elapsed(), r.model.step);
}
