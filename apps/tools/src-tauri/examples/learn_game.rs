//! A developer experiment, not part of Tools: can a small model learn a game from the perfect
//! player's lessons? Teaches a new Nova-2-size model (1.2M parameters) one game's lessons with
//! the Engine directly, then measures it with Tools' own scoreboard, the Engine's Nexus line
//! writer standing in for the Yendory Network. Optionally saves it as a model in a models folder.
//!
//! cargo run --release --example learn_game -- <game> <sessions> <steps> [models folder]
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use yendory_engine::learn::{batch_loss_and_gradients, encode_item, parse_lesson, Rng};
use yendory_engine::{nexus, Ablation, Bicameral, Config, Focus};
use yendory_tools_lib::{lessons, score};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let game = args.get(1).map(String::as_str).unwrap_or("nim");
    let sessions: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(400);
    let set = lessons::teacher_lessons(game, sessions, 1).unwrap();
    let items = parse_lesson(&set.text, "lessons.txt").unwrap();
    println!("{game}: {} sessions ({} held-out turns left out)", items.len(), set.held_out_skipped);

    let mut model = Bicameral::with_config(Config::default(), 7);
    let mut rng = Rng::new(3);
    let rows: Vec<_> = items
        .iter()
        .filter_map(|it| {
            encode_item(&model.tokenizer, &it.segments, model.config.max_tokens, &mut rng).ok().flatten()
        })
        .collect();
    let line = |m: &Bicameral| {
        let m = m.clone();
        move |ctx: &str| Ok((nexus::next_line(&m, ctx, 64, Ablation::default())?, false))
    };
    let stop = AtomicBool::new(false);
    let show = |m: &Bicameral, label: &str| {
        let (t, moves, held) = score::measure(game, 10, 99, line(m), |_, _| {}, &stop).unwrap();
        println!(
            "{label}: {:?}; legal {}/{}, best {}/{}, held-out best {}/{}",
            t.iter()
                .map(|(k, x)| format!("{k} {}-{}-{} ({} forfeits)", x.wins, x.draws, x.losses, x.forfeits))
                .collect::<Vec<_>>(),
            moves.legal,
            moves.turns,
            moves.best,
            moves.turns,
            held.best,
            held.turns
        );
    };
    show(&model, "untaught");
    let started = Instant::now();
    let batch = 24.min(rows.len());
    for step in 0..steps {
        let chosen: Vec<_> =
            (0..batch).map(|_| rows[(rng.next_u64() % rows.len() as u64) as usize].clone()).collect();
        let (loss, grads) = batch_loss_and_gradients(&model, &chosen, Ablation::default()).unwrap();
        let lr = if step < steps * 3 / 4 { 3e-3 } else { 1e-3 };
        model.adam_step(&grads, lr, Focus::Joint).unwrap();
        if step % 50 == 0 || step + 1 == steps {
            println!("step {step}: loss {loss:.3} ({:.0}s)", started.elapsed().as_secs_f32());
        }
    }
    show(&model, "taught");

    if let Some(folder) = args.get(4) {
        let id = format!(
            "m{:x}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        );
        let dir = std::path::Path::new(folder).join(&id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.nnova2"), model.to_bytes().unwrap()).unwrap();
        std::fs::write(dir.join("archive.json"), b"[]").unwrap();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let meta = serde_json::json!({
            "id": id, "name": format!("{game} player"), "created": now, "updated": now, "step": model.step,
            "parameters": model.parameter_count(),
            "groups": [model.group_parameters(Focus::Left), model.group_parameters(Focus::Right), model.group_parameters(Focus::Interconnect)],
            "words": model.tokenizer.words().len(), "slots": model.word_capacity(), "wordLimit": null,
            "context": model.config.max_tokens, "taught": [], "origin": "new", "read": [], "conversation": [],
            "architecture": model.architecture(), "blocks": model.config.blocks,
        });
        std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta).unwrap()).unwrap();
        println!("saved {}", dir.display());
    }
}
