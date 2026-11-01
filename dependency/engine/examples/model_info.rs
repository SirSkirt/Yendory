//! What a checkpoint is: its shape, vocabulary, step and a few greedy replies.
//! cargo run --release --example model_info -- <model.nnova2> [prompt...]
use yendory_engine::learn::eval::reply;
use yendory_engine::learn::lessons::chat_prompt;
use yendory_engine::{Ablation, Bicameral};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t = std::time::Instant::now();
    let m = Bicameral::load(&a[1]).unwrap();
    println!("loaded in {:.1}s", t.elapsed().as_secs_f64());
    println!("{:#?}", m.config);
    println!("neuro design: {}", m.neuro.is_some());
    println!("parameters {}, step {}", m.parameter_count(), m.step);
    println!("learned words {} of {} slots", m.tokenizer.words().len(), m.word_capacity());
    let w = m.tokenizer.words();
    println!("first words: {:?}", &w[..w.len().min(40)]);
    println!("last words: {:?}", &w[w.len().saturating_sub(40)..]);
    let prompts: Vec<String> = if a.len() > 2 {
        a[2..].to_vec()
    } else {
        [
            "hello",
            "What is your name?",
            "How are you today?",
            "Where is the cat?",
            "Do you like apples or bread?",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    };
    for p in prompts {
        let r = reply(&m, &chat_prompt(&p), 48, Ablation::default()).unwrap();
        println!("U: {p}\nN: {r}");
    }
}
