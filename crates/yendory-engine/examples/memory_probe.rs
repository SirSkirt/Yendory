//! Can a network learn to use Study Mode memories? Each training question has two memory lines
//! in front of it: the fact it asks about and an unrelated one. Which animal a name is changes
//! from example to example, so the memory is the only way to know the answer. Afterwards the
//! model is asked new random facts with and without the memory lines.
//!
//! cargo run --release --example memory_probe -- <steps> <threads>
use yendory_engine::learn::eval::reply;
use yendory_engine::learn::lessons::Segment;
use yendory_engine::learn::{batch_loss_and_gradients, encode_item, Rng};
use yendory_engine::memory::MEMORY;
use yendory_engine::{Ablation, Bicameral, Config, Focus};

const NAMES: [&str; 12] =
    ["Max", "Bella", "Ada", "Ben", "Cleo", "Dan", "Eve", "Finn", "Gus", "Hana", "Ivo", "Jo"];
const ANIMALS: [&str; 8] = ["dog", "cat", "cow", "hen", "fox", "owl", "pig", "bee"];

fn example(rng: &mut Rng) -> (String, String, String) {
    let n = NAMES[rng.below(NAMES.len())];
    let a = ANIMALS[rng.below(ANIMALS.len())];
    let mut m = NAMES[rng.below(NAMES.len())];
    while m == n {
        m = NAMES[rng.below(NAMES.len())];
    }
    let b = ANIMALS[rng.below(ANIMALS.len())];
    let facts = if rng.below(2) == 0 {
        format!("{MEMORY}{n} is a {a}.\n{MEMORY}{m} is a {b}.\n")
    } else {
        format!("{MEMORY}{m} is a {b}.\n{MEMORY}{n} is a {a}.\n")
    };
    (facts, format!("What is {n}?"), format!("A {a}."))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1500);
    yendory_engine::learn::set_cpu_threads(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0));
    let mut model = Bicameral::with_config(Config::default(), 21);
    let mut rng = Rng::new(7);
    let test: Vec<_> = {
        let mut r = Rng::new(99);
        (0..64).map(|_| example(&mut r)).collect()
    };
    let score = |model: &Bicameral, with: bool| {
        let right = test
            .iter()
            .filter(|(f, q, a)| {
                let prompt = format!("{}U: {q}\nN: ", if with { f.as_str() } else { "" });
                reply(model, &prompt, 12, Ablation::default()).map(|r| r == *a).unwrap_or(false)
            })
            .count();
        right as f32 / test.len() as f32
    };
    let t = std::time::Instant::now();
    for step in 0..=steps {
        if step % 250 == 0 {
            println!(
                "[{:>5.0?}] step {step}: with memory {:.0}%, without {:.0}% (chance {:.0}%)",
                t.elapsed(),
                100. * score(&model, true),
                100. * score(&model, false),
                100. / ANIMALS.len() as f32
            );
        }
        if step == steps {
            break;
        }
        let rows: Vec<_> = (0..16)
            .filter_map(|_| {
                let (facts, q, a) = example(&mut rng);
                let segs = vec![
                    Segment { text: facts, weight: 0. },
                    Segment { text: format!("U: {q}\nN: "), weight: 0.15 },
                    Segment { text: format!("{a}\n"), weight: 1.0 },
                ];
                encode_item(&model.tokenizer, &segs, model.config.max_tokens, &mut rng).unwrap()
            })
            .collect();
        let (_, grads) = batch_loss_and_gradients(&model, &rows, Ablation::default()).unwrap();
        model.adam_step(&grads, if step < steps * 3 / 4 { 2e-3 } else { 5e-4 }, Focus::Joint).unwrap();
    }
}
