//! Review Point 1: Nova Bicameral against Nova Neuro-Core at the same size (Nova-2 size,
//! ~1.18M parameters each), same data, same reading budget, same seeds.
//!
//! Training: generated practice text (all six families, difficulty 2, a fixed seed), read with
//! the ordinary corpus pre-training (held-out checks, best checkpoint kept).
//! Evaluation, never trained on:
//! * the corpus's own held-out documents (loss);
//! * in-distribution questions: new examples of every family from another seed, any example
//!   whose question appears in the training text removed;
//! * "who is where" stories with **unseen names** (none of the generator's names), and with
//!   **unseen names and unseen places** (nothing in the story was ever read).
//!
//! Greedy replies are graded exactly, and "where" answers also by content (the right place,
//! no other place). Neuro is also evaluated with each module cut (ablations).
//!
//! NC_SEED=<n> cargo run --release --example neuro_compare -- <bicameral|neuro> <out.json> <threads> <tokens>
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::corpus::{pretrain, Corpus, PretrainConfig};
use yendory_engine::learn::generate::{example, generate, Family, GenerateConfig};
use yendory_engine::learn::Rng;
use yendory_engine::model::neuro::presets;
use yendory_engine::{Ablation, Bicameral, Config, TokenId};

const DATA_SEED: u64 = 1234;
const EVAL_SEED: u64 = 98_765;
const SEEN_PLACES: [&str; 10] =
    ["kitchen", "garden", "office", "bedroom", "hallway", "bathroom", "park", "school", "library", "shop"];
const NEW_NAMES: [&str; 12] =
    ["Bram", "Cleo", "Dov", "Elin", "Fynn", "Gus", "Hana", "Ines", "Jory", "Kaz", "Lior", "Mabel"];
const NEW_PLACES: [&str; 8] = ["attic", "cellar", "porch", "studio", "lobby", "pantry", "garage", "barn"];
const SEEN_NAMES: [&str; 24] = [
    "Ann", "Ben", "Cara", "Dan", "Eva", "Finn", "Gia", "Hugo", "Ivy", "Jack", "Kim", "Leo", "Mia", "Noah",
    "Owen", "Pia", "Ray", "Sara", "Tom", "Uma", "Vic", "Wes", "Yara", "Zoe",
];
const MOVES: [&str; 4] = ["went to", "walked to", "moved to", "ran to"];

struct Question {
    family: String,
    prompt: String,
    answer: String,
    /// The place a "where" question must name.
    place: Option<String>,
}

/// A "who is where" story in the generator's own grammar, from the given names and places.
fn story(rng: &mut Rng, names: &[&str], places: &[&str]) -> Question {
    let mut people: Vec<&str> = names.to_vec();
    for i in 0..3 {
        let j = i + rng.below(people.len() - i);
        people.swap(i, j);
    }
    people.truncate(3);
    let mut at: Vec<Option<&str>> = vec![None; 3];
    let mut lines = Vec::new();
    for _ in 0..(5 + rng.below(3)) {
        let p = rng.below(3);
        let to = places[rng.below(places.len())];
        at[p] = Some(to);
        lines.push(format!("{} {} the {to}.", people[p], MOVES[rng.below(MOVES.len())]));
    }
    let known: Vec<usize> = (0..3).filter(|&i| at[i].is_some()).collect();
    let p = known[rng.below(known.len())];
    let here = at[p].unwrap();
    Question {
        family: String::new(),
        prompt: format!("U: {} Where is {}?\nN: ", lines.join(" "), people[p]),
        answer: format!("{} is in the {here}.", people[p]),
        place: Some(here.to_string()),
    }
}

fn split_qa(text: &str) -> Option<(String, String)> {
    let i = text.find("\nN: ")?;
    Some((format!("{}\nN: ", &text[..i]), text[i + 4..].trim_end().to_string()))
}

#[derive(Default, serde::Serialize, Clone)]
struct Score {
    questions: usize,
    exact: usize,
    place_right: usize,
    place_questions: usize,
    answer_loss: f32,
}

/// The score of a set, and each question's own score (one reply and one loss per question).
/// Questions are scored in parallel (each thread its own questions, one Engine thread each);
/// every question's score is the same as when scored alone.
fn evaluate(model: &Bicameral, qs: &[Question], abl: Ablation, threads: usize) -> (Score, Vec<Score>) {
    let chunk = qs.len().div_ceil(threads.max(1)).max(1);
    let per: Vec<Score> = std::thread::scope(|sc| {
        let jobs: Vec<_> =
            qs.chunks(chunk).map(|part| sc.spawn(move || score_each(model, part, abl))).collect();
        jobs.into_iter().flat_map(|j| j.join().unwrap()).collect()
    });
    let mut total = Score::default();
    for s in &per {
        total.questions += 1;
        total.exact += s.exact;
        total.place_right += s.place_right;
        total.place_questions += s.place_questions;
        total.answer_loss += s.answer_loss;
    }
    total.answer_loss /= total.questions.max(1) as f32;
    (total, per)
}

fn score_each(model: &Bicameral, qs: &[Question], abl: Ablation) -> Vec<Score> {
    let all_places: Vec<&str> = SEEN_PLACES.iter().chain(NEW_PLACES.iter()).copied().collect();
    let mut per = Vec::new();
    for q in qs {
        let mut s = Score { questions: 1, ..Score::default() };
        let ids = model.tokenizer.encode(&q.prompt).unwrap();
        let out = model.generate_cached_ids(&ids, 32, abl).unwrap();
        let reply = model.tokenizer.decode(&out).unwrap();
        let reply = reply.trim_end_matches('\n');
        s.exact = usize::from(reply == q.answer);
        if let Some(place) = &q.place {
            s.place_questions = 1;
            let words: Vec<String> =
                reply.split(|c: char| !c.is_ascii_alphanumeric()).map(|w| w.to_ascii_lowercase()).collect();
            let named: Vec<&&str> = all_places.iter().filter(|p| words.iter().any(|w| w == **p)).collect();
            s.place_right = usize::from(named.len() == 1 && **named[0] == *place.as_str());
        }
        // Teacher-forced loss of the answer tokens (prompt not scored).
        let a = model.tokenizer.encode(&format!("{}\n", q.answer)).unwrap();
        let mut all: Vec<TokenId> = ids.iter().chain(a.iter()).copied().collect();
        let mut mask: Vec<f32> = vec![0.; ids.len()];
        mask.extend(std::iter::repeat(1.).take(a.len()));
        let max = model.config.max_tokens + 1;
        if all.len() > max {
            let cut = all.len() - max;
            all.drain(..cut);
            mask.drain(..cut);
        }
        let (x, y, m) = (&all[..all.len() - 1], &all[1..], &mask[1..]);
        s.answer_loss = model.loss(x, y, m, abl).unwrap();
        per.push(s);
    }
    per
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arch = args[1].as_str();
    let out = &args[2];
    let threads: usize = args[3].parse().unwrap();
    yendory_engine::learn::set_cpu_threads(threads);
    let tokens: u64 = args[4].parse().unwrap();
    let seed: u64 = std::env::var("NC_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(7);
    let start = match arch {
        "bicameral" => Bicameral::with_config(Config::default(), seed),
        "neuro" => {
            let p = presets().into_iter().next().unwrap();
            Bicameral::with_design(p.config, Some(p.design), seed)
        }
        _ => panic!("arch: bicameral or neuro"),
    };
    let text = generate(&GenerateConfig {
        families: Family::ALL.to_vec(),
        difficulty: 2,
        bytes: 4_000_000,
        seed: DATA_SEED,
    })
    .unwrap();
    let corpus = Corpus::prepare(&text);
    let t = std::time::Instant::now();
    let cfg = PretrainConfig { tokens, evaluations: 16, seed, ..PretrainConfig::default() };
    let mut evals = Vec::new();
    let r = pretrain(&start, &corpus, &cfg, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(false)).unwrap();
    for e in &r.evaluations {
        evals.push(
            serde_json::json!({"tokens": e.tokens, "held_out_loss": e.held_out_loss, "accepted": e.accepted}),
        );
    }
    let seconds = t.elapsed().as_secs_f64();
    eprintln!("read {} tokens in {seconds:.0}s", r.tokens);
    let model = r.model;
    // Scoring: questions in parallel, one Engine thread each.
    yendory_engine::learn::set_cpu_threads(1);

    // Evaluation sets (never trained on).
    let mut rng = Rng::new(EVAL_SEED);
    let mut in_dist = Vec::new();
    for family in Family::ALL {
        let mut n = 0;
        while n < 40 {
            let ex = example(family, 2, &mut rng);
            let Some((prompt, answer)) = split_qa(&ex) else { continue };
            let question = prompt.trim_end_matches("\nN: ").trim_start_matches("U: ");
            if text.contains(question) {
                continue; // in the training text: not a fair test
            }
            let place =
                (family == Family::World && question.contains("Where is") && !answer.contains("cannot"))
                    .then(|| answer.trim_end_matches('.').rsplit(' ').next().unwrap_or("").to_string())
                    .filter(|p| SEEN_PLACES.contains(&p.as_str()));
            in_dist.push(Question { family: family.name().into(), prompt, answer, place });
            n += 1;
        }
    }
    let unseen_names: Vec<Question> = (0..80).map(|_| story(&mut rng, &NEW_NAMES, &SEEN_PLACES)).collect();
    let unseen_all: Vec<Question> = (0..80).map(|_| story(&mut rng, &NEW_NAMES, &NEW_PLACES)).collect();
    let seen_where: Vec<Question> = (0..80)
        .map(|_| story(&mut rng, &SEEN_NAMES, &SEEN_PLACES))
        .filter(|q| !text.contains(q.prompt.trim_start_matches("U: ").trim_end_matches("\nN: ")))
        .collect();
    let sets: Vec<(&str, &[Question])> = vec![
        ("in_distribution", &in_dist),
        ("where_seen_names_new_stories", &seen_where),
        ("where_unseen_names", &unseen_names),
        ("where_unseen_names_and_places", &unseen_all),
    ];
    let score_all = |abl: Ablation| -> serde_json::Value {
        let mut o = serde_json::Map::new();
        for (name, qs) in &sets {
            let t = std::time::Instant::now();
            let (total, per) = evaluate(&model, qs, abl, threads);
            eprintln!("  {name}: {} questions in {:.0}s", qs.len(), t.elapsed().as_secs_f64());
            o.insert(name.to_string(), serde_json::to_value(total).unwrap());
            if *name == "in_distribution" {
                // Per family, from the same replies.
                let mut fam = serde_json::Map::new();
                for family in Family::ALL {
                    let mut f = Score::default();
                    for (q, r) in qs.iter().zip(&per) {
                        if q.family == family.name() {
                            f.questions += 1;
                            f.exact += r.exact;
                            f.place_right += r.place_right;
                            f.place_questions += r.place_questions;
                            f.answer_loss += r.answer_loss;
                        }
                    }
                    f.answer_loss /= f.questions.max(1) as f32;
                    fam.insert(family.name().into(), serde_json::to_value(f).unwrap());
                }
                o.insert("by_family".into(), serde_json::Value::Object(fam));
            }
        }
        serde_json::Value::Object(o)
    };
    eprintln!("full model");
    let full = score_all(Ablation::default());
    let mut ablations = serde_json::Map::new();
    if arch == "neuro" {
        for (name, abl) in [
            ("no_working_memory", Ablation { no_memory: true, ..Ablation::default() }),
            ("no_associative_memory", Ablation { no_associative: true, ..Ablation::default() }),
            ("no_controller", Ablation { no_controller: true, ..Ablation::default() }),
            ("one_round", Ablation { one_round: true, ..Ablation::default() }),
            ("no_modules_into_transformer", Ablation { no_afferent: true, ..Ablation::default() }),
        ] {
            eprintln!("ablation {name}");
            ablations.insert(name.into(), score_all(abl));
        }
    } else {
        ablations.insert(
            "no_dense_into_transformer".into(),
            score_all(Ablation { no_afferent: true, ..Ablation::default() }),
        );
    }
    let summary = serde_json::json!({
        "architecture": arch, "seed": seed, "parameters": model.parameter_count(),
        "tokens_read": r.tokens, "reading_seconds": seconds, "tokens_per_second": r.tokens as f64 / seconds,
        "baseline_loss": r.baseline_loss, "best_held_out_loss": r.best_loss, "evaluations": evals,
        "activation_floats_per_token": Bicameral::activation_floats_for(&model.config, model.neuro.as_ref()),
        "scores": full, "ablations": ablations,
    });
    std::fs::write(out, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    model.save(format!("{out}.model")).unwrap();
    println!("{arch} seed {seed}: held-out loss {:.4} in {:.0}s", r.best_loss, seconds);
}
