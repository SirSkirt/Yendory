//! Cost of a per-network trace of one turn on a real model: time and size of the trace, against
//! untraced generation, and that the traced reply is the untraced one.
//! cargo run --release --example trace_profile -- <model> [turns] [trace.json]
use std::time::Instant;
use yendory_engine::model::inspect::TraceReply;
use yendory_engine::{Ablation, Bicameral};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = Bicameral::load(&a[1]).unwrap();
    let turns: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(6);
    let said = [
        ("Hi! My name is Sam and I have a red bike.", "Hi Sam! I like red bikes."),
        ("I keep the bike in the garage.", "That sounds like a good place."),
        ("Yesterday I rode it to the park with my sister.", "That sounds like fun."),
        ("Actually, my bike is blue, not red.", "Oh, I see."),
        ("We had lunch by the lake.", "That sounds lovely."),
        ("My sister likes to read books.", "Reading is great."),
    ];
    let mut prompt = String::new();
    for (u, n) in said.iter().cycle().take(turns) {
        prompt.push_str(&format!("U: {u}\nN: {n}\n"));
    }
    prompt.push_str("U: What colour is my bike?\nN: ");
    let ids = m.tokenizer.encode(&prompt).unwrap();
    println!("model step {}, prompt {} tokens", m.step, ids.len());
    let t0 = Instant::now();
    let plain = m.generate_cached_ids(&ids, 32, Ablation::default()).unwrap();
    let plain_s = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let mut choose = |p: &[f32], _: &[u32]| yendory_engine::model::argmax(p);
    let mut tr =
        m.trace_turn(&ids, TraceReply::Generate { max_tokens: 32, choose: &mut choose }, |_| true).unwrap();
    let traced_s = t0.elapsed().as_secs_f64();
    let same = tr.steps.iter().map(|s| s.chosen.id).collect::<Vec<_>>() == plain;
    let t0 = Instant::now();
    m.measure_influence(&mut tr).unwrap();
    let infl_s = t0.elapsed().as_secs_f64();
    let json = serde_json::to_string(&tr).unwrap();
    if let Some(out) = a.get(3) {
        std::fs::write(out, &json).unwrap();
    }
    println!("reply: {:?}", tr.reply);
    println!(
        "untraced {plain_s:.1} s, traced {traced_s:.1} s ({:.2}x), influence of {} networks {infl_s:.1} s; same reply: {same}",
        traced_s / plain_s,
        tr.influence.len()
    );
    println!(
        "trace: {} steps, {} lines, {:.2} MB as JSON",
        tr.steps.len(),
        tr.lines.len(),
        json.len() as f64 / 1e6
    );
    let s = &tr.steps[0];
    let message = tr.lines.iter().find(|l| l.role == "message").map(|l| l.index).unwrap();
    for st in s.transformer.stages.iter().take(3) {
        let on_msg: f32 = st.lines.iter().filter(|(l, _)| *l == message).map(|(_, w)| w).sum();
        println!(
            "block {} round {}: {:.3} of attention on the current message, mean distance {:.1} tokens",
            st.block + 1,
            st.round,
            on_msg,
            st.distance
        );
    }
    for (i, l) in s.memory.iter().enumerate() {
        let strongest: Vec<_> = {
            let mut v = l.lines.clone();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            v.into_iter().take(3).collect()
        };
        println!(
            "working memory layer {i}: update {:.3}, overwritten {:.3}, strongest lines {:?}",
            l.update, l.overwrite, strongest
        );
    }
    for r in &s.rounds {
        println!(
            "round {}: gates {:?}, retrieval entropy {:.2} bits, similar lines {:?}",
            r.round,
            r.modules.iter().map(|m| (m.name.clone(), m.gate)).collect::<Vec<_>>(),
            r.retrieval_entropy,
            &r.similar_lines[..r.similar_lines.len().min(3)]
        );
    }
    for i in &s.influence {
        println!("influence {:<16} change {:.3} flips {}", i.network, i.change, i.flips);
    }
}
