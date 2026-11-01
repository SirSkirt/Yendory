//! Answers the GPU confirms (eval::confirm): every one is exactly what greedy writing on the CPU
//! gives, and an evaluation that skips writing them gives the same report as one that writes
//! every answer. Run with `--features gpu` (a software GPU is enough).
#![cfg(feature = "gpu")]
use yendory_engine::learn::builtin::builtin_courses;
use yendory_engine::learn::eval::{confirm, evaluate, evaluate_confirmed, reply, MAX_REPLY_TOKENS};
use yendory_engine::learn::lessons::{chat_prompt, parse_lesson, Item, Kind};
use yendory_engine::learn::{set_device, Device, Trainer};
use yendory_engine::model::neuro::presets;
use yendory_engine::{Ablation, Bicameral, Focus};

#[test]
fn confirmed_answers_are_what_the_cpu_writes() {
    std::env::set_var("YENDORY_GPU_SOFTWARE", "1");
    if yendory_engine::learn::gpu_name().is_err() {
        println!("no GPU available; skipped");
        return;
    }
    let fixture = Bicameral::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/compat/fixtures/nova2-step15000.nnova2"
    ))
    .unwrap();
    let p = presets().into_iter().next().unwrap();
    let mut neuro = Bicameral::with_design(p.config, Some(p.design), 3);
    // Short replies that end with a line break, like a trained model's.
    for _ in 0..40 {
        for t in [
            "U: Where is the key?\nN: In the box.\n",
            "U: hi\nN: Hello!\n",
            "U: What colour is the sky?\nN: Blue.\n",
        ] {
            neuro.train_text(t, 3e-3, Focus::Joint).unwrap();
        }
    }
    check("step 15000", &fixture);
    check("Nova Neuro", &neuro);
}

fn check(name: &str, model: &Bicameral) {
    let model = model.clone();
    // Held-out questions of the built-in courses (the models get few of them right) ...
    let mut items: Vec<Item> = Vec::new();
    for c in builtin_courses() {
        for f in c.lessons.iter().take(2) {
            let parsed = parse_lesson(&f.text, &f.name).unwrap();
            items.extend(parsed.into_iter().filter(|i| i.held_out && i.answer.is_some()).take(6));
        }
    }
    // ... and the same questions with the model's own reply as the expected answer (it
    // gets all of these right), plus two-turn conversations built the same way.
    let own: Vec<Item> = items
        .iter()
        .map(|it| {
            let prompt = format!(
                "{}{}",
                it.context.as_deref().unwrap_or(""),
                chat_prompt(it.prompt.as_ref().unwrap())
            );
            let mine = reply(&model, &prompt, MAX_REPLY_TOKENS, Ablation::default()).unwrap();
            Item { answer: Some(mine), key: None, ..it.clone() }
        })
        .collect();
    let mut talks = Vec::new();
    for pair in own.chunks(2).filter(|p| p.len() == 2) {
        let users: Vec<String> = pair.iter().map(|i| i.prompt.clone().unwrap()).collect();
        let mut history = String::new();
        let mut replies = Vec::new();
        for u in &users {
            history.push_str(&chat_prompt(u));
            let r = reply(&model, &history, MAX_REPLY_TOKENS, Ablation::default()).unwrap();
            history.push_str(&r);
            history.push('\n');
            replies.push(r);
        }
        talks.push(Item {
            kind: Kind::Conversation,
            users,
            replies,
            prompt: None,
            answer: None,
            key: None,
            ..pair[0].clone()
        });
    }
    items.extend(own);
    items.extend(talks);
    let refs: Vec<&Item> = items.iter().collect();

    let cpu = evaluate(&model, &refs, Ablation::default()).unwrap();
    set_device(Device::Gpu(yendory_engine::gpu::Api::Auto));
    let trainer = Trainer::new(&model);
    let confirmed = confirm(&trainer, &model, &refs).unwrap();

    set_device(Device::Cpu);
    assert!(trainer.on_gpu());
    let fast = evaluate_confirmed(&model, &refs, Ablation::default(), &confirmed).unwrap();

    // Every confirmed answer is one the CPU writes exactly.
    let mut answer_of = std::collections::HashMap::new();
    let mut k = 0;
    for (i, it) in refs.iter().enumerate() {
        let n = if it.kind == Kind::Conversation { it.replies.len() } else { 1 };
        for j in 0..n {
            answer_of.insert((i, j), k);
            k += 1;
        }
    }
    for key in &confirmed.0 {
        let o = &cpu.outcomes[answer_of[key]];
        assert!(
            o.correct && o.reply == o.expected,
            "confirmed {key:?} but the CPU wrote {:?} for {:?}",
            o.reply,
            o.expected
        );
    }
    assert_eq!(fast, cpu, "the same report");
    let right = cpu.overall().correct;
    println!(
        "{name}: {} answers, {right} right, {} confirmed on the GPU",
        cpu.overall().total,
        confirmed.len()
    );
    // The model's own replies are confirmed, apart from the closest calls.
    assert!(confirmed.len() * 10 >= right * 8, "{} of {right}", confirmed.len());
}
