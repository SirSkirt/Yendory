//! Reply limits fitted to each answer (eval::reply_budget): a right answer longer than 24 tokens
//! can be answered right, the grading of short answers is unchanged, and with the switch off
//! every reply stops at 24 tokens as before.
use yendory_engine::learn::eval::{
    evaluate, reply, reply_budget, reply_room, set_fitted_replies, MAX_FITTED_REPLY, MAX_REPLY_TOKENS,
};
use yendory_engine::learn::lessons::{chat_prompt, Item, Kind};
use yendory_engine::{Ablation, Bicameral};

#[test]
fn long_answers_can_be_answered_right() {
    // Budgets: the answer spelled out letter by letter plus the line break, more for keyed
    // answers, never under 24 or over the cap.
    assert_eq!(reply_budget("Blue.", false), MAX_REPLY_TOKENS);
    let long = "I pick the water, because it will quench my thirst.";
    assert_eq!(reply_budget(long, false), long.len() + 1);
    assert_eq!(reply_budget(long, true), (long.len() + 1) * 3 / 2);
    assert_eq!(reply_budget(&"a".repeat(5000), false), MAX_FITTED_REPLY);

    let model = Bicameral::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/compat/fixtures/nova2-step15000.nnova2"
    ))
    .unwrap();
    // Teach a copy two long answers, then ask for them.
    let mut model = model;
    let pairs = [
        (
            "Tell me about the farm.",
            "The farm has a red barn, three brown cows, a big tractor and a small pond.",
        ),
        (
            "What is in the box?",
            "There is a blue hat, a yellow kite, two old books and a tiny silver key in it.",
        ),
    ];
    for _ in 0..60 {
        for (q, a) in pairs {
            model
                .train_text(&format!("{}{a}\n", chat_prompt(q)), 3e-3, yendory_engine::Focus::Joint)
                .unwrap();
        }
    }
    let base = Item {
        kind: Kind::Dialogue,
        source: "t".into(),
        held_out: true,
        segments: Vec::new(),
        prompt: None,
        answer: None,
        contradicted: None,
        users: Vec::new(),
        replies: Vec::new(),
        key: None,
        reading: None,
        context: None,
        spelling: false,
    };
    let mut items = Vec::new();
    for (q, a) in pairs {
        let tokens = model.tokenizer.encode(a).unwrap().len();
        let written = reply(&model, &chat_prompt(q), 200, Ablation::default()).unwrap();
        println!("{q}: {tokens} tokens; writes {written:?}");
        assert!(tokens > MAX_REPLY_TOKENS);
        if written == a {
            items.push(Item { prompt: Some(q.into()), answer: Some(a.into()), ..base.clone() });
        }
    }
    println!("{} long answers", items.len());
    assert!(!items.is_empty(), "the model learned a long answer");
    let refs: Vec<&Item> = items.iter().collect();

    let room = reply_room(&model, &refs).unwrap();
    println!("{room:?}");
    assert_eq!(room.too_long, 0);
    assert!(room.longest_answer > MAX_REPLY_TOKENS && room.longest_limit >= room.longest_answer);
    let fitted = evaluate(&model, &refs, Ablation::default()).unwrap();
    assert_eq!(fitted.overall().correct, items.len(), "every long answer is right with room to write it");

    set_fitted_replies(false);
    let capped = evaluate(&model, &refs, Ablation::default()).unwrap();
    let room_capped = reply_room(&model, &refs).unwrap();
    set_fitted_replies(true);
    assert_eq!(capped.overall().correct, 0, "cut at 24 tokens, none can be right");
    assert_eq!(room_capped.too_long, items.len());
}
