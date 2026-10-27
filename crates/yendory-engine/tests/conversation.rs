//! Conversation practice on the step-15000 checkpoint with Classroom lessons: the Classroom
//! check is measured on held-out questions, earlier lessons are mixed in, and only checkpoints
//! that lower the held-out loss of Nova's lines while keeping the Classroom answers are kept.
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::builtin::builtin_courses;
use yendory_engine::learn::conversation::{
    converse, parse_dialogue, ConversationConfig, ConversationEvent, ConversationSet,
};
use yendory_engine::learn::parse_lesson;
use yendory_engine::model::Bicameral;

#[test]
fn practice_checks_classroom_answers_and_keeps_better_checkpoints() {
    let model = Bicameral::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/compat/fixtures/nova2-step15000.nnova2"
    ))
    .unwrap();
    let english = &builtin_courses()[0];
    let mut earlier = Vec::new();
    for f in &english.lessons[..2] {
        earlier.extend(parse_lesson(&f.text, &f.name).unwrap());
    }
    let topics =
        ["the park", "the sea", "a book", "the rain", "my garden", "the market", "a song", "the hills"];
    let mut dialogues = Vec::new();
    for (i, t) in topics.iter().cycle().take(160).enumerate() {
        let doc = format!(
            "Ann: Hi Ben! Did you see {t} today?\nBen: Hi Ann. Yes, I saw {t} this morning.\nAnn: Did you like it?\nBen: I did, it was lovely number {i}."
        );
        dialogues.push(parse_dialogue(&doc).unwrap());
    }
    let sets = [ConversationSet { name: "practice".into(), dialogues, nova: None, hold_out: true }];
    let cfg = ConversationConfig {
        tokens: 60_000,
        batch_size: 8,
        learning_rate: 1e-3,
        evaluations: 3,
        guard_questions: 30,
        own_name: "Nova".into(),
        ..ConversationConfig::default()
    };
    let mut started = None;
    let r = converse(
        &model,
        &sets,
        &earlier,
        &cfg,
        &mut |e| {
            if let ConversationEvent::Started { classroom_total, baseline_classroom, earlier_items, .. } = e {
                started = Some((*classroom_total, *baseline_classroom, *earlier_items));
            }
        },
        &mut |_| Ok(()),
        &AtomicBool::new(false),
    )
    .unwrap();
    let (total, baseline, items) = started.unwrap();
    eprintln!(
        "Classroom {baseline}/{total} before; checks: {:?}",
        r.checks.iter().map(|c| (c.held_out_loss, c.classroom_correct, c.accepted)).collect::<Vec<_>>()
    );
    assert!(total > 0 && items > 0);
    assert_eq!((r.baseline_classroom, r.classroom_total), (baseline, total));
    let floor = baseline - (cfg.guard_drop * baseline as f32).floor() as usize;
    for c in &r.checks {
        if c.accepted {
            assert!(c.classroom_correct >= floor, "an accepted checkpoint keeps the Classroom answers");
        }
        // Nova's speaker is called Nova in the samples' prompts.
        assert!(c.samples.iter().all(|s| s.prompt.contains("Hi Nova!")), "{:?}", c.samples);
    }
    assert!(r.best_loss <= r.baseline_loss);
}
