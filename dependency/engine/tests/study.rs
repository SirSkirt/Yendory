//! Practice, reflection and Review Training on a small model: each training pass goes through
//! the ordinary session (held-out questions judge it), keeps the origin of every target, and
//! reports before and after.
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::designer::AnswerSource;
use yendory_engine::learn::parse_lesson;
use yendory_engine::learn::practice::{
    answer_all, train as train_practice, Mark, PracticeItem, TargetOrigin,
};
use yendory_engine::learn::reflect::{connections, reflect, train as train_reflection, Statement};
use yendory_engine::learn::review::{assess, compare, generate, lesson_items, ReviewOptions};
use yendory_engine::learn::session::SessionConfig;
use yendory_engine::memory::MemoryStore;
use yendory_engine::{Bicameral, Config};

fn cfg() -> SessionConfig {
    SessionConfig {
        epochs: 3,
        batch_size: 8,
        learning_rate: 2e-3,
        foundation: false,
        rehearsal_share: 0.,
        grow_vocabulary: false,
        heldout_loss_tolerance: 1.,
        protect_correct_answers: false,
        retry: yendory_engine::learn::session::RetryPolicy {
            retries: 0,
            requeue: false,
            min_learning_rate: 1e-6,
        },
        seed: 5,
        ..SessionConfig::default()
    }
}

#[test]
fn practice_trains_on_marked_answers_with_their_origin() {
    let model = Bicameral::with_config(Config::default(), 3);
    let lesson = parse_lesson(
        "[Dialogue Lesson]\n[examples]\nIs Max a dog?  Yes.\nIs Bella a cat?  Yes.\n[/examples]\n[/Dialogue Lesson]\n\
         [Evaluation]\nIs Rex a dog?  Yes.\nIs Tom a cat?  Yes.\n[/Evaluation]\n",
        "pets.txt",
    )
    .unwrap();
    let item = |q: &str, e: &str| PracticeItem {
        question: q.into(),
        expected: Some(e.into()),
        answer: String::new(),
        memories: vec![],
        mark: None,
        correction: String::new(),
        note: String::new(),
        target: None,
        after: None,
        grade: None,
        auto: None,
    };
    let mut items =
        vec![item("Is Max a dog?", "Yes."), item("Is Bella a cat?", "Yes."), item("Is Kim a hen?", "")];
    answer_all(&model, &mut items, None, false, 0).unwrap();
    assert!(items.iter().all(|i| i.memories.is_empty()));
    items[0].mark = Some(Mark::Right);
    items[1].mark = Some(Mark::Wrong);
    items[1].correction = "Yes, Bella is a cat.".into();
    items[2].mark = Some(Mark::Wrong); // no correction and no lesson answer: not trained
    let stop = AtomicBool::new(false);
    let r = train_practice(&model, &mut items, &lesson, &[], &cfg(), &mut |_| {}, &stop).unwrap().unwrap();
    assert_eq!(items[0].target.as_ref().unwrap().1, TargetOrigin::NovaConfirmed);
    assert_eq!(items[1].target, Some(("Yes, Bella is a cat.".into(), TargetOrigin::TeacherCorrection)));
    assert!(items[2].target.is_none());
    assert_eq!(r.baseline.exact_total, 2, "judged by the lesson's held-out questions");
    answer_all(&r.model, &mut items, None, true, 0).unwrap();
    assert!(items.iter().all(|i| i.after.is_some()));
}

#[test]
fn reflection_and_review_train_through_the_session() {
    let model = Bicameral::with_config(Config::default(), 8);
    let st = |t: &str| Statement { text: t.into(), memory: None, source: "farm".into() };
    let statements = [
        st("Max is a dog."),
        st("Max likes the park."),
        st("Dogs like bones."),
        st("Bella likes milk."),
        st("Bella is a cat."),
    ];
    let conns = connections(&model, &statements, 6).unwrap();
    assert!(conns.len() >= 3);
    let mut obs = reflect(&model, conns, false).unwrap();
    let stop = AtomicBool::new(false);
    assert!(
        train_reflection(&model, &mut obs, false, &[], &cfg(), &mut |_| {}, &stop).is_err(),
        "nothing approved"
    );
    for (k, o) in obs.iter_mut().enumerate() {
        o.mark = Some(Mark::Wrong);
        o.correction = format!("Connection {k}.");
    }
    let (r, rec) = train_reflection(&model, &mut obs, false, &[], &cfg(), &mut |_| {}, &stop).unwrap();
    assert!(rec.held_out >= 1 && rec.trained >= 2);
    assert!(obs.iter().filter(|o| o.held_out).all(|o| o.target.is_some()));
    assert!(rec.material_before > 0.);
    let _ = r;

    // Review Training: generated, checked, assessed, trained, assessed again, compared.
    let mut store = MemoryStore::default();
    for t in ["Max is a dog.", "Bella is a cat.", "Rex can swim."] {
        store.add_teacher(t, 0, 0);
    }
    let ids: Vec<u64> = store.entries.iter().map(|m| m.id).collect();
    let mut draft = generate(&store, &ids, ReviewOptions { difficulty: 2 }, "Review");
    assert!(
        lesson_items(&draft).unwrap().iter().all(|i| i.prompt.is_none()),
        "unchecked answers are not taught"
    );
    for b in &mut draft.blocks {
        for i in &mut b.items {
            assert_eq!(i.answer_source, AnswerSource::Generated);
            i.verified = true;
        }
    }
    let items = lesson_items(&draft).unwrap();
    let before = assess(&model, &items, Some(&mut store), 1).unwrap();
    assert!(before.total >= 3 && before.with_memory.is_some());
    let r = yendory_engine::learn::session::teach_with(&model, &items, None, &[], &cfg(), &mut |_| {}, &stop)
        .unwrap();
    let after = assess(&r.model, &items, None, 2).unwrap();
    let c = compare(&before, &after);
    assert_eq!(
        c.improved.len() + c.regressed.len() + c.still_right.len() + c.still_wrong.len(),
        before.total
    );
    assert!(c.concepts.iter().any(|(t, ..)| t == "max"));
}
