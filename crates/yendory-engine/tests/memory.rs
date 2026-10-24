//! Study Mode memory with a real (small) model: what Nova keeps is ranked by her own surprise,
//! held-out questions are never stored, lessons train with memory copies of their questions,
//! grading stays without memory, and the score with memory is measured separately.
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::course::{teach_course_studying, CourseEvent, CourseFile};
use yendory_engine::learn::parse_lesson;
use yendory_engine::learn::session::SessionConfig;
use yendory_engine::memory::{candidates, surprise, MemoryStore, Studying, KNOWN_LOSS, MEMORY};
use yendory_engine::{Bicameral, Config, Focus};

fn lesson() -> String {
    let reading = "Max is a dog. Dogs are animals. Animals need water. Bella is a cat. Cats like milk. \
                   The farm has a red barn. Hens lay eggs on the farm. Cows give milk every morning.";
    let mut qa = String::new();
    for (q, a) in [
        ("Is Max a dog?", "Yes."),
        ("Is Bella a cat?", "Yes."),
        ("Do animals need water?", "Yes."),
        ("Do cats like milk?", "Yes."),
        ("Is the barn red?", "Yes."),
        ("Do hens lay eggs?", "Yes."),
        ("Do cows give milk?", "Yes."),
        ("Are dogs animals?", "Yes."),
    ] {
        qa.push_str(&format!("{q}  {a}\n"));
    }
    format!(
        "[Reading]\nMode: combined\n{reading}\n[/Reading]\n[Dialogue Lesson]\n[examples]\n{qa}[/examples]\n[/Dialogue Lesson]\n\
         [Evaluation]\nDoes Max need water?  Yes.\nDoes Bella like milk?  Yes.\n[/Evaluation]\n"
    )
}

#[test]
fn nova_keeps_what_she_cannot_predict() {
    let mut model = Bicameral::with_config(Config::default(), 5);
    let passage =
        "The cat sat on the mat today. A bird flew over the old house. The river runs past the mill.";
    // Teach one sentence until the network predicts it well.
    for _ in 0..300 {
        model.train_text("The cat sat on the mat today.\n", 3e-3, Focus::Joint).unwrap();
    }
    let known = surprise(&model, "The cat sat on the mat today.").unwrap();
    let unknown = surprise(&model, "The river runs past the mill.").unwrap();
    assert!(known < KNOWN_LOSS && unknown > known, "known {known} unknown {unknown}");
    let mut store = MemoryStore::default();
    let report = store.study(&model, passage, "t", "", &[], 1).unwrap();
    assert_eq!(report.candidates, 3);
    let kept: Vec<&str> = report.stored.iter().map(|(_, t, _)| t.as_str()).collect();
    assert!(
        !kept.contains(&"The cat sat on the mat today."),
        "a sentence she predicts is not kept: {kept:?}"
    );
    assert!(kept.len() <= 2, "at most half a passage is kept (rounded up)");
    // Studying the same passage again merges instead of duplicating.
    let again = store.study(&model, passage, "t", "", &[], 2).unwrap();
    assert!(again.stored.is_empty());
    assert_eq!(again.merged.len(), kept.len());
    assert!(store.entries.iter().all(|m| m.seen == 2));
    assert_eq!(candidates("Is it raining? It is raining today in town."), ["It is raining today in town."]);
}

#[test]
fn study_mode_teaching_keeps_grading_without_memory() {
    let model = Bicameral::with_config(Config::default(), 9);
    let files = vec![CourseFile { name: "farm.txt".into(), text: lesson() }];
    let items = parse_lesson(&files[0].text, "farm.txt").unwrap();
    assert_eq!(items.iter().filter(|i| i.held_out).count(), 2);
    let cfg = SessionConfig {
        epochs: 2,
        batch_size: 8,
        learning_rate: 2e-3,
        foundation: false,
        education: None,
        keep_dense_reliance: 0.,
        patience: 2,
        rehearsal_share: 0.,
        eval_limit: None,
        heldout_loss_tolerance: 1.,
        retention_loss_tolerance: 1.,
        retention_panel: 0,
        protect_correct_answers: false,
        protect_answers_net: false,
        retention_relative: None,
        retention_units: 16,
        grow_vocabulary: false,
        word_limit: None,
        min_word_count: 12,
        seed: 3,
        strategy: Default::default(),
        retry: yendory_engine::learn::session::RetryPolicy {
            retries: 0,
            requeue: false,
            min_learning_rate: 1e-6,
        },
        pass_mark: 0.5,
        min_loss_improvement: 0.01,
        dropout: 0.,
        reading_share: 0.,
        reading: None,
        blend: 0.,
        recover_earlier: false,
        earlier_answers_net: false,
        train_earlier: false,
    };
    let mut events = Vec::new();
    let stop = AtomicBool::new(false);
    let mut studying = Studying::new(MemoryStore::default(), 100);
    let result = teach_course_studying(
        &model,
        &files,
        &[],
        &cfg,
        Some(&mut studying),
        &mut |e| events.push(e.clone()),
        &mut |_, _| Ok(()),
        &stop,
    )
    .unwrap();
    let store = studying.store;
    let studied = events.iter().find_map(|e| match e {
        CourseEvent::Studied { report, copies, .. } => Some((report.clone(), *copies)),
        _ => None,
    });
    let (report, copies) = studied.expect("the reading was studied before teaching");
    assert_eq!(report.candidates, 8);
    assert!(!report.stored.is_empty() && report.stored.len() <= 4, "{report:?}");
    // Nothing held out for grading is stored.
    for m in &store.entries {
        assert!(!m.text.contains("Does Max need water"));
        assert_eq!(m.sources[0].lesson, "farm.txt");
        assert!(!m.text.contains(MEMORY));
    }
    assert!(copies > 0, "questions related to the memories train with them in front");
    assert!(store.log.iter().any(|r| r.purpose == "training"));
    let record = &result.files[0];
    // Grading (final_score) is without memory; the score with memory is reported separately.
    let fs = record.final_score.as_ref().unwrap();
    assert_eq!(fs.exact_total, 2);
    assert_eq!(record.with_memory.map(|w| w.1), Some(2));
    assert!(store.log.iter().any(|r| r.purpose == "evaluation"));
}
