//! Comparing settings (learn::compare): the model is never changed, every variant starts from it,
//! the test half is never part of what a session sees, and every epoch is measured.
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::compare::{compare, split_test, CompareEvent, Variant};
use yendory_engine::learn::course::CourseFile;
use yendory_engine::learn::lessons::parse_lesson;
use yendory_engine::learn::session::{Event, SessionConfig};
use yendory_engine::Bicameral;

fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/compat")
}

#[test]
fn variants_start_from_the_same_model_and_are_tested_on_unseen_questions() {
    let model = Bicameral::load(root().join("fixtures/nova2-step15000.nnova2")).unwrap();
    let read = |f: &str| std::fs::read_to_string(root().join("fixtures/lessons").join(f)).unwrap();
    let lesson = yendory_engine::learn::builtin::builtin_courses().remove(0).lessons.remove(0);
    let files = vec![CourseFile { name: lesson.name.clone(), text: lesson.text.clone() }];
    let archive = parse_lesson(&read("sample_course_a.txt"), "a.txt").unwrap();
    let base = SessionConfig {
        epochs: 2,
        batch_size: 8,
        learning_rate: 1e-3,
        foundation: false,
        train_earlier: true,
        ..SessionConfig::default()
    };
    let variants = vec![
        Variant { name: "A".into(), cfg: base.clone() },
        Variant { name: "B".into(), cfg: SessionConfig { probation_epochs: 2, ..base } },
    ];
    // Questions the sessions judge by: the held-out ones left after the test half is split off.
    let (_, test_half) = split_test(parse_lesson(&files[0].text, &files[0].name).unwrap());
    let mut events = Vec::new();
    let before = model.clone();
    let results = compare(
        &model,
        &files,
        &archive,
        &variants,
        1,
        &mut |e| events.push(e.clone()),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(model.weights, before.weights, "the model is never changed");
    assert_eq!(results.len(), 2);
    let Some(CompareEvent::Started { new_test, earlier_test, start, start_context, .. }) = events.first()
    else {
        panic!("starts with the test halves")
    };
    assert_eq!(*new_test, test_half.len());
    assert!(*earlier_test > 0 && start.new_total == *new_test && start_context.is_some());
    for (i, r) in results.iter().enumerate() {
        println!("{}: {:?} gained {} lost {} rejections {:?}", r.name, r.end, r.gained, r.lost, r.rejections);
        assert_eq!(r.trajectory.len() as u32, r.epochs, "every epoch tested");
        assert_eq!(r.end.new_total, *new_test);
        assert!(r.context.is_some());
        // The judged questions never include the test half.
        let judged = events.iter().find_map(|e| match e {
            CompareEvent::Session { index, event: Event::Started { heldout_units, .. } } if *index == i => {
                Some(*heldout_units)
            }
            _ => None,
        });
        let all_held =
            parse_lesson(&files[0].text, &files[0].name).unwrap().iter().filter(|x| x.held_out).count()
                + archive.iter().filter(|x| x.held_out).count();
        assert!(judged.unwrap() + test_half.len() <= all_held, "test questions are not judged");
    }
}
