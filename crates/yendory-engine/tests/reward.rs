//! Rewards on real teaching sessions: recording changes nothing about training, reward-guided
//! training schedules units by need within the same budget, and protected exam questions never
//! reach the ledger.
use std::sync::atomic::AtomicBool;
use yendory_engine::learn::course::CourseFile;
use yendory_engine::learn::curriculum::{prepare, CourseDef};
use yendory_engine::learn::lessons::{canonical, parse_lesson, Item};
use yendory_engine::learn::program::{run_course, CourseHost, CourseRun, ProgramEvent, RunOptions, Snapshot};
use yendory_engine::learn::reward::{
    question_id, question_text, Origin, RewardConfig, RewardLedger, Rewards,
};
use yendory_engine::learn::session::{teach_rewarded, teach_with, Event, SessionConfig};
use yendory_engine::{Bicameral, Config};

fn lesson() -> Vec<Item> {
    let names = ["Max", "Bella", "Rex", "Tom", "Kim", "Sam", "Ann", "Bob", "Joe", "Liz", "Ned", "Pam"];
    let places = ["park", "kitchen", "garden", "hallway"];
    let mut s = String::from("[Dialogue Lesson]\n[examples]\n");
    for (i, n) in names.iter().enumerate() {
        for (j, p) in places.iter().enumerate() {
            s.push_str(&format!("Where is {n} {j}?  In the {p}.\n"));
        }
        s.push_str(&format!("Is {n} {i} here?  Yes.\n"));
    }
    s.push_str("[/examples]\n[/Dialogue Lesson]\n[Evaluation]\nWhere is Uma 1?  In the kitchen.\nIs Uma here?  Yes.\nWhere is Vic 3?  In the hallway.\n[/Evaluation]\n");
    parse_lesson(&s, "places.txt").unwrap()
}

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
        seed: 11,
        ..SessionConfig::default()
    }
}

#[test]
fn recording_rewards_does_not_change_ordinary_training() {
    let model = Bicameral::with_config(Config::default(), 21);
    let items = lesson();
    let stop = AtomicBool::new(false);
    let plain = teach_with(&model, &items, None, &[], &cfg(), &mut |_| {}, &stop).unwrap();
    let mut rewards = Rewards::new(RewardConfig::default(), RewardLedger::default(), "m");
    let mut events = Vec::new();
    let rec = teach_rewarded(
        &model,
        &items,
        None,
        &[],
        &cfg(),
        Some(&mut rewards),
        &mut |e| events.push(e.clone()),
        &stop,
    )
    .unwrap();
    assert_eq!(plain.model.weights, rec.model.weights, "Stage 1 only measures");
    assert_eq!(plain.final_score, rec.final_score);
    assert!(!events.iter().any(|e| matches!(e, Event::Emphasis { .. })), "no emphasis without Stage 2");
    assert_eq!(events.iter().filter(|e| matches!(e, Event::Rewarded { .. })).count(), 2);
    let l = &rewards.ledger;
    assert_eq!(l.sessions.len(), 1);
    assert!(l.questions.values().any(|q| q.unseen) && l.questions.values().any(|q| q.trained));
    // Held-out questions are recorded as never-trained (generalization), not progress.
    for e in l.audit.iter().filter(|e| e.origin == Origin::HeldOut) {
        assert_eq!(e.progress, 0.);
    }
    assert!(
        l.audit.iter().filter(|e| e.origin == Origin::Training).count() <= RewardConfig::default().quiz_limit
    );
}

#[test]
fn reward_guided_training_practises_gaps_more_within_the_same_budget() {
    let model = Bicameral::with_config(Config::default(), 22);
    let items = lesson();
    let stop = AtomicBool::new(false);
    let config = RewardConfig { weighted_training: true, quiz_limit: 200, ..RewardConfig::default() };
    let mut rewards = Rewards::new(config, RewardLedger::default(), "m");
    let mut plans = Vec::new();
    let r = teach_rewarded(
        &model,
        &items,
        None,
        &[],
        &cfg(),
        Some(&mut rewards),
        &mut |e| {
            if let Event::Emphasis { plan, .. } = e {
                plans.push(plan.clone());
            }
        },
        &stop,
    )
    .unwrap();
    assert_eq!(plans.len() as u32, r.epochs.len() as u32, "one plan per epoch");
    let train = items.iter().filter(|i| !i.held_out).count();
    for p in &plans {
        assert_eq!(p.units, train);
        assert!(
            (p.copies as i64 - train as i64).abs() <= (train / 5) as i64,
            "about the same number of units: {} vs {train}",
            p.copies
        );
        assert!(p.top.iter().all(|t| !t.reasons.is_empty()), "every emphasis has its reasons");
        assert!(p.top[0].value >= p.low[0].value);
    }
    // Held-out questions never enter the schedule.
    let held: Vec<String> = items.iter().filter(|i| i.held_out).map(question_text).collect();
    assert!(plans.iter().all(|p| p.top.iter().chain(&p.low).all(|t| !held.contains(&t.question))));
    // Every taught unit was quizzed; an untrained model gets most of them wrong, so gaps lead.
    assert!(plans[0].quizzed >= train - 1);
    assert!(plans[0].top[0].reasons.iter().any(|x| x.contains('%')));
}

struct Host<'a> {
    rewards: Option<Rewards>,
    returned: Option<Rewards>,
    stop: &'a AtomicBool,
}
impl CourseHost for Host<'_> {
    fn event(&mut self, e: &ProgramEvent) {
        if let ProgramEvent::Graded { .. } = e {
            self.stop.store(true, std::sync::atomic::Ordering::Release);
        }
    }
    fn save(&mut self, _: &Bicameral, _: &[Item], _: &CourseRun) -> Result<(), String> {
        Ok(())
    }
    fn save_best(&mut self, _: &Snapshot) -> Result<(), String> {
        Ok(())
    }
    fn take_rewards(&mut self) -> Option<Rewards> {
        self.rewards.take()
    }
    fn return_rewards(&mut self, r: Rewards) {
        self.returned = Some(r);
    }
}

#[test]
fn course_exams_are_protected_from_the_ledger() {
    let mut text = String::from("[Dialogue Lesson]\n[examples]\n");
    for i in 0..120 {
        text.push_str(&format!("What is item {i}?  It is thing {}.\n", i % 7));
    }
    text.push_str("[/examples]\n[/Dialogue Lesson]\n");
    let mut def = CourseDef::new("c", "Things");
    def.settings.epochs = 1;
    def.settings.retries = 0;
    def.settings.requeue = false;
    def.settings.foundation = false;
    def.add_files(vec![CourseFile { name: "1 Transformer - things.txt".into(), text }]);
    let prep = prepare(&def);
    assert!(prep.check.problems.is_empty(), "{:?}", prep.check.problems);
    let exam: Vec<String> = prep
        .exam
        .iter()
        .flat_map(|s| &s.items)
        .map(|i| canonical(i.prompt.as_deref().unwrap_or("")))
        .collect();
    assert!(!exam.is_empty());
    let model = Bicameral::with_config(Config::default(), 5);
    let mut run = CourseRun::new(&prep, false, 3);
    let config = RewardConfig { weighted_training: true, ..RewardConfig::default() };
    let stop = AtomicBool::new(false);
    let mut host = Host {
        rewards: Some(Rewards::new(config, RewardLedger::default(), "m")),
        returned: None,
        stop: &stop,
    };
    run_course(&model, &[], &prep, &mut run, None, &RunOptions::default(), &mut host, &stop).unwrap();
    let r = host.returned.expect("the ledger is given back");
    assert!(!r.ledger.sessions.is_empty());
    for q in r.ledger.questions.values() {
        assert!(!exam.contains(&canonical(&q.question)), "exam question in the ledger: {}", q.question);
    }
    assert!(r.ledger.audit.iter().all(|e| !exam.contains(&canonical(&e.question))));
    // An exam item handed to the ledger directly is refused.
    let exam_item = prep.exam[0].items[0].clone();
    assert!(r.is_protected(&exam_item));
    assert!(!r.ledger.questions.contains_key(&question_id(&exam_item)));
}
