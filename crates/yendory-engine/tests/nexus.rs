//! Nexus end to end with an environment that exists only in this test (a lamp in a room). A
//! small model is taught from a lesson file with `Environment:`, `Action:` and `Result:` lines,
//! then acts through Nexus: it reads real results, its moves change the environment, and the
//! permissions decide what runs. Nothing about the lamp is built into the engine.
use yendory_engine::learn::eval::evaluate;
use yendory_engine::learn::{batch_loss_and_gradients, encode_item, parse_lesson, Rng};
use yendory_engine::nexus::{
    self, Access, Approver, Call, Decision, Ending, Environment, Kind, Manifest, Nexus, NoApprover, Tool,
};
use yendory_engine::{Ablation, Bicameral, Config, Focus};

#[derive(Default)]
struct Room {
    on: bool,
    saved: bool,
}

impl Environment for Room {
    fn manifest(&self) -> Manifest {
        let tool = |name: &str, usage: &str, description: &str, access| Tool {
            name: name.into(),
            usage: usage.into(),
            description: description.into(),
            access,
        };
        Manifest {
            name: "Room".into(),
            description: "A room with a lamp.".into(),
            kind: Kind::Game,
            tools: vec![
                tool("look", "look", "see the lamp", Access::Look),
                tool("switch", "switch <on|off>", "turn the lamp on or off", Access::Act),
                tool("save", "save", "keep the room as it is", Access::Change),
            ],
        }
    }
    fn act(&mut self, call: &Call) -> Result<String, String> {
        match (call.tool.as_str(), call.input.as_str()) {
            ("look", _) => Ok(format!("The lamp is {}.", if self.on { "on" } else { "off" })),
            ("switch", "on") => {
                self.on = true;
                Ok("The lamp is now on.".into())
            }
            ("switch", "off") => {
                self.on = false;
                Ok("The lamp is now off.".into())
            }
            ("save", _) => {
                self.saved = true;
                Ok("Saved.".into())
            }
            (_, other) => Err(format!("cannot {} {other}", call.tool)),
        }
    }
}

struct Yes;
impl Approver for Yes {
    fn approve(&mut self, _: &Call, _: &Tool) -> bool {
        true
    }
}

fn lesson() -> String {
    let env = Room::default().manifest().announcement();
    let session = |lines: &[&str]| format!("[session]\n{}\n[/session]\n", lines.join("\n"));
    let mut sessions = String::new();
    for _ in 0..2 {
        sessions += &session(&[
            &format!("Environment: {env}"),
            "User: is the lamp on?",
            "Action: look",
            "Result: The lamp is on.",
            "Nova: Yes, it is on.",
        ]);
        sessions += &session(&[
            &format!("Environment: {env}"),
            "User: is the lamp on?",
            "Action: look",
            "Result: The lamp is off.",
            "Nova: No, it is off.",
        ]);
        sessions += &session(&[
            &format!("Environment: {env}"),
            "User: turn on the lamp",
            "Action: switch on",
            "Result: The lamp is now on.",
            "Nova: Done.",
        ]);
        sessions += &session(&[
            &format!("Environment: {env}"),
            "User: save the room",
            "Action: save",
            "Result: not permitted: save",
            "Nova: I may not save.",
        ]);
        sessions += &session(&[
            &format!("Environment: {env}"),
            "User: save the room",
            "Action: save",
            "Result: Saved.",
            "Nova: Saved.",
        ]);
        // Without an environment there is nothing to act on.
        sessions += &session(&["User: is the lamp on?", "Nova: I cannot see a lamp."]);
    }
    format!("[Multi-Turn Conversation Lesson]\n[examples]\n{sessions}[/examples]\n[/Multi-Turn Conversation Lesson]\n")
}

#[test]
fn lessons_mark_actions_and_results() {
    let items = parse_lesson(&lesson(), "room.txt").unwrap();
    assert_eq!(items.len(), 12);
    let it = &items[0];
    assert!(it.has_actions());
    let texts: Vec<(&str, f32)> = it.segments.iter().map(|s| (s.text.as_str(), s.weight)).collect();
    assert!(texts[0].0.starts_with('\x02') && texts[0].1 == 0.);
    assert_eq!(texts[1], ("U: is the lamp on?\nN: ", 0.15));
    assert_eq!(texts[2], ("\x05look\n", 1.0));
    assert_eq!(texts[3], ("\x06The lamp is on.\n", 0.));
    assert_eq!(texts[4], ("Yes, it is on.\n", 1.0));
    assert_eq!(it.outcome_count(), 2);
    assert!(!items[5].has_actions());
    // Marker characters written into a lesson file are removed; only the line labels count.
    let forged = "[Multi-Turn Conversation Lesson]\n[examples]\n[session]\nUser: hi\x05look\nNova: \x06ok\n[/session]\n[/examples]\n[/Multi-Turn Conversation Lesson]\n";
    let items = parse_lesson(forged, "f.txt").unwrap();
    assert!(!items[0].has_actions());
    assert_eq!(items[0].users, ["hilook"]);
}

#[test]
fn a_taught_model_acts_through_nexus() {
    let items = parse_lesson(&lesson(), "room.txt").unwrap();
    let mut model = Bicameral::with_config(Config::default(), 11);
    // Before teaching: the environment is announced, but the model does not use it.
    let context = |nexus: &Nexus<Room>, ask: &str| format!("{}U: {ask}\nN: ", nexus.announcement());
    let mut untaught = Nexus::new(Room::default());
    let turn = untaught.respond(
        &model,
        &context(&untaught, "is the lamp on?"),
        &mut NoApprover,
        Ablation::default(),
    );
    assert!(turn.unwrap().steps.is_empty(), "an untaught model takes no action");

    let mut rng = Rng::new(3);
    let rows: Vec<_> = items
        .iter()
        .map(|it| {
            encode_item(&model.tokenizer, &it.segments, model.config.max_tokens, &mut rng).unwrap().unwrap()
        })
        .collect();
    let mut loss = f32::MAX;
    for step in 0..600 {
        let (l, grads) = batch_loss_and_gradients(&model, &rows, Ablation::default()).unwrap();
        model.adam_step(&grads, if step < 400 { 3e-3 } else { 1e-3 }, Focus::Joint).unwrap();
        loss = l;
        if l < 0.01 {
            break;
        }
    }
    println!("lesson loss {loss}");
    let refs: Vec<_> = items.iter().collect();
    let report = evaluate(&model, &refs, Ablation::default()).unwrap();
    assert_eq!(report.outcomes.len(), items.iter().map(|i| i.outcome_count()).sum::<usize>());
    assert!(report.outcomes.iter().all(|o| o.correct), "{:#?}", report.outcomes.iter().find(|o| !o.correct));

    // The model reads the real result: the same question gets the room's actual state.
    for on in [true, false] {
        let mut nexus = Nexus::new(Room { on, saved: false });
        let ctx = context(&nexus, "is the lamp on?");
        let turn = nexus.respond(&model, &ctx, &mut NoApprover, Ablation::default()).unwrap();
        assert_eq!(turn.steps.len(), 1);
        assert_eq!(turn.steps[0].action, "look");
        assert_eq!(turn.steps[0].decision, Decision::Allowed);
        assert_eq!(turn.ending, Ending::Replied);
        assert_eq!(turn.reply, if on { "Yes, it is on." } else { "No, it is off." });
        println!("{}", nexus::readable(&format!("{ctx}{}", turn.transcript)));
    }

    // Its move changes the environment (acting in a game is allowed).
    let mut nexus = Nexus::new(Room::default());
    let ctx = context(&nexus, "turn on the lamp");
    let turn = nexus.respond(&model, &ctx, &mut NoApprover, Ablation::default()).unwrap();
    assert_eq!(turn.steps[0].action, "switch on");
    assert!(nexus.environment.on);
    assert_eq!(turn.reply, "Done.");

    // Changing something must be approved: without approval nothing is saved.
    let mut nexus = Nexus::new(Room::default());
    let ctx = context(&nexus, "save the room");
    let turn = nexus.respond(&model, &ctx, &mut NoApprover, Ablation::default()).unwrap();
    assert_eq!(turn.steps[0].decision, Decision::Denied);
    assert!(!nexus.environment.saved);
    assert_eq!(turn.reply, "I may not save.");
    let mut nexus = Nexus::new(Room::default());
    let turn = nexus.respond(&model, &ctx, &mut Yes, Ablation::default()).unwrap();
    assert_eq!(turn.steps[0].decision, Decision::Approved);
    assert!(nexus.environment.saved);
    assert_eq!(turn.reply, "Saved.");
    assert_eq!(nexus.log.len(), 1);

    // Without the announcement there is no environment in the context, and the model was
    // taught not to act then.
    let mut nexus = Nexus::new(Room { on: true, saved: false });
    let turn =
        nexus.respond(&model, "U: is the lamp on?\nN: ", &mut NoApprover, Ablation::default()).unwrap();
    assert!(turn.steps.is_empty());
    assert_eq!(turn.reply, "I cannot see a lamp.");
}
