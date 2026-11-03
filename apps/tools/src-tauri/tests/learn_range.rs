//! Does a network learn to play the range from pixels? (docs/GAME_TRAINING.md, phase 4)
//!
//! There is no person in a test, so a **test-only teacher** plays the range: it reads the
//! range's truth (where the targets are), turns toward the nearest, walks toward it and fires
//! when it lines up, through the safety layer like any input. Its play is recorded through the
//! ordinary recorder (listening to XTEST, which only tests may). The network then learns from
//! those recordings alone (pictures and controls), and is measured on layouts it never saw
//! against the untrained network and the frequency baseline. The teacher never plays during the
//! measurement and nothing of it exists outside this test. Then generations: the network goes on
//! learning from its own best play on the range (the teacher takes no part), and the champion
//! that results is measured against generation 0.
//!
//! Long (about forty minutes, in real time): run it on purpose, under an X display:
//! `xvfb-run -a -s "-screen 0 1920x1080x24" cargo test --release --test learn_range -- --ignored --nocapture`
#![cfg(target_os = "linux")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use yendory_tools_lib::live::autoplay::{Brain, Decide};
use yendory_tools_lib::live::controls::{Button, Control};
use yendory_tools_lib::live::evaluate::{evaluate, EvalSetup};
use yendory_tools_lib::live::generations::{self, Arena, GenSettings};
use yendory_tools_lib::live::policy::{Policy, Size};
use yendory_tools_lib::live::profile::{self, range_profile};
use yendory_tools_lib::live::record::{record, Setup};
use yendory_tools_lib::live::recording::{self, Split};
use yendory_tools_lib::live::safety::Controller;
use yendory_tools_lib::live::train::{self, Settings};
use yendory_tools_lib::live::{skills, sys, RangeProcess};
use yendory_tools_lib::range::{angle_diff, TURN_PER_UNIT};

const EXE: &str = env!("CARGO_BIN_EXE_yendory-tools");

fn num(v: &serde_json::Value, k: &str) -> f32 {
    v[k].as_f64().unwrap_or(0.0) as f32
}

/// The teacher: plays from the range's truth for `seconds`, through the safety layer. It aims at
/// the target in line of sight nearest its view, walks toward it, and fires when lined up; with
/// none in sight it turns to look for one.
fn teach(range: &Mutex<RangeProcess>, window: u64, seconds: f64, stop: &AtomicBool) {
    let p = range_profile();
    let c =
        Mutex::new(Controller::new(sys::open().unwrap(), window, p.permitted().unwrap(), p.limits.clone()));
    let start = Instant::now();
    let mut firing = 0;
    while start.elapsed().as_secs_f64() < seconds && !stop.load(Ordering::Acquire) {
        let tick = Instant::now();
        let s = range.lock().unwrap().state().unwrap();
        let (x, y, a) = (num(&s, "x"), num(&s, "y"), num(&s, "angle"));
        let targets: Vec<(f32, f32)> = serde_json::from_value(s["targets"].clone()).unwrap_or_default();
        let visible: Vec<bool> = serde_json::from_value(s["visible"].clone()).unwrap_or_default();
        let best = targets
            .iter()
            .zip(&visible)
            .filter(|(_, v)| **v)
            .map(|(&(tx, ty), _)| (angle_diff((ty - y).atan2(tx - x), a), (tx - x).hypot(ty - y)))
            .min_by(|p, q| p.0.abs().total_cmp(&q.0.abs()));
        let mut c = c.lock().unwrap();
        let (w, fire) = (Control::Key("w"), Control::Button(Button::Left));
        if firing > 0 {
            firing -= 1;
            if firing == 0 {
                c.release(fire, "");
            }
        }
        match best {
            Some((diff, dist)) => {
                let units = (diff / TURN_PER_UNIT).clamp(-40.0, 40.0);
                if units.abs() >= 1.0 {
                    c.motion(units, 0.0);
                }
                if diff.abs() < 0.3 && dist > 3.0 {
                    c.press(w);
                } else {
                    c.release(w, "");
                }
                if firing == 0 && diff.abs() < (0.25 / dist).atan() * 0.7 {
                    c.press(fire);
                    firing = 2;
                }
            }
            None => {
                // Nothing in sight: turn to look.
                c.release(w, "");
                c.motion(30.0, 0.0);
            }
        }
        c.tick();
        drop(c);
        std::thread::sleep(Duration::from_millis(50).saturating_sub(tick.elapsed()));
    }
    c.lock().unwrap().release_all("teaching ended");
}

#[test]
#[ignore = "about forty minutes in real time; run on purpose (see the top of the file)"]
fn a_network_learns_to_hit_targets_from_recorded_play() {
    assert!(std::env::var_os("DISPLAY").is_some(), "needs an X display");
    let data = std::env::temp_dir().join(format!("tools-learn-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let prof = profile::save(&data, range_profile()).unwrap();
    let range = RangeProcess::start(EXE.as_ref(), 1, |_| {}).unwrap();
    let pid = range.pid();
    let range = Arc::new(Mutex::new(range));
    let window = loop {
        if let Some(w) =
            sys::open().unwrap().windows().unwrap().into_iter().find(|w| prof.matches(w) && w.pid == pid)
        {
            break w.id;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut d = sys::open().unwrap();
    d.bring_to_front(window).unwrap();
    std::thread::sleep(Duration::from_millis(500));

    // Recordings of the teacher: several layouts for training, others held out for testing.
    let plan: Vec<(u64, f64, Split)> =
        (0..9).map(|i| (10 + i, 30.0, Split::Train)).chain([(50, 30.0, Split::Test)]).collect();
    for (seed, seconds, split) in plan {
        range.lock().unwrap().reset(seed).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        let (tx, rx) = std::sync::mpsc::channel();
        let listening = sys::listen(true, tx).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let (r2, s2) = (range.clone(), stop.clone());
        let teacher = std::thread::spawn(move || {
            teach(&r2, window, seconds, &s2);
            s2.store(true, Ordering::Release);
        });
        let mut desk = sys::open().unwrap();
        let setup =
            Setup { data: &data, profile: &prof, window, split, notes: format!("teacher, seed {seed}") };
        let info = record(setup, desk.as_mut(), rx, &stop, |_| {}).unwrap();
        teacher.join().unwrap();
        drop(listening);
        let v = recording::validate(&data, &prof.id, &info.id).unwrap();
        assert!(v.ok, "{v:?}");
        eprintln!("recorded seed {seed} ({split:?}): {} ticks, active {:?}", v.ticks, v.active_ticks);
    }
    let teacher_hits = range.lock().unwrap().state().unwrap()["hits"].as_u64().unwrap_or(0);
    eprintln!("the teacher's hits in its last episode (30 s): {teacher_hits}");
    assert!(teacher_hits >= 5, "the teacher itself must play well enough to learn from");

    // Learning: from the recordings only.
    // A new skill, as a Neuro model gets one (kept in memory here: no model or Network needed).
    let skill = skills::new_skill(&data, &prof, Size::Small, 7).unwrap();
    eprintln!("mouse steps: {:?}", skill.spec.arch.mouse_bins);
    let settings = Settings {
        steps: 1200,
        batch: 16,
        length: 8,
        learning_rate: 5e-4,
        measure_every: 100,
        ..Settings::default()
    };
    let t0 = Instant::now();
    let trained = train::train(
        &data,
        &prof,
        skill,
        settings,
        false,
        None,
        &AtomicBool::new(false),
        |p| {
            if let Some(m) = &p.measure {
                eprintln!(
                    "step {}: train {:.3}, held out {:.3} (baseline {:.3}){}",
                    p.step,
                    p.train_loss,
                    m.loss,
                    m.baseline_loss,
                    if p.accepted { " accepted" } else { "" }
                );
            }
        },
        |_, _| Ok(()),
    )
    .unwrap();
    eprintln!("trained in {:.0} s", t0.elapsed().as_secs_f64());
    let best = train::runs(&trained.spec).last().unwrap().best.clone().expect("a held-out measurement");
    assert!(best.loss < best.baseline_loss, "held-out {} vs baseline {}", best.loss, best.baseline_loss);

    // Measuring on layouts nobody played, against the deciders that can't have learned.
    let policy = trained.policy().unwrap();
    let untrained = Policy::new(trained.spec.arch.clone(), 99).unwrap();
    let freq =
        train::frequencies(&trained.spec, &train::load_clips(&data, &trained.spec, Split::Train).unwrap());
    let mut brains = vec![
        ("trained".to_string(), Brain::net(policy)),
        ("untrained".to_string(), Brain::net(untrained)),
        ("frequencies".to_string(), Brain::Frequencies { freq, bins: trained.spec.arch.mouse_bins.clone() }),
    ];
    let setup = EvalSetup {
        data: &data,
        profile: &prof,
        window,
        episodes: 4,
        seconds: 20.0,
        decide: Decide::Likely,
        first_seed: yendory_tools_lib::live::evaluate::SEEDS_FROM,
        record: false,
    };
    let rows = {
        let mut r = range.lock().unwrap();
        evaluate(setup, &mut r, &mut brains, &AtomicBool::new(false), |p| eprintln!("{p:?}")).unwrap()
    };
    for r in &rows {
        eprintln!(
            "{}: {:.1} hits a minute, {:.1} shots a minute, episodes {:?}",
            r.who,
            r.hits_per_minute,
            r.shots_per_minute,
            r.episodes.iter().map(|e| e.hits).collect::<Vec<_>>()
        );
    }
    assert!(rows[0].hits_per_minute > rows[1].hits_per_minute, "trained vs untrained");
    assert!(rows[0].hits_per_minute > rows[2].hits_per_minute, "trained vs frequencies");

    // Generations: it goes on learning from its own best play (the teacher takes no part).
    let gen0 = trained.policy().unwrap();
    let range = Mutex::new(Some(Arc::try_unwrap(range).ok().unwrap().into_inner().unwrap()));
    let arena = Arena { data: &data, profile: &prof, window, range: &range, name: "nova".into() };
    let settings = GenSettings { generations: 4, episodes: 2, seconds: 20.0, ..GenSettings::default() };
    let t0 = Instant::now();
    let champion = generations::run(
        &arena,
        trained,
        settings,
        None,
        &AtomicBool::new(false),
        |p| {
            if p.phase == "done" {
                eprintln!("generation {} of {}: {}", p.run, p.of, p.detail);
            }
        },
        |_| Ok(()),
    )
    .unwrap();
    eprintln!("generations took {:.0} s", t0.elapsed().as_secs_f64());
    for a in generations::attempts(&champion) {
        eprintln!(
            "attempt {}: players {:?}, learned from {:?}, champion {:.1} vs new {:.1}, accepted {} {}",
            a.attempt, a.candidates, a.elite, a.champion_score, a.child_score, a.accepted, a.note
        );
    }
    // The champion after generations against generation 0, on the measuring layouts.
    let mut pair = vec![
        (format!("generation {}", champion.spec.lineage.generation), Brain::net(champion.policy().unwrap())),
        ("generation 0".to_string(), Brain::net(gen0)),
    ];
    let setup = EvalSetup {
        data: &data,
        profile: &prof,
        window,
        episodes: 8,
        seconds: 20.0,
        decide: Decide::Likely,
        first_seed: yendory_tools_lib::live::evaluate::SEEDS_FROM,
        record: false,
    };
    let rows = {
        let mut slot = range.lock().unwrap();
        evaluate(setup, slot.as_mut().unwrap(), &mut pair, &AtomicBool::new(false), |_| {}).unwrap()
    };
    for r in &rows {
        eprintln!(
            "{}: {:.1} hits a minute, {:.1} shots a minute, episodes {:?}",
            r.who,
            r.hits_per_minute,
            r.shots_per_minute,
            r.episodes.iter().map(|e| e.hits).collect::<Vec<_>>()
        );
    }
    let _ = std::fs::remove_dir_all(&data);
}
