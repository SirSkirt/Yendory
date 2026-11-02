//! Checks for each phase of game training, run from the window on the owner's own computer, so
//! every phase can be verified on its own (docs/GAME_TRAINING.md). Each check says what it did
//! and what it found. All input goes through the safety layer, as everywhere.
use super::autoplay::{self, Brain, Decide};
use super::policy::{check_arch, gradient_check, random_sequence, Policy};
use super::recording::{self, Split};
use super::safety::{Controller, STOPS};
use super::{alive, data_dir, player, profile, script, sys, train, Live, RangeProcess};
use crate::Res;
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Check {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

fn check(name: &str, ok: bool, detail: impl Into<String>) -> Check {
    Check { name: name.into(), ok, detail: detail.into() }
}

fn num(v: &serde_json::Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or(0.0)
}

/// The test range (started if needed) and its window.
fn range_window(app: &AppHandle) -> Res<u64> {
    let data = data_dir(app)?;
    if profile::load(&data, "yendory-range").is_err() {
        profile::save(&data, profile::range_profile())?;
    }
    let live = app.state::<Live>();
    let pid = {
        let mut slot = live.range.lock().unwrap();
        if alive(&mut slot).is_none() {
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            *slot = Some(RangeProcess::start(&exe, 1, |_| {})?);
        }
        slot.as_ref().map(|r| r.pid()).unwrap_or(0)
    };
    let p = profile::range_profile();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(w) = sys::open()?.windows()?.into_iter().find(|w| p.matches(w) && w.pid == pid) {
            return Ok(w.id);
        }
        if Instant::now() > deadline {
            return Err("The range's window did not appear".into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn front(window: u64) -> Res<()> {
    let mut d = sys::open()?;
    d.bring_to_front(window)?;
    let deadline = Instant::now() + Duration::from_secs(3);
    while !d.in_front(window) {
        if Instant::now() > deadline {
            return Err("The range did not come to the front".into());
        }
        std::thread::sleep(Duration::from_millis(30));
    }
    std::thread::sleep(Duration::from_millis(300));
    Ok(())
}

fn range_state(app: &AppHandle) -> Res<serde_json::Value> {
    let live = app.state::<Live>();
    let mut slot = live.range.lock().unwrap();
    alive(&mut slot).ok_or("The range closed")?.state()
}

fn controller(window: u64) -> Res<Mutex<Controller>> {
    let p = profile::range_profile();
    Ok(Mutex::new(Controller::new(sys::open()?, window, p.permitted()?, p.limits.clone())))
}

/// Phase 2: the game's picture, and input through the safety layer, on the range.
fn phase2(app: &AppHandle) -> Res<Vec<Check>> {
    let mut out = Vec::new();
    let window = range_window(app)?;
    front(window)?;
    let mut desk = sys::open()?;
    let before = desk.grab(window, 128, 72)?;
    let varied = before.iter().any(|&b| b != before[0]);
    out.push(check(
        "The game's picture can be taken",
        varied,
        if varied {
            "a 128×72 picture of the range, not blank"
        } else {
            "the picture is blank: keep the range visible on screen"
        },
    ));

    let s0 = range_state(app)?;
    let c = controller(window)?;
    script::run(
        &c,
        &script::parse("hold w 300; move 100 0; wait 100; click left; hold w+shift 200; press e").unwrap(),
    );
    std::thread::sleep(Duration::from_millis(250));
    let s1 = range_state(app)?;
    let presses = num(&s1, "presses") - num(&s0, "presses");
    let mouse = num(&s1, "mouseX") - num(&s0, "mouseX");
    let shots = num(&s1, "shots") - num(&s0, "shots");
    out.push(check(
        "Key and button presses arrive",
        presses == 4.0,
        format!("the range received {presses} presses of the 4 sent (W, the click, W and Shift)"),
    ));
    out.push(check(
        "Mouse movement arrives exactly",
        mouse == 100.0,
        format!("the range received {mouse} units across of the 100 sent"),
    ));
    out.push(check("A click fires", shots == 1.0, format!("{shots} shot")));
    let held: Vec<String> = s1["held"]
        .as_object()
        .map(|o| o.iter().filter(|(_, v)| v.as_bool() == Some(true)).map(|(k, _)| k.clone()).collect())
        .unwrap_or_default();
    out.push(check(
        "Nothing is left held",
        held.is_empty() && c.lock().unwrap().held().is_empty(),
        if held.is_empty() {
            "nothing held in the range or in Tools".to_string()
        } else {
            format!("still held in the range: {}", held.join(", "))
        },
    ));
    let refused: Vec<String> = c
        .lock()
        .unwrap()
        .log
        .iter()
        .filter(|e| !e.sent)
        .map(|e| format!("{} ({})", e.action, e.note))
        .collect();
    out.push(check(
        "A key outside the profile is refused",
        refused.iter().any(|r| r.starts_with("press e")),
        refused.join("; "),
    ));
    let after = desk.grab(window, 128, 72)?;
    out.push(check(
        "The picture changes when the view turns",
        after != before,
        "compared the picture before and after the actions",
    ));

    // Another window comes to the front: everything is released and input pauses.
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut other = RangeProcess::start(&exe, 2, |_| {})?;
    let p = profile::range_profile();
    let pid = other.pid();
    let deadline = Instant::now() + Duration::from_secs(20);
    let other_window = loop {
        if let Some(w) = sys::open()?.windows()?.into_iter().find(|w| p.matches(w) && w.pid == pid) {
            break w.id;
        }
        if Instant::now() > deadline {
            return Err("A second range window (for the focus check) did not appear".into());
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    front(window)?;
    let c = controller(window)?;
    let pressed = c.lock().unwrap().press(super::controls::Control::Key("w"));
    front(other_window)?;
    c.lock().unwrap().tick();
    let released = c.lock().unwrap().held().is_empty();
    let paused = c.lock().unwrap().paused.is_some();
    let refused = !c.lock().unwrap().press(super::controls::Control::Key("w"));
    let other_got = other.state().map(|s| num(&s, "presses")).unwrap_or(-1.0);
    drop(other);
    out.push(check(
        "Leaving the front releases everything and pauses",
        pressed && released && paused && refused && other_got == 0.0,
        format!("W pressed: {pressed}; released when another window came forward: {released}; paused: {paused}; the other window received {other_got} presses"),
    ));
    let _ = front(window);
    let problem = match &*app.state::<Live>().stop_keys.lock().unwrap() {
        Some(Ok(())) => String::new(),
        Some(Err(e)) => e.clone(),
        None => "not started".into(),
    };
    out.push(check(
        "The emergency stop keys are watched",
        problem.is_empty(),
        if problem.is_empty() {
            "Ctrl+Alt+End is watched in every program (press it in the check below to try it)".into()
        } else {
            problem
        },
    ));
    Ok(out)
}

/// Phase 3: what Tools records, and the person's latest recording.
fn phase3(app: &AppHandle, game: &str) -> Res<Vec<Check>> {
    let mut out = Vec::new();
    // Tools' own (virtual) input is never heard as a person's.
    let window = range_window(app)?;
    front(window)?;
    let (tx, rx) = std::sync::mpsc::channel();
    let listening = sys::listen(false, tx)?;
    let c = controller(window)?;
    script::run(&c, &script::parse("hold w 200; move 40 0; click left").unwrap());
    std::thread::sleep(Duration::from_millis(300));
    drop(listening);
    let heard = rx.try_iter().count();
    out.push(check(
        "Tools never records its own input as yours",
        heard == 0,
        format!(
            "sent W, a mouse move and a click to the range; the recorder's listener heard {heard} of them"
        ),
    ));

    let data = data_dir(app)?;
    let p = profile::load(&data, game)?;
    let latest = recording::list(&data, game).into_iter().find(|r| r.player == "person");
    let Some(r) = latest else {
        out.push(check(
            "Your latest recording",
            false,
            format!(
                "no recording of you playing {} yet: record half a minute on 2.3, then run this again",
                p.name
            ),
        ));
        return Ok(out);
    };
    let v = recording::validate(&data, game, &r.id)?;
    out.push(check(
        "Your latest recording reads back and matches your raw input",
        v.ok,
        format!(
            "{} pictures; unreadable {}; steps not matching the raw input {}",
            v.ticks, v.pictures_bad, v.ticks_disagreeing
        ),
    ));
    let fps_ok = v.fps >= p.capture.fps as f64 * 0.8;
    out.push(check(
        "It kept up the picture rate",
        fps_ok,
        format!(
            "{:.1} pictures a second of {} asked for; longest gap {:.0} ms",
            v.fps, p.capture.fps, v.longest_gap_ms
        ),
    ));
    let used: Vec<String> =
        v.active_ticks.iter().filter(|x| x.1 > 0).map(|x| format!("{} ({} steps)", x.0, x.1)).collect();
    out.push(check(
        "Your controls were recorded",
        !used.is_empty(),
        if used.is_empty() {
            "no control was active in any step: was the game in front while you played?".to_string()
        } else {
            used.join(", ")
        },
    ));
    if p.mouse {
        out.push(check(
            "Your mouse movement was recorded",
            v.mouse_x.abs() + v.mouse_y.abs() > 0.0,
            format!("{:.0} units across, {:.0} down in all", v.mouse_x, v.mouse_y),
        ));
    }
    Ok(out)
}

/// Phase 4: the network's maths, saving, learning, and playing through the safety layer.
fn phase4(app: &AppHandle, game: &str, player_id: Option<&str>) -> Res<Vec<Check>> {
    let mut out = Vec::new();
    let (checked, worst) = gradient_check(42);
    out.push(check("The network's gradients are right", worst < 2e-2, format!("{checked} weights in every kind of layer compared with finite differences; worst relative error {worst:.4}")));

    let a = check_arch();
    let p = Policy::new(a.clone(), 3)?;
    let q = Policy::from_bytes(a.clone(), &p.to_bytes())?;
    let (inputs, _) = random_sequence(&a, 4, 1);
    let m = vec![0f32; a.memory];
    let g = vec![0f32; a.goal];
    let same = p.forward(&inputs[0], &m, &g).logits == q.forward(&inputs[0], &m, &g).logits;
    out.push(check(
        "Saved weights load back exactly",
        same,
        "saved, loaded, and compared the outputs for the same picture",
    ));

    let data = data_dir(app)?;
    let prof = profile::load(&data, game)?;
    match player_id {
        Some(id) => {
            let info = player::load_info(&data, game, id)?;
            player::fits(&info, &prof)?;
            let clips = train::load_clips(&data, &info, Split::Train)?;
            if clips.is_empty() {
                out.push(check(
                    "Training lowers the loss on your recordings",
                    false,
                    "no recordings of you marked for training",
                ));
            } else {
                // A short training on a copy (nothing is saved): the loss on a fixed batch must fall.
                let (mut policy, _) = player::load_latest(&data, &info)?;
                let mut adam = super::policy::Adam::new(policy.parameters());
                let mut batch = Vec::new();
                for c in clips.iter() {
                    let n = c.targets.len();
                    for start in (0..n.saturating_sub(10)).step_by((n / 4).max(1)).take(4) {
                        let inputs: Vec<Vec<f32>> = (start..start + 6)
                            .map(|t| {
                                super::policy::encode(
                                    &info.arch,
                                    &[&c.pictures[t.saturating_sub(1)], &c.pictures[t]],
                                )
                            })
                            .collect();
                        let targets = (start..start + 6).map(|t| c.targets[t + 1]).collect::<Vec<_>>();
                        batch.push((inputs, targets));
                    }
                }
                let loss = |p: &Policy| {
                    batch
                        .iter()
                        .map(|(i, t)| p.sequence_loss(i, t, &mut vec![0f32; p.parameters()], 1.0).total)
                        .sum::<f64>()
                        / batch.len() as f64
                };
                let first = loss(&policy);
                for _ in 0..30 {
                    let mut grad = vec![0f32; policy.parameters()];
                    for (i, t) in &batch {
                        policy.sequence_loss(i, t, &mut grad, 1.0 / batch.len() as f32);
                    }
                    adam.step(&mut policy, &grad, 1e-3, 0.0);
                }
                let last = loss(&policy);
                out.push(check("Training lowers the loss on your recordings", last < first, format!("{} sequences from your recordings: loss {first:.3} → {last:.3} after 30 steps (on a copy; nothing saved)", batch.len())));
            }
        }
        None => out.push(check(
            "Training lowers the loss on your recordings",
            false,
            "make a player on 2.5 first, then choose it here",
        )),
    }

    // A network plays the range through the safety layer, and stopping lets go of everything.
    let window = range_window(app)?;
    front(window)?;
    let rp = profile::range_profile();
    let arch = super::policy::Arch::new(
        rp.capture.width as usize,
        rp.capture.height as usize,
        3,
        rp.keys.len(),
        rp.buttons.len(),
        vec![-20.0, 0.0, 20.0],
        super::policy::Size::Small,
    );
    let mut net = Policy::new(arch, 7)?;
    let kb = net.slot("key.b").map(|s| s.range()).unwrap_or(0..0);
    // Bias it to walk, so its decisions show in the range.
    net.w[kb.start] += 6.0;
    let mut brain = Brain::net(net);
    let s0 = range_state(app)?;
    let c = controller(window)?;
    let mut desk = sys::open()?;
    let setup = autoplay::Setup {
        data: &data,
        profile: &rp,
        window,
        decide: Decide::Likely,
        record_as: None,
        seconds: Some(3.0),
        seed: 1,
    };
    let played = autoplay::play(setup, &mut brain, desk.as_mut(), &c, &AtomicBool::new(false), |_| {})?;
    std::thread::sleep(Duration::from_millis(200));
    let s1 = range_state(app)?;
    let moved = (num(&s1, "x") - num(&s0, "x")).hypot(num(&s1, "y") - num(&s0, "y"));
    let held_after =
        s1["held"].as_object().map(|o| o.values().any(|v| v.as_bool() == Some(true))).unwrap_or(false);
    out.push(check(
        "A network plays the range through the safety layer",
        played.ticks > 30 && (num(&s1, "presses") > num(&s0, "presses")) && !held_after,
        format!("{} decisions in 3 s ({}); the range received {} presses and moved {moved:.1} cells; held afterwards: {held_after}", played.ticks, played.ended, num(&s1, "presses") - num(&s0, "presses")),
    ));
    Ok(out)
}

/// Runs a phase's automatic checks (2, 3 or 4). `game` and `player` are for phases 3 and 4.
#[tauri::command]
pub async fn checks_run(
    app: AppHandle,
    phase: u8,
    game: Option<String>,
    player: Option<String>,
) -> Res<Vec<Check>> {
    let live = app.state::<Live>();
    if live.running.swap(true, Ordering::AcqRel) {
        return Err("Something is already running (actions, a recording, or play)".into());
    }
    let game = game.unwrap_or_else(|| "yendory-range".into());
    let result = match phase {
        2 => phase2(&app),
        3 => phase3(&app, &game),
        4 => phase4(&app, &game, player.as_deref()),
        _ => Err("No such phase".into()),
    };
    live.running.store(false, Ordering::Release);
    result
}

/// Waits (15 s at most) for the emergency stop keys to be pressed.
#[tauri::command]
pub async fn checks_stop_keys() -> Check {
    let before = STOPS.load(Ordering::Acquire);
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if STOPS.load(Ordering::Acquire) != before {
            return check("The emergency stop keys work", true, "Ctrl+Alt+End was seen");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    check("The emergency stop keys work", false, "Ctrl+Alt+End was not seen within 15 seconds")
}
