//! Commands for Gaming Nova: players, training (generation 0), play, and measuring on the range.
use super::autoplay::{self, Brain, Decide};
use super::evaluate::{self, EvalSetup};
use super::policy::Size;
use super::recording::Split;
use super::safety::Controller;
use super::train::{self, Settings};
use super::{alive, data_dir, player, profile, recording, sys, Live};
use crate::Res;
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Players {
    players: Vec<player::PlayerInfo>,
    /// The person's recordings of the game that can teach (training) and measure (testing).
    training: Recs,
    testing: Recs,
    training_now: Option<String>,
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Recs {
    count: usize,
    ticks: u64,
    seconds: f64,
}

fn recs(data: &std::path::Path, id: &str, split: Split) -> Recs {
    let mut r = Recs::default();
    for i in recording::list(data, id).into_iter().filter(|i| i.player == "person" && i.split == split) {
        r.count += 1;
        r.ticks += i.ticks;
        r.seconds += (i.duration_ms.saturating_sub(i.paused_ms)) as f64 / 1000.0;
    }
    r
}

#[tauri::command]
pub async fn players(app: AppHandle, id: String) -> Res<Players> {
    let data = data_dir(&app)?;
    profile::load(&data, &id)?;
    let live = app.state::<Live>();
    let training_now = live.training_stop.lock().unwrap().as_ref().map(|_| String::new());
    Ok(Players {
        players: player::list(&data, &id),
        training: recs(&data, &id, Split::Train),
        testing: recs(&data, &id, Split::Test),
        training_now,
    })
}

#[tauri::command]
pub async fn player_create(app: AppHandle, id: String, name: String, size: Size) -> Res<player::PlayerInfo> {
    let data = data_dir(&app)?;
    let p = profile::load(&data, &id)?;
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);
    player::create(&data, &p, &name, size, seed)
}

/// Deletes a player after asking. Ok(false) when the person said no.
#[tauri::command]
pub async fn player_delete(app: AppHandle, id: String, player: String) -> Res<bool> {
    if app.state::<Live>().training_stop.lock().unwrap().is_some() {
        return Err("Wait for the training to end first".into());
    }
    let data = data_dir(&app)?;
    let info = player::load_info(&data, &id, &player)?;
    if !super::ask(
        &app,
        "Delete the player",
        &format!("Delete the player “{}” and its weights? It can't be undone.", info.name),
        "Delete",
    ) {
        return Ok(false);
    }
    player::delete(&data, &id, &player).map(|_| true)
}

/// Starts training a player; progress arrives as `train-progress`, the end as `train-done`
/// (the player's info) or `train-failed`.
#[tauri::command]
pub async fn train_start(
    app: AppHandle,
    id: String,
    player: String,
    settings: Settings,
    fresh: bool,
) -> Res<()> {
    settings.check()?;
    let data = data_dir(&app)?;
    let p = profile::load(&data, &id)?;
    let info = player::load_info(&data, &id, &player)?;
    player::fits(&info, &p)?;
    let live = app.state::<Live>();
    let stop = Arc::new(AtomicBool::new(false));
    {
        let mut slot = live.training_stop.lock().unwrap();
        if slot.is_some() {
            return Err("A training is already running".into());
        }
        *slot = Some(stop.clone());
    }
    let app2 = app.clone();
    std::thread::spawn(move || {
        let a3 = app2.clone();
        let who = player.clone();
        let result = train::train(&data, &p, &player, settings, fresh, &stop, move |pr| {
            let _ = a3.emit("train-progress", serde_json::json!({"player": who, "progress": pr}));
        });
        app2.state::<Live>().training_stop.lock().unwrap().take();
        match result {
            Ok(info) => {
                let _ = app2.emit("train-done", info);
            }
            Err(e) => {
                let _ = app2.emit("train-failed", e);
            }
        }
    });
    Ok(())
}

#[tauri::command]
pub fn train_stop(live: tauri::State<'_, Live>) {
    if let Some(s) = live.training_stop.lock().unwrap().as_ref() {
        s.store(true, Ordering::Release);
    }
}

/// What play needs before it starts: the desktop, the game's window, the safety layer.
type Started = (Box<dyn sys::Desktop>, u64, Arc<Mutex<Controller>>);

/// Lets a player play the profile's game by itself, through the safety layer, until stopped,
/// after the person allows it in a native dialog (every time). Ok(false) when they said no.
#[tauri::command]
pub async fn play_start(
    app: AppHandle,
    id: String,
    player: String,
    decide: Decide,
    record: bool,
) -> Res<bool> {
    let data = data_dir(&app)?;
    let p = profile::load(&data, &id)?;
    let info = player::load_info(&data, &id, &player)?;
    player::fits(&info, &p)?;
    let msg = format!(
        "Allow {} to control {} with the keyboard and mouse?\n\nOnly the game's permitted controls, only while it is in front. Ctrl+Alt+End stops it at once.",
        info.name, p.name
    );
    if !super::ask(&app, "Let a network play", &msg, "Allow") {
        return Ok(false);
    }
    let policy = player::load_policy(&data, &info)?;
    let live = app.state::<Live>();
    if live.running.swap(true, Ordering::AcqRel) {
        return Err("Something is already running (actions, a recording, or play)".into());
    }
    let started = (|| -> Res<Started> {
        let mut desk = sys::open()?;
        let window =
            desk.windows()?.into_iter().find(|w| p.matches(w)).ok_or("The game's window isn't open")?.id;
        desk.bring_to_front(window)?;
        for _ in 0..50 {
            if desk.in_front(window) {
                break;
            }
            std::thread::sleep(Duration::from_millis(30));
        }
        if !desk.in_front(window) {
            return Err("The game did not come to the front: click on it, then try again".into());
        }
        let c = Arc::new(Mutex::new(Controller::new(sys::open()?, window, p.permitted()?, p.limits.clone())));
        Ok((desk, window, c))
    })();
    let (mut desk, window, controller) = match started {
        Ok(x) => x,
        Err(e) => {
            live.running.store(false, Ordering::Release);
            return Err(e);
        }
    };
    let stop = Arc::new(AtomicBool::new(false));
    *live.play_stop.lock().unwrap() = Some(stop.clone());
    *live.active.lock().unwrap() = Some(controller.clone());
    let app2 = app.clone();
    std::thread::spawn(move || {
        let mut brain = Brain::net(policy);
        let setup = autoplay::Setup {
            data: &data,
            profile: &p,
            window,
            decide,
            record_as: record.then(|| info.name.clone()),
            seconds: None,
            seed: info.steps ^ 0x5eed,
        };
        let a3 = app2.clone();
        let result = autoplay::play(setup, &mut brain, desk.as_mut(), &controller, &stop, move |s| {
            let _ = a3.emit("play-status", s);
        });
        let live = app2.state::<Live>();
        live.active.lock().unwrap().take();
        live.play_stop.lock().unwrap().take();
        drop(controller);
        live.running.store(false, Ordering::Release);
        match result {
            Ok(r) => {
                let _ = app2.emit("play-done", r);
            }
            Err(e) => {
                let _ = app2.emit("play-failed", e);
            }
        }
    });
    Ok(true)
}

#[tauri::command]
pub fn play_stop(live: tauri::State<'_, Live>) {
    if let Some(s) = live.play_stop.lock().unwrap().as_ref() {
        s.store(true, Ordering::Release);
    }
}

/// Measures a player on the test range against the untrained network and the frequency
/// baseline, on the same unseen layouts. Progress: `eval-progress`; the end: `eval-done` (the
/// rows) or `eval-failed`. The range must have been started from Tools.
#[tauri::command]
pub async fn evaluate_start(
    app: AppHandle,
    player: String,
    episodes: usize,
    seconds: f64,
    decide: Decide,
) -> Res<bool> {
    let id = "yendory-range".to_string();
    let data = data_dir(&app)?;
    let p = profile::load(&data, &id)?;
    let info = player::load_info(&data, &id, &player)?;
    player::fits(&info, &p)?;
    if !(1..=50).contains(&episodes) || !(5.0..=55.0).contains(&seconds) {
        return Err("Episodes: 1 to 50, each 5 to 55 seconds".into());
    }
    let msg = format!(
        "Allow {}, an untrained network and the frequency baseline to control the test range in turn ({} episodes of {} s each)?\n\nCtrl+Alt+End stops it at once.",
        info.name, episodes, seconds
    );
    if !super::ask(&app, "Measure on the range", &msg, "Allow") {
        return Ok(false);
    }
    let trained = player::load_policy(&data, &info)?;
    let untrained = super::policy::Policy::new(info.arch.clone(), info.seed ^ 0xabcdef)?;
    let freq = train::frequencies(&info, &train::load_clips(&data, &info, Split::Train)?);
    let live = app.state::<Live>();
    {
        let mut slot = live.range.lock().unwrap();
        if alive(&mut slot).is_none() {
            return Err("Start the test range from Tools first (2.1, Start the test range)".into());
        }
    }
    if live.running.swap(true, Ordering::AcqRel) {
        return Err("Something is already running (actions, a recording, or play)".into());
    }
    let window = {
        let pid = live.range.lock().unwrap().as_ref().map(|r| r.pid()).unwrap_or(0);
        sys::open()
            .and_then(|mut d| d.windows())
            .ok()
            .and_then(|ws| ws.into_iter().find(|w| p.matches(w) && w.pid == pid))
            .map(|w| w.id)
    };
    let Some(window) = window else {
        live.running.store(false, Ordering::Release);
        return Err("The range's window isn't open".into());
    };
    let stop = Arc::new(AtomicBool::new(false));
    *live.play_stop.lock().unwrap() = Some(stop.clone());
    let app2 = app.clone();
    std::thread::spawn(move || {
        let mut brains = vec![
            (info.name.clone(), Brain::net(trained)),
            ("untrained network".to_string(), Brain::net(untrained)),
            (
                "frequency baseline (ignores the picture)".to_string(),
                Brain::Frequencies { bins: info.arch.mouse_bins.clone(), freq },
            ),
        ];
        let live = app2.state::<Live>();
        let result = {
            let mut slot = live.range.lock().unwrap();
            match alive(&mut slot) {
                Some(range) => {
                    let a3 = app2.clone();
                    let setup = EvalSetup { data: &data, profile: &p, window, episodes, seconds, decide };
                    evaluate::evaluate(setup, range, &mut brains, &stop, move |pr| {
                        let _ = a3.emit("eval-progress", pr);
                    })
                }
                None => Err("The range closed".to_string()),
            }
        };
        live.play_stop.lock().unwrap().take();
        live.running.store(false, Ordering::Release);
        match result {
            Ok(rows) => {
                let _ = app2.emit("eval-done", rows);
            }
            Err(e) => {
                let _ = app2.emit("eval-failed", e);
            }
        }
    });
    Ok(true)
}
