//! Commands for Gaming Nova: the open model's game skills, training them (generation 0), play,
//! and measuring on the range. Skills live inside the model (a Nova Neuro or Bicameral model made in Yendory
//! Hub); Tools reads and saves them through the Network (`skills.rs`).
use super::autoplay::{self, Brain, Decide};
use super::evaluate::{self, EvalSetup};
use super::generations;
use super::policy::{Size, SkillSpec};
use super::recording::Split;
use super::safety::Controller;
use super::train::{self, Settings};
use super::{alive, cortex, data_dir, player, profile, recording, skills, sys, Live};
use crate::Res;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

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

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillsView {
    /// The open model (none: open one), whether it can keep skills, and why not.
    model: Option<crate::OpenModel>,
    problem: Option<String>,
    /// The open model's skills (all games).
    skills: Vec<SkillSpec>,
    /// Players kept in Tools by 0.1.4 for this game (to move into a model).
    old_players: Vec<player::PlayerInfo>,
    /// The person's recordings of the game that can teach (training) and measure (testing).
    training: Recs,
    testing: Recs,
    training_now: bool,
    generations_now: bool,
    /// A new skill of each size for the game as it is now: its weights, and why it can't be
    /// kept in a model (None: it can).
    sizes: Vec<SizeView>,
    /// The CPU threads training uses (all of them).
    threads: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SizeView {
    size: Size,
    weights: usize,
    problem: Option<String>,
}

#[tauri::command]
pub async fn skills_view(app: AppHandle, id: String) -> Res<SkillsView> {
    let data = data_dir(&app)?;
    let p = profile::load(&data, &id)?;
    let sizes = skills::weights_for(&data, &p)
        .into_iter()
        .map(|(size, weights)| SizeView { size, weights, problem: skills::fits_network(weights, &p).err() })
        .collect();
    let model = crate::open_model_now(&app);
    let problem = skills::open_model(&app).err();
    let skills = if problem.is_none() { skills::list(&app)? } else { vec![] };
    Ok(SkillsView {
        model,
        problem,
        skills,
        old_players: player::list(&data, &id),
        training: recs(&data, &id, Split::Train),
        testing: recs(&data, &id, Split::Test),
        training_now: app.state::<Live>().training_stop.lock().unwrap().is_some(),
        generations_now: app.state::<Live>().generations.load(Ordering::Acquire),
        sizes,
        threads: train::cpu_threads(),
    })
}

/// Gives the open model a new, untrained skill for the profile's game.
#[tauri::command]
pub async fn skill_create(app: AppHandle, id: String, size: Size) -> Res<SkillSpec> {
    let data = data_dir(&app)?;
    let p = profile::load(&data, &id)?;
    let m = skills::open_model(&app)?;
    if skills::list(&app)?.iter().any(|k| k.game == id) {
        return Err(format!("{} already has a skill for {}", m.name, p.name));
    }
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);
    let skill = skills::new_skill(&data, &p, size, seed)?;
    skills::fits_network(skill.accepted.len(), &p)?;
    crate::debug::info(format!(
        "Adding a {:?} skill for {} to {}: {} weights",
        size,
        p.name,
        m.name,
        skill.accepted.len()
    ));
    skills::save(&app, &m.id, &skill)?;
    Ok(skill.spec)
}

/// Removes the open model's skill for a game, after asking. Ok(false) when the person said no.
#[tauri::command]
pub async fn skill_delete(app: AppHandle, id: String) -> Res<bool> {
    if app.state::<Live>().busy_with_model() {
        return Err("Wait for the training or play to end first".into());
    }
    let m = skills::open_model(&app)?;
    let p = profile::load(&data_dir(&app)?, &id)?;
    let msg = format!("Remove {}'s skill for {}? Its training is lost; it can't be undone.", m.name, p.name);
    if !super::ask(&app, "Remove the game skill", &msg, "Remove") {
        return Ok(false);
    }
    skills::delete(&app, &id).map(|_| true)
}

/// Moves a player kept in Tools (0.1.4) into the open model as its skill for the game, after
/// asking; the player's folder is removed once the model has it. Ok(false) when the person said no.
#[tauri::command]
pub async fn player_move(app: AppHandle, id: String, player: String) -> Res<bool> {
    let data = data_dir(&app)?;
    let info = player::load_info(&data, &id, &player)?;
    let m = skills::open_model(&app)?;
    if skills::list(&app)?.iter().any(|k| k.game == id) {
        return Err(format!(
            "{} already has a skill for this game: remove it first to move {} in",
            m.name, info.name
        ));
    }
    let msg = format!("Move the player “{}” into {} as its skill for this game? It is then kept in the model, not in Tools.", info.name, m.name);
    if !super::ask(&app, "Move into the model", &msg, "Move") {
        return Ok(false);
    }
    let skill = player::to_skill(&data, &info)?;
    skills::save(&app, &m.id, &skill)?;
    player::delete(&data, &id, &player)?;
    Ok(true)
}

/// Deletes a player kept in Tools (0.1.4), after asking.
#[tauri::command]
pub async fn player_delete(app: AppHandle, id: String, player: String) -> Res<bool> {
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

/// How a linked skill's core is trained (from the page).
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CoreChoice {
    /// The core's weights learn too (else it plays its part, frozen).
    pub train: bool,
    pub learning_rate: f32,
}

/// The stretch a linked skill was last trained with (the core starts again every stretch).
fn stretch_of(spec: &SkillSpec) -> usize {
    train::runs(spec).last().map(|r| r.settings.length).unwrap_or(Settings::default().length)
}

/// The open model's core, for a linked skill (None for a skill without a link).
fn core_for(app: &AppHandle, model_id: &str, spec: &SkillSpec) -> Res<Option<yendory_engine::Bicameral>> {
    if spec.arch.cortex.is_none() {
        return Ok(None);
    }
    cortex::load(app, model_id).map(Some)
}

/// Links the open model's skill for the game to the model's own core (`on`), or removes the
/// link (after the person agrees: the link's training is lost). The skill's own weights stay.
#[tauri::command]
pub async fn skill_link(app: AppHandle, id: String, on: bool) -> Res<bool> {
    let m = skills::open_model(&app)?;
    let skill = skills::get(&app, &id)?;
    let next = if on {
        if skill.spec.arch.cortex.is_some() {
            return Err("The skill is already linked to the core".into());
        }
        let core = cortex::load(&app, &m.id)?;
        let (width, right) = core.sense_widths();
        skill.with_cortex(Some(yendory_engine::skill::CortexLink { width, right }))?
    } else {
        if skill.spec.arch.cortex.is_none() {
            return Ok(true);
        }
        let msg = format!(
            "Remove the link between {}'s skill for {} and its core? What the link has learned is lost; the skill's own weights stay.",
            m.name, skill.spec.name
        );
        if !super::ask(&app, "Remove the link to the core", &msg, "Remove") {
            return Ok(false);
        }
        skill.with_cortex(None)?
    };
    skills::save(&app, &m.id, &next)?;
    crate::debug::info(format!(
        "{} {}'s skill for {}: {} weights",
        if on { "Linked to the core" } else { "Unlinked from the core" },
        m.name,
        next.spec.name,
        next.accepted.len()
    ));
    Ok(true)
}

/// Trains the open model's skill for the game; every measurement that is kept is saved into the
/// model through the Network. Progress: `train-progress`; the end: `train-done` (the skill's
/// description) or `train-failed`.
#[tauri::command]
pub async fn train_start(
    app: AppHandle,
    id: String,
    settings: Settings,
    fresh: bool,
    core: Option<CoreChoice>,
) -> Res<()> {
    settings.check()?;
    let data = data_dir(&app)?;
    let p = profile::load(&data, &id)?;
    let m = skills::open_model(&app)?;
    let skill = skills::get(&app, &id)?;
    skills::fits(&skill.spec, &p)?;
    let choice = core.unwrap_or_default();
    let core = match core_for(&app, &m.id, &skill.spec)? {
        Some(model) => {
            if choice.train && !(1e-7..=1e-2).contains(&choice.learning_rate) {
                return Err("The core's learning rate: 0.0000001 to 0.01".into());
            }
            Some(train::CoreTraining { model, train: choice.train, learning_rate: choice.learning_rate })
        }
        None => None,
    };
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
        let (a3, a4) = (app2.clone(), app2.clone());
        let game = id.clone();
        let model_id = m.id.clone();
        let result = train::train(
            &data,
            &p,
            skill,
            settings,
            fresh,
            core,
            &stop,
            move |pr| {
                let _ = a3.emit("train-progress", serde_json::json!({"game": game, "progress": pr}));
            },
            move |k, core| match core {
                // The core learned too: the whole model, with this skill in it.
                Some(c) => {
                    let mut whole = c.clone();
                    match whole.skills.iter_mut().find(|s| s.spec.game == k.spec.game) {
                        Some(s) => *s = k.clone(),
                        None => whole.skills.push(k.clone()),
                    }
                    cortex::save(&a4, &model_id, &whole).map(|_| ())
                }
                None => skills::save(&a4, &model_id, k).map(|_| ()),
            },
        );
        app2.state::<Live>().training_stop.lock().unwrap().take();
        match result {
            Ok(k) => {
                let _ = app2.emit("train-done", k.spec);
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

/// Lets the open model play the profile's game by itself with its skill for it, through the
/// safety layer, until stopped, after the person allows it in a native dialog (every time).
/// Ok(false) when they said no.
#[tauri::command]
pub async fn play_start(app: AppHandle, id: String, decide: Decide, record: bool) -> Res<bool> {
    let data = data_dir(&app)?;
    let p = profile::load(&data, &id)?;
    let m = skills::open_model(&app)?;
    let skill = skills::get(&app, &id)?;
    skills::fits(&skill.spec, &p)?;
    let policy = skill.policy()?;
    let core = core_for(&app, &m.id, &skill.spec)?;
    let stretch = stretch_of(&skill.spec);
    let msg = format!(
        "Allow {} to control {} with the keyboard and mouse?\n\nOnly the game's permitted controls, only while it is in front. Ctrl+Alt+End stops it at once.",
        m.name, p.name
    );
    if !super::ask(&app, "Let a network play", &msg, "Allow") {
        return Ok(false);
    }
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
        let mut brain = match core {
            Some(c) => match cortex::Cortex::new(c, stretch).and_then(|cx| Brain::with_core(policy, cx)) {
                Ok(b) => b,
                Err(e) => {
                    let live = app2.state::<Live>();
                    live.active.lock().unwrap().take();
                    live.play_stop.lock().unwrap().take();
                    live.running.store(false, Ordering::Release);
                    let _ = app2.emit("play-failed", e);
                    return;
                }
            },
            None => Brain::net(policy),
        };
        let setup = autoplay::Setup {
            data: &data,
            profile: &p,
            window,
            decide,
            record_as: record.then(|| m.name.clone()),
            seconds: None,
            seed: skill.spec.steps ^ 0x5eed,
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

/// Measures the open model's range skill on the test range against the untrained network and
/// the frequency baseline, on the same unseen layouts, after the person allows it. Progress:
/// `eval-progress`; the end: `eval-done` (the rows) or `eval-failed`. The range must have been
/// started from Tools.
#[tauri::command]
pub async fn evaluate_start(app: AppHandle, episodes: usize, seconds: f64, decide: Decide) -> Res<bool> {
    let id = "yendory-range".to_string();
    let data = data_dir(&app)?;
    let p = profile::load(&data, &id)?;
    let m = skills::open_model(&app)?;
    let skill = skills::get(&app, &id)?;
    skills::fits(&skill.spec, &p)?;
    if !(1..=50).contains(&episodes) || !(5.0..=55.0).contains(&seconds) {
        return Err("Episodes: 1 to 50, each 5 to 55 seconds".into());
    }
    let msg = format!(
        "Allow {}, an untrained network and the frequency baseline to control the test range in turn ({} episodes of {} s each)?\n\nCtrl+Alt+End stops it at once.",
        m.name, episodes, seconds
    );
    if !super::ask(&app, "Measure on the range", &msg, "Allow") {
        return Ok(false);
    }
    let trained = skill.policy()?;
    let core = core_for(&app, &m.id, &skill.spec)?;
    let with_core = match core {
        Some(c) => Some(Brain::with_core(trained.clone(), cortex::Cortex::new(c, stretch_of(&skill.spec))?)?),
        None => None,
    };
    let untrained = super::policy::Policy::new(skill.spec.arch.clone(), skill.spec.seed ^ 0xabcdef)?;
    let freq = train::frequencies(&skill.spec, &train::load_clips(&data, &skill.spec, Split::Train)?);
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
        // A linked skill plays with its core, and again with the core cut (the skill alone),
        // beside the deciders that can't have learned.
        let mut brains = Vec::new();
        match with_core {
            Some(b) => {
                brains.push((format!("{} (with its core)", m.name), b));
                brains.push((format!("{} (its core cut)", m.name), Brain::net(trained)));
            }
            None => brains.push((m.name.clone(), Brain::net(trained))),
        }
        brains.push(("untrained network".to_string(), Brain::net(untrained)));
        brains.push((
            "frequency baseline (ignores the picture)".to_string(),
            Brain::Frequencies { bins: skill.spec.arch.mouse_bins.clone(), freq },
        ));
        let live = app2.state::<Live>();
        let result = {
            let mut slot = live.range.lock().unwrap();
            match alive(&mut slot) {
                Some(range) => {
                    let a3 = app2.clone();
                    let setup = EvalSetup {
                        data: &data,
                        profile: &p,
                        window,
                        episodes,
                        seconds,
                        decide,
                        first_seed: evaluate::SEEDS_FROM,
                        record: false,
                    };
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

/// Runs generations of the open model's range skill on the test range (it learns from its own
/// best play; the champion is replaced only by a network that beats it), after the person allows
/// it. Progress: `gen-progress`; the end: `gen-done` (the skill's description) or `gen-failed`.
/// The range must have been started from Tools: it reports the score.
#[tauri::command]
pub async fn generations_start(app: AppHandle, settings: generations::GenSettings) -> Res<bool> {
    settings.check()?;
    let id = "yendory-range".to_string();
    let data = data_dir(&app)?;
    let p = profile::load(&data, &id)?;
    let m = skills::open_model(&app)?;
    let skill = skills::get(&app, &id)?;
    skills::fits(&skill.spec, &p)?;
    let core = core_for(&app, &m.id, &skill.spec)?;
    let minutes = settings.play_seconds() / 60.0;
    let msg = format!(
        "Allow {} to control the test range for {} generation{} (about {:.0} minutes of play each, plus learning)?\n\nIts candidates play in turn, then the new network and the champion. Only the range's permitted controls, only while it is in front. Ctrl+Alt+End stops it at once.",
        m.name,
        settings.generations,
        if settings.generations == 1 { "" } else { "s" },
        minutes.max(1.0)
    );
    if !super::ask(&app, "Run generations", &msg, "Allow") {
        return Ok(false);
    }
    let live = app.state::<Live>();
    {
        let mut slot = live.range.lock().unwrap();
        if alive(&mut slot).is_none() {
            return Err("Start the test range from Tools first (2.1, Start the test range)".into());
        }
    }
    let stop = Arc::new(AtomicBool::new(false));
    {
        let mut slot = live.training_stop.lock().unwrap();
        if slot.is_some() {
            return Err("A training is already running".into());
        }
        if live.running.swap(true, Ordering::AcqRel) {
            return Err("Something is already running (actions, a recording, or play)".into());
        }
        *slot = Some(stop.clone());
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
        live.training_stop.lock().unwrap().take();
        live.running.store(false, Ordering::Release);
        return Err("The range's window isn't open".into());
    };
    *live.play_stop.lock().unwrap() = Some(stop.clone());
    live.generations.store(true, Ordering::Release);
    let app2 = app.clone();
    std::thread::spawn(move || {
        let live = app2.state::<Live>();
        let arena =
            generations::Arena { data: &data, profile: &p, window, range: &live.range, name: m.name.clone() };
        let (a3, a4) = (app2.clone(), app2.clone());
        let model_id = m.id.clone();
        let result = generations::run(
            &arena,
            skill,
            settings,
            core,
            &stop,
            move |pr| {
                let _ = a3.emit("gen-progress", pr);
            },
            move |k| skills::save(&a4, &model_id, k).map(|_| ()),
        );
        live.play_stop.lock().unwrap().take();
        live.training_stop.lock().unwrap().take();
        live.generations.store(false, Ordering::Release);
        live.running.store(false, Ordering::Release);
        match result {
            Ok(k) => {
                let _ = app2.emit("gen-done", k.spec);
            }
            Err(e) => {
                let _ = app2.emit("gen-failed", e);
            }
        }
    });
    Ok(true)
}

#[tauri::command]
pub fn generations_stop(live: tauri::State<'_, Live>) {
    if let Some(s) = live.training_stop.lock().unwrap().as_ref() {
        s.store(true, Ordering::Release);
    }
}
