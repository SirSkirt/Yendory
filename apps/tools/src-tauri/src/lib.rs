//! Yendory Tools backend: games (and later apps) that trained Yendory models play and use
//! through Nexus. Tools holds no model: it opens one through the Yendory Network, which writes
//! every line the model writes; the games, their perfect players, the lessons and the
//! scoreboard are here (docs/GAMES.md).
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_dialog::DialogExt;
use yendory_client::Client;

pub mod debug;
pub mod games;
mod handoff;
pub mod lessons;
pub mod live;
pub mod play;
pub mod range;
pub mod score;
pub mod turn;

use games::{Opponent, Rng};
use play::Match;

pub(crate) type Res<T> = Result<T, String>;
pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn data_dir(app: &AppHandle) -> Res<PathBuf> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// The model Tools has open (through the Network).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenModel {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) architecture: String,
    parameters: u64,
    step: u64,
    context: u64,
}

/// A Network call from anywhere in Tools (the connection is made, or remade, as needed).
/// The most the Network reads as one message (yendory-client and the Network: 64 MB a line).
pub(crate) const NETWORK_MESSAGE_LIMIT: usize = 64 << 20;

pub(crate) fn network_call(app: &AppHandle, method: &str, params: Value) -> Res<Value> {
    app.state::<Tools>().call(app, method, params)
}

/// Sends `data` to the Network in parts; the upload's id, for a method that takes one.
pub(crate) fn network_upload(app: &AppHandle, data: &[u8]) -> Res<u64> {
    let started = std::time::Instant::now();
    let client = app.state::<Tools>().network(app)?;
    let r = client.upload(data).map_err(|e| e.message);
    match &r {
        Ok(_) => debug::info(format!(
            "Network upload ({}): sent in {:.2} s",
            debug::size(data.len()),
            started.elapsed().as_secs_f64()
        )),
        Err(e) => debug::warn(format!("Network upload ({}): failed: {e}", debug::size(data.len()))),
    }
    r
}

/// Reads `total` bytes from the Network in parts (`method` takes `offset` and `length`).
pub(crate) fn network_read_parts(app: &AppHandle, method: &str, params: Value, total: usize) -> Res<Vec<u8>> {
    let started = std::time::Instant::now();
    let client = app.state::<Tools>().network(app)?;
    let r = client.read_parts(method, params, total).map_err(|e| e.message);
    match &r {
        Ok(_) => debug::info(format!(
            "Network {method} ({}): read in parts in {:.2} s",
            debug::size(total),
            started.elapsed().as_secs_f64()
        )),
        Err(e) => debug::warn(format!("Network {method} ({}): failed: {e}", debug::size(total))),
    }
    r
}

/// Downloads in parts what `method` offers (`model.checkpoint`).
pub(crate) fn network_download(app: &AppHandle, method: &str, params: Value) -> Res<Vec<u8>> {
    let started = std::time::Instant::now();
    let client = app.state::<Tools>().network(app)?;
    let r = client.download(method, params).map_err(|e| e.message);
    match &r {
        Ok(b) => debug::info(format!(
            "Network {method} ({}): downloaded in parts in {:.2} s",
            debug::size(b.len()),
            started.elapsed().as_secs_f64()
        )),
        Err(e) => debug::warn(format!("Network {method}: failed: {e}")),
    }
    r
}

/// The model Tools has open, if any.
pub(crate) fn open_model_now(app: &AppHandle) -> Option<OpenModel> {
    app.state::<Tools>().open.lock().unwrap().clone()
}

#[derive(Default)]
pub struct Tools {
    network: Mutex<Option<Arc<Client>>>,
    open: Mutex<Option<OpenModel>>,
    matches: Mutex<HashMap<u64, Match>>,
    next_match: AtomicU64,
    measuring: AtomicBool,
    stop: Arc<AtomicBool>,
    handed: Mutex<Option<handoff::Asked>>,
}

impl Tools {
    /// The connection to the Network, made (and the Network started) when needed. After a new
    /// connection the open model is opened again (a model is open per connection).
    fn network(&self, app: &AppHandle) -> Res<Arc<Client>> {
        let mut slot = self.network.lock().unwrap();
        if let Some(c) = slot.as_ref().filter(|c| c.is_alive()) {
            return Ok(c.clone());
        }
        debug::info("Connecting to the Yendory Network (starting it if needed)");
        let client = Arc::new(
            Client::start_and_connect("Yendory Tools", VERSION, Duration::from_secs(15)).map_err(|e| {
                debug::error(format!("The Yendory Network: {}", e.message));
                e.message
            })?,
        );
        let h = &client.hello;
        debug::info(format!("Connected to the Yendory Network {} (Engine {})", h.network, h.engine));
        if let Some(events) = client.take_events() {
            let app = app.clone();
            std::thread::spawn(move || {
                for e in events {
                    let _ = app.emit("network-event", json!({"name": e.name, "data": e.data}));
                }
            });
        }
        if let Some(m) = self.open.lock().unwrap().as_ref() {
            client.call("models.open", json!({"id": m.id})).map_err(|e| e.message)?;
        }
        *slot = Some(client.clone());
        Ok(client)
    }

    fn call(&self, app: &AppHandle, method: &str, params: Value) -> Res<Value> {
        // The Network reads one message of at most 64 MB; a longer one is never answered (the
        // call would wait for ever), so it is refused here with the reason.
        let bytes = serde_json::to_vec(&params).map(|v| v.len()).unwrap_or(0) + method.len() + 64;
        if bytes > NETWORK_MESSAGE_LIMIT {
            let e = format!(
                "Too large to send to the Yendory Network: {} in one message, and it takes at most {}",
                debug::size(bytes),
                debug::size(NETWORK_MESSAGE_LIMIT)
            );
            debug::error(format!("Network {method}: {e}"));
            return Err(e);
        }
        let started = std::time::Instant::now();
        let client = self.network(app)?;
        let result = match client.call(method, params.clone()) {
            Err(e) if e.code == "disconnected" => {
                debug::warn(format!("Network {method}: the connection was lost; connecting again"));
                *self.network.lock().unwrap() = None;
                self.network(app)?.call(method, params).map_err(|e| e.message)
            }
            other => other.map_err(|e| e.message),
        };
        let took = started.elapsed().as_secs_f64();
        match &result {
            Ok(_) => {
                debug::info(format!("Network {method} ({}): answered in {took:.2} s", debug::size(bytes)))
            }
            Err(e) => debug::warn(format!(
                "Network {method} ({}): failed after {took:.2} s: {e}",
                debug::size(bytes)
            )),
        }
        result
    }

    /// Writes the open model's next line after `context` (one line; whether it was cut off).
    fn line_writer<'a>(&'a self, app: &'a AppHandle) -> Res<impl FnMut(&str) -> Res<(String, bool)> + 'a> {
        let id = self.open.lock().unwrap().as_ref().map(|m| m.id.clone()).ok_or("Open a model first")?;
        Ok(move |context: &str| {
            let r = self.call(
                app,
                "model.reply",
                json!({"id": id, "prompt": context, "maxTokens": turn::LINE_TOKENS}),
            )?;
            Ok((r["text"].as_str().unwrap_or("").to_string(), r["cut"].as_bool().unwrap_or(false)))
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The Network and models
// ---------------------------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct About {
    tools: &'static str,
    network: Option<String>,
    engine: Option<String>,
    network_error: Option<String>,
    models_folder: String,
    data_folder: String,
}

#[tauri::command]
async fn about(app: AppHandle, tools: tauri::State<'_, Tools>) -> Res<About> {
    let (network, engine, network_error) = match tools.network(&app) {
        Ok(c) => (Some(c.hello.network.clone()), Some(c.hello.engine.clone()), None),
        Err(e) => (None, None, Some(e)),
    };
    Ok(About {
        tools: VERSION,
        network,
        engine,
        network_error,
        models_folder: yendory_client::paths::models_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        data_folder: data_dir(&app)?.display().to_string(),
    })
}

fn open_model_of(meta: &Value) -> OpenModel {
    let n = |k: &str| meta[k].as_u64().unwrap_or(0);
    OpenModel {
        id: meta["id"].as_str().unwrap_or("").into(),
        name: meta["name"].as_str().unwrap_or("").into(),
        architecture: meta["architecture"].as_str().unwrap_or("bicameral").into(),
        parameters: n("parameters"),
        step: n("step"),
        context: n("context"),
    }
}

/// The models in the suite's models folder, newest first.
#[tauri::command]
async fn models(app: AppHandle, tools: tauri::State<'_, Tools>) -> Res<Vec<Value>> {
    let list = tools.call(&app, "models.list", json!({}))?;
    Ok(list
        .as_array()
        .map(|a| {
            a.iter()
                .map(|e| {
                    let mut v = serde_json::to_value(open_model_of(&e["meta"])).unwrap_or_default();
                    v["openBy"] = e["openBy"].clone();
                    v
                })
                .collect()
        })
        .unwrap_or_default())
}

/// Opens a model through the Network (the one open before is closed).
#[tauri::command]
async fn open_model(app: AppHandle, tools: tauri::State<'_, Tools>, id: String) -> Res<OpenModel> {
    if tools.measuring.load(Ordering::Acquire) {
        return Err("Wait until the measurement has finished, or stop it".into());
    }
    if app.state::<live::Live>().busy_with_model() {
        return Err("Wait until the game training, play or measuring has finished, or stop it".into());
    }
    let opened = tools.call(&app, "models.open", json!({"id": id}))?;
    let model = open_model_of(&opened["meta"]);
    let before = tools.open.lock().unwrap().replace(model.clone());
    if let Some(b) = before.filter(|b| b.id != model.id) {
        let _ = tools.call(&app, "models.close", json!({"id": b.id}));
    }
    Ok(model)
}

#[tauri::command]
fn current_model(tools: tauri::State<'_, Tools>) -> Option<OpenModel> {
    tools.open.lock().unwrap().clone()
}

// ---------------------------------------------------------------------------------------------
// Games and matches
// ---------------------------------------------------------------------------------------------

#[tauri::command]
fn games() -> Vec<Value> {
    games::CATALOGUE
        .iter()
        .map(|i| {
            let mut v = serde_json::to_value(i).unwrap_or_default();
            v["announcement"] = Value::String(i.manifest().announcement());
            v
        })
        .collect()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MatchView {
    id: u64,
    game: &'static str,
    opponent: Opponent,
    finished: bool,
    outcome: Option<games::Outcome>,
    forfeit: bool,
    needs_person: bool,
    model_to_move: bool,
    legal: Vec<String>,
    board: Value,
    log: Vec<play::Entry>,
    /// The line that opens the model's next turn.
    prompt: String,
}

fn view(id: u64, m: &Match) -> MatchView {
    MatchView {
        id,
        game: m.game.info().id,
        opponent: m.opponent,
        finished: m.finished(),
        outcome: m.outcome(),
        forfeit: m.forfeit,
        needs_person: m.needs_person(),
        model_to_move: m.model_to_move(),
        legal: if m.needs_person() { m.game.legal() } else { Vec::new() },
        board: m.game.board(),
        log: m.log.clone(),
        prompt: m.prompt(),
    }
}

#[tauri::command]
fn new_match(
    tools: tauri::State<'_, Tools>,
    game: String,
    opponent: Opponent,
    model_first: bool,
) -> Res<MatchView> {
    let seed = now() ^ (tools.next_match.load(Ordering::Relaxed) << 20);
    let mut rng = Rng::new(seed);
    let g = play::new_game(&game, &mut rng, model_first, None)?;
    let id = tools.next_match.fetch_add(1, Ordering::Relaxed) + 1;
    let m = Match::new(g, opponent, rng.next_u64());
    let v = view(id, &m);
    let mut all = tools.matches.lock().unwrap();
    // Only the newest few are kept.
    all.retain(|k, _| *k + 8 > id);
    all.insert(id, m);
    Ok(v)
}

/// The next step of a match: the model's turn (written by the Network) or a computed opponent's.
#[tauri::command]
async fn match_step(app: AppHandle, tools: tauri::State<'_, Tools>, id: u64) -> Res<MatchView> {
    let mut m = tools.matches.lock().unwrap().remove(&id).ok_or("This game is no longer open")?;
    let result = if m.model_to_move() {
        tools.line_writer(&app).and_then(|w| m.model_turn(w))
    } else if m.needs_person() {
        Err("It is your move".into())
    } else if m.finished() {
        Ok(())
    } else {
        m.opponent_turn(None)
    };
    let v = view(id, &m);
    tools.matches.lock().unwrap().insert(id, m);
    result.map(|_| v)
}

#[tauri::command]
fn person_move(tools: tauri::State<'_, Tools>, id: u64, input: String) -> Res<MatchView> {
    let mut all = tools.matches.lock().unwrap();
    let m = all.get_mut(&id).ok_or("This game is no longer open")?;
    m.opponent_turn(Some(&input))?;
    Ok(view(id, m))
}

// ---------------------------------------------------------------------------------------------
// The scoreboard
// ---------------------------------------------------------------------------------------------

fn scores_file(app: &AppHandle, model: &str) -> Res<PathBuf> {
    if model.is_empty() || !model.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err("Invalid model id".into());
    }
    let dir = data_dir(app)?.join("scores");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir.join(format!("{model}.jsonl")))
}

fn read_reports(path: &Path) -> Vec<score::Report> {
    std::fs::read_to_string(path)
        .map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
        .unwrap_or_default()
}

/// Every measurement of a model, oldest first.
#[tauri::command]
fn reports(app: AppHandle, model: String) -> Res<Vec<score::Report>> {
    Ok(read_reports(&scores_file(&app, &model)?))
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct MeasureProgress {
    game: String,
    done: u32,
    total: u32,
}

/// Measures the open model on a game in the background ("measure-progress", then
/// "measure-done" with the report or "measure-failed").
#[tauri::command]
fn measure(app: AppHandle, tools: tauri::State<'_, Tools>, game: String, games: u32) -> Res<()> {
    let model = tools.open.lock().unwrap().clone().ok_or("Open a model first")?;
    games::info(&game).ok_or("Unknown game")?;
    if tools.measuring.swap(true, Ordering::AcqRel) {
        return Err("A measurement is already running".into());
    }
    tools.stop.store(false, Ordering::Release);
    let games = games.clamp(2, 200);
    std::thread::spawn(move || {
        let tools = app.state::<Tools>();
        let seed = now();
        let result = (|| -> Res<score::Report> {
            let writer = tools.line_writer(&app)?;
            let (tallies, moves, held_out) = score::measure(
                &game,
                games,
                seed,
                writer,
                |done, total| {
                    let _ = app.emit("measure-progress", MeasureProgress { game: game.clone(), done, total });
                },
                &tools.stop,
            )?;
            let report = score::Report {
                game: game.clone(),
                model: model.id.clone(),
                model_name: model.name.clone(),
                step: model.step,
                time: now(),
                seed,
                tallies,
                moves,
                held_out,
            };
            if report.tallies.iter().any(|t| t.1.games > 0) {
                use std::io::Write;
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(scores_file(&app, &model.id)?)
                    .map_err(|e| e.to_string())?;
                writeln!(f, "{}", serde_json::to_string(&report).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            }
            Ok(report)
        })();
        tools.measuring.store(false, Ordering::Release);
        match result {
            Ok(r) => {
                let _ = app.emit("measure-done", r);
            }
            Err(e) => {
                let _ = app.emit("measure-failed", e);
            }
        }
    });
    Ok(())
}

#[tauri::command]
fn stop_measure(tools: tauri::State<'_, Tools>) {
    tools.stop.store(true, Ordering::Release);
}

// ---------------------------------------------------------------------------------------------
// Lessons
// ---------------------------------------------------------------------------------------------

#[tauri::command]
async fn lessons_preview(game: String, count: usize, seed: u64) -> Res<lessons::LessonSet> {
    lessons::teacher_lessons(&game, count.clamp(1, 20_000), seed)
}

/// Writes the lessons to a .txt file the owner chooses (Studio's Classroom can teach it).
#[tauri::command]
async fn save_lessons(app: AppHandle, game: String, count: usize, seed: u64) -> Res<Option<String>> {
    let set = lessons::teacher_lessons(&game, count.clamp(1, 20_000), seed)?;
    let name = games::info(&game).map(|i| i.name).unwrap_or("Game").replace(' ', "-");
    let Some(picked) = app
        .dialog()
        .file()
        .add_filter("Lesson file", &["txt"])
        .set_file_name(format!("{name}-lessons-{}.txt", set.sessions))
        .blocking_save_file()
    else {
        return Ok(None);
    };
    let path = picked.into_path().map_err(|e| e.to_string())?;
    std::fs::write(&path, set.text.as_bytes()).map_err(|e| e.to_string())?;
    Ok(Some(path.display().to_string()))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default();
    #[cfg(desktop)]
    let builder = builder
        .plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| handoff::second_start(app, argv)));
    builder
        .plugin(tauri_plugin_dialog::init())
        .manage(Tools::default())
        .manage(live::Live::default())
        .setup(|app| {
            debug::start(app.handle());
            *app.state::<Tools>().handed.lock().unwrap() = handoff::model_argument(std::env::args());
            live::watch_stop_keys(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            debug::debug_lines,
            about,
            models,
            open_model,
            current_model,
            handoff::handed_model,
            games,
            new_match,
            match_step,
            person_move,
            reports,
            measure,
            stop_measure,
            lessons_preview,
            save_lessons,
            live::live_info,
            live::live_windows,
            live::live_profiles,
            live::live_save_profile,
            live::live_delete_profile,
            live::live_find_window,
            live::live_front,
            live::live_preview,
            live::live_run,
            live::live_stop,
            live::range_start,
            live::range_state,
            live::range_reset,
            live::range_stop,
            live::record_start,
            live::record_stop,
            live::recordings,
            live::recording_ticks,
            live::recording_frame,
            live::recording_validate,
            live::recording_update,
            live::recording_delete,
            live::nova::skills_view,
            live::nova::skill_create,
            live::nova::skill_delete,
            live::nova::player_move,
            live::nova::player_delete,
            live::nova::train_start,
            live::nova::train_stop,
            live::nova::play_start,
            live::nova::play_stop,
            live::nova::evaluate_start,
            live::nova::skill_link,
            live::nova::generations_start,
            live::nova::generations_stop,
            live::checks::checks_run,
            live::checks::checks_stop_keys,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Yendory Tools");
}
