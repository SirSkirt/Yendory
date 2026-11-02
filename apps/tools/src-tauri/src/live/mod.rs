//! Real games (docs/GAME_TRAINING.md): game profiles, the game's window and picture, virtual
//! keyboard and mouse input through the safety layer, the emergency stop, and the test range.
pub mod controls;
pub mod profile;
pub mod safety;
pub mod script;
pub mod sys;

use crate::Res;
use profile::Profile;
use safety::{Controller, STOPS};
use serde::Serialize;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};

/// RGB to grey (ITU-R BT.601 weights).
pub fn to_gray(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|p| ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114 + 500) / 1000) as u8)
        .collect()
}

/// The range's last reported state and how many it has reported, with a signal for each new one.
type Reports = Arc<(Mutex<(u64, Option<Value>)>, Condvar)>;

/// The test range, run as a child process; its reports arrive as JSON lines.
pub struct RangeProcess {
    child: Child,
    stdin: ChildStdin,
    /// The last `state` it reported, and how many it has reported.
    state: Reports,
}

impl RangeProcess {
    /// Starts `exe --range` (Tools itself) and passes its reports to `on_event`.
    pub fn start(
        exe: &std::path::Path,
        seed: u64,
        on_event: impl Fn(Value) + Send + 'static,
    ) -> Res<RangeProcess> {
        let mut child = Command::new(exe)
            .args(["--range", "--seed", &seed.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("The range did not start: {e}"))?;
        let stdin = child.stdin.take().ok_or("The range has no input")?;
        let stdout = child.stdout.take().ok_or("The range has no output")?;
        let state: Reports = Arc::default();
        let s = state.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
                if v["ev"] == "state" {
                    let mut g = s.0.lock().unwrap();
                    g.0 += 1;
                    g.1 = Some(v["state"].clone());
                    s.1.notify_all();
                } else {
                    on_event(v);
                }
            }
        });
        Ok(RangeProcess { child, stdin, state })
    }

    pub fn running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn send(&mut self, line: &str) -> Res<()> {
        writeln!(self.stdin, "{line}")
            .and_then(|_| self.stdin.flush())
            .map_err(|_| "The range has closed".to_string())
    }

    /// Asks for the truth and waits for it (a second at most).
    pub fn state(&mut self) -> Res<Value> {
        let before = self.state.0.lock().unwrap().0;
        self.send("state")?;
        let (lock, cv) = &*self.state;
        let (g, _) =
            cv.wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(1), |g| g.0 == before).unwrap();
        if g.0 == before {
            return Err("The range did not answer".into());
        }
        Ok(g.1.clone().unwrap_or(Value::Null))
    }

    pub fn reset(&mut self, seed: u64) -> Res<()> {
        self.send(&format!("reset {seed}"))
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for RangeProcess {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < deadline {
            if !matches!(self.child.try_wait(), Ok(None)) {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Default)]
pub struct Live {
    /// The controller of the actions running now (the emergency stop releases through it).
    active: Mutex<Option<Arc<Mutex<Controller>>>>,
    running: AtomicBool,
    range: Mutex<Option<RangeProcess>>,
    /// Whether the emergency stop keys are watched (the reason when not).
    stop_keys: Mutex<Option<Res<()>>>,
}

impl Live {
    /// The emergency stop: everything held is released and everything running ends.
    pub fn emergency_stop(&self) {
        STOPS.fetch_add(1, Ordering::AcqRel);
        let active = self.active.lock().unwrap().clone();
        if let Some(c) = active {
            c.lock().unwrap().release_all("the emergency stop was pressed");
        }
    }
}

/// Starts watching the emergency stop keys (at start).
pub fn watch_stop_keys(app: &AppHandle) {
    let handle = app.clone();
    let result = sys::stop_key_watch(Box::new(move || {
        handle.state::<Live>().emergency_stop();
        let _ = handle.emit("live-stopped", ());
    }));
    *app.state::<Live>().stop_keys.lock().unwrap() = Some(result);
}

fn data_dir(app: &AppHandle) -> Res<std::path::PathBuf> {
    crate::data_dir(app)
}

// ---------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveInfo {
    keys: &'static [controls::KeyInfo],
    buttons: Vec<&'static str>,
    stop_keys: &'static str,
    /// Empty when the stop keys are watched; otherwise why not.
    stop_keys_problem: String,
    running: bool,
    held: Vec<&'static str>,
    platform: &'static str,
}

#[tauri::command]
pub fn live_info(live: tauri::State<'_, Live>) -> LiveInfo {
    let problem = match &*live.stop_keys.lock().unwrap() {
        Some(Ok(())) => String::new(),
        Some(Err(e)) => e.clone(),
        None => "not started".into(),
    };
    let held = live.active.lock().unwrap().as_ref().map(|c| c.lock().unwrap().held()).unwrap_or_default();
    LiveInfo {
        keys: controls::KEYS,
        buttons: controls::Button::ALL.iter().map(|b| b.name()).collect(),
        stop_keys: controls::STOP_KEYS,
        stop_keys_problem: problem,
        running: live.running.load(Ordering::Acquire),
        held,
        platform: if cfg!(windows) { "windows" } else { "x11" },
    }
}

#[tauri::command]
pub async fn live_windows() -> Res<Vec<sys::WindowInfo>> {
    let mut w = sys::open()?.windows()?;
    w.sort_by_key(|w| w.title.to_lowercase());
    Ok(w)
}

#[tauri::command]
pub fn live_profiles(app: AppHandle) -> Res<Vec<Profile>> {
    Ok(profile::list(&data_dir(&app)?))
}

#[tauri::command]
pub fn live_save_profile(app: AppHandle, profile: Profile) -> Res<Profile> {
    profile::save(&data_dir(&app)?, profile)
}

#[tauri::command]
pub fn live_delete_profile(app: AppHandle, id: String) -> Res<()> {
    profile::delete(&data_dir(&app)?, &id)
}

/// The profile's window, if it is open now.
#[tauri::command]
pub async fn live_find_window(app: AppHandle, id: String) -> Res<Option<sys::WindowInfo>> {
    let p = profile::load(&data_dir(&app)?, &id)?;
    Ok(sys::open()?.windows()?.into_iter().find(|w| p.matches(w)))
}

#[tauri::command]
pub async fn live_front(window: u64) -> Res<()> {
    sys::open()?.bring_to_front(window)
}

/// The window's picture as the network would see it: 8 bytes (width and height as 16-bit
/// little-endian numbers, channels, then the milliseconds the capture took), then the pixels.
#[tauri::command]
pub async fn live_preview(window: u64, width: u32, height: u32, gray: bool) -> Res<tauri::ipc::Response> {
    let (width, height) = (width.clamp(16, 640), height.clamp(9, 480));
    let started = Instant::now();
    let rgb = sys::open()?.grab(window, width, height)?;
    let ms = started.elapsed().as_millis().min(255) as u8;
    let px = if gray { to_gray(&rgb) } else { rgb };
    let mut out = Vec::with_capacity(8 + px.len());
    out.extend_from_slice(&(width as u16).to_le_bytes());
    out.extend_from_slice(&(height as u16).to_le_bytes());
    out.extend_from_slice(&[if gray { 1 } else { 3 }, ms, 0, 0]);
    out.extend_from_slice(&px);
    Ok(tauri::ipc::Response::new(out))
}

/// Runs test actions on the profile's game through the safety layer. The game is brought to
/// the front first; the log arrives as `live-log` events and `live-done` at the end.
#[tauri::command]
pub async fn live_run(app: AppHandle, id: String, window: u64, actions: String) -> Res<()> {
    let p = profile::load(&data_dir(&app)?, &id)?;
    let steps = script::parse(&actions)?;
    let live = app.state::<Live>();
    if live.running.swap(true, Ordering::AcqRel) {
        return Err("Actions are already running".into());
    }
    let start = (|| -> Res<Arc<Mutex<Controller>>> {
        let mut desk = sys::open()?;
        desk.bring_to_front(window)?;
        // Give the game a moment to take the keyboard.
        let deadline = Instant::now() + Duration::from_millis(1500);
        while !desk.in_front(window) {
            if Instant::now() > deadline {
                return Err("The game did not come to the front: click on it, then try again".into());
            }
            std::thread::sleep(Duration::from_millis(30));
        }
        std::thread::sleep(Duration::from_millis(150));
        Ok(Arc::new(Mutex::new(Controller::new(desk, window, p.permitted()?, p.limits.clone()))))
    })();
    let controller = match start {
        Ok(c) => c,
        Err(e) => {
            live.running.store(false, Ordering::Release);
            return Err(e);
        }
    };
    *live.active.lock().unwrap() = Some(controller.clone());
    let app2 = app.clone();
    std::thread::spawn(move || {
        let done = Arc::new(AtomicBool::new(false));
        // Pass the log on as it grows.
        let (c2, d2, a2) = (controller.clone(), done.clone(), app2.clone());
        let feeder = std::thread::spawn(move || {
            let mut shown = 0;
            loop {
                let finished = d2.load(Ordering::Acquire);
                let new: Vec<safety::LogEntry> = {
                    let c = c2.lock().unwrap();
                    let n = c.log.len();
                    let v = c.log[shown.min(n)..].to_vec();
                    shown = n;
                    v
                };
                if !new.is_empty() {
                    let _ = a2.emit("live-log", new);
                }
                if finished {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        script::run(&controller, &steps);
        let stopped = controller.lock().unwrap().stopped();
        done.store(true, Ordering::Release);
        let _ = feeder.join();
        let live = app2.state::<Live>();
        *live.active.lock().unwrap() = None;
        drop(controller);
        live.running.store(false, Ordering::Release);
        let _ = app2.emit("live-done", json!({"stopped": stopped}));
    });
    Ok(())
}

/// The emergency stop, from the window's button.
#[tauri::command]
pub fn live_stop(app: AppHandle, live: tauri::State<'_, Live>) {
    live.emergency_stop();
    let _ = app.emit("live-stopped", ());
}

fn alive(slot: &mut Option<RangeProcess>) -> Option<&mut RangeProcess> {
    let running = slot.as_mut().is_some_and(|r| r.running());
    if running {
        slot.as_mut()
    } else {
        None
    }
}

/// Starts the test range (or says it is running) and makes sure its profile exists.
#[tauri::command]
pub async fn range_start(app: AppHandle, seed: Option<u64>) -> Res<Value> {
    let data = data_dir(&app)?;
    if profile::load(&data, "yendory-range").is_err() {
        profile::save(&data, profile::range_profile())?;
    }
    let live = app.state::<Live>();
    let mut slot = live.range.lock().unwrap();
    if let Some(r) = alive(&mut slot) {
        return Ok(json!({"pid": r.pid(), "already": true}));
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let handle = app.clone();
    let r = RangeProcess::start(&exe, seed.unwrap_or(1), move |v| {
        let _ = handle.emit("range-event", v);
    })?;
    let pid = r.pid();
    *slot = Some(r);
    Ok(json!({"pid": pid, "already": false}))
}

/// What the range reports about itself (the truth, for checking inputs).
#[tauri::command]
pub async fn range_state(app: AppHandle) -> Res<Value> {
    let live = app.state::<Live>();
    let mut slot = live.range.lock().unwrap();
    let r = alive(&mut slot).ok_or("The range is not running")?;
    r.state()
}

#[tauri::command]
pub async fn range_reset(app: AppHandle, seed: u64) -> Res<()> {
    let live = app.state::<Live>();
    let mut slot = live.range.lock().unwrap();
    alive(&mut slot).ok_or("The range is not running")?.reset(seed)
}

#[tauri::command]
pub fn range_stop(live: tauri::State<'_, Live>) {
    live.range.lock().unwrap().take();
}
