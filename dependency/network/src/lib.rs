//! Yendory Network: the background service that runs the Yendory Engine for every program of the
//! suite (docs/NETWORK.md). Programs connect with `yendory-client`; the Network owns the models
//! folder's writes and keeps open models in memory.
//!
//! - One copy runs at a time (a lock on `service.lock` in its folder).
//! - It listens on 127.0.0.1 only, on a port the system picks, and writes the port and a fresh
//!   secret token to `service.json` (readable by this user only). A connection that doesn't send
//!   the token first is closed.
//! - It stops by itself after `idle_seconds` with no program connected and nothing open, and when
//!   asked (`shutdown`) by the Hub before an update.
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use yendory_client::{paths, ServiceInfo, API};
use yendory_engine::{Ablation, Bicameral};

pub mod library;
use library::Res;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const ENGINE: &str = env!("YENDORY_ENGINE_VERSION");

/// Longest line before `hello` (4 KiB) and after it (64 MiB).
const MAX_HELLO: u64 = 4 << 10;
const MAX_LINE: u64 = 64 << 20;
/// The most one upload may hold (a skill's four arrays at the Engine's largest skill).
const MAX_UPLOAD: usize = 1100 << 20;
/// The most one part of a transfer may carry (`upload.part`, `model.skill.read`).
const MAX_PART: usize = 16 << 20;

pub struct Options {
    /// Stop after this long with no program connected and nothing open.
    pub idle_seconds: u64,
}

/// `--version`: what this build is.
pub fn version_json() -> String {
    json!({"network": VERSION, "api": API, "engine": ENGINE}).to_string()
}

fn network_dir() -> Res<PathBuf> {
    let dir = paths::network_dir().ok_or("No data folder on this system")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Appends a line to the Network's log (`network.log`, the previous one kept as `network.old.log`).
pub fn log(line: &str) {
    let Ok(dir) = network_dir() else { return };
    let path = dir.join("network.log");
    if std::fs::metadata(&path).map(|m| m.len() > 1 << 20).unwrap_or(false) {
        let _ = std::fs::rename(&path, dir.join("network.old.log"));
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{} {line}", library::now());
    }
}

/// The secret a program must send first: 32 random bytes as hex.
fn new_token() -> Res<String> {
    let mut b = [0u8; 32];
    getrandom::fill(&mut b).map_err(|e| format!("No randomness for the token: {e}"))?;
    Ok(b.iter().map(|x| format!("{x:02x}")).collect())
}

/// Writes `service.json` readable by this user only.
fn write_private(path: &std::path::Path, bytes: &[u8]) -> Res<()> {
    let tmp = path.with_extension("tmp");
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(&tmp).map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    drop(f);
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

struct Conn {
    name: String,
    version: String,
    writer: Arc<Mutex<TcpStream>>,
}

/// A model held in memory, and the programs that opened it.
struct Open {
    model: Arc<Mutex<Bicameral>>,
    clients: HashSet<u64>,
}

/// An upload: the size announced, and the bytes that have arrived.
type Upload = (usize, Vec<u8>);
/// A download: the bytes being read in parts.
type Download = Arc<Vec<u8>>;

struct State {
    models: PathBuf,
    port: u16,
    clients: Mutex<HashMap<u64, Conn>>,
    open: Mutex<HashMap<String, Open>>,
    /// Writes to the models folder happen one at a time.
    library: Mutex<()>,
    next_client: AtomicU64,
    busy: AtomicUsize,
    last_activity: Mutex<Instant>,
    stopping: AtomicBool,
    started: Instant,
    service_file: PathBuf,
    token: String,
    /// Replies being written (`model.chat` with a `stream`), by program and stream, and the flag
    /// that stops each one (`model.stop`).
    streams: Mutex<HashMap<(u64, String), Arc<AtomicBool>>>,
    /// Transfers in parts (`upload.begin`, `upload.part`), by program and upload: the size
    /// announced and what has arrived. Dropped when the program disconnects.
    uploads: Mutex<HashMap<(u64, u64), Upload>>,
    /// Downloads in parts (`model.checkpoint`, `download.read`), by program and download.
    downloads: Mutex<HashMap<(u64, u64), Download>>,
    next_upload: AtomicU64,
}

impl State {
    fn touch(&self) {
        *self.last_activity.lock().unwrap() = Instant::now();
    }
    fn send(writer: &Arc<Mutex<TcpStream>>, msg: &Value) {
        let mut line = msg.to_string().into_bytes();
        line.push(b'\n');
        let mut w = writer.lock().unwrap();
        let _ = w.write_all(&line).and_then(|_| w.flush());
    }
    /// Sends an event to every connected program.
    fn broadcast(&self, event: &str, data: Value) {
        let msg = json!({"event": event, "data": data});
        let writers: Vec<_> = self.clients.lock().unwrap().values().map(|c| c.writer.clone()).collect();
        for w in writers {
            State::send(&w, &msg);
        }
    }
    /// Names of the programs that have a model open.
    fn opened_by(&self, id: &str) -> Vec<String> {
        let open = self.open.lock().unwrap();
        let clients = self.clients.lock().unwrap();
        let mut names: Vec<String> = open
            .get(id)
            .map(|o| o.clients.iter().filter_map(|c| clients.get(c).map(|c| c.name.clone())).collect())
            .unwrap_or_default();
        names.sort();
        names.dedup();
        names
    }
    /// Refuses a change to a model another program has open.
    fn not_open_elsewhere(&self, id: &str, me: u64, change: &str) -> Res<()> {
        let open = self.open.lock().unwrap();
        if let Some(o) = open.get(id) {
            if o.clients.iter().any(|c| *c != me) {
                drop(open);
                return Err(format!(
                    "{} has this model open. Close it there first: the {change} would be undone when it saves.",
                    self.opened_by(id).join(" and ")
                ));
            }
        }
        Ok(())
    }
    /// A model this program has open (methods on a model need it open first).
    fn held(&self, id: &str, client: u64) -> Result<Arc<Mutex<Bicameral>>, (&'static str, String)> {
        let open = self.open.lock().unwrap();
        open.get(id)
            .filter(|o| o.clients.contains(&client))
            .map(|o| o.model.clone())
            .ok_or_else(|| refused("Open the model first".into()))
    }
    /// A program left: its open models are closed (dropped when no one else has them).
    fn forget_client(&self, id: u64) {
        self.clients.lock().unwrap().remove(&id);
        self.uploads.lock().unwrap().retain(|(c, _), _| *c != id);
        self.downloads.lock().unwrap().retain(|(c, _), _| *c != id);
        self.streams.lock().unwrap().retain(|(c, _), stop| {
            // Its replies stop at the next token.
            if *c == id {
                stop.store(true, Ordering::Release);
            }
            *c != id
        });
        let mut open = self.open.lock().unwrap();
        for o in open.values_mut() {
            o.clients.remove(&id);
        }
        open.retain(|_, o| !o.clients.is_empty());
        drop(open);
        self.touch();
    }
    fn stop(&self, why: &str) {
        if self.stopping.swap(true, Ordering::AcqRel) {
            return;
        }
        log(&format!("stopping: {why}"));
        self.broadcast("network.stopping", json!({"reason": why}));
        let _ = std::fs::remove_file(&self.service_file);
        // Wake the accept loop so it sees `stopping`.
        let _ = TcpStream::connect(("127.0.0.1", self.port));
    }
}

// ---------------------------------------------------------------------------------------------
// Serving
// ---------------------------------------------------------------------------------------------

/// Runs the Network until it is idle or asked to stop. Returns at once (Ok) when another copy
/// is already running.
pub fn serve(opts: Options) -> Res<()> {
    let dir = network_dir()?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join("service.lock"))
        .map_err(|e| e.to_string())?;
    // Another copy holds the lock: if it is running (its service.json is there), this copy
    // leaves. If it is only stopping (being replaced by an update), wait for it a little.
    let until = Instant::now() + Duration::from_secs(10);
    loop {
        match lock.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::WouldBlock) => {
                if dir.join(paths::SERVICE_FILE).exists() || Instant::now() > until {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(std::fs::TryLockError::Error(e)) => return Err(format!("The lock could not be taken: {e}")),
        }
    }
    let models = paths::models_dir().ok_or("No data folder on this system")?;
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| format!("No local port: {e}"))?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let token = new_token()?;
    let service_file = dir.join(paths::SERVICE_FILE);
    let info = ServiceInfo {
        pid: std::process::id(),
        port,
        token: token.clone(),
        version: VERSION.into(),
        api: API,
        engine: ENGINE.into(),
    };
    write_private(&service_file, &serde_json::to_vec_pretty(&info).map_err(|e| e.to_string())?)?;
    log(&format!("started: Network {VERSION}, Engine {ENGINE}, port {port}"));
    let state = Arc::new(State {
        models,
        port,
        clients: Mutex::new(HashMap::new()),
        open: Mutex::new(HashMap::new()),
        library: Mutex::new(()),
        next_client: AtomicU64::new(1),
        busy: AtomicUsize::new(0),
        last_activity: Mutex::new(Instant::now()),
        stopping: AtomicBool::new(false),
        started: Instant::now(),
        service_file,
        token,
        streams: Mutex::new(HashMap::new()),
        uploads: Mutex::new(HashMap::new()),
        downloads: Mutex::new(HashMap::new()),
        next_upload: AtomicU64::new(1),
    });
    {
        let state = state.clone();
        let idle = Duration::from_secs(opts.idle_seconds.max(1));
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_millis(500));
            if state.stopping.load(Ordering::Acquire) {
                return;
            }
            let quiet = state.clients.lock().unwrap().is_empty()
                && state.open.lock().unwrap().is_empty()
                && state.busy.load(Ordering::Acquire) == 0;
            if quiet && state.last_activity.lock().unwrap().elapsed() >= idle {
                state.stop("idle");
                return;
            }
        });
    }
    for stream in listener.incoming() {
        if state.stopping.load(Ordering::Acquire) {
            break;
        }
        let Ok(stream) = stream else { continue };
        let state = state.clone();
        std::thread::spawn(move || connection(state, stream));
    }
    // Answers already on their way get a moment to arrive.
    std::thread::sleep(Duration::from_millis(200));
    drop(lock);
    Ok(())
}

fn error(code: &str, message: impl Into<String>) -> Value {
    json!({"code": code, "message": message.into()})
}

fn connection(state: Arc<State>, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let Ok(read_half) = stream.try_clone() else { return };
    let writer = Arc::new(Mutex::new(stream));
    let mut reader = BufReader::new(read_half);

    // The first message must be `hello` with the token, within 10 seconds.
    let _ = reader.get_ref().set_read_timeout(Some(Duration::from_secs(10)));
    let mut line = String::new();
    if (&mut reader).take(MAX_HELLO).read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let first: Value = serde_json::from_str(&line).unwrap_or_default();
    let id = first.get("id").cloned().unwrap_or(Value::Null);
    let params = first.get("params").cloned().unwrap_or_default();
    if first.get("method").and_then(Value::as_str) != Some("hello")
        || params.get("token").and_then(Value::as_str) != Some(state.token.as_str())
    {
        State::send(
            &writer,
            &json!({"id": id, "error": error("unauthorized", "This program may not use the Yendory Network")}),
        );
        return;
    }
    let client = state.next_client.fetch_add(1, Ordering::Relaxed);
    let name = params
        .get("client")
        .and_then(Value::as_str)
        .unwrap_or("A program")
        .chars()
        .take(60)
        .collect::<String>();
    let version = params
        .get("clientVersion")
        .and_then(Value::as_str)
        .unwrap_or("")
        .chars()
        .take(30)
        .collect::<String>();
    state
        .clients
        .lock()
        .unwrap()
        .insert(client, Conn { name: name.clone(), version, writer: writer.clone() });
    state.touch();
    log(&format!("connected: {name} (#{client})"));
    State::send(
        &writer,
        &json!({"id": id, "result": {
            "network": VERSION, "api": API, "engine": ENGINE,
            "modelsFolder": state.models.display().to_string(), "clientId": client,
        }}),
    );

    let _ = reader.get_ref().set_read_timeout(None);
    loop {
        let mut line = String::new();
        match (&mut reader).take(MAX_LINE).read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                State::send(
                    &writer,
                    &json!({"id": null, "error": error("invalid", "A message could not be read")}),
                );
                continue;
            }
        };
        state.touch();
        let state = state.clone();
        let writer = writer.clone();
        state.busy.fetch_add(1, Ordering::AcqRel);
        std::thread::spawn(move || {
            let id = msg.get("id").cloned().unwrap_or(Value::Null);
            let method = msg.get("method").and_then(Value::as_str).unwrap_or("").to_string();
            let params = msg.get("params").cloned().unwrap_or_default();
            let reply = match dispatch(&state, client, &method, &params) {
                Ok(result) => json!({"id": id, "result": result}),
                Err((code, message)) => json!({"id": id, "error": error(code, message)}),
            };
            State::send(&writer, &reply);
            state.busy.fetch_sub(1, Ordering::AcqRel);
            state.touch();
            if method == "shutdown" && reply.get("result").is_some() {
                state.stop("asked by a program");
            }
        });
    }
    log(&format!("disconnected: {name} (#{client})"));
    state.forget_client(client);
}

// ---------------------------------------------------------------------------------------------
// Methods (API 1; docs/NETWORK.md)
// ---------------------------------------------------------------------------------------------

type Answer = Result<Value, (&'static str, String)>;

fn refused(e: String) -> (&'static str, String) {
    ("refused", e)
}
fn invalid(e: impl Into<String>) -> (&'static str, String) {
    ("invalid", e.into())
}
fn str_param<'a>(p: &'a Value, key: &str) -> Result<&'a str, (&'static str, String)> {
    p.get(key).and_then(Value::as_str).ok_or_else(|| invalid(format!("`{key}` is missing")))
}

fn dispatch(state: &State, client: u64, method: &str, p: &Value) -> Answer {
    let models = &state.models;
    match method {
        "ping" => Ok(json!("pong")),
        "status" => {
            let clients = state.clients.lock().unwrap();
            let open = state.open.lock().unwrap();
            Ok(json!({
                "network": VERSION, "api": API, "engine": ENGINE,
                "modelsFolder": models.display().to_string(),
                "uptime": state.started.elapsed().as_secs(),
                "clients": clients.iter().map(|(id, c)| json!({"id": id, "name": c.name, "version": c.version})).collect::<Vec<_>>(),
                "open": open.iter().map(|(id, o)| json!({"id": id, "clients": o.clients.iter().collect::<Vec<_>>()})).collect::<Vec<_>>(),
            }))
        }
        "shutdown" => {
            // Refused while another program is connected or a model is open: the Hub asks
            // before replacing the Network, and must not cut another program off.
            let others: Vec<String> = state
                .clients
                .lock()
                .unwrap()
                .iter()
                .filter(|(id, _)| **id != client)
                .map(|(_, c)| c.name.clone())
                .collect();
            if !others.is_empty() && !p.get("force").and_then(Value::as_bool).unwrap_or(false) {
                return Err(refused(format!(
                    "{} is using the Yendory Network. Close it first.",
                    others.join(" and ")
                )));
            }
            Ok(json!(true))
        }
        "models.list" => {
            let list = library::list(models).map_err(|e| ("failed", e))?;
            Ok(Value::Array(
                list.into_iter()
                    .map(|mut e| {
                        let id = e["meta"]["id"].as_str().unwrap_or("").to_string();
                        e["openBy"] = json!(state.opened_by(&id));
                        e
                    })
                    .collect(),
            ))
        }
        "models.options" => Ok(serde_json::to_value(library::options()).unwrap_or_default()),
        "models.plan" => {
            // The exact plan of a Neuro brain chosen network by network: each network's
            // weights, the total and the file's size (nothing is made).
            let choice: library::NewModel =
                serde_json::from_value(p.clone()).map_err(|e| invalid(e.to_string()))?;
            library::plan(&choice).map_err(refused)
        }
        "models.create" => {
            let choice: library::NewModel =
                serde_json::from_value(p.clone()).map_err(|e| invalid(e.to_string()))?;
            let _w = state.library.lock().unwrap();
            let made = library::create(models, &choice).map_err(refused)?;
            state.broadcast("models.changed", json!({"id": made["meta"]["id"], "change": "created"}));
            Ok(made)
        }
        "models.import" => {
            let path = PathBuf::from(str_param(p, "path")?);
            let _w = state.library.lock().unwrap();
            let made =
                library::import(models, &path, p.get("name").and_then(Value::as_str)).map_err(refused)?;
            state.broadcast("models.changed", json!({"id": made["meta"]["id"], "change": "imported"}));
            Ok(made)
        }
        "models.rename" => {
            let id = str_param(p, "id")?;
            state.not_open_elsewhere(id, client, "new name").map_err(refused)?;
            let _w = state.library.lock().unwrap();
            let changed = library::rename(models, id, str_param(p, "name")?).map_err(refused)?;
            state.broadcast("models.changed", json!({"id": id, "change": "renamed"}));
            Ok(changed)
        }
        "models.delete" => {
            let id = str_param(p, "id")?;
            if !state.opened_by(id).is_empty() {
                return Err(refused(format!(
                    "{} has this model open. Close it there first.",
                    state.opened_by(id).join(" and ")
                )));
            }
            let _w = state.library.lock().unwrap();
            library::delete(models, id).map_err(refused)?;
            state.broadcast("models.changed", json!({"id": id, "change": "deleted"}));
            Ok(json!(true))
        }
        "models.export" => {
            let id = str_param(p, "id")?;
            let path = PathBuf::from(str_param(p, "path")?);
            let bytes = library::export(models, id, &path).map_err(refused)?;
            Ok(json!({"path": path.display().to_string(), "bytes": bytes}))
        }
        "models.open" => {
            let id = str_param(p, "id")?.to_string();
            let mut open = state.open.lock().unwrap();
            if let Some(o) = open.get_mut(&id) {
                o.clients.insert(client);
            } else {
                drop(open);
                let (_, model) = library::load(models, &id).map_err(refused)?;
                open = state.open.lock().unwrap();
                open.entry(id.clone())
                    .or_insert_with(|| Open { model: Arc::new(Mutex::new(model)), clients: HashSet::new() })
                    .clients
                    .insert(client);
            }
            drop(open);
            let dir = library::model_dir(models, &id).map_err(invalid)?;
            let meta: Value =
                serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap_or_default())
                    .unwrap_or_default();
            Ok(library::entry(&dir, meta))
        }
        "models.close" => {
            let id = str_param(p, "id")?;
            let mut open = state.open.lock().unwrap();
            if let Some(o) = open.get_mut(id) {
                o.clients.remove(&client);
                if o.clients.is_empty() {
                    open.remove(id);
                }
            }
            Ok(json!(true))
        }
        "model.reply" => {
            // The model's continuation of a prompt (Studio formats the prompt).
            let id = str_param(p, "id")?;
            let prompt = str_param(p, "prompt")?;
            let max = p.get("maxTokens").and_then(Value::as_u64).unwrap_or(48).clamp(1, 4096) as usize;
            let model = state.held(id, client)?;
            let model = model.lock().unwrap();
            let (text, cut) =
                yendory_engine::learn::eval::reply_limited(&model, prompt, max, Ablation::default())
                    .map_err(refused)?;
            Ok(json!({"text": text, "cut": cut}))
        }
        "model.chat" => {
            // A conversation reply as Studio's chat writes it: the model's continuation of the
            // prompt, each token chosen with `sampling` (Engine `Sampling`; greedy when absent),
            // up to a line break or `maxTokens`. With `stream`, each token's text is sent to this
            // program as it is written (`model.token` events: `stream`, `text`; the final line
            // break is not sent), and `model.stop` with the same `stream` stops it after the
            // token being written.
            let id = str_param(p, "id")?;
            let prompt = str_param(p, "prompt")?;
            let max = p.get("maxTokens").and_then(Value::as_u64).unwrap_or(48).clamp(1, 4096) as usize;
            let sampling: yendory_engine::learn::sampling::Sampling = match p.get("sampling") {
                Some(s) if !s.is_null() => {
                    serde_json::from_value(s.clone()).map_err(|e| invalid(e.to_string()))?
                }
                _ => Default::default(),
            };
            let stream: String =
                p.get("stream").and_then(Value::as_str).unwrap_or("").chars().take(80).collect();
            // The weights as they are now (a reload while writing does not change this reply).
            let model = state.held(id, client)?.lock().unwrap().clone();
            let stop = Arc::new(AtomicBool::new(false));
            let writer = state.clients.lock().unwrap().get(&client).map(|c| c.writer.clone());
            if !stream.is_empty() {
                state.streams.lock().unwrap().insert((client, stream.clone()), stop.clone());
            }
            let reply = yendory_engine::learn::sampling::reply_with(&model, prompt, max, &sampling, |text| {
                if let (Some(w), false) = (&writer, stream.is_empty()) {
                    State::send(
                        w,
                        &json!({"event": "model.token", "data": {"stream": stream, "text": text}}),
                    );
                }
                !stop.load(Ordering::Acquire)
            });
            if !stream.is_empty() {
                state.streams.lock().unwrap().remove(&(client, stream.clone()));
            }
            let reply = reply.map_err(refused)?;
            Ok(json!({
                "text": reply.text, "tokens": reply.tokens, "finished": reply.finished,
                "stopped": reply.stopped, "step": model.step, "sampling": sampling,
            }))
        }
        "model.stop" => {
            // Stops this program's reply on `stream`; false when none is being written.
            let stream = str_param(p, "stream")?;
            let found = state.streams.lock().unwrap().get(&(client, stream.to_string())).cloned();
            if let Some(stop) = &found {
                stop.store(true, Ordering::Release);
            }
            Ok(json!(found.is_some()))
        }
        "models.reload" => {
            // Reads an open model's file again (a program that wrote it itself, such as Studio
            // after training, says so); every program that has it open gets the new weights.
            let id = str_param(p, "id")?;
            let held = state.held(id, client)?;
            let (_, model) = library::load(models, id).map_err(refused)?;
            *held.lock().unwrap() = model;
            let dir = library::model_dir(models, id).map_err(invalid)?;
            let meta: Value =
                serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap_or_default())
                    .unwrap_or_default();
            Ok(library::entry(&dir, meta))
        }
        "model.skills" => {
            // The open model's game skills (their descriptions, no weights).
            let id = str_param(p, "id")?;
            let model = state.held(id, client)?;
            let model = model.lock().unwrap();
            Ok(json!(model.skills.iter().map(|k| &k.spec).collect::<Vec<_>>()))
        }
        "model.skill" => {
            // One skill with its weights: the accepted ones, the latest, and the optimizer's
            // moments and step (floats in base64, yendory_client::floats). With `parts`, only
            // its description, step and size in bytes: the weights are read with
            // `model.skill.read` (a large skill does not fit one message).
            let id = str_param(p, "id")?;
            let game = str_param(p, "game")?;
            let model = state.held(id, client)?;
            let model = model.lock().unwrap();
            let k = model
                .skills
                .iter()
                .find(|k| k.spec.game == game)
                .ok_or_else(|| refused("This model has no skill for that game".into()))?;
            if p.get("parts").and_then(Value::as_bool) == Some(true) {
                return Ok(json!({"spec": k.spec, "t": k.adam.t, "bytes": 16 * k.accepted.len()}));
            }
            use yendory_client::floats::encode;
            Ok(json!({
                "spec": k.spec, "accepted": encode(&k.accepted), "latest": encode(&k.latest),
                "m": encode(&k.adam.m), "v": encode(&k.adam.v), "t": k.adam.t,
            }))
        }
        "model.skill.read" => {
            // Part of a skill's weights: bytes `offset..offset + length` of its accepted
            // weights, latest weights, Adam m and v, one after another (f32 little-endian).
            let id = str_param(p, "id")?;
            let game = str_param(p, "game")?;
            let offset = p.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
            let length = p.get("length").and_then(Value::as_u64).unwrap_or(0) as usize;
            if length == 0 || length > MAX_PART || !offset.is_multiple_of(4) || !length.is_multiple_of(4) {
                return Err(invalid(format!("Read 4 to {MAX_PART} bytes at a time, at whole values")));
            }
            let model = state.held(id, client)?;
            let model = model.lock().unwrap();
            let k = model
                .skills
                .iter()
                .find(|k| k.spec.game == game)
                .ok_or_else(|| refused("This model has no skill for that game".into()))?;
            let n = k.accepted.len();
            if offset + length > 16 * n {
                return Err(invalid("Past the end of the skill's weights".to_string()));
            }
            let arrays = [&k.accepted, &k.latest, &k.adam.m, &k.adam.v];
            let mut out = Vec::with_capacity(length);
            for i in offset / 4..(offset + length) / 4 {
                out.extend_from_slice(&arrays[i / n][i % n].to_le_bytes());
            }
            Ok(json!({"data": yendory_client::bytes::encode(&out), "bytes": 16 * n}))
        }
        "model.checkpoint" => {
            // The open model's whole checkpoint (its weights, optimizer, words and skills, as
            // its file holds them), to be read in parts with `download.read`: a program that
            // trains the model itself (Yendory Tools: the core playing a game).
            let id = str_param(p, "id")?;
            let held = state.held(id, client)?;
            let bytes = held.lock().unwrap().to_bytes().map_err(refused)?;
            let n = bytes.len();
            let d = state.next_upload.fetch_add(1, Ordering::Relaxed);
            let mut downloads = state.downloads.lock().unwrap();
            // One download of a checkpoint at a time for each program.
            downloads.retain(|(c, _), _| *c != client);
            downloads.insert((client, d), Arc::new(bytes));
            Ok(json!({"download": d, "bytes": n}))
        }
        "download.read" => {
            let d =
                p.get("download").and_then(Value::as_u64).ok_or_else(|| invalid("`download` is missing"))?;
            let offset = p.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
            let length = p.get("length").and_then(Value::as_u64).unwrap_or(0) as usize;
            let data = state
                .downloads
                .lock()
                .unwrap()
                .get(&(client, d))
                .cloned()
                .ok_or_else(|| invalid("No such download"))?;
            if length == 0 || length > MAX_PART || offset + length > data.len() {
                return Err(invalid(format!("Read 1 to {MAX_PART} bytes within the download")));
            }
            if offset + length == data.len() {
                // Read to its end: let it go.
                state.downloads.lock().unwrap().remove(&(client, d));
            }
            Ok(
                json!({"data": yendory_client::bytes::encode(&data[offset..offset + length]), "bytes": data.len()}),
            )
        }
        "model.checkpoint.save" => {
            // Replaces the open model's weights, optimizer and skills with a checkpoint of the
            // same model (same architecture, sizes and words) sent as an upload, and saves its
            // file. Refused when another program has the model open.
            let id = str_param(p, "id")?;
            state.not_open_elsewhere(id, client, "training").map_err(refused)?;
            let up = p.get("upload").and_then(Value::as_u64).ok_or_else(|| invalid("`upload` is missing"))?;
            let (size, data) = state
                .uploads
                .lock()
                .unwrap()
                .remove(&(client, up))
                .ok_or_else(|| invalid("No such upload"))?;
            if data.len() != size {
                return Err(invalid("The upload did not all arrive"));
            }
            let next = Bicameral::from_bytes(&data).map_err(invalid)?;
            drop(data);
            let held = state.held(id, client)?;
            let _w = state.library.lock().unwrap();
            let mut model = held.lock().unwrap();
            if next.config != model.config
                || next.neuro != model.neuro
                || next.tokenizer.words() != model.tokenizer.words()
            {
                return Err(refused(
                    "The checkpoint is not this model's shape (its sizes, design or words differ)".into(),
                ));
            }
            let entry = library::save_checkpoint(models, id, &next).map_err(refused)?;
            *model = next;
            drop(model);
            state.broadcast("models.changed", json!({"id": id, "change": "trained"}));
            Ok(entry)
        }
        "upload.begin" => {
            let bytes = p.get("bytes").and_then(Value::as_u64).unwrap_or(0) as usize;
            if bytes == 0 || bytes > MAX_UPLOAD {
                return Err(invalid(format!("An upload holds 1 to {MAX_UPLOAD} bytes")));
            }
            let id = state.next_upload.fetch_add(1, Ordering::Relaxed);
            state.uploads.lock().unwrap().insert((client, id), (bytes, Vec::new()));
            Ok(json!({"upload": id}))
        }
        "upload.part" => {
            let id = p.get("upload").and_then(Value::as_u64).ok_or_else(|| invalid("`upload` is missing"))?;
            let data = yendory_client::bytes::decode(str_param(p, "data")?).map_err(invalid)?;
            if data.len() > MAX_PART {
                return Err(invalid(format!("A part holds at most {MAX_PART} bytes")));
            }
            let mut uploads = state.uploads.lock().unwrap();
            let (size, got) =
                uploads.get_mut(&(client, id)).ok_or_else(|| invalid("No such upload".to_string()))?;
            if got.len() + data.len() > *size {
                return Err(invalid("More than the upload announced"));
            }
            if got.capacity() == 0 {
                got.reserve_exact(*size);
            }
            got.extend_from_slice(&data);
            Ok(json!({"received": got.len()}))
        }
        "model.skill.save" => {
            // Adds or replaces the skill for `spec.game` and saves the model's file (Bicameral or
            // Neuro). Refused when another program has the model open (its next save would undo
            // this). The weights come in the message (base64) or, for a large skill, as an
            // upload sent in parts (`upload`).
            let id = str_param(p, "id")?;
            state.not_open_elsewhere(id, client, "game skill").map_err(refused)?;
            let spec: yendory_engine::skill::SkillSpec =
                serde_json::from_value(p.get("spec").cloned().unwrap_or_default())
                    .map_err(|e| invalid(format!("`spec`: {e}")))?;
            let t = p.get("t").and_then(Value::as_u64).unwrap_or(0);
            let skill = if let Some(up) = p.get("upload").and_then(Value::as_u64) {
                // The weights sent in parts: accepted, latest, Adam m and v, one after another.
                let (size, data) = state
                    .uploads
                    .lock()
                    .unwrap()
                    .remove(&(client, up))
                    .ok_or_else(|| invalid("No such upload"))?;
                let n = yendory_engine::skill::layout_len(&spec.arch);
                if data.len() != size || size != 16 * n {
                    return Err(invalid("The upload is not the skill's weights (or did not all arrive)"));
                }
                let mut all = yendory_client::floats::from_bytes(&data).map_err(invalid)?;
                drop(data);
                let v = all.split_off(3 * n);
                let m = all.split_off(2 * n);
                let latest = all.split_off(n);
                yendory_engine::skill::Skill {
                    spec,
                    accepted: all,
                    latest,
                    adam: yendory_engine::skill::Adam { m, v, t },
                }
            } else {
                use yendory_client::floats::decode;
                let arr = |k: &str| decode(str_param(p, k)?).map_err(invalid);
                yendory_engine::skill::Skill {
                    spec,
                    accepted: arr("accepted")?,
                    latest: arr("latest")?,
                    adam: yendory_engine::skill::Adam { m: arr("m")?, v: arr("v")?, t },
                }
            };
            skill.check().map_err(invalid)?;
            let held = state.held(id, client)?;
            let _w = state.library.lock().unwrap();
            let mut model = held.lock().unwrap();
            let mut next = model.clone();
            match next.skills.iter_mut().find(|k| k.spec.game == skill.spec.game) {
                Some(k) => *k = skill,
                None => next.skills.push(skill),
            }
            let entry = library::save_skills(models, id, &next).map_err(refused)?;
            *model = next;
            drop(model);
            state.broadcast("models.changed", json!({"id": id, "change": "skills"}));
            Ok(entry)
        }
        "model.skill.delete" => {
            let id = str_param(p, "id")?;
            let game = str_param(p, "game")?;
            state.not_open_elsewhere(id, client, "game skill").map_err(refused)?;
            let held = state.held(id, client)?;
            let _w = state.library.lock().unwrap();
            let mut model = held.lock().unwrap();
            let mut next = model.clone();
            let before = next.skills.len();
            next.skills.retain(|k| k.spec.game != game);
            if next.skills.len() == before {
                return Err(refused("This model has no skill for that game".into()));
            }
            let entry = library::save_skills(models, id, &next).map_err(refused)?;
            *model = next;
            drop(model);
            state.broadcast("models.changed", json!({"id": id, "change": "skills"}));
            Ok(entry)
        }
        "hello" => Err(invalid("Already connected")),
        _ => Err(("unknown_method", format!("The Yendory Network {VERSION} has no method `{method}`"))),
    }
}
