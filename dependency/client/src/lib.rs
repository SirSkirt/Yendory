//! Yendory client: how a program of the Yendory suite finds, starts and talks to the Yendory
//! Network, the background service that runs the Engine for every program (docs/NETWORK.md).
//!
//! The Network listens on a local port (127.0.0.1 only). While it runs, its folder holds
//! `service.json` with the port and a secret token; a program reads it, connects, and proves it
//! may talk to the Network by sending the token in its first message (`hello`). Messages are one
//! JSON object per line:
//!
//! - request:  `{"id": 1, "method": "models.list", "params": {}}`
//! - response: `{"id": 1, "result": …}` or `{"id": 1, "error": {"code": "…", "message": "…"}}`
//! - event:    `{"event": "models.changed", "data": {…}}` (sent by the Network unasked)
//!
//! When the Network is not running, [`Client::start_and_connect`] starts the copy the Hub
//! installed (`install.json` in the same folder names it).
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// The protocol version this client speaks. A Network with another major API is refused.
pub const API: u32 = 1;

/// Longest line a client accepts from the Network (64 MiB).
const MAX_LINE: u64 = 64 << 20;

/// Where the suite keeps its data.
pub mod paths {
    use std::path::PathBuf;

    /// Identifiers of the suite's data folders (the programs' Tauri identifiers).
    pub const NETWORK_ID: &str = "org.yendory.network";
    pub const STUDIO_ID: &str = "org.yendory.studio";
    pub const HUB_ID: &str = "org.yendory.hub";

    /// The user's data folder: `%APPDATA%` on Windows, `~/.local/share` on Linux,
    /// `~/Library/Application Support` on macOS (where Tauri puts every app's data). The
    /// environment variable `YENDORY_DATA_DIR` replaces it (tests and development).
    pub fn data_dir() -> Option<PathBuf> {
        match std::env::var_os("YENDORY_DATA_DIR") {
            Some(d) if !d.is_empty() => Some(PathBuf::from(d)),
            _ => dirs::data_dir(),
        }
    }
    /// The Network's folder: `service.json` (while it runs), `install.json`, its lock and log.
    pub fn network_dir() -> Option<PathBuf> {
        data_dir().map(|d| d.join(NETWORK_ID))
    }
    /// The suite's models folder: Studio's `models` folder (docs/HANDOFF.md).
    pub fn models_dir() -> Option<PathBuf> {
        data_dir().map(|d| d.join(STUDIO_ID).join("models"))
    }
    pub const SERVICE_FILE: &str = "service.json";
    pub const INSTALL_FILE: &str = "install.json";
}

/// What a running Network writes to `service.json`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ServiceInfo {
    pub pid: u32,
    pub port: u16,
    pub token: String,
    pub version: String,
    pub api: u32,
    pub engine: String,
}

/// What the Hub writes to `install.json` when it installs the Network.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InstallInfo {
    pub version: String,
    /// The Network's executable.
    pub path: PathBuf,
}

/// The Network's answer to `hello`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Hello {
    pub network: String,
    pub api: u32,
    pub engine: String,
    pub models_folder: String,
    pub client_id: u64,
}

/// An error from the Network, or from reaching it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Error {
    /// `unreachable`, `not_installed`, `refused`, `not_found`, `invalid`, `failed`,
    /// `unauthorized`, `unknown_method`, `incompatible`, `disconnected`.
    pub code: String,
    pub message: String,
}
impl Error {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Error { code: code.into(), message: message.into() }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Error {}

/// Something the Network sent unasked.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub name: String,
    pub data: Value,
}

type Pending = Arc<Mutex<HashMap<u64, mpsc::Sender<Result<Value, Error>>>>>;

/// A connection to the Network. Calls may be made from several threads at once.
pub struct Client {
    writer: Mutex<TcpStream>,
    pending: Pending,
    next: AtomicU64,
    alive: Arc<AtomicBool>,
    events: Mutex<Option<mpsc::Receiver<Event>>>,
    pub hello: Hello,
}

/// Reads `<Network folder>/<name>` as JSON.
fn read_file<T: for<'de> Deserialize<'de>>(name: &str) -> Option<T> {
    let path = paths::network_dir()?.join(name);
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// The running Network's address and token, if its `service.json` is there.
pub fn service_info() -> Option<ServiceInfo> {
    read_file(paths::SERVICE_FILE)
}

/// The Network the Hub installed, if any.
pub fn install_info() -> Option<InstallInfo> {
    read_file(paths::INSTALL_FILE)
}

/// Starts the Network in the background. `path`: its executable (`None`: the one in
/// `install.json`, or `YENDORY_NETWORK` when set, for development). The Network runs on its own:
/// it keeps running after this program closes, while other programs use it.
pub fn start(path: Option<&Path>) -> Result<(), Error> {
    let exe = match path {
        Some(p) => p.to_path_buf(),
        None => match std::env::var_os("YENDORY_NETWORK") {
            Some(p) if !p.is_empty() => PathBuf::from(p),
            _ => {
                install_info()
                    .ok_or_else(|| {
                        Error::new(
                            "not_installed",
                            "The Yendory Network is not installed. Install it from Yendory Hub.",
                        )
                    })?
                    .path
            }
        },
    };
    if !exe.is_file() {
        return Err(Error::new(
            "not_installed",
            format!("The Yendory Network is missing ({}). Reinstall it from Yendory Hub.", exe.display()),
        ));
    }
    let mut cmd = std::process::Command::new(&exe);
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    if let Some(dir) = exe.parent() {
        cmd.current_dir(dir);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // No console window, and not tied to this program's console.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Its own process group: closing this program's terminal doesn't stop it.
        cmd.process_group(0);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| Error::new("failed", format!("The Yendory Network could not be started: {e}")))?;
    // Collect its exit if it ends while this program runs (no finished process is left behind).
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

impl Client {
    /// Connects to the running Network. `name` and `version` say which program is asking.
    pub fn connect(name: &str, version: &str) -> Result<Client, Error> {
        let info =
            service_info().ok_or_else(|| Error::new("unreachable", "The Yendory Network is not running"))?;
        Self::connect_to(&info, name, version)
    }

    /// Connects to the Network, starting it first when it is not running, and waits up to
    /// `wait` for it to answer.
    pub fn start_and_connect(name: &str, version: &str, wait: Duration) -> Result<Client, Error> {
        if let Ok(c) = Self::connect(name, version) {
            return Ok(c);
        }
        start(None)?;
        let until = Instant::now() + wait;
        loop {
            std::thread::sleep(Duration::from_millis(150));
            match Self::connect(name, version) {
                Ok(c) => return Ok(c),
                Err(e) if e.code == "incompatible" || e.code == "unauthorized" => return Err(e),
                Err(e) if Instant::now() > until => {
                    return Err(Error::new(
                        "unreachable",
                        format!("The Yendory Network was started but did not answer ({})", e.message),
                    ))
                }
                Err(_) => {}
            }
        }
    }

    fn connect_to(info: &ServiceInfo, name: &str, version: &str) -> Result<Client, Error> {
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], info.port));
        let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))
            .map_err(|e| Error::new("unreachable", format!("The Yendory Network is not answering: {e}")))?;
        let _ = stream.set_nodelay(true);
        let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
        let reader = stream.try_clone().map_err(|e| Error::new("unreachable", e.to_string()))?;
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let (events_tx, events_rx) = mpsc::channel();
        {
            let pending = pending.clone();
            let alive = alive.clone();
            std::thread::spawn(move || read_loop(reader, pending, events_tx, alive));
        }
        let mut client = Client {
            writer: Mutex::new(stream),
            pending,
            next: AtomicU64::new(1),
            alive,
            events: Mutex::new(Some(events_rx)),
            hello: Hello {
                network: String::new(),
                api: 0,
                engine: String::new(),
                models_folder: String::new(),
                client_id: 0,
            },
        };
        let answer = client.call(
            "hello",
            json!({"token": info.token, "client": name, "clientVersion": version, "api": API}),
        )?;
        let hello: Hello = serde_json::from_value(answer).map_err(|e| Error::new("failed", e.to_string()))?;
        if hello.api != API {
            return Err(Error::new(
                "incompatible",
                format!(
                    "This program speaks the Yendory Network's API {API}, but the Network {} speaks API {}. Update both from Yendory Hub.",
                    hello.network, hello.api
                ),
            ));
        }
        client.hello = hello;
        Ok(client)
    }

    /// Calls a method and waits for its answer.
    pub fn call(&self, method: &str, params: Value) -> Result<Value, Error> {
        if !self.is_alive() {
            return Err(Error::new("disconnected", "The connection to the Yendory Network was lost"));
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel();
        self.pending.lock().unwrap().insert(id, tx);
        let mut line = serde_json::to_vec(&json!({"id": id, "method": method, "params": params}))
            .map_err(|e| Error::new("invalid", e.to_string()))?;
        line.push(b'\n');
        let sent = {
            let mut w = self.writer.lock().unwrap();
            w.write_all(&line).and_then(|_| w.flush())
        };
        if let Err(e) = sent {
            self.pending.lock().unwrap().remove(&id);
            self.alive.store(false, Ordering::Release);
            return Err(Error::new(
                "disconnected",
                format!("The connection to the Yendory Network was lost: {e}"),
            ));
        }
        rx.recv().unwrap_or_else(|_| {
            Err(Error::new("disconnected", "The connection to the Yendory Network was lost"))
        })
    }

    /// The Network's events, for one reader (the first caller gets them).
    pub fn take_events(&self) -> Option<mpsc::Receiver<Event>> {
        self.events.lock().unwrap().take()
    }

    /// Whether the connection is still open.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        if let Ok(w) = self.writer.lock() {
            let _ = w.shutdown(Shutdown::Both);
        }
    }
}

fn read_loop(stream: TcpStream, pending: Pending, events: mpsc::Sender<Event>, alive: Arc<AtomicBool>) {
    let mut reader = BufReader::new(stream);
    loop {
        let mut line = String::new();
        match (&mut reader).take(MAX_LINE).read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else { continue };
        if let Some(name) = msg.get("event").and_then(Value::as_str) {
            let _ = events
                .send(Event { name: name.into(), data: msg.get("data").cloned().unwrap_or(Value::Null) });
            continue;
        }
        let Some(id) = msg.get("id").and_then(Value::as_u64) else { continue };
        let result = match msg.get("error") {
            Some(e) if !e.is_null() => Err(serde_json::from_value(e.clone())
                .unwrap_or_else(|_| Error::new("failed", "The Yendory Network sent an unreadable error"))),
            _ => Ok(msg.get("result").cloned().unwrap_or(Value::Null)),
        };
        if let Some(tx) = pending.lock().unwrap().remove(&id) {
            let _ = tx.send(result);
        }
    }
    alive.store(false, Ordering::Release);
    // Every call still waiting learns the connection is gone.
    for (_, tx) in pending.lock().unwrap().drain() {
        let _ = tx.send(Err(Error::new("disconnected", "The connection to the Yendory Network was lost")));
    }
}
