//! The Network as programs see it: the real binary, started in a temporary data folder, driven
//! through `yendory-client`.
use serde_json::json;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use yendory_client::{paths, Client, ServiceInfo};

struct Network {
    child: Child,
    data: PathBuf,
}

impl Network {
    fn start(name: &str, idle: u64) -> Network {
        let data = std::env::temp_dir().join(format!("yendory-service-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        std::fs::create_dir_all(&data).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_yendory-network"))
            .env("YENDORY_DATA_DIR", &data)
            .env("YENDORY_NETWORK_IDLE", idle.to_string())
            .spawn()
            .unwrap();
        let n = Network { child, data };
        n.wait_for(|| n.info().is_some(), "service.json");
        n
    }
    fn info(&self) -> Option<ServiceInfo> {
        let path = self.data.join(paths::NETWORK_ID).join(paths::SERVICE_FILE);
        serde_json::from_slice(&std::fs::read(path).ok()?).ok()
    }
    fn wait_for(&self, ok: impl Fn() -> bool, what: &str) {
        let until = Instant::now() + Duration::from_secs(20);
        while !ok() {
            assert!(Instant::now() < until, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    /// A client of this Network (the data folder is chosen by environment, so set it here).
    fn connect(&self, name: &str) -> Result<Client, yendory_client::Error> {
        std::env::set_var("YENDORY_DATA_DIR", &self.data);
        Client::connect(name, "test")
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.data);
    }
}

// The tests share YENDORY_DATA_DIR (process environment), so they run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn programs_share_one_network_and_its_models() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let n = Network::start("share", 60);
    let hub = n.connect("Yendory Hub").unwrap();
    let studio = n.connect("Yendory Studio").unwrap();
    assert_eq!(hub.hello.api, yendory_client::API);
    assert_ne!(hub.hello.client_id, studio.hello.client_id);
    let studio_events = studio.take_events().unwrap();

    // The Hub makes a model; Studio hears about it and sees it.
    let made = hub
        .call(
            "models.create",
            json!({"name": "Robin", "architecture": "bicameral", "preset": "nova2", "context": 256}),
        )
        .unwrap();
    let id = made["meta"]["id"].as_str().unwrap().to_string();
    let e = studio_events.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!((e.name.as_str(), e.data["id"].as_str()), ("models.changed", Some(id.as_str())));
    let list = studio.call("models.list", json!({})).unwrap();
    assert_eq!(list.as_array().unwrap().len(), 1);

    // Studio opens it and asks it something; the untrained model answers something.
    studio.call("models.open", json!({"id": id})).unwrap();
    let r = studio.call("model.reply", json!({"id": id, "prompt": "U: hello\nN: ", "maxTokens": 8})).unwrap();
    assert!(r["text"].is_string());
    // The Hub can't reply without opening, nor rename or delete what Studio has open.
    assert_eq!(hub.call("model.reply", json!({"id": id, "prompt": "x"})).unwrap_err().code, "refused");
    let err = hub.call("models.rename", json!({"id": id, "name": "Wren"})).unwrap_err();
    assert!(err.message.contains("Yendory Studio has this model open"), "{}", err.message);
    assert_eq!(hub.call("models.delete", json!({"id": id})).unwrap_err().code, "refused");
    let listed = hub.call("models.list", json!({})).unwrap();
    assert_eq!(listed[0]["openBy"], json!(["Yendory Studio"]));
    // The Hub can't stop the Network under Studio.
    assert_eq!(hub.call("shutdown", json!({})).unwrap_err().code, "refused");

    // Studio leaves: its model is closed, and the Hub may change it.
    drop(studio);
    n.wait_for(
        || hub.call("models.list", json!({})).unwrap()[0]["openBy"] == json!([]),
        "Studio's model to close",
    );
    hub.call("models.rename", json!({"id": id, "name": "Wren"})).unwrap();
    hub.call("models.delete", json!({"id": id})).unwrap();
    assert_eq!(hub.call("models.list", json!({})).unwrap(), json!([]));
    assert_eq!(hub.call("no.such", json!({})).unwrap_err().code, "unknown_method");
}

#[test]
fn a_wrong_token_is_refused() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let n = Network::start("token", 60);
    let mut info = n.info().unwrap();
    info.token = "0".repeat(64);
    let path = n.data.join(paths::NETWORK_ID).join(paths::SERVICE_FILE);
    let real = std::fs::read(&path).unwrap();
    std::fs::write(&path, serde_json::to_vec(&info).unwrap()).unwrap();
    let err = n.connect("Intruder").err().expect("refused");
    assert_eq!(err.code, "unauthorized");
    std::fs::write(&path, real).unwrap();
    assert!(n.connect("Yendory Hub").is_ok());
}

#[test]
fn one_copy_runs_and_it_stops_when_idle() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut n = Network::start("idle", 2);
    let first = n.info().unwrap();
    // A second copy sees the first and leaves at once, without touching service.json.
    let status = Command::new(env!("CARGO_BIN_EXE_yendory-network"))
        .env("YENDORY_DATA_DIR", &n.data)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(n.info().unwrap(), first);
    // A connected program keeps it running past the idle time.
    let hub = n.connect("Yendory Hub").unwrap();
    std::thread::sleep(Duration::from_secs(3));
    assert_eq!(hub.call("ping", json!({})).unwrap(), json!("pong"));
    // Alone and idle, it stops and removes service.json.
    drop(hub);
    n.wait_for(|| n.info().is_none(), "the idle Network to stop");
    let code = n.child.wait().unwrap();
    assert!(code.success());
}

#[test]
fn shutdown_when_asked_and_alone() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut n = Network::start("shutdown", 60);
    let hub = n.connect("Yendory Hub").unwrap();
    let events = hub.take_events().unwrap();
    assert_eq!(hub.call("shutdown", json!({})).unwrap(), json!(true));
    let e = events.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(e.name, "network.stopping");
    assert!(n.child.wait().unwrap().success());
    assert!(n.info().is_none());
}

#[test]
fn version_is_reported_without_starting() {
    let out = Command::new(env!("CARGO_BIN_EXE_yendory-network")).arg("--version").output().unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["network"], env!("CARGO_PKG_VERSION"));
    assert_eq!(v["api"], yendory_client::API);
}
