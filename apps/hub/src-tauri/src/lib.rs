//! Yendory Hub backend: the suite's front door. It keeps models on shelves, installs and updates
//! the suite's components (the Yendory Network, Studio, Tools, and the Hub itself) from signed
//! packages, and starts programs with a model to open (docs/HANDOFF.md).
//!
//! The Hub holds no Engine: every model is made, imported, renamed, deleted and exported through
//! the Yendory Network, the background service that runs the Engine (docs/NETWORK.md).
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};
use yendory_client::Client;

mod library;
mod packages;
mod programs;
mod shelves;
mod switch;

pub(crate) type Res<T> = Result<T, String>;

pub(crate) const HUB_VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// The Hub's own data folder (shelves, settings, its log).
pub(crate) fn hub_dir(app: &AppHandle) -> Res<PathBuf> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Writes a file so that a crash or power loss leaves either the old file or the new one.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Res<()> {
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    drop(f);
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }
    Ok(())
}

/// A JSON file, or `null` when it is missing or unreadable.
pub(crate) fn read_json(path: &Path) -> serde_json::Value {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

/// What the Hub is doing that must not be interrupted, and its link to the Network.
#[derive(Default)]
pub struct Hub {
    /// The component being downloaded and installed (one at a time).
    pub(crate) installing: Mutex<Option<String>>,
    pub(crate) cancel: Arc<AtomicBool>,
    /// Each component's newest package (its `package.json`), as last fetched.
    pub(crate) manifests: Mutex<std::collections::BTreeMap<String, serde_json::Value>>,
    /// The connection to the Yendory Network.
    pub(crate) network: Mutex<Option<Arc<Client>>>,
    /// A downloaded Hub update waiting for a restart.
    pub(crate) restart_ready: Mutex<Option<String>>,
}

impl Hub {
    /// The open connection to the Network, connecting (and starting it) when needed.
    pub(crate) fn network(&self, app: &AppHandle) -> Res<Arc<Client>> {
        let mut slot = self.network.lock().unwrap();
        if let Some(c) = slot.as_ref().filter(|c| c.is_alive()) {
            return Ok(c.clone());
        }
        let client = Arc::new(
            Client::start_and_connect("Yendory Hub", HUB_VERSION, Duration::from_secs(15))
                .map_err(|e| e.message)?,
        );
        // The Network's events reach the window (models changed, the Network stopping).
        if let Some(events) = client.take_events() {
            let app = app.clone();
            std::thread::spawn(move || {
                for e in events {
                    let _ = app.emit("network-event", serde_json::json!({"name": e.name, "data": e.data}));
                }
                let _ = app
                    .emit("network-event", serde_json::json!({"name": "network.disconnected", "data": null}));
            });
        }
        *slot = Some(client.clone());
        Ok(client)
    }

    /// Calls the Network; a lost connection is made again once.
    pub(crate) fn call(
        &self,
        app: &AppHandle,
        method: &str,
        params: serde_json::Value,
    ) -> Res<serde_json::Value> {
        let client = self.network(app)?;
        match client.call(method, params.clone()) {
            Err(e) if e.code == "disconnected" => {
                *self.network.lock().unwrap() = None;
                self.network(app)?.call(method, params).map_err(|e| e.message)
            }
            other => other.map_err(|e| e.message),
        }
    }

    /// Drops the connection (before the Network is replaced).
    pub(crate) fn disconnect(&self) {
        *self.network.lock().unwrap() = None;
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct About {
    hub: &'static str,
    models_folder: String,
    hub_folder: String,
    programs_folder: String,
    /// The Network's and the Engine's versions, when the Network answers.
    network: Option<String>,
    engine: Option<String>,
    network_error: Option<String>,
    /// The Hub was just updated from this version (shown once).
    updated_from: Option<String>,
}

static UPDATED_FROM: Mutex<Option<String>> = Mutex::new(None);

/// Versions and folders, for the sidebar and Settings. Connects to the Network (starting it).
#[tauri::command]
async fn about(app: AppHandle, hub: tauri::State<'_, Hub>) -> Res<About> {
    let (network, engine, network_error) = match hub.network(&app) {
        Ok(c) => (Some(c.hello.network.clone()), Some(c.hello.engine.clone()), None),
        Err(e) => (None, None, Some(e)),
    };
    Ok(About {
        hub: HUB_VERSION,
        models_folder: yendory_client::paths::models_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        hub_folder: hub_dir(&app)?.display().to_string(),
        programs_folder: packages::programs_dir(&app)?.display().to_string(),
        network,
        engine,
        network_error,
        updated_from: UPDATED_FROM.lock().unwrap().take(),
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // `--updated-from <version>`: the Hub restarted after updating itself.
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--updated-from") {
        *UPDATED_FROM.lock().unwrap() = args.get(i + 1).cloned();
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Hub::default())
        .setup(|app| {
            packages::clean_after_update();
            // A Studio data folder left aside by an interrupted switch is put back.
            switch::recover(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            about,
            library::list_models,
            library::model_options,
            library::create_model,
            library::plan_model,
            library::import_model,
            library::rename_model,
            library::delete_model,
            library::export_model,
            shelves::shelves,
            shelves::add_shelf,
            shelves::rename_shelf,
            shelves::delete_shelf,
            shelves::move_model,
            programs::components,
            programs::launch,
            programs::install_component,
            programs::remove_component,
            programs::cancel_install,
            programs::restart_hub,
            programs::locate_program,
            programs::forget_location,
            programs::open_releases,
            switch::switch_studio,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Yendory Hub");
}
