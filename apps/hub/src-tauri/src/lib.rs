//! Yendory Hub backend: the suite's front door. It creates and imports models with the Yendory
//! Engine, keeps them on shelves, and installs and launches the other programs, handing them a
//! model to open (docs/HANDOFF.md).
//!
//! Models live where Studio keeps them: `<data folder>/org.yendory.studio/models/<id>/`
//! (`model.nnova2`, `meta.json`, `archive.json`, `history.jsonl`). Every program of the suite
//! reads that folder. The Hub's own files (shelves, settings, its log) are in its own data folder.
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Manager};

mod library;
mod programs;
mod shelves;
mod update;

pub(crate) type Res<T> = Result<T, String>;

/// Studio's identifier: its data folder holds the suite's models.
pub(crate) const STUDIO_ID: &str = "org.yendory.studio";

pub(crate) fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Studio's data folder (`app_data_dir` as Studio sees it).
pub(crate) fn studio_data_dir(app: &AppHandle) -> Res<PathBuf> {
    Ok(app.path().data_dir().map_err(|e| e.to_string())?.join(STUDIO_ID))
}

/// The Hub's own data folder.
pub(crate) fn hub_dir(app: &AppHandle) -> Res<PathBuf> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Writes a file so that a crash or power loss leaves either the old file or the new one, never
/// a mix (the same way Studio writes its files).
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

/// A JSON file of the Hub's, or `null` when it is missing or unreadable.
pub(crate) fn read_json(path: &Path) -> serde_json::Value {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

/// What the Hub is doing that must not be interrupted (one install or update at a time).
#[derive(Default)]
pub struct Hub {
    /// The program being downloaded and installed, if any.
    pub(crate) installing: Mutex<Option<String>>,
    pub(crate) cancel: Arc<AtomicBool>,
    /// Each program's newest release (its `latest.json`), as last fetched.
    pub(crate) manifests: Mutex<std::collections::BTreeMap<String, serde_json::Value>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct About {
    hub: &'static str,
    engine: String,
    models_folder: String,
    hub_folder: String,
}

/// Versions and folders, for the sidebar and Settings.
#[tauri::command]
fn about(app: AppHandle) -> Res<About> {
    Ok(About {
        hub: env!("CARGO_PKG_VERSION"),
        engine: env!("YENDORY_ENGINE_VERSION").to_string(),
        models_folder: library::models_dir(&app)?.display().to_string(),
        hub_folder: hub_dir(&app)?.display().to_string(),
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default().plugin(tauri_plugin_dialog::init());
    #[cfg(desktop)]
    let builder = builder.plugin(tauri_plugin_updater::Builder::new().build());
    builder
        .manage(Hub::default())
        .invoke_handler(tauri::generate_handler![
            about,
            library::list_models,
            library::model_options,
            library::create_model,
            library::import_model,
            library::rename_model,
            library::delete_model,
            library::export_model,
            shelves::shelves,
            shelves::add_shelf,
            shelves::rename_shelf,
            shelves::delete_shelf,
            shelves::move_model,
            programs::programs,
            programs::launch,
            programs::install_program,
            programs::cancel_install,
            programs::locate_program,
            programs::forget_location,
            programs::open_releases,
            update::check_hub_update,
            update::install_hub_update,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Yendory Hub");
}
