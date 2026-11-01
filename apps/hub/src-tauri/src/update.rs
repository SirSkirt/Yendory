//! The Hub's own updates: it asks the release page for its newest build (`latest.json` on the
//! `hub-updates` release, written by the Hub's release workflow), downloads it, checks its
//! signature against the public key in `tauri.conf.json`, installs it over this version and
//! restarts. Installing over the old version keeps the Hub's data folder (shelves, settings) and
//! never touches the models folder.
use super::*;
use tauri::Emitter;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInfo {
    current: String,
    version: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct UpdateProgress {
    received: u64,
    total: Option<u64>,
}

/// The newest build of the Hub, when it is newer than this one (`None`: up to date).
#[tauri::command]
pub async fn check_hub_update(app: AppHandle) -> Res<Option<UpdateInfo>> {
    #[cfg(desktop)]
    {
        use tauri_plugin_updater::UpdaterExt;
        let update = app
            .updater()
            .map_err(|e| e.to_string())?
            .check()
            .await
            .map_err(|e| format!("Could not check for updates: {e}"))?;
        Ok(update.map(|u| UpdateInfo { current: u.current_version.clone(), version: u.version.clone() }))
    }
    #[cfg(not(desktop))]
    {
        let _ = app;
        Err("Updates are installed from the release page on this device".into())
    }
}

/// Downloads the newest Hub, checks its signature, installs it over this one and restarts.
/// Refused while the Hub is installing a program (it would be cut off).
#[tauri::command]
pub async fn install_hub_update(app: AppHandle, hub: tauri::State<'_, Hub>) -> Res<()> {
    if hub.installing.lock().unwrap().is_some() {
        return Err("Wait until the program being installed has finished".into());
    }
    #[cfg(desktop)]
    {
        use tauri_plugin_updater::UpdaterExt;
        let Some(update) = app
            .updater()
            .map_err(|e| e.to_string())?
            .check()
            .await
            .map_err(|e| format!("Could not check for updates: {e}"))?
        else {
            return Err("The Hub is already up to date".into());
        };
        let mut received = 0u64;
        let progress = app.clone();
        update
            .download_and_install(
                move |chunk, total| {
                    received += chunk as u64;
                    let _ = progress.emit("hub-update-progress", UpdateProgress { received, total });
                },
                || {},
            )
            .await
            .map_err(|e| format!("The update could not be installed: {e}"))?;
        // On Windows the installer has taken over by now; elsewhere, start the new version.
        app.restart();
    }
    #[cfg(not(desktop))]
    {
        let _ = app;
        Err("Updates are installed from the release page on this device".into())
    }
}
