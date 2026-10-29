//! Updates on the desktop: Studio asks the release page for the newest build
//! (`latest.json` on the `studio-updates` release, written by the release workflow), and can
//! download and install it in place, then restart. Every update is signed with the project's
//! updater key and checked against the public key in `tauri.conf.json` before it is installed,
//! so only builds made by the release workflow are accepted. Installing over the old version
//! keeps everything in the app's data folder: models, datasets, chats and settings.
//! On Android, updates are installed from the APK as before.
use super::*;

/// A newer build than the one running.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInfo {
    current: String,
    version: String,
    notes: Option<String>,
    date: Option<String>,
}

/// Download progress of an update.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct UpdateProgress {
    received: u64,
    total: Option<u64>,
}

/// The newest build, when it is newer than this one (`None`: up to date).
#[tauri::command]
pub async fn check_update(app: AppHandle) -> Res<Option<UpdateInfo>> {
    #[cfg(desktop)]
    {
        use tauri_plugin_updater::UpdaterExt;
        let update = app
            .updater()
            .map_err(|e| e.to_string())?
            .check()
            .await
            .map_err(|e| format!("Could not check for updates: {e}"))?;
        Ok(update.map(|u| UpdateInfo {
            current: u.current_version.clone(),
            version: u.version.clone(),
            notes: u.body.clone(),
            date: u.date.map(|d| d.to_string()),
        }))
    }
    #[cfg(not(desktop))]
    {
        let _ = app;
        Err("On Android, install the new APK from the release page".into())
    }
}

/// Downloads the newest build, checks its signature, installs it over this one and restarts.
/// Refused while teaching, reading or downloading (they would be cut off).
#[tauri::command]
pub async fn install_update(app: AppHandle, studio: State<'_, Studio>) -> Res<()> {
    not_teaching(&studio)?;
    if studio.downloads.lock().unwrap_or_else(|e| e.into_inner()).current.is_some() {
        return Err("Wait until the downloads have finished".into());
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
            return Err("Studio is already up to date".into());
        };
        let mut received = 0u64;
        let progress = app.clone();
        update
            .download_and_install(
                move |chunk, total| {
                    received += chunk as u64;
                    let _ = progress.emit("update-progress", UpdateProgress { received, total });
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
        Err("On Android, install the new APK from the release page".into())
    }
}
