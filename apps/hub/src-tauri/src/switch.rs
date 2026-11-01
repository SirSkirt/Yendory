//! Switching a program installed the old way to the Hub: Studio from its own installer (Windows),
//! a system package (.deb), the Hub 0.1 AppImage, or a file the owner chose.
//!
//! The owner's data is never at the mercy of an old uninstaller:
//! 1. the Hub's package is downloaded and checked first (a failure changes nothing);
//! 2. Studio's data folders are moved aside (a rename, instant), so no uninstaller can reach them;
//! 3. the old copy is uninstalled;
//! 4. the data folders are put back;
//! 5. the Hub's copy is made the one to run, with its menu shortcut.
//!
//! A marker (`switch.json` in the Hub's folder) records the folders moved aside; if the Hub stops
//! before step 4, the next start puts them back ([`recover`]).
use super::*;
use programs::{component, Legacy};
use serde_json::{json, Value};

fn marker(app: &AppHandle) -> Res<PathBuf> {
    Ok(hub_dir(app)?.join("switch.json"))
}

/// Studio's data folders: roaming and local (the same folder on Linux).
fn data_folders(app: &AppHandle) -> Vec<PathBuf> {
    let id = yendory_client::paths::STUDIO_ID;
    let mut out = Vec::new();
    if let Some(d) = yendory_client::paths::data_dir() {
        out.push(d.join(id));
    }
    if std::env::var_os("YENDORY_DATA_DIR").is_none() {
        if let Ok(d) = app.path().local_data_dir() {
            out.push(d.join(id));
        }
    }
    out.sort();
    out.dedup();
    out.into_iter().filter(|p| p.exists()).collect()
}

/// Puts back data folders moved aside by a switch that did not finish.
pub(crate) fn recover(app: &AppHandle) {
    let Ok(path) = marker(app) else { return };
    let moved = read_json(&path);
    let Some(list) = moved["moved"].as_array() else { return };
    for m in list {
        let (Some(orig), Some(aside)) = (m["from"].as_str(), m["aside"].as_str()) else { continue };
        let _ = put_back(Path::new(orig), Path::new(aside));
    }
    let _ = std::fs::remove_file(path);
}

/// Moves a folder back to its place. Something an uninstaller left at the place goes aside.
fn put_back(orig: &Path, aside: &Path) -> Res<()> {
    if !aside.exists() {
        return Ok(());
    }
    if orig.exists() {
        let left = orig.with_extension(format!("left-by-uninstall-{}", now()));
        std::fs::rename(orig, &left).map_err(|e| e.to_string())?;
    }
    std::fs::rename(aside, orig)
        .map_err(|e| format!("{} could not be put back from {}: {e}", orig.display(), aside.display()))
}

/// Removes one copy installed the old way. The data folders are already aside.
fn remove_legacy(app: &AppHandle, l: &Legacy) -> Res<()> {
    match l.kind {
        "located" => programs::forget_located(app, "studio"),
        "hub-0.1" => {
            std::fs::remove_file(&l.path)
                .map_err(|e| format!("The old Studio AppImage could not be removed: {e}"))?;
            let rec = hub_dir(app)?.join("programs.json");
            let mut r = read_json(&rec);
            if let Some(m) = r.as_object_mut() {
                m.remove("studio");
                write_atomic(&rec, &serde_json::to_vec_pretty(&r).map_err(|e| e.to_string())?)?;
            }
            Ok(())
        }
        "installer" => {
            #[cfg(windows)]
            {
                let uninstaller = l
                    .uninstall
                    .as_deref()
                    .map(|s| s.trim().trim_matches('"').to_string())
                    .ok_or("The old Studio has no uninstaller")?;
                let folder = l.folder.clone().ok_or("The old Studio's folder is unknown")?;
                // `_?=` keeps the uninstaller in place, so the Hub waits for it to finish; `/S`
                // without a window.
                let status = packages::hidden(&uninstaller, &["/S", &format!("_?={folder}")])?;
                if !status.success() {
                    return Err(format!(
                        "The old Studio's uninstaller stopped with code {:?}",
                        status.code()
                    ));
                }
                let _ = std::fs::remove_dir_all(&folder);
                Ok(())
            }
            #[cfg(not(windows))]
            Err("Not a Windows install".into())
        }
        "system" => {
            #[cfg(target_os = "linux")]
            {
                // A system package needs the system's permission (its own password prompt).
                let status = std::process::Command::new("pkexec")
                    .args(["apt-get", "remove", "-y", "yendory-studio"])
                    .status()
                    .map_err(|_| "Remove it with: sudo apt remove yendory-studio".to_string())?;
                if !status.success() {
                    return Err(
                        "The system package was not removed (the system refused or it was cancelled)".into(),
                    );
                }
                Ok(())
            }
            #[cfg(not(target_os = "linux"))]
            Err("Not a system package".into())
        }
        other => Err(format!("Unknown kind of install: {other}")),
    }
}

/// Switches Studio, installed the old way, to the Hub. Returns every component's state.
#[tauri::command]
pub async fn switch_studio(app: AppHandle, hub: tauri::State<'_, Hub>) -> Res<Vec<programs::ComponentView>> {
    let c = component("studio")?;
    {
        let mut busy = hub.installing.lock().unwrap();
        if busy.is_some() {
            return Err("Wait until the other install has finished".into());
        }
        *busy = Some(c.id.into());
    }
    hub.cancel.store(false, std::sync::atomic::Ordering::Release);
    let result = switch_now(&app, &hub);
    *hub.installing.lock().unwrap() = None;
    result?;
    programs::components(app, hub, false).await
}

fn switch_now(app: &AppHandle, hub: &Hub) -> Res<()> {
    let c = component("studio")?;
    let olds = programs::legacy(app, c);
    if olds.is_empty() {
        return Err("No Studio installed on its own was found".into());
    }
    if programs::running(c.id) {
        return Err("Close Yendory Studio first, then switch".into());
    }
    // 1. The Hub's package, checked, before anything is removed.
    let manifest = packages::fetch_manifest(c.id)?.ok_or("Studio has not published a Hub package yet")?;
    let version = manifest["version"].as_str().ok_or("The package has no version")?.to_string();
    if let Some(need) = manifest["requires"]["hub"].as_str().filter(|v| !packages::at_least(HUB_VERSION, v)) {
        return Err(format!("Studio {version} needs Hub {need} or newer: update the Hub first"));
    }
    if let Some(reqs) = manifest["requires"].as_object() {
        for (id, need) in reqs {
            let (Ok(dep), Some(need)) = (component(id), need.as_str()) else { continue };
            if dep.kind != programs::Kind::Service {
                continue;
            }
            let have =
                packages::installed(app, dep.id).and_then(|e| e["version"].as_str().map(str::to_string));
            if !have.is_some_and(|h| packages::at_least(&h, need)) {
                programs::install_one(app, hub, dep, 1)?;
            }
        }
    }
    let key = packages::platform().ok_or("No packages for this computer")?;
    let build = manifest["platforms"][key].clone();
    if build.is_null() {
        return Err(format!("Studio {version} has no package for this computer"));
    }
    let data = packages::download(app, c.id, &build, &hub.cancel)?;
    packages::progress(app, c.id, "switching", Some("Putting your Studio data aside".into()));
    let exe = packages::place(app, c.id, &version, &build, &data)?;
    let folder = packages::programs_dir(app)?.join(c.id).join(&version);

    // 2. The data folders aside, recorded first so a crash can be recovered.
    let stamp = now();
    let moves: Vec<(PathBuf, PathBuf)> = data_folders(app)
        .into_iter()
        .map(|p| (p.clone(), p.with_extension(format!("keep-{stamp}"))))
        .collect();
    write_atomic(
        &marker(app)?,
        &serde_json::to_vec_pretty(&json!({"moved": moves.iter().map(|(f, a)| json!({"from": f, "aside": a})).collect::<Vec<Value>>()}))
            .map_err(|e| e.to_string())?,
    )?;
    let mut done: Vec<&(PathBuf, PathBuf)> = Vec::new();
    for m in &moves {
        if let Err(e) = std::fs::rename(&m.0, &m.1) {
            for d in &done {
                let _ = put_back(&d.0, &d.1);
            }
            let _ = std::fs::remove_file(marker(app)?);
            let _ = std::fs::remove_dir_all(&folder);
            return Err(format!(
                "Studio's data folder is in use and could not be set aside safely ({e}). Close every Yendory program and try again; nothing was changed."
            ));
        }
        done.push(m);
    }

    // 3. The old copies go.
    packages::progress(app, c.id, "switching", Some("Removing the old Studio".into()));
    let removed: Res<()> = olds.iter().try_for_each(|l| remove_legacy(app, l));

    // 4. The data comes back, whatever happened.
    let mut restore_error = None;
    for m in &moves {
        if let Err(e) = put_back(&m.0, &m.1) {
            restore_error = Some(e);
        }
    }
    if restore_error.is_none() {
        let _ = std::fs::remove_file(marker(app)?);
    }
    if let Some(e) = restore_error {
        return Err(format!("{e}. Your data is safe in that folder; the Hub tries again at its next start."));
    }
    if let Err(e) = removed {
        let _ = std::fs::remove_dir_all(&folder);
        return Err(format!(
            "{e}. Your data is unchanged; the old Studio may be partly removed: try the switch again."
        ));
    }

    // 5. The Hub's copy is the one to run.
    packages::progress(app, c.id, "switching", Some("Installing Studio through the Hub".into()));
    packages::shortcut(
        c.name,
        &exe,
        exe.parent().map(|d| d.join("icon.png")).filter(|p| p.is_file()).as_deref(),
    )?;
    packages::commit(app, c.id, &version, &exe)?;
    let kinds: Vec<&str> = olds.iter().map(|l| l.kind).collect();
    if let Ok(dir) = hub_dir(app) {
        if let Ok(mut f) =
            std::fs::OpenOptions::new().create(true).append(true).open(dir.join("installs.jsonl"))
        {
            let _ = writeln!(
                f,
                "{}",
                json!({"time": now(), "id": "studio", "version": version, "switchedFrom": kinds})
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_goes_back_even_over_what_an_uninstaller_left() {
        let base = std::env::temp_dir().join(format!("yendory-switch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let orig = base.join("org.yendory.studio");
        let aside = base.join("org.yendory.studio.keep-1");
        std::fs::create_dir_all(orig.join("models/m1")).unwrap();
        std::fs::write(orig.join("models/m1/meta.json"), b"{}").unwrap();
        std::fs::rename(&orig, &aside).unwrap();
        // An uninstaller made a new, empty folder at the place.
        std::fs::create_dir_all(&orig).unwrap();
        put_back(&orig, &aside).unwrap();
        assert!(orig.join("models/m1/meta.json").is_file());
        assert!(!aside.exists());
        let left: Vec<_> = std::fs::read_dir(&base)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(left.iter().any(|n| n.contains("left-by-uninstall")), "{left:?}");
        // Nothing aside: nothing to do.
        put_back(&orig, &aside).unwrap();
        let _ = std::fs::remove_dir_all(&base);
    }
}
