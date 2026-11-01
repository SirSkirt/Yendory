//! The suite's programs: where each one is installed, whether it is running, launching it with a
//! model to open (`--model <id>`, docs/HANDOFF.md), and installing or updating it from the
//! release page. Installers come from the program's `latest.json` (written by its release
//! workflow) and are checked against the project's updater key before they run, so only builds
//! made by the release workflow are installed.
use super::*;
use serde_json::{json, Value};
use std::io::Read;
use std::sync::atomic::Ordering;
use tauri::Emitter;
use tauri_plugin_dialog::DialogExt;

const RELEASES: &str = "https://github.com/SirSkirt/Yendory/releases";

/// The project's updater public key (the same as Studio's and the Hub's `tauri.conf.json`).
const UPDATER_KEY: &str = "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IDdFODg4NzFBRTE1MTU5NzAKUldSd1dWSGhHb2VJZm9ZTklCb3A5L3lqd05wS3NnV0dpM0NkRFhGRVdTYjhNQ3lCaXU4MU1DazQK";

struct Program {
    id: &'static str,
    name: &'static str,
    does: &'static str,
    /// Where its release workflow publishes `latest.json`; `None`: not released yet.
    manifest: Option<&'static str>,
    /// Its installer's product name (Windows: the uninstall entry's name and install folder).
    product: &'static str,
    /// Process and executable names, lowercase, without `.exe`.
    binaries: &'static [&'static str],
}

const CATALOGUE: &[Program] = &[
    Program {
        id: "studio",
        name: "Yendory Studio",
        does: "Teaches models: reading, lessons, courses, exams",
        manifest: Some("https://github.com/SirSkirt/Yendory/releases/download/studio-updates/latest.json"),
        product: "Yendory Studio",
        binaries: &["yendory-studio", "yendory studio"],
    },
    Program {
        id: "tools",
        name: "Yendory Tools",
        does: "Puts trained models to work in apps and games, through Nexus",
        manifest: None,
        product: "Yendory Tools",
        binaries: &["yendory-tools", "yendory tools"],
    },
];

fn program(id: &str) -> Res<&'static Program> {
    CATALOGUE.iter().find(|p| p.id == id).ok_or_else(|| "Unknown program".into())
}

/// This computer's key in a `latest.json` (`None`: no builds for it).
fn platform() -> Option<&'static str> {
    if cfg!(all(windows, target_arch = "x86_64")) {
        Some("windows-x86_64")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("linux-x86_64")
    } else {
        None
    }
}

/// Whether a program of the suite is running on this computer.
pub(crate) fn running(id: &str) -> bool {
    let Ok(p) = program(id) else { return false };
    use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};
    let mut sys = System::new();
    sys.refresh_processes_specifics(ProcessesToUpdate::All, true, ProcessRefreshKind::nothing());
    sys.processes().values().any(|proc| {
        let name = proc.name().to_string_lossy().to_lowercase();
        // An AppImage's process is named after its file ("Yendory-Studio.AppImage").
        proc.status() != sysinfo::ProcessStatus::Zombie
            && p.binaries.iter().any(|b| name.starts_with(b) || name.starts_with(&b.replace(' ', "-")))
    })
}

// ---------------------------------------------------------------------------------------------
// The Hub's records: chosen locations (settings.json) and what it installed (programs.json)
// ---------------------------------------------------------------------------------------------

fn settings_file(app: &AppHandle) -> Res<PathBuf> {
    Ok(hub_dir(app)?.join("settings.json"))
}
fn installed_file(app: &AppHandle) -> Res<PathBuf> {
    Ok(hub_dir(app)?.join("programs.json"))
}
fn update_json(path: &Path, f: impl FnOnce(&mut serde_json::Map<String, Value>)) -> Res<()> {
    let mut v = read_json(path);
    if !v.is_object() {
        v = json!({});
    }
    f(v.as_object_mut().expect("an object"));
    write_atomic(path, &serde_json::to_vec_pretty(&v).map_err(|e| e.to_string())?)
}

/// Where the Hub keeps the programs it installs itself (Linux: AppImages).
fn programs_dir(app: &AppHandle) -> Res<PathBuf> {
    let dir = hub_dir(app)?.join("programs");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}
fn appimage_path(app: &AppHandle, p: &Program) -> Res<PathBuf> {
    Ok(programs_dir(app)?.join(format!("{}.AppImage", p.product.replace(' ', "-"))))
}

// ---------------------------------------------------------------------------------------------
// Finding an installed program
// ---------------------------------------------------------------------------------------------

/// How a program was found.
#[derive(Clone, Copy, PartialEq)]
enum Source {
    /// The owner chose the file (Settings → Locate).
    Located,
    /// The Hub installed it (Linux AppImage).
    Hub,
    /// Its own installer (Windows).
    #[cfg_attr(not(windows), allow(dead_code))]
    Installer,
    /// A system package (Linux .deb).
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    System,
}
impl Source {
    fn key(self) -> &'static str {
        match self {
            Source::Located => "located",
            Source::Hub => "hub",
            Source::Installer => "installer",
            Source::System => "system",
        }
    }
}

struct Found {
    path: PathBuf,
    source: Source,
    version: Option<String>,
}

#[cfg(windows)]
fn from_registry(p: &Program) -> Option<Found> {
    use winreg::enums::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
    use winreg::RegKey;
    let clean = |s: String| s.trim().trim_matches('"').trim_end_matches(",0").trim_matches('"').to_string();
    for root in [HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE] {
        let Ok(key) = RegKey::predef(root)
            .open_subkey(format!(r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{}", p.product))
        else {
            continue;
        };
        let version: Option<String> = key.get_value("DisplayVersion").ok();
        let icon = key.get_value::<String, _>("DisplayIcon").map(clean).unwrap_or_default();
        let mut candidates = vec![PathBuf::from(icon)];
        if let Ok(loc) = key.get_value::<String, _>("InstallLocation").map(clean) {
            for b in exe_names(p) {
                candidates.push(Path::new(&loc).join(b));
            }
        }
        if let Some(path) = candidates.into_iter().find(|c| c.is_file()) {
            return Some(Found { path, source: Source::Installer, version });
        }
    }
    None
}

#[cfg(windows)]
fn exe_names(p: &Program) -> Vec<String> {
    vec![format!("{}.exe", p.binaries[0]), format!("{}.exe", p.product)]
}

/// Default install folders, when the uninstall entry is missing.
#[cfg(windows)]
fn from_folders(p: &Program) -> Option<Found> {
    let roots =
        ["LOCALAPPDATA", "ProgramFiles"].iter().filter_map(|v| std::env::var_os(v)).map(PathBuf::from);
    for root in roots {
        for b in exe_names(p) {
            let path = root.join(p.product).join(b);
            if path.is_file() {
                return Some(Found { path, source: Source::Installer, version: None });
            }
        }
    }
    None
}

/// A system package (.deb): the binary in /usr/bin and the package's version.
#[cfg(target_os = "linux")]
fn from_system(p: &Program) -> Option<Found> {
    let name = p.binaries[0];
    let path = PathBuf::from("/usr/bin").join(name);
    if !path.is_file() {
        return None;
    }
    let version = std::process::Command::new("dpkg-query")
        .args(["-W", "-f=${Version}", name])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|v| !v.is_empty());
    Some(Found { path, source: Source::System, version })
}

fn find(app: &AppHandle, p: &Program) -> Option<Found> {
    let recorded = read_json(&installed_file(app).ok()?);
    let recorded_version = |path: &Path| {
        let r = &recorded[p.id];
        (r["path"].as_str() == Some(&path.display().to_string()))
            .then(|| r["version"].as_str().map(str::to_string))
            .flatten()
    };
    let located = read_json(&settings_file(app).ok()?)["locations"][p.id].as_str().map(PathBuf::from);
    if let Some(path) = located.filter(|l| l.is_file()) {
        return Some(Found { version: recorded_version(&path), path, source: Source::Located });
    }
    if let Ok(path) = appimage_path(app, p) {
        if path.is_file() {
            return Some(Found { version: recorded_version(&path), path, source: Source::Hub });
        }
    }
    #[cfg(windows)]
    if let Some(f) = from_registry(p).or_else(|| from_folders(p)) {
        return Some(f);
    }
    #[cfg(target_os = "linux")]
    if let Some(f) = from_system(p) {
        return Some(f);
    }
    None
}

// ---------------------------------------------------------------------------------------------
// The programs page
// ---------------------------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgramView {
    id: &'static str,
    name: &'static str,
    does: &'static str,
    released: bool,
    installed: bool,
    path: Option<String>,
    /// located, hub, installer or system.
    source: Option<&'static str>,
    version: Option<String>,
    /// The newest release's version, when it has been fetched.
    latest: Option<String>,
    /// A newer release than the installed version (known versions only).
    update: bool,
    running: bool,
    /// The Hub can install (or update) it on this computer, with the reason when it can't.
    can_install: bool,
    install_note: Option<String>,
    /// Being downloaded or installed now.
    installing: bool,
}

/// "0.9.1" → [0, 9, 1] (anything after a '-' or '+' is ignored).
fn version_parts(v: &str) -> Vec<u64> {
    v.split(['-', '+']).next().unwrap_or("").split('.').map(|x| x.parse().unwrap_or(0)).collect()
}
fn newer(latest: &str, installed: &str) -> bool {
    version_parts(latest) > version_parts(installed)
}

fn fetch_manifest(url: &str) -> Res<Value> {
    let resp = ureq::get(url)
        .timeout(std::time::Duration::from_secs(15))
        .call()
        .map_err(|e| format!("Could not reach the release page: {e}"))?;
    let text = resp.into_string().map_err(|e| format!("The release page could not be read: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("The release page's latest.json could not be read: {e}"))
}

fn view(app: &AppHandle, hub: &Hub, p: &'static Program) -> ProgramView {
    let found = find(app, p);
    let manifest = hub.manifests.lock().unwrap().get(p.id).cloned();
    let latest = manifest.as_ref().and_then(|m| m["version"].as_str()).map(str::to_string);
    let version = found.as_ref().and_then(|f| f.version.clone());
    let has_build = |m: &Value| platform().is_some_and(|k| m["platforms"][k]["url"].is_string());
    let install_note = if p.manifest.is_none() {
        Some("Not released yet".to_string())
    } else if platform().is_none() {
        Some(
            "The Hub can install it on 64-bit Windows and Linux only; download it from the release page"
                .into(),
        )
    } else if found.as_ref().is_some_and(|f| f.source == Source::System) {
        Some(
            "Installed as a system package: update it by installing the new .deb from the release page"
                .into(),
        )
    } else if found.as_ref().is_some_and(|f| f.source == Source::Located) {
        Some(
            "You chose this file in Settings: update it the way you installed it, or forget the location"
                .into(),
        )
    } else if manifest.as_ref().is_some_and(|m| !has_build(m)) {
        Some("The newest release has no build for this computer".into())
    } else {
        None
    };
    ProgramView {
        id: p.id,
        name: p.name,
        does: p.does,
        released: p.manifest.is_some(),
        installed: found.is_some(),
        path: found.as_ref().map(|f| f.path.display().to_string()),
        source: found.as_ref().map(|f| f.source.key()),
        update: match (&latest, &version) {
            (Some(l), Some(v)) => newer(l, v),
            _ => false,
        },
        version,
        latest,
        running: running(p.id),
        can_install: install_note.is_none(),
        install_note,
        installing: hub.installing.lock().unwrap().as_deref() == Some(p.id),
    }
}

/// Every program of the suite. `check`: fetch each one's newest release first (needs the internet;
/// a failure leaves the last known release and is not an error).
#[tauri::command]
pub async fn programs(app: AppHandle, hub: tauri::State<'_, Hub>, check: bool) -> Res<Vec<ProgramView>> {
    if check {
        for p in CATALOGUE {
            if let Some(url) = p.manifest {
                if let Ok(m) = fetch_manifest(url) {
                    hub.manifests.lock().unwrap().insert(p.id.into(), m);
                }
            }
        }
    }
    Ok(CATALOGUE.iter().map(|p| view(&app, &hub, p)).collect())
}

/// Starts a program, with a model to open when one is given.
#[tauri::command]
pub fn launch(app: AppHandle, id: String, model: Option<String>) -> Res<()> {
    let p = program(&id)?;
    let found = find(&app, p).ok_or_else(|| format!("{} is not installed", p.name))?;
    if running(p.id) {
        return Err(match model {
            Some(_) => format!(
                "{} is already open. Switch to it and open the model there: a program that is already running can't be handed a model yet.",
                p.name
            ),
            None => format!("{} is already open", p.name),
        });
    }
    let mut cmd = std::process::Command::new(&found.path);
    if let Some(m) = &model {
        let dir = library::model_dir(&app, m)?;
        if !dir.join("meta.json").is_file() {
            return Err("This model is no longer in the models folder".into());
        }
        cmd.args(["--model", m]);
    }
    if let Some(dir) = found.path.parent() {
        cmd.current_dir(dir);
    }
    let mut child = cmd.spawn().map_err(|e| format!("{} could not be started: {e}", p.name))?;
    // Collect its exit when it closes, so no finished process is left behind (and taken for a
    // running one).
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    if let Some(m) = &model {
        library::log(&app, m, "Hub", &format!("Opened in {}", p.name.trim_start_matches("Yendory ")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Installing and updating
// ---------------------------------------------------------------------------------------------

/// Checks a download against its signature from `latest.json` and the project's updater key.
fn verify(data: &[u8], signature: &str) -> Res<()> {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD;
    let decode = |s: &str| {
        b64.decode(s.trim())
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .ok_or_else(|| "The installer's signature could not be read".to_string())
    };
    let key = minisign_verify::PublicKey::decode(&decode(UPDATER_KEY)?).map_err(|e| e.to_string())?;
    let sig = minisign_verify::Signature::decode(&decode(signature)?)
        .map_err(|_| "The installer's signature could not be read".to_string())?;
    key.verify(data, &sig, true).map_err(|_| {
        "The downloaded installer is not signed by the Yendory project; it was not installed".to_string()
    })
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Progress {
    id: String,
    /// downloading, checking, installing.
    stage: &'static str,
    received: u64,
    total: Option<u64>,
}

fn download(app: &AppHandle, id: &str, url: &str, cancel: &AtomicBool) -> Res<Vec<u8>> {
    // No limit on the whole download (installers are large), only on connecting and on a stall.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(20))
        .timeout_read(std::time::Duration::from_secs(60))
        .build();
    let resp = agent.get(url).call().map_err(|e| format!("The download could not start: {e}"))?;
    let total = resp.header("Content-Length").and_then(|v| v.parse::<u64>().ok());
    let mut reader = resp.into_reader();
    let mut data = Vec::with_capacity(total.unwrap_or(0).min(1 << 30) as usize);
    let mut buf = vec![0u8; 1 << 16];
    let mut last = std::time::Instant::now();
    loop {
        if cancel.load(Ordering::Acquire) {
            return Err("Cancelled".into());
        }
        let n = reader.read(&mut buf).map_err(|e| format!("The download was interrupted: {e}"))?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        if last.elapsed().as_millis() > 150 {
            last = std::time::Instant::now();
            let _ = app.emit(
                "install-progress",
                Progress { id: id.into(), stage: "downloading", received: data.len() as u64, total },
            );
        }
    }
    if total.is_some_and(|t| t != data.len() as u64) {
        return Err("The download ended early; try again".into());
    }
    Ok(data)
}

fn install_now(app: &AppHandle, hub: &Hub, p: &'static Program) -> Res<()> {
    let url = p.manifest.ok_or("Not released yet")?;
    if running(p.id) {
        return Err(format!("Close {} first, then install again", p.name));
    }
    let manifest = fetch_manifest(url)?;
    hub.manifests.lock().unwrap().insert(p.id.into(), manifest.clone());
    let v = view(app, hub, p);
    if !v.can_install {
        return Err(v.install_note.unwrap_or_else(|| "It can't be installed here".into()));
    }
    let key = platform().ok_or("No builds for this computer")?;
    let build = &manifest["platforms"][key];
    let (Some(file_url), Some(signature)) = (build["url"].as_str(), build["signature"].as_str()) else {
        return Err("The newest release has no build for this computer".into());
    };
    let version = manifest["version"].as_str().unwrap_or("").to_string();
    let data = download(app, p.id, file_url, &hub.cancel)?;
    let _ = app.emit(
        "install-progress",
        Progress {
            id: p.id.into(),
            stage: "checking",
            received: data.len() as u64,
            total: Some(data.len() as u64),
        },
    );
    verify(&data, signature)?;
    let _ = app.emit(
        "install-progress",
        Progress {
            id: p.id.into(),
            stage: "installing",
            received: data.len() as u64,
            total: Some(data.len() as u64),
        },
    );
    if cfg!(windows) {
        // The program's own installer, in its passive mode (a progress bar, no questions). It
        // installs over an earlier version and keeps that version's data.
        let name = file_url.rsplit('/').next().unwrap_or("setup.exe").replace("%20", " ");
        let path = std::env::temp_dir().join(name);
        write_atomic(&path, &data)?;
        let mut child = std::process::Command::new(&path)
            .arg("/P")
            .spawn()
            .map_err(|e| format!("The installer could not be started: {e}"))?;
        std::thread::spawn(move || {
            let _ = child.wait();
        });
    } else {
        let path = appimage_path(app, p)?;
        write_atomic(&path, &data)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .map_err(|e| e.to_string())?;
        }
        let shown = path.display().to_string();
        update_json(&installed_file(app)?, |m| {
            m.insert(p.id.into(), json!({"path": shown, "version": version, "time": now()}));
        })?;
    }
    Ok(())
}

/// Downloads the newest release of a program for this computer, checks its signature and
/// installs it (Windows: runs its installer; Linux: keeps the AppImage in the Hub's folder).
#[tauri::command]
pub async fn install_program(
    app: AppHandle,
    hub: tauri::State<'_, Hub>,
    id: String,
) -> Res<Vec<ProgramView>> {
    let p = program(&id)?;
    {
        let mut busy = hub.installing.lock().unwrap();
        if let Some(other) = busy.as_deref() {
            return Err(format!(
                "Wait until {} has finished installing",
                program(other).map(|p| p.name).unwrap_or("the other program")
            ));
        }
        *busy = Some(p.id.into());
    }
    hub.cancel.store(false, Ordering::Release);
    let result = install_now(&app, &hub, p);
    *hub.installing.lock().unwrap() = None;
    result?;
    Ok(CATALOGUE.iter().map(|p| view(&app, &hub, p)).collect())
}

#[tauri::command]
pub fn cancel_install(hub: tauri::State<'_, Hub>) {
    hub.cancel.store(true, Ordering::Release);
}

/// Lets the owner point the Hub at a program installed somewhere it does not look.
#[tauri::command]
pub async fn locate_program(app: AppHandle, id: String) -> Res<bool> {
    let p = program(&id)?;
    let Some(picked) = app.dialog().file().set_title(format!("Where is {}?", p.name)).blocking_pick_file()
    else {
        return Ok(false);
    };
    let path = picked.into_path().map_err(|e| e.to_string())?;
    let shown = path.display().to_string();
    update_json(&settings_file(&app)?, |m| {
        let locations = m.entry("locations").or_insert_with(|| json!({}));
        if !locations.is_object() {
            *locations = json!({});
        }
        locations[p.id] = Value::String(shown);
    })?;
    Ok(true)
}

#[tauri::command]
pub fn forget_location(app: AppHandle, id: String) -> Res<()> {
    let p = program(&id)?;
    update_json(&settings_file(&app)?, |m| {
        if let Some(l) = m.get_mut("locations").and_then(Value::as_object_mut) {
            l.remove(p.id);
        }
    })
}

/// Opens the release page in the browser.
#[tauri::command]
pub fn open_releases() -> Res<()> {
    let result = if cfg!(windows) {
        std::process::Command::new("cmd").args(["/C", "start", "", RELEASES]).spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(RELEASES).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(RELEASES).spawn()
    };
    let mut child = result.map_err(|_| format!("Open {RELEASES} in your browser"))?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_by_number() {
        assert!(newer("0.10.0", "0.9.9"));
        assert!(newer("1.0.0", "0.9.9"));
        assert!(!newer("0.9.1", "0.9.1"));
        assert!(!newer("0.9.1-dev.98", "0.9.1"));
        assert!(newer("0.9.2", "0.9.1-dev.98"));
    }

    #[test]
    fn a_tampered_installer_is_refused() {
        // A signature in the format latest.json carries, for other data: refused.
        let sig = "dW50cnVzdGVkIGNvbW1lbnQ6IHNpZ25hdHVyZSBmcm9tIHRhdXJpIHNlY3JldCBrZXkKUlVSd1dWSGhHb2VJZnJTNE9QMGNDRUtnZURxREtaZUhVQUlHQnVGSThCMU9jbitCVjhnNncyazZDREFoYlhmSldkQnU5bjUwVSs2cmpNYWJUZ1lCUlpabjR5VkpJRHVZa0E4PQp0cnVzdGVkIGNvbW1lbnQ6IHRpbWVzdGFtcDoxNzkzMzkxMDMzCWZpbGU6WWVuZG9yeSBTdHVkaW9fMC45LjFfeDY0LXNldHVwLmV4ZQl2ZXJzaW9uOjAuOS4xCjUwT2NUa09VTU5JRVZidUFHSGQ2RFZuMnYweWg4S2NWdGhSbTE0WkpHQTJPUk9QQmFFVkQxVG5FUkpkcytLTlBoQmxLaTNuVHRxRmJGMXh4amI4REJ3PT0K";
        assert!(verify(b"not the installer", sig).unwrap_err().contains("not signed"));
        assert!(verify(b"x", "garbage").is_err());
    }

    /// With YENDORY_HUB_INSTALLER (a downloaded installer) and YENDORY_HUB_SIGNATURE (its
    /// signature from latest.json): the real file passes. Run by hand:
    /// `cargo test --release -- --ignored real_installer`.
    #[test]
    #[ignore]
    fn a_real_installer_passes() {
        let data = std::fs::read(std::env::var("YENDORY_HUB_INSTALLER").unwrap()).unwrap();
        verify(&data, &std::env::var("YENDORY_HUB_SIGNATURE").unwrap()).unwrap();
    }
}
