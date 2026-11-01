//! The suite's components as the Hub manages them: the Hub itself, the Yendory Network, Studio
//! and Tools. For each: where it is, whether it runs, the newest package, and installing,
//! updating, removing and launching it. A Studio installed the old way (its own installer, a
//! system package, the Hub 0.1 AppImage, or a file the owner chose) is found too, launched as it
//! is, and offered the switch to the Hub (switch.rs).
use super::*;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use tauri_plugin_dialog::DialogExt;

const RELEASE_PAGE: &str = "https://github.com/SirSkirt/Yendory/releases";

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    Hub,
    Service,
    Program,
}

pub(crate) struct Component {
    pub id: &'static str,
    pub name: &'static str,
    pub does: &'static str,
    pub kind: Kind,
    /// Process and executable names, lowercase, without `.exe`.
    pub binaries: &'static [&'static str],
}

pub(crate) const CATALOGUE: &[Component] = &[
    Component {
        id: "hub",
        name: "Yendory Hub",
        does: "This program: your models on shelves, and the suite's installer",
        kind: Kind::Hub,
        binaries: &["yendory-hub", "yendory hub"],
    },
    Component {
        id: "network",
        name: "Yendory Network",
        does: "Runs the Yendory Engine for every program, in the background",
        kind: Kind::Service,
        binaries: &["yendory-network"],
    },
    Component {
        id: "studio",
        name: "Yendory Studio",
        does: "Teaches models: reading, lessons, courses, exams",
        kind: Kind::Program,
        binaries: &["yendory-studio", "yendory studio"],
    },
    Component {
        id: "tools",
        name: "Yendory Tools",
        does: "Puts trained models to work in apps and games, through Nexus",
        kind: Kind::Program,
        binaries: &["yendory-tools", "yendory tools"],
    },
];

pub(crate) fn component(id: &str) -> Res<&'static Component> {
    CATALOGUE.iter().find(|c| c.id == id).ok_or_else(|| "Unknown program".into())
}

/// Studio hands a model to its running copy from this version on (single-instance hand-off).
const STUDIO_FORWARDS_FROM: &str = "0.9.3";

/// Whether a component of the suite is running on this computer.
pub(crate) fn running(id: &str) -> bool {
    let Ok(c) = component(id) else { return false };
    if c.kind == Kind::Service {
        return yendory_client::service_info().is_some_and(|s| pid_alive(s.pid));
    }
    use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};
    let mut sys = System::new();
    sys.refresh_processes_specifics(ProcessesToUpdate::All, true, ProcessRefreshKind::nothing());
    sys.processes().values().any(|proc| {
        let name = proc.name().to_string_lossy().to_lowercase();
        // An AppImage's process is named after its file ("Yendory-Studio.AppImage").
        proc.status() != sysinfo::ProcessStatus::Zombie
            && c.binaries.iter().any(|b| name.starts_with(b) || name.starts_with(&b.replace(' ', "-")))
    })
}

fn pid_alive(pid: u32) -> bool {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
    let mut sys = System::new();
    let pid = Pid::from_u32(pid);
    sys.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, ProcessRefreshKind::nothing());
    sys.process(pid).is_some_and(|p| p.status() != sysinfo::ProcessStatus::Zombie)
}

// ---------------------------------------------------------------------------------------------
// Copies installed the old way
// ---------------------------------------------------------------------------------------------

/// A copy of a program the Hub did not install.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Legacy {
    /// installer (Windows, its own setup), system (.deb), hub-0.1 (the Hub 0.1 AppImage),
    /// located (a file the owner chose).
    pub kind: &'static str,
    pub path: String,
    pub version: Option<String>,
    /// Windows: the uninstaller and the install folder.
    #[serde(skip)]
    #[cfg_attr(not(windows), allow(dead_code))]
    pub uninstall: Option<String>,
    #[serde(skip)]
    #[cfg_attr(not(windows), allow(dead_code))]
    pub folder: Option<String>,
}

fn settings_file(app: &AppHandle) -> Res<PathBuf> {
    Ok(hub_dir(app)?.join("settings.json"))
}
fn update_json(path: &Path, f: impl FnOnce(&mut serde_json::Map<String, Value>)) -> Res<()> {
    let mut v = read_json(path);
    if !v.is_object() {
        v = json!({});
    }
    f(v.as_object_mut().expect("an object"));
    write_atomic(path, &serde_json::to_vec_pretty(&v).map_err(|e| e.to_string())?)
}

/// The Hub 0.1 AppImage of Studio (Linux), and its record.
pub(crate) fn hub01_appimage(app: &AppHandle) -> Option<(PathBuf, Option<String>)> {
    let dir = hub_dir(app).ok()?;
    let path = dir.join("programs").join("Yendory-Studio.AppImage");
    if !path.is_file() {
        return None;
    }
    let version = read_json(&dir.join("programs.json"))["studio"]["version"].as_str().map(str::to_string);
    Some((path, version))
}

#[cfg(windows)]
fn from_registry(product: &str, binaries: &[&str]) -> Option<Legacy> {
    use winreg::enums::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
    use winreg::RegKey;
    let clean = |s: String| s.trim().trim_matches('"').trim_end_matches(",0").trim_matches('"').to_string();
    for root in [HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE] {
        let Ok(key) = RegKey::predef(root)
            .open_subkey(format!(r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{product}"))
        else {
            continue;
        };
        let version: Option<String> = key.get_value("DisplayVersion").ok();
        let icon = key.get_value::<String, _>("DisplayIcon").map(clean).unwrap_or_default();
        let folder = key.get_value::<String, _>("InstallLocation").map(clean).ok();
        let uninstall = key.get_value::<String, _>("UninstallString").ok();
        let mut candidates = vec![PathBuf::from(&icon)];
        if let Some(loc) = &folder {
            candidates.push(Path::new(loc).join(format!("{}.exe", binaries[0])));
            candidates.push(Path::new(loc).join(format!("{product}.exe")));
        }
        let path = candidates.into_iter().find(|c| c.is_file());
        let folder =
            folder.or_else(|| path.as_ref().and_then(|p| p.parent()).map(|p| p.display().to_string()));
        if path.is_some() || uninstall.is_some() {
            return Some(Legacy {
                kind: "installer",
                path: path.map(|p| p.display().to_string()).unwrap_or_default(),
                version,
                uninstall,
                folder,
            });
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn from_system(binary: &str) -> Option<Legacy> {
    let path = PathBuf::from("/usr/bin").join(binary);
    if !path.is_file() {
        return None;
    }
    let version = std::process::Command::new("dpkg-query")
        .args(["-W", "-f=${Version}", binary])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|v| !v.is_empty());
    Some(Legacy { kind: "system", path: path.display().to_string(), version, uninstall: None, folder: None })
}

/// Every copy of a program the Hub did not install.
pub(crate) fn legacy(app: &AppHandle, c: &Component) -> Vec<Legacy> {
    let mut out = Vec::new();
    if c.kind != Kind::Program {
        return out;
    }
    if let Some(p) = read_json(&settings_file(app).unwrap_or_default())["locations"][c.id]
        .as_str()
        .filter(|p| Path::new(p).is_file())
    {
        out.push(Legacy { kind: "located", path: p.into(), version: None, uninstall: None, folder: None });
    }
    if c.id == "studio" {
        if let Some((path, version)) = hub01_appimage(app) {
            out.push(Legacy {
                kind: "hub-0.1",
                path: path.display().to_string(),
                version,
                uninstall: None,
                folder: None,
            });
        }
    }
    #[cfg(windows)]
    if let Some(l) = from_registry(c.name, c.binaries) {
        out.push(l);
    }
    #[cfg(target_os = "linux")]
    if let Some(l) = from_system(c.binaries[0]) {
        out.push(l);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// The Downloads & updates page
// ---------------------------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComponentView {
    id: &'static str,
    name: &'static str,
    does: &'static str,
    /// hub, service or program.
    kind: &'static str,
    /// Installed by the Hub (or the Hub itself).
    managed: bool,
    /// Installed by the Hub, or found installed the old way.
    installed: bool,
    version: Option<String>,
    path: Option<String>,
    /// Copies installed the old way (they can be switched to the Hub).
    legacy: Vec<Legacy>,
    /// The newest package's version, when it has been fetched; `released`: whether there is one.
    latest: Option<String>,
    released: Option<bool>,
    update: bool,
    running: bool,
    /// The Hub can install or update it now, with the reason when it can't.
    can_install: bool,
    install_note: Option<String>,
    installing: bool,
    /// The Hub: a downloaded update waits for a restart.
    restart_ready: Option<String>,
}

fn view(app: &AppHandle, hub: &Hub, c: &'static Component) -> ComponentView {
    let manifest = hub.manifests.lock().unwrap().get(c.id).cloned();
    let released = manifest.as_ref().map(|m| !m.is_null());
    let latest = manifest.as_ref().and_then(|m| m["version"].as_str()).map(str::to_string);
    let managed_entry = if c.kind == Kind::Hub { None } else { packages::installed(app, c.id) };
    let legacy = legacy(app, c);
    let (managed, version, path) = match (c.kind, &managed_entry) {
        (Kind::Hub, _) => (
            true,
            Some(HUB_VERSION.to_string()),
            std::env::current_exe().ok().map(|p| p.display().to_string()),
        ),
        (_, Some(e)) => {
            (true, e["version"].as_str().map(str::to_string), e["path"].as_str().map(str::to_string))
        }
        (_, None) => match legacy.first() {
            Some(l) => (false, l.version.clone(), Some(l.path.clone())),
            None => (false, None, None),
        },
    };
    let installed = managed || !legacy.is_empty();
    let build_here = |m: &Value| match c.kind {
        Kind::Hub => packages::self_platform(&packages::self_install())
            .is_some_and(|k| m["platforms"][k]["url"].is_string()),
        _ => packages::platform().is_some_and(|k| m["platforms"][k]["url"].is_string()),
    };
    let needs_hub = manifest
        .as_ref()
        .and_then(|m| m["requires"]["hub"].as_str())
        .filter(|v| !packages::at_least(HUB_VERSION, v));
    let install_note = if released == Some(false) {
        Some("Not released yet".to_string())
    } else if c.kind == Kind::Hub {
        match packages::self_install() {
            packages::SelfInstall::Unsupported(why) => Some(why),
            _ => None,
        }
    } else if packages::platform().is_none() {
        Some("The Hub installs programs on 64-bit Windows and Linux".into())
    } else if !managed && !legacy.is_empty() {
        Some("Installed on its own: switch it to the Hub to have the Hub keep it updated".into())
    } else {
        None
    }
    .or_else(|| needs_hub.map(|v| format!("Needs Hub {v} or newer: update the Hub first")))
    .or_else(|| {
        manifest
            .as_ref()
            .filter(|m| !m.is_null() && !build_here(m))
            .map(|_| "The newest release has no package for this computer".into())
    });
    let restart_ready = if c.kind == Kind::Hub { hub.restart_ready.lock().unwrap().clone() } else { None };
    ComponentView {
        id: c.id,
        name: c.name,
        does: c.does,
        kind: match c.kind {
            Kind::Hub => "hub",
            Kind::Service => "service",
            Kind::Program => "program",
        },
        update: managed
            && restart_ready.is_none()
            && matches!((&latest, &version), (Some(l), Some(v)) if packages::newer(l, v)),
        managed,
        installed,
        version,
        path,
        legacy,
        latest,
        released,
        running: c.kind == Kind::Hub || running(c.id),
        can_install: install_note.is_none() && released == Some(true),
        install_note,
        installing: hub.installing.lock().unwrap().as_deref() == Some(c.id),
        restart_ready,
    }
}

/// Every component. `check`: fetch each one's newest package first (needs the internet; a
/// failure keeps what was known and is not an error).
#[tauri::command]
pub async fn components(app: AppHandle, hub: tauri::State<'_, Hub>, check: bool) -> Res<Vec<ComponentView>> {
    if check {
        for c in CATALOGUE {
            if let Ok(m) = packages::fetch_manifest(c.id) {
                hub.manifests.lock().unwrap().insert(c.id.into(), m.unwrap_or(Value::Null));
            }
        }
    }
    Ok(CATALOGUE.iter().map(|c| view(&app, &hub, c)).collect())
}

// ---------------------------------------------------------------------------------------------
// Launching
// ---------------------------------------------------------------------------------------------

/// Starts a program, with a model to open when one is given.
#[tauri::command]
pub async fn launch(
    app: AppHandle,
    hub: tauri::State<'_, Hub>,
    id: String,
    model: Option<String>,
) -> Res<()> {
    let c = component(&id)?;
    if c.kind != Kind::Program {
        return Err("Only programs are started from the Hub".into());
    }
    let v = view(&app, &hub, c);
    let path = v.path.filter(|_| v.installed).ok_or_else(|| format!("{} is not installed", c.name))?;
    if running(c.id) {
        // Studio 0.9.3 and later hand a second start to the running copy.
        let forwards = c.id == "studio"
            && v.version.as_deref().is_some_and(|ver| packages::at_least(ver, STUDIO_FORWARDS_FROM));
        if !forwards {
            return Err(match model {
                Some(_) => format!("{} is already open. Switch to it and open the model there.", c.name),
                None => format!("{} is already open", c.name),
            });
        }
    }
    let mut cmd = std::process::Command::new(&path);
    if let Some(m) = &model {
        if m.is_empty() || !m.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '-') {
            return Err("Invalid model id".into());
        }
        cmd.args(["--model", m]);
    }
    if let Some(dir) = Path::new(&path).parent() {
        cmd.current_dir(dir);
    }
    let mut child = cmd.spawn().map_err(|e| format!("{} could not be started: {e}", c.name))?;
    // Collect its exit when it closes, so no finished process is left behind.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    if let Some(m) = &model {
        library::log(&app, m, "Hub", &format!("Opened in {}", c.name.trim_start_matches("Yendory ")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Installing, updating and removing
// ---------------------------------------------------------------------------------------------

/// The icon a package may carry for menu shortcuts.
fn package_icon(exe: &Path) -> Option<PathBuf> {
    exe.parent().map(|d| d.join("icon.png")).filter(|p| p.is_file())
}

/// Starts the Network just installed and checks it is the version expected.
fn check_network(exe: &Path, version: &str) -> Res<()> {
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--version");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().map_err(|e| format!("The new Yendory Network could not be started: {e}"))?;
    let v: Value = serde_json::from_slice(&out.stdout).unwrap_or_default();
    if v["network"].as_str() != Some(version) {
        return Err(format!("The package said Yendory Network {version}, but it holds another program"));
    }
    if v["api"].as_u64() != Some(yendory_client::API as u64) {
        return Err("This Yendory Network speaks another API than this Hub: update the Hub first".into());
    }
    Ok(())
}

/// Stops the running Network before it is replaced (refused while another program uses it).
fn stop_network(app: &AppHandle, hub: &Hub) -> Res<()> {
    let Some(pid) = yendory_client::service_info().map(|s| s.pid).filter(|p| pid_alive(*p)) else {
        return Ok(());
    };
    if let Err(e) = hub.call(app, "shutdown", json!({})) {
        if !e.contains("lost") {
            return Err(e);
        }
    }
    hub.disconnect();
    // Its process must be gone (it holds the lock that keeps one copy running), not only its
    // service.json.
    let until = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while pid_alive(pid) {
        if std::time::Instant::now() > until {
            return Err("The Yendory Network did not stop; try again".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(150));
    }
    Ok(())
}

/// Installs or updates one component from its newest package (not the Hub itself).
pub(crate) fn install_one(app: &AppHandle, hub: &Hub, c: &'static Component, depth: u8) -> Res<String> {
    let manifest =
        packages::fetch_manifest(c.id)?.ok_or_else(|| format!("{} is not released yet", c.name))?;
    hub.manifests.lock().unwrap().insert(c.id.into(), manifest.clone());
    let version = manifest["version"].as_str().ok_or("The package has no version")?.to_string();
    if let Some(need) = manifest["requires"]["hub"].as_str().filter(|v| !packages::at_least(HUB_VERSION, v)) {
        return Err(format!("{} {version} needs Hub {need} or newer: update the Hub first", c.name));
    }
    // What it needs from the suite comes first (Studio needs the Network).
    if let Some(reqs) = manifest["requires"].as_object() {
        for (id, need) in reqs {
            let (Ok(dep), Some(need)) = (component(id), need.as_str()) else { continue };
            if dep.kind == Kind::Hub || dep.id == c.id || depth > 2 {
                continue;
            }
            let have =
                packages::installed(app, dep.id).and_then(|e| e["version"].as_str().map(str::to_string));
            if !have.is_some_and(|h| packages::at_least(&h, need)) {
                install_one(app, hub, dep, depth + 1)?;
            }
        }
    }
    let key = packages::platform().ok_or("No packages for this computer")?;
    let build = manifest["platforms"][key].clone();
    if build.is_null() {
        return Err(format!("{} {version} has no package for this computer", c.name));
    }
    if c.kind == Kind::Program && running(c.id) {
        return Err(format!("Close {} first, then try again", c.name));
    }
    let data = packages::download(app, c.id, &build, &hub.cancel)?;
    packages::progress(app, c.id, "installing", None);
    let exe = packages::place(app, c.id, &version, &build, &data)?;
    let folder = packages::programs_dir(app)?.join(c.id).join(&version);
    let activated = (|| {
        match c.kind {
            Kind::Service => {
                check_network(&exe, &version)?;
                stop_network(app, hub)?;
                let dir = yendory_client::paths::network_dir().ok_or("No data folder")?;
                std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                let info = yendory_client::InstallInfo { version: version.clone(), path: exe.clone() };
                write_atomic(
                    &dir.join(yendory_client::paths::INSTALL_FILE),
                    &serde_json::to_vec_pretty(&info).map_err(|e| e.to_string())?,
                )?;
            }
            Kind::Program => packages::shortcut(c.name, &exe, package_icon(&exe).as_deref())?,
            Kind::Hub => {}
        }
        packages::commit(app, c.id, &version, &exe)
    })();
    if let Err(e) = activated {
        // The new version is not used; the one that was there stays.
        if packages::installed(app, c.id).and_then(|e| e["version"].as_str().map(str::to_string)).as_deref()
            != Some(&version)
        {
            let _ = std::fs::remove_dir_all(&folder);
        }
        return Err(e);
    }
    if c.kind == Kind::Service {
        // The new Network starts now, so the Hub keeps working.
        hub.network(app)?;
    }
    Ok(version)
}

/// Downloads and installs the Hub's own newest package; it takes effect at the restart.
fn update_self(app: &AppHandle, hub: &Hub) -> Res<String> {
    let manifest = packages::fetch_manifest("hub")?.ok_or("The Hub has no package yet")?;
    hub.manifests.lock().unwrap().insert("hub".into(), manifest.clone());
    let version = manifest["version"].as_str().ok_or("The package has no version")?.to_string();
    if !packages::newer(&version, HUB_VERSION) {
        return Err("This is the newest Hub".into());
    }
    let how = packages::self_install();
    let key = packages::self_platform(&how).ok_or_else(|| match &how {
        packages::SelfInstall::Unsupported(why) => why.clone(),
        _ => "No Hub package for this computer".into(),
    })?;
    let build = manifest["platforms"][&key].clone();
    if build.is_null() {
        return Err("The newest Hub has no package for this computer".into());
    }
    let data = packages::download(app, "hub", &build, &hub.cancel)?;
    packages::progress(app, "hub", "installing", None);
    let exe = packages::replace_self(&how, &build, &data, &version)?;
    *hub.restart_ready.lock().unwrap() = Some(version.clone());
    RESTART_INTO.lock().unwrap().replace(exe);
    Ok(version)
}

static RESTART_INTO: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Installs, updates or repairs a component. Returns every component's state.
#[tauri::command]
pub async fn install_component(
    app: AppHandle,
    hub: tauri::State<'_, Hub>,
    id: String,
) -> Res<Vec<ComponentView>> {
    let c = component(&id)?;
    {
        let mut busy = hub.installing.lock().unwrap();
        if let Some(other) = busy.as_deref() {
            return Err(format!(
                "Wait until {} has finished",
                component(other).map(|c| c.name).unwrap_or("the other install")
            ));
        }
        *busy = Some(c.id.into());
    }
    hub.cancel.store(false, Ordering::Release);
    let result = if c.kind == Kind::Hub {
        update_self(&app, &hub)
    } else if c.kind == Kind::Program
        && !packages::installed(&app, c.id).is_some()
        && !legacy(&app, c).is_empty()
    {
        Err(format!(
            "{} is installed on its own. Switch it to the Hub instead, so there is one copy.",
            c.name
        ))
    } else {
        install_one(&app, &hub, c, 0)
    };
    *hub.installing.lock().unwrap() = None;
    let version = result?;
    if c.kind != Kind::Hub {
        log_install(&app, c, &version);
    }
    Ok(CATALOGUE.iter().map(|c| view(&app, &hub, c)).collect())
}

fn log_install(app: &AppHandle, c: &Component, version: &str) {
    let path = match hub_dir(app) {
        Ok(d) => d.join("installs.jsonl"),
        Err(_) => return,
    };
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{}", json!({"time": now(), "id": c.id, "version": version}));
    }
}

/// Removes a program the Hub installed (its models and data are kept).
#[tauri::command]
pub async fn remove_component(
    app: AppHandle,
    hub: tauri::State<'_, Hub>,
    id: String,
) -> Res<Vec<ComponentView>> {
    let c = component(&id)?;
    if c.kind != Kind::Program {
        return Err(format!("{} can't be removed from the Hub", c.name));
    }
    if running(c.id) {
        return Err(format!("Close {} first", c.name));
    }
    packages::uninstall(&app, c.id)?;
    packages::remove_shortcut(c.name);
    Ok(CATALOGUE.iter().map(|c| view(&app, &hub, c)).collect())
}

#[tauri::command]
pub fn cancel_install(hub: tauri::State<'_, Hub>) {
    hub.cancel.store(true, Ordering::Release);
}

/// Starts the updated Hub and closes this one.
#[tauri::command]
pub fn restart_hub(app: AppHandle) -> Res<()> {
    let exe = RESTART_INTO.lock().unwrap().clone().ok_or("No update is waiting")?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.args(["--updated-from", HUB_VERSION]);
    cmd.spawn().map_err(|e| format!("The new Hub could not be started: {e}"))?;
    app.exit(0);
    Ok(())
}

/// Lets the owner point the Hub at a program installed somewhere it does not look.
#[tauri::command]
pub async fn locate_program(app: AppHandle, id: String) -> Res<bool> {
    let c = component(&id)?;
    let Some(picked) = app.dialog().file().set_title(format!("Where is {}?", c.name)).blocking_pick_file()
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
        locations[c.id] = Value::String(shown);
    })?;
    Ok(true)
}

pub(crate) fn forget_located(app: &AppHandle, id: &str) -> Res<()> {
    update_json(&settings_file(app)?, |m| {
        if let Some(l) = m.get_mut("locations").and_then(Value::as_object_mut) {
            l.remove(id);
        }
    })
}

#[tauri::command]
pub fn forget_location(app: AppHandle, id: String) -> Res<()> {
    forget_located(&app, component(&id)?.id)
}

/// Opens the release page in the browser.
#[tauri::command]
pub fn open_releases() -> Res<()> {
    let result = if cfg!(windows) {
        std::process::Command::new("cmd").args(["/C", "start", "", RELEASE_PAGE]).spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(RELEASE_PAGE).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(RELEASE_PAGE).spawn()
    };
    let mut child = result.map_err(|_| format!("Open {RELEASE_PAGE} in your browser"))?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}
