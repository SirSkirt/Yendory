//! Yendory packages (docs/PACKAGES.md): how the Hub installs and updates the suite's components
//! without any installer window of their own.
//!
//! Each component publishes `package.json` on its `<id>-updates` release: its version, what it
//! needs (`requires`), and for each platform a signed zip and the file in it to run. The Hub
//! downloads the zip, checks its signature against the project's updater key, unpacks it into
//! `programs/<id>/<version>/` in its own folder, and records it in `programs/programs.json`. The
//! previous version's folder is removed once nothing runs from it.
use super::*;
use serde_json::{json, Value};
use std::io::Read;
use std::sync::atomic::Ordering;

const RELEASES: &str = "https://github.com/SirSkirt/Yendory/releases/download";

/// The project's updater public key (minisign; the same key signs every package).
const UPDATER_KEY: &str = "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IDdFODg4NzFBRTE1MTU5NzAKUldSd1dWSGhHb2VJZm9ZTklCb3A5L3lqd05wS3NnV0dpM0NkRFhGRVdTYjhNQ3lCaXU4MU1DazQK";

/// Development and tests only (feature `dev-overrides`): another release page and key.
fn override_env(name: &str) -> Option<String> {
    if cfg!(feature = "dev-overrides") {
        std::env::var(name).ok().filter(|v| !v.is_empty())
    } else {
        None
    }
}

pub(crate) fn manifest_url(id: &str) -> String {
    let base = override_env("YENDORY_RELEASES").unwrap_or_else(|| RELEASES.into());
    format!("{base}/{id}-updates/package.json")
}

/// This computer's key in a package's `platforms` (`None`: no packages for it).
pub(crate) fn platform() -> Option<&'static str> {
    if cfg!(all(windows, target_arch = "x86_64")) {
        Some("windows-x86_64")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("linux-x86_64")
    } else {
        None
    }
}

/// "0.9.1" → [0, 9, 1] (anything after a '-' or '+' is ignored).
pub(crate) fn version_parts(v: &str) -> Vec<u64> {
    v.split(['-', '+']).next().unwrap_or("").split('.').map(|x| x.parse().unwrap_or(0)).collect()
}
pub(crate) fn newer(a: &str, b: &str) -> bool {
    version_parts(a) > version_parts(b)
}
pub(crate) fn at_least(have: &str, need: &str) -> bool {
    version_parts(have) >= version_parts(need)
}

fn agent() -> ureq::Agent {
    // No limit on a whole download (packages are large), only on connecting and on a stall.
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(20))
        .timeout_read(std::time::Duration::from_secs(60))
        .build()
}

/// A component's newest package. `Ok(None)`: it hasn't published one (not released yet).
pub(crate) fn fetch_manifest(id: &str) -> Res<Option<Value>> {
    match agent().get(&manifest_url(id)).call() {
        Ok(resp) => {
            let text = resp.into_string().map_err(|e| format!("The release page could not be read: {e}"))?;
            let m: Value = serde_json::from_str(&text)
                .map_err(|e| format!("{id}'s package.json could not be read: {e}"))?;
            if m["format"].as_u64() != Some(1) || m["id"].as_str() != Some(id) {
                return Err(format!(
                    "{id}'s package.json is in a format this Hub doesn't know; update the Hub"
                ));
            }
            Ok(Some(m))
        }
        Err(ureq::Error::Status(404, _)) => Ok(None),
        Err(e) => Err(format!("Could not reach the release page: {e}")),
    }
}

/// Checks downloaded bytes against their signature (from `package.json`) and the updater key.
pub(crate) fn verify(data: &[u8], signature: &str) -> Res<()> {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD;
    let decode = |s: &str| {
        b64.decode(s.trim())
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .ok_or_else(|| "The package's signature could not be read".to_string())
    };
    let key_text = override_env("YENDORY_UPDATER_KEY").unwrap_or_else(|| UPDATER_KEY.into());
    let key = minisign_verify::PublicKey::decode(&decode(&key_text)?).map_err(|e| e.to_string())?;
    let sig = minisign_verify::Signature::decode(&decode(signature)?)
        .map_err(|_| "The package's signature could not be read".to_string())?;
    key.verify(data, &sig, true)
        .map_err(|_| "The download is not signed by the Yendory project, so it was not installed".to_string())
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Progress {
    pub id: String,
    /// downloading, checking, installing, removing, switching.
    pub stage: &'static str,
    pub received: u64,
    pub total: Option<u64>,
    pub note: Option<String>,
}

pub(crate) fn progress(app: &AppHandle, id: &str, stage: &'static str, note: Option<String>) {
    let _ = app.emit("install-progress", Progress { id: id.into(), stage, received: 0, total: None, note });
}

/// Downloads a package for this computer and checks its signature.
pub(crate) fn download(app: &AppHandle, id: &str, build: &Value, cancel: &AtomicBool) -> Res<Vec<u8>> {
    let url = build["url"].as_str().ok_or("The package has no download")?;
    let signature =
        build["signature"].as_str().ok_or("The package is not signed, so it was not installed")?;
    let resp = agent().get(url).call().map_err(|e| format!("The download could not start: {e}"))?;
    let total =
        resp.header("Content-Length").and_then(|v| v.parse::<u64>().ok()).or_else(|| build["size"].as_u64());
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
                Progress {
                    id: id.into(),
                    stage: "downloading",
                    received: data.len() as u64,
                    total,
                    note: None,
                },
            );
        }
    }
    if total.is_some_and(|t| t != data.len() as u64) {
        return Err("The download ended early; try again".into());
    }
    progress(app, id, "checking", None);
    verify(&data, signature)?;
    Ok(data)
}

// ---------------------------------------------------------------------------------------------
// Installed components: programs/<id>/<version>/, recorded in programs/programs.json
// ---------------------------------------------------------------------------------------------

/// Where the Hub keeps the components it installs (not roaming on Windows).
pub(crate) fn programs_dir(app: &AppHandle) -> Res<PathBuf> {
    let dir = app.path().app_local_data_dir().map_err(|e| e.to_string())?.join("programs");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn registry_file(app: &AppHandle) -> Res<PathBuf> {
    Ok(programs_dir(app)?.join("programs.json"))
}

/// A component the Hub installed: `{version, path, folder, installed}`.
pub(crate) fn installed(app: &AppHandle, id: &str) -> Option<Value> {
    let r = read_json(&registry_file(app).ok()?);
    let entry = r.get(id)?.clone();
    let present = entry["path"].as_str().is_some_and(|p| Path::new(p).is_file());
    present.then_some(entry)
}

fn record(app: &AppHandle, id: &str, entry: Option<Value>) -> Res<()> {
    let path = registry_file(app)?;
    let mut r = read_json(&path);
    if !r.is_object() {
        r = json!({});
    }
    match entry {
        Some(e) => r[id] = e,
        None => {
            r.as_object_mut().map(|m| m.remove(id));
        }
    }
    write_atomic(&path, &serde_json::to_vec_pretty(&r).map_err(|e| e.to_string())?)
}

/// Unpacks a package's zip into `dir` (which must not exist). Paths that would leave the folder
/// are refused; Unix permissions are kept.
pub(crate) fn unpack(data: &[u8], dir: &Path) -> Res<()> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(data))
        .map_err(|e| format!("The package could not be opened: {e}"))?;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    for i in 0..zip.len() {
        let mut f = zip.by_index(i).map_err(|e| e.to_string())?;
        let rel = f.enclosed_name().ok_or("The package holds a path outside its folder")?;
        let out = dir.join(rel);
        if f.is_dir() {
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut w = std::fs::File::create(&out).map_err(|e| e.to_string())?;
        std::io::copy(&mut f, &mut w).map_err(|e| e.to_string())?;
        w.sync_all().map_err(|e| e.to_string())?;
        #[cfg(unix)]
        if let Some(mode) = f.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&out, std::fs::Permissions::from_mode(mode & 0o777))
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// The file to run in a package (`run`), which must be a plain relative path inside it.
pub(crate) fn run_path(build: &Value) -> Res<PathBuf> {
    let run = build["run"].as_str().ok_or("The package doesn't say what to run")?;
    let p = Path::new(run);
    if p.is_absolute() || p.components().any(|c| !matches!(c, std::path::Component::Normal(_))) {
        return Err("The package names a file to run outside its folder".into());
    }
    Ok(p.to_path_buf())
}

/// Puts a downloaded package in place as `programs/<id>/<version>/` and returns the file to run.
/// Nothing is recorded yet (see [`commit`]).
pub(crate) fn place(app: &AppHandle, id: &str, version: &str, build: &Value, data: &[u8]) -> Res<PathBuf> {
    if version.is_empty() || !version.chars().all(|c| c.is_ascii_alphanumeric() || ".-+".contains(c)) {
        return Err("The package's version is not valid".into());
    }
    let run = run_path(build)?;
    let base = programs_dir(app)?.join(id);
    let partial = base.join(format!("{version}.partial"));
    let _ = std::fs::remove_dir_all(&partial);
    if let Err(e) = unpack(data, &partial) {
        let _ = std::fs::remove_dir_all(&partial);
        return Err(e);
    }
    let exe = partial.join(&run);
    if !exe.is_file() {
        let _ = std::fs::remove_dir_all(&partial);
        return Err(format!("The package has no {}", run.display()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&exe).map(|m| m.permissions().mode()).unwrap_or(0o644);
        let _ = std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(mode | 0o755));
    }
    let folder = base.join(version);
    if folder.exists() {
        // The same version again (a repair): the old copy goes aside first.
        let aside = base.join(format!("{version}.old-{}", now()));
        std::fs::rename(&folder, &aside)
            .map_err(|e| format!("The installed copy is in use ({e}); close it and try again"))?;
    }
    std::fs::rename(&partial, &folder).map_err(|e| e.to_string())?;
    Ok(folder.join(run))
}

/// Records an installed version as the one to run, and removes the component's other versions
/// (a folder still in use is left for the next time).
pub(crate) fn commit(app: &AppHandle, id: &str, version: &str, exe: &Path) -> Res<()> {
    let folder = programs_dir(app)?.join(id).join(version);
    record(
        app,
        id,
        Some(
            json!({"version": version, "path": exe.display().to_string(), "folder": folder.display().to_string(), "installed": now()}),
        ),
    )?;
    prune(app, id, Some(version));
    Ok(())
}

/// Removes a component's version folders other than `keep`.
pub(crate) fn prune(app: &AppHandle, id: &str, keep: Option<&str>) {
    let Ok(base) = programs_dir(app).map(|d| d.join(id)) else { return };
    let Ok(entries) = std::fs::read_dir(&base) else { return };
    for e in entries.flatten() {
        if Some(e.file_name().to_string_lossy().as_ref()) != keep {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// Forgets a component and removes its files.
pub(crate) fn uninstall(app: &AppHandle, id: &str) -> Res<()> {
    record(app, id, None)?;
    prune(app, id, None);
    let _ = std::fs::remove_dir(programs_dir(app)?.join(id));
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Shortcuts: the Start menu (Windows) and the applications menu (Linux)
// ---------------------------------------------------------------------------------------------

/// Makes or replaces a program's menu shortcut (it points at the installed version, so it is
/// written again after every update).
pub(crate) fn shortcut(name: &str, exe: &Path, icon: Option<&Path>) -> Res<()> {
    #[cfg(windows)]
    {
        let dir = PathBuf::from(std::env::var_os("APPDATA").ok_or("No Start menu")?)
            .join(r"Microsoft\Windows\Start Menu\Programs");
        let lnk = dir.join(format!("{name}.lnk"));
        let q = |p: &Path| p.display().to_string().replace('\'', "''");
        let script = format!(
            "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{}');$s.TargetPath='{}';$s.WorkingDirectory='{}';$s.Save()",
            q(&lnk),
            q(exe),
            q(exe.parent().unwrap_or(exe))
        );
        let _ = icon;
        hidden("powershell", &["-NoProfile", "-NonInteractive", "-Command", &script])?;
        Ok(())
    }
    #[cfg(target_os = "linux")]
    {
        let dir = dirs_home().join(".local/share/applications");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let file = dir.join(format!("{}.desktop", name.to_lowercase().replace(' ', "-")));
        let mut text = format!(
            "[Desktop Entry]\nType=Application\nName={name}\nExec=\"{}\" %U\nTerminal=false\nCategories=Education;\n",
            exe.display()
        );
        if let Some(icon) = icon.filter(|i| i.is_file()) {
            text.push_str(&format!("Icon={}\n", icon.display()));
        }
        write_atomic(&file, text.as_bytes())
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = (name, exe, icon);
        Ok(())
    }
}

pub(crate) fn remove_shortcut(name: &str) {
    #[cfg(windows)]
    if let Some(d) = std::env::var_os("APPDATA") {
        let _ = std::fs::remove_file(
            PathBuf::from(d).join(format!(r"Microsoft\Windows\Start Menu\Programs\{name}.lnk")),
        );
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::fs::remove_file(
            dirs_home()
                .join(".local/share/applications")
                .join(format!("{}.desktop", name.to_lowercase().replace(' ', "-"))),
        );
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    let _ = name;
}

#[cfg(target_os = "linux")]
fn dirs_home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp"))
}

/// Runs a helper without a window and waits for it.
#[cfg(windows)]
pub(crate) fn hidden(program: &str, args: &[&str]) -> Res<std::process::ExitStatus> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    std::process::Command::new(program)
        .args(args)
        .creation_flags(CREATE_NO_WINDOW)
        .status()
        .map_err(|e| format!("{program} could not be run: {e}"))
}

// ---------------------------------------------------------------------------------------------
// The Hub updating itself
// ---------------------------------------------------------------------------------------------

/// How this Hub was installed, which decides how it replaces itself.
pub(crate) enum SelfInstall {
    /// Windows, per-user install folder (writable): the exe is swapped in place.
    Folder(PathBuf),
    /// Linux AppImage: the AppImage file is swapped in place.
    AppImage(PathBuf),
    /// Linux system package: the new .deb is installed with the system's permission prompt.
    Deb,
    /// A development build, or a folder the Hub may not write.
    Unsupported(String),
}

pub(crate) fn self_install() -> SelfInstall {
    if let Some(app_image) = std::env::var_os("APPIMAGE").map(PathBuf::from).filter(|p| p.is_file()) {
        return SelfInstall::AppImage(app_image);
    }
    let Ok(exe) = std::env::current_exe() else {
        return SelfInstall::Unsupported("The Hub can't find itself".into());
    };
    if cfg!(debug_assertions) {
        return SelfInstall::Unsupported("A development build doesn't update itself".into());
    }
    if cfg!(target_os = "linux") && exe.starts_with("/usr") {
        return SelfInstall::Deb;
    }
    if cfg!(windows) {
        let dir = exe.parent().map(Path::to_path_buf).unwrap_or_default();
        let probe = dir.join(".yendory-write-test");
        if std::fs::write(&probe, b"").is_ok() {
            let _ = std::fs::remove_file(probe);
            return SelfInstall::Folder(exe);
        }
        return SelfInstall::Unsupported(
            "The Hub is installed where it may not write; install the new version from the release page"
                .into(),
        );
    }
    SelfInstall::Unsupported(
        "The Hub can update itself only when installed from its .exe, .deb or AppImage".into(),
    )
}

/// The platform key for the Hub's own package on this computer.
pub(crate) fn self_platform(how: &SelfInstall) -> Option<String> {
    let p = platform()?;
    match how {
        SelfInstall::Folder(_) => Some(p.to_string()),
        SelfInstall::AppImage(_) => Some(format!("{p}-appimage")),
        SelfInstall::Deb => Some(format!("{p}-deb")),
        SelfInstall::Unsupported(_) => None,
    }
}

/// Puts a verified Hub package in place of the running Hub. Returns the file to start.
pub(crate) fn replace_self(how: &SelfInstall, build: &Value, data: &[u8], version: &str) -> Res<PathBuf> {
    match how {
        SelfInstall::Folder(exe) | SelfInstall::AppImage(exe) => {
            // The new file comes from the package's zip.
            let tmp = std::env::temp_dir().join(format!("yendory-hub-{version}-{}", now()));
            let _ = std::fs::remove_dir_all(&tmp);
            unpack(data, &tmp)?;
            let new = tmp.join(run_path(build)?);
            if !new.is_file() {
                return Err("The Hub package has no Hub in it".into());
            }
            let staged = exe.with_extension("new");
            std::fs::copy(&new, &staged).map_err(|e| e.to_string())?;
            let _ = std::fs::remove_dir_all(&tmp);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755))
                    .map_err(|e| e.to_string())?;
            }
            // A running program's file can be renamed (Windows) or replaced (Linux), not
            // overwritten: the old one steps aside and is removed at the next start.
            let old = exe.with_extension("old");
            let _ = std::fs::remove_file(&old);
            std::fs::rename(exe, &old).map_err(|e| format!("The Hub could not replace itself: {e}"))?;
            if let Err(e) = std::fs::rename(&staged, exe) {
                let _ = std::fs::rename(&old, exe);
                return Err(format!("The Hub could not replace itself: {e}"));
            }
            #[cfg(windows)]
            set_windows_version(version);
            Ok(exe.clone())
        }
        SelfInstall::Deb => {
            #[cfg(target_os = "linux")]
            {
                let file = std::env::temp_dir().join(format!("yendory-hub_{version}.deb"));
                write_atomic(&file, data)?;
                // System packages need the system's permission (its own password prompt).
                let status = std::process::Command::new("pkexec")
                    .arg("dpkg")
                    .arg("-i")
                    .arg(&file)
                    .status()
                    .map_err(|_| format!("Install {} with your package manager", file.display()))?;
                if !status.success() {
                    return Err(
                        "The new Hub was not installed (the system refused or it was cancelled)".into()
                    );
                }
                std::env::current_exe().map_err(|e| e.to_string())
            }
            #[cfg(not(target_os = "linux"))]
            {
                let _ = (build, data, version);
                Err("Not a system package".into())
            }
        }
        SelfInstall::Unsupported(why) => Err(why.clone()),
    }
}

/// Windows lists the Hub in Apps & features with its version; keep it right after an update.
#[cfg(windows)]
fn set_windows_version(version: &str) {
    use winreg::enums::{HKEY_CURRENT_USER, KEY_SET_VALUE};
    use winreg::RegKey;
    if let Ok(key) = RegKey::predef(HKEY_CURRENT_USER).open_subkey_with_flags(
        r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Yendory Hub",
        KEY_SET_VALUE,
    ) {
        let _ = key.set_value("DisplayVersion", &version);
    }
}

/// After an update: the old Hub file that stepped aside is removed.
pub(crate) fn clean_after_update() {
    let mut files = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        files.push(exe.with_extension("old"));
    }
    if let Some(a) = std::env::var_os("APPIMAGE") {
        files.push(PathBuf::from(a).with_extension("old"));
    }
    for f in files {
        // The old Hub may still be closing: try for a few seconds in the background.
        std::thread::spawn(move || {
            for _ in 0..20 {
                if !f.exists() || std::fs::remove_file(&f).is_ok() {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zip_of(files: &[(&str, &[u8], u32)]) -> Vec<u8> {
        let mut out = std::io::Cursor::new(Vec::new());
        {
            let mut z = zip::ZipWriter::new(&mut out);
            for (name, data, mode) in files {
                let opts = zip::write::SimpleFileOptions::default().unix_permissions(*mode);
                z.start_file(*name, opts).unwrap();
                z.write_all(data).unwrap();
            }
            z.finish().unwrap();
        }
        out.into_inner()
    }

    #[test]
    fn versions_compare_by_number() {
        assert!(newer("0.10.0", "0.9.9"));
        assert!(!newer("0.9.1", "0.9.1"));
        assert!(newer("0.9.2", "0.9.1-dev.98"));
        assert!(at_least("0.9.3", "0.9.3"));
        assert!(!at_least("0.9.2", "0.9.3"));
    }

    #[test]
    fn packages_unpack_inside_their_folder_only() {
        let dir = std::env::temp_dir().join(format!("yendory-unpack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        unpack(&zip_of(&[("bin/run", b"#!/bin/sh\n", 0o755), ("icon.png", b"x", 0o644)]), &dir).unwrap();
        assert!(dir.join("bin/run").is_file());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(dir.join("bin/run")).unwrap().permissions().mode() & 0o777, 0o755);
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(unpack(&zip_of(&[("../escape", b"x", 0o644)]), &dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
        assert!(run_path(&json!({"run": "../x"})).is_err());
        assert!(run_path(&json!({"run": "/bin/sh"})).is_err());
        assert_eq!(
            run_path(&json!({"run": "yendory-studio.exe"})).unwrap(),
            PathBuf::from("yendory-studio.exe")
        );
    }

    #[test]
    fn a_tampered_package_is_refused() {
        // A real signature (Studio 0.9.1's installer), for other bytes: refused.
        let sig = "dW50cnVzdGVkIGNvbW1lbnQ6IHNpZ25hdHVyZSBmcm9tIHRhdXJpIHNlY3JldCBrZXkKUlVSd1dWSGhHb2VJZnJTNE9QMGNDRUtnZURxREtaZUhVQUlHQnVGSThCMU9jbitCVjhnNncyazZDREFoYlhmSldkQnU5bjUwVSs2cmpNYWJUZ1lCUlpabjR5VkpJRHVZa0E4PQp0cnVzdGVkIGNvbW1lbnQ6IHRpbWVzdGFtcDoxNzkzMzkxMDMzCWZpbGU6WWVuZG9yeSBTdHVkaW9fMC45LjFfeDY0LXNldHVwLmV4ZQl2ZXJzaW9uOjAuOS4xCjUwT2NUa09VTU5JRVZidUFHSGQ2RFZuMnYweWg4S2NWdGhSbTE0WkpHQTJPUk9QQmFFVkQxVG5FUkpkcytLTlBoQmxLaTNuVHRxRmJGMXh4amI4REJ3PT0K";
        assert!(verify(b"not the package", sig).unwrap_err().contains("not signed"));
        assert!(verify(b"x", "garbage").is_err());
    }
}
