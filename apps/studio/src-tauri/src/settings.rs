//! Studio's settings file (`<app data>/settings.json`), versioned (0.9).
//!
//! - Every write keeps the previous file as `settings.bak.json` (when it could be read), then
//!   replaces `settings.json` atomically.
//! - A `settings.json` that cannot be read is never written over: the backup is used instead
//!   (and said so), and the damaged file is kept as `settings.damaged-<time>.json` before anything
//!   new is saved. Studio then runs on the backup or, with neither, on its defaults, and says so.
//! - `schemaVersion` says which layout the file has. Opening an older file keeps a copy of it
//!   (`settings.v<n>.json`) and upgrades it in place; settings Studio does not know (a newer
//!   version's) are kept as they are.
use super::*;
use std::path::Path;

/// The settings layout this Studio writes. 1: before 0.8.2 (no `schemaVersion`). 2: versioned.
pub(crate) const SETTINGS_VERSION: u64 = 2;
const FILE: &str = "settings.json";
const BACKUP: &str = "settings.bak.json";

type Map = serde_json::Map<String, serde_json::Value>;

/// Messages about the settings file for the owner (shown once by the page).
static NOTICES: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// One writer at a time, so two settings saved together both stay.
static LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn notice(text: String) {
    let mut n = NOTICES.lock().unwrap_or_else(|e| e.into_inner());
    if !n.contains(&text) {
        n.push(text);
    }
}

fn parse(bytes: &[u8]) -> Option<Map> {
    serde_json::from_slice::<serde_json::Value>(bytes).ok()?.as_object().cloned()
}

/// What reading the settings found.
#[derive(Debug, PartialEq)]
pub(crate) enum Found {
    /// No settings file yet.
    Nothing,
    Fine,
    /// The file could not be read; these are the backup's settings.
    FromBackup,
    /// Neither the file nor a backup could be read: defaults.
    Damaged,
}

/// The settings in `dir`: the file, or its backup when the file is damaged.
pub(crate) fn read_in(dir: &Path) -> (Map, Found) {
    let backup = || std::fs::read(dir.join(BACKUP)).ok().and_then(|b| parse(&b));
    match std::fs::read(dir.join(FILE)) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (Map::new(), Found::Nothing),
        Ok(bytes) if parse(&bytes).is_some() => (parse(&bytes).unwrap_or_default(), Found::Fine),
        _ => match backup() {
            Some(m) => (m, Found::FromBackup),
            None => (Map::new(), Found::Damaged),
        },
    }
}

/// Sets one setting in `dir`, keeping the previous file as the backup (or, when it was damaged,
/// under its own name: it is never written over).
pub(crate) fn write_in(dir: &Path, key: &str, value: serde_json::Value) -> Res<()> {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let (mut map, found) = read_in(dir);
    let file = dir.join(FILE);
    match found {
        Found::Fine => {
            if let Ok(bytes) = std::fs::read(&file) {
                write_atomic(&dir.join(BACKUP), &bytes)?;
            }
        }
        Found::FromBackup | Found::Damaged => {
            let kept = dir.join(format!("settings.damaged-{}.json", now()));
            std::fs::rename(&file, &kept).map_err(|e| {
                format!("The settings file could not be read, and could not be set aside to keep it ({e}); nothing was saved")
            })?;
            notice(format!(
                "The settings file could not be read. It is kept as {} and was not written over.",
                kept.file_name().and_then(|n| n.to_str()).unwrap_or("settings.damaged.json")
            ));
        }
        Found::Nothing => {}
    }
    map.insert(key.into(), value);
    let version = map.get("schemaVersion").and_then(|v| v.as_u64()).unwrap_or(0).max(SETTINGS_VERSION);
    map.insert("schemaVersion".into(), version.into());
    write_atomic(&file, &serde_json::to_vec_pretty(&map).map_err(|e| e.to_string())?)
}

/// Removes `key` from the settings file and from every copy of it (the backup, the copies kept
/// when the layout was upgraded): for a secret that must not stay in any of them.
pub(crate) fn scrub_in(dir: &Path, key: &str) -> Res<()> {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let entries = std::fs::read_dir(dir).map_err(|e| e.to_string())?;
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let copy =
            name == FILE || name == BACKUP || (name.starts_with("settings.v") && name.ends_with(".json"));
        if !copy {
            continue;
        }
        let Some(mut map) = std::fs::read(e.path()).ok().and_then(|b| parse(&b)) else { continue };
        if map.remove(key).is_some() {
            write_atomic(&e.path(), &serde_json::to_vec_pretty(&map).map_err(|e| e.to_string())?)?;
        }
    }
    Ok(())
}

/// Brings the settings in `dir` to this version's layout (at start-up). Returns what the owner
/// should know. Running it again changes nothing.
pub(crate) fn migrate_in(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let (map, found) = read_in(dir);
    match found {
        Found::Nothing => return out,
        Found::FromBackup => out.push(
            "The settings file could not be read; Studio uses the backup copy from before the last change. The damaged file is kept when a setting is next saved.".into(),
        ),
        Found::Damaged => {
            out.push(
                "The settings file could not be read and there is no backup: Studio uses its defaults for now. The file is kept (not written over) when a setting is next saved.".into(),
            );
            return out;
        }
        Found::Fine => {}
    }
    let version = map.get("schemaVersion").and_then(|v| v.as_u64()).unwrap_or(1);
    if version > SETTINGS_VERSION {
        out.push(format!(
            "The settings were saved by a newer Studio (layout {version}); settings this version does not know are kept as they are."
        ));
        return out;
    }
    if version < SETTINGS_VERSION && found == Found::Fine {
        // The original, as it was, before anything changes.
        let copy = dir.join(format!("settings.v{version}.json"));
        if !copy.exists() {
            if let Err(e) = std::fs::copy(dir.join(FILE), &copy) {
                out.push(format!("The settings could not be upgraded: a copy of them could not be kept ({e}). They are unchanged."));
                return out;
            }
        }
        // 1 → 2: the same settings, with their layout's version.
        if let Err(e) = write_in(dir, "schemaVersion", SETTINGS_VERSION.into()) {
            out.push(format!(
                "The settings could not be upgraded ({e}). They are unchanged; the copy {} is kept.",
                copy.display()
            ));
        }
    }
    out
}

fn data_dir(app: &AppHandle) -> Res<std::path::PathBuf> {
    app.path().app_data_dir().map_err(|e| e.to_string())
}

pub(crate) fn read(app: &AppHandle) -> Map {
    data_dir(app).map(|d| read_in(&d).0).unwrap_or_default()
}

pub(crate) fn write(app: &AppHandle, key: &str, value: serde_json::Value) -> Res<()> {
    write_in(&data_dir(app)?, key, value)
}

/// At start-up: the settings brought to this version's layout.
pub(crate) fn migrate(app: &AppHandle) {
    if let Ok(dir) = data_dir(app) {
        for n in migrate_in(&dir) {
            notice(n);
        }
    }
}

/// What the owner should know about the settings file (each message once).
#[tauri::command]
pub fn settings_notices() -> Vec<String> {
    std::mem::take(&mut *NOTICES.lock().unwrap_or_else(|e| e.into_inner()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp() -> std::path::PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir().join(format!("yendory-settings-{nanos:x}"));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Settings from before 0.8.2 are kept as they were (a copy) and gain their version; nothing
    /// else changes, and doing it again changes nothing.
    #[test]
    fn old_settings_are_upgraded_and_kept() {
        let dir = temp();
        let old = br#"{"trainingDevice":"gpu","reading":{"values":{"readTokens":"500000"}},"futureThing":7}"#;
        std::fs::write(dir.join(FILE), old).unwrap();
        assert!(migrate_in(&dir).is_empty());
        let (m, found) = read_in(&dir);
        assert_eq!(found, Found::Fine);
        assert_eq!(m["schemaVersion"], 2);
        assert_eq!(m["trainingDevice"], "gpu");
        assert_eq!(m["reading"]["values"]["readTokens"], "500000");
        assert_eq!(m["futureThing"], 7, "unknown settings stay");
        assert_eq!(std::fs::read(dir.join("settings.v1.json")).unwrap(), old);
        let before = std::fs::read(dir.join(FILE)).unwrap();
        assert!(migrate_in(&dir).is_empty());
        assert_eq!(std::fs::read(dir.join(FILE)).unwrap(), before, "idempotent");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A damaged settings file is never written over: the backup is used, and the damaged file
    /// is kept under its own name when a setting is saved.
    #[test]
    fn a_damaged_file_falls_back_to_the_backup_and_is_kept() {
        let dir = temp();
        write_in(&dir, "cpuThreads", 4.into()).unwrap();
        write_in(&dir, "gpuChecks", false.into()).unwrap();
        // The backup holds the file before the last change.
        assert_eq!(parse(&std::fs::read(dir.join(BACKUP)).unwrap()).unwrap()["cpuThreads"], 4);
        std::fs::write(dir.join(FILE), b"{\"cpuThreads\": 4, \"gpuCh").unwrap();
        let (m, found) = read_in(&dir);
        assert_eq!(found, Found::FromBackup);
        assert_eq!(m["cpuThreads"], 4);
        assert!(!migrate_in(&dir).is_empty(), "the owner is told");
        write_in(&dir, "studyMode", true.into()).unwrap();
        let kept: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("settings.damaged-"))
            .collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(std::fs::read(kept[0].path()).unwrap(), b"{\"cpuThreads\": 4, \"gpuCh");
        let (m, found) = read_in(&dir);
        assert_eq!(found, Found::Fine);
        assert_eq!((m["cpuThreads"].clone(), m["studyMode"].clone()), (4.into(), true.into()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn neither_file_nor_backup_means_defaults_and_nothing_is_lost() {
        let dir = temp();
        std::fs::write(dir.join(FILE), b"\x00\x01 not json").unwrap();
        let (m, found) = read_in(&dir);
        assert!(m.is_empty() && found == Found::Damaged);
        assert!(!migrate_in(&dir).is_empty());
        assert_eq!(std::fs::read(dir.join(FILE)).unwrap(), b"\x00\x01 not json", "start-up writes nothing");
        write_in(&dir, "x", 1.into()).unwrap();
        let kept = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .find(|e| e.file_name().to_string_lossy().starts_with("settings.damaged-"))
            .unwrap();
        assert_eq!(std::fs::read(kept.path()).unwrap(), b"\x00\x01 not json");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Settings from a newer Studio are read and kept, never downgraded.
    #[test]
    fn newer_settings_are_kept() {
        let dir = temp();
        std::fs::write(dir.join(FILE), br#"{"schemaVersion": 9, "newKey": {"a": 1}}"#).unwrap();
        assert!(!migrate_in(&dir).is_empty());
        write_in(&dir, "cpuThreads", 2.into()).unwrap();
        let (m, _) = read_in(&dir);
        assert_eq!((m["schemaVersion"].clone(), m["newKey"]["a"].clone()), (9.into(), 1.into()));
        let _ = std::fs::remove_dir_all(dir);
    }
}
