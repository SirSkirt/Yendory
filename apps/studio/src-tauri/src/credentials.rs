//! The Hugging Face access token (0.8.4): kept in the system's credential store (Windows
//! Credential Manager; on Linux the Secret Service, your desktop's keyring; the Keychain on
//! macOS). Where there is none (Android, or a Linux session without a keyring) it is kept in a
//! file of its own in Studio's data folder that only your user account can read, and the page
//! says so. It is never kept in the settings file: a token stored there by an earlier version is
//! moved at start-up, and removed from the settings file and its backup copies.
//!
//! The token is sent only to huggingface.co (dataset downloads and the account check), never
//! back to the page, and never written to logs or history.
use super::*;
use std::path::Path;

#[cfg(desktop)]
const SERVICE: &str = "org.yendory.studio";
#[cfg(desktop)]
const ACCOUNT: &str = "huggingface-token";
/// The fallback file (in Studio's data folder).
const FILE: &str = "hf-token.secret";

/// Where the token is kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Kept {
    System,
    File,
}

impl Kept {
    pub fn describe(self) -> &'static str {
        match self {
            Kept::System if cfg!(windows) => "Windows Credential Manager",
            Kept::System if cfg!(target_os = "macos") => "the macOS Keychain",
            Kept::System => "your desktop's keyring (Secret Service)",
            Kept::File => "a file in Studio's data folder that only your user account can read (no system credential store was available)",
        }
    }
}

/// The system store, or why it cannot be used.
trait Store {
    fn set(&self, token: &str) -> Result<(), String>;
    fn get(&self) -> Result<Option<String>, String>;
    fn delete(&self) -> Result<(), String>;
}

#[cfg(desktop)]
struct System;

#[cfg(desktop)]
impl Store for System {
    fn set(&self, token: &str) -> Result<(), String> {
        keyring::Entry::new(SERVICE, ACCOUNT).and_then(|e| e.set_password(token)).map_err(|e| e.to_string())
    }
    fn get(&self) -> Result<Option<String>, String> {
        match keyring::Entry::new(SERVICE, ACCOUNT).and_then(|e| e.get_password()) {
            Ok(t) => Ok(Some(t)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }
    fn delete(&self) -> Result<(), String> {
        match keyring::Entry::new(SERVICE, ACCOUNT).and_then(|e| e.delete_credential()) {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// No system store on this platform.
#[cfg(not(desktop))]
struct System;

#[cfg(not(desktop))]
impl Store for System {
    fn set(&self, _: &str) -> Result<(), String> {
        Err("no system credential store on this platform".into())
    }
    fn get(&self) -> Result<Option<String>, String> {
        Ok(None)
    }
    fn delete(&self) -> Result<(), String> {
        Ok(())
    }
}

fn write_private(path: &Path, text: &str) -> Res<()> {
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("tmp");
    {
        let mut o = std::fs::OpenOptions::new();
        o.create(true).write(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
        let mut f = o.open(&tmp).map_err(|e| e.to_string())?;
        f.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

fn save_with(store: &dyn Store, dir: &Path, token: &str) -> Res<Kept> {
    match store.set(token) {
        Ok(()) => {
            let _ = std::fs::remove_file(dir.join(FILE));
            Ok(Kept::System)
        }
        Err(_) => {
            write_private(&dir.join(FILE), token)?;
            Ok(Kept::File)
        }
    }
}

fn load_with(store: &dyn Store, dir: &Path) -> Option<(String, Kept)> {
    if let Ok(Some(t)) = store.get() {
        return Some((t, Kept::System));
    }
    let t = std::fs::read_to_string(dir.join(FILE)).ok()?;
    let t = t.trim().to_string();
    (!t.is_empty()).then_some((t, Kept::File))
}

fn delete_with(store: &dyn Store, dir: &Path) -> Res<()> {
    let _ = std::fs::remove_file(dir.join(FILE));
    store.delete().map_err(|e| format!("The token could not be removed from the credential store: {e}"))
}

/// A token an earlier version kept in the settings file: moved to the credential store, then
/// removed from the settings file and every backup copy of it. Returns what the owner should
/// know. Nothing changes when there is none (or it cannot be moved).
fn migrate_with(store: &dyn Store, dir: &Path) -> Vec<String> {
    let (map, _) = settings::read_in(dir);
    let Some(token) = map.get("hfToken").and_then(|v| v.as_str()).map(str::trim).filter(|t| !t.is_empty())
    else {
        // An empty leftover key is removed quietly.
        if map.contains_key("hfToken") {
            let _ = settings::scrub_in(dir, "hfToken");
        }
        return Vec::new();
    };
    match save_with(store, dir, token) {
        Ok(kept) => match settings::scrub_in(dir, "hfToken") {
            Ok(()) => vec![format!(
                "Your Hugging Face token was moved out of the settings file into {}.",
                kept.describe()
            )],
            Err(e) => vec![format!(
                "Your Hugging Face token is now kept in {}, but it could not be removed from the settings file ({e}).",
                kept.describe()
            )],
        },
        Err(e) => vec![format!(
            "Your Hugging Face token could not be moved out of the settings file ({e}); it is still used from there."
        )],
    }
}

fn data_dir(app: &AppHandle) -> Res<std::path::PathBuf> {
    app.path().app_data_dir().map_err(|e| e.to_string())
}

/// At start-up: an old token moved out of the settings file, then the token loaded.
pub(crate) fn start(app: &AppHandle) -> Vec<String> {
    let Ok(dir) = data_dir(app) else { return Vec::new() };
    let notices = migrate_with(&System, &dir);
    let loaded = load_with(&System, &dir).or_else(|| {
        // Not movable: still read from the settings file, as before.
        settings::read_in(&dir).0.get("hfToken").and_then(|v| v.as_str()).map(|t| (t.to_string(), Kept::File))
    });
    *KEPT.lock().unwrap_or_else(|e| e.into_inner()) = loaded.as_ref().map(|(_, k)| *k);
    reading::set_token_in_memory(loaded.map(|(t, _)| t));
    notices
}

/// Where the token is kept now (None: no token).
static KEPT: Mutex<Option<Kept>> = Mutex::new(None);

pub(crate) fn save(app: &AppHandle, token: &str) -> Res<Kept> {
    let kept = save_with(&System, &data_dir(app)?, token)?;
    *KEPT.lock().unwrap_or_else(|e| e.into_inner()) = Some(kept);
    Ok(kept)
}

pub(crate) fn delete(app: &AppHandle) -> Res<()> {
    *KEPT.lock().unwrap_or_else(|e| e.into_inner()) = None;
    delete_with(&System, &data_dir(app)?)
}

pub(crate) fn kept() -> Option<Kept> {
    *KEPT.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A credential store in memory (or one that is unavailable).
    struct Fake {
        value: RefCell<Option<String>>,
        available: bool,
    }
    impl Store for Fake {
        fn set(&self, token: &str) -> Result<(), String> {
            if !self.available {
                return Err("unavailable".into());
            }
            *self.value.borrow_mut() = Some(token.into());
            Ok(())
        }
        fn get(&self) -> Result<Option<String>, String> {
            if !self.available {
                return Err("unavailable".into());
            }
            Ok(self.value.borrow().clone())
        }
        fn delete(&self) -> Result<(), String> {
            *self.value.borrow_mut() = None;
            Ok(())
        }
    }

    fn temp() -> std::path::PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir().join(format!("yendory-cred-{nanos:x}"));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn files_with(dir: &Path, needle: &str) -> Vec<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .filter(|e| std::fs::read_to_string(e.path()).is_ok_and(|t| t.contains(needle)))
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect()
    }

    /// A token in the settings file (and so in its backup and the copy kept when the settings
    /// were upgraded) ends up only in the credential store; the other settings stay.
    #[test]
    fn an_old_token_moves_to_the_store_and_leaves_no_copy() {
        let dir = temp();
        std::fs::write(dir.join("settings.json"), br#"{"hfToken": "hf_secret123", "cpuThreads": 4}"#)
            .unwrap();
        settings::migrate_in(&dir); // keeps settings.v1.json (with the token in it)
        settings::write_in(&dir, "gpuChecks", false.into()).unwrap(); // a backup with it too
        assert!(files_with(&dir, "hf_secret123").len() >= 3);
        let store = Fake { value: RefCell::new(None), available: true };
        let notes = migrate_with(&store, &dir);
        assert!(notes[0].contains("moved out of the settings file"), "{notes:?}");
        assert_eq!(store.value.borrow().as_deref(), Some("hf_secret123"));
        assert!(
            files_with(&dir, "hf_secret123").is_empty(),
            "no copy left: {:?}",
            files_with(&dir, "hf_secret123")
        );
        let (m, _) = settings::read_in(&dir);
        assert_eq!(
            (m.get("cpuThreads").cloned(), m.get("gpuChecks").cloned()),
            (Some(4.into()), Some(false.into()))
        );
        assert_eq!(load_with(&store, &dir), Some(("hf_secret123".into(), Kept::System)));
        assert!(migrate_with(&store, &dir).is_empty(), "nothing to do again");
        delete_with(&store, &dir).unwrap();
        assert_eq!(load_with(&store, &dir), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Without a system store the token goes to its own file (readable only by the owner on
    /// Linux and macOS), never to the settings file.
    #[test]
    fn without_a_system_store_the_token_has_its_own_file() {
        let dir = temp();
        let none = Fake { value: RefCell::new(None), available: false };
        assert_eq!(save_with(&none, &dir, "hf_abc").unwrap(), Kept::File);
        assert_eq!(load_with(&none, &dir), Some(("hf_abc".into(), Kept::File)));
        assert_eq!(files_with(&dir, "hf_abc"), [FILE]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.join(FILE)).unwrap().permissions().mode();
            assert_eq!(mode & 0o077, 0, "only the owner can read it: {mode:o}");
        }
        // A store that becomes available takes it over, and the file goes.
        let store = Fake { value: RefCell::new(None), available: true };
        assert_eq!(save_with(&store, &dir, "hf_abc").unwrap(), Kept::System);
        assert!(!dir.join(FILE).exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
