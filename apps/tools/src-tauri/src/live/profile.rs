//! Game profiles: everything Tools knows about one game, kept in Tools' data folder
//! (`games/<id>/profile.json`): how to find its window, the controls a network may use, the
//! mouse's sensitivity, what is captured and how often, and the safety limits.
use super::controls::{key, Button};
use super::safety::{Limits, Permitted};
use super::sys::WindowInfo;
use crate::Res;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Capture {
    /// The picture the network sees, in pixels.
    pub width: u32,
    pub height: u32,
    /// Grey instead of colour (a third of the size).
    pub gray: bool,
    /// Pictures (and decisions) a second.
    pub fps: u32,
}

impl Default for Capture {
    fn default() -> Self {
        Capture { width: 128, height: 72, gray: false, fps: 20 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Profile {
    pub id: String,
    pub name: String,
    /// The game's window: its title contains this (any case)...
    pub window_title: String,
    /// ...and, when set, its program is this (`Ravenfield.exe`).
    pub window_process: String,
    pub keys: Vec<String>,
    pub buttons: Vec<Button>,
    pub mouse: bool,
    /// Mouse units sent per unit the network asks for.
    pub mouse_scale: f32,
    pub capture: Capture,
    pub limits: Limits,
    /// The most its recordings may take on disk (megabytes).
    pub storage_mb: u64,
    pub notes: String,
}

impl Default for Profile {
    fn default() -> Self {
        Profile {
            id: String::new(),
            name: String::new(),
            window_title: String::new(),
            window_process: String::new(),
            keys: vec!["w".into(), "a".into(), "s".into(), "d".into()],
            buttons: vec![Button::Left],
            mouse: true,
            mouse_scale: 1.0,
            capture: Capture::default(),
            limits: Limits::default(),
            storage_mb: 4096,
            notes: String::new(),
        }
    }
}

/// The test range's profile (created when the range is first started).
pub fn range_profile() -> Profile {
    Profile {
        id: "yendory-range".into(),
        name: "Yendory Range".into(),
        window_title: crate::range::app::TITLE.into(),
        window_process: String::new(),
        keys: ["w", "a", "s", "d", "shift"].iter().map(|s| s.to_string()).collect(),
        buttons: vec![Button::Left],
        notes:
            "Tools' own test game: W A S D to move, Shift to run, the mouse to turn, the left button to fire."
                .into(),
        ..Profile::default()
    }
}

/// A file-safe id from a name.
pub fn slug(name: &str) -> String {
    let mut s: String =
        name.to_ascii_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    s.trim_matches('-').chars().take(48).collect()
}

impl Profile {
    /// Checks and tidies the profile; the reason when it can't be used.
    pub fn check(mut self) -> Res<Profile> {
        self.name = self.name.trim().to_string();
        if self.name.is_empty() {
            return Err("Give the game a name".into());
        }
        if self.id.is_empty() {
            self.id = slug(&self.name);
        }
        if self.id.is_empty() || self.id != slug(&self.id) {
            return Err("The profile's id may only hold a-z, 0-9 and -".into());
        }
        self.window_title = self.window_title.trim().to_string();
        self.window_process = self.window_process.trim().to_string();
        if self.window_title.is_empty() && self.window_process.is_empty() {
            return Err("Choose the game's window".into());
        }
        let mut keys = Vec::new();
        for k in &self.keys {
            let info = key(k).ok_or_else(|| format!("{k} is not a key Tools can press"))?;
            if !keys.contains(&info.name.to_string()) {
                keys.push(info.name.to_string());
            }
        }
        self.keys = keys;
        self.buttons.sort();
        self.buttons.dedup();
        if self.keys.is_empty() && self.buttons.is_empty() && !self.mouse {
            return Err("Permit at least one control".into());
        }
        let c = &mut self.capture;
        if !(16..=640).contains(&c.width) || !(9..=480).contains(&c.height) {
            return Err("The picture must be 16 to 640 pixels wide and 9 to 480 high".into());
        }
        if !(1..=60).contains(&c.fps) {
            return Err("Pictures a second: 1 to 60".into());
        }
        if !(0.05..=20.0).contains(&self.mouse_scale) {
            return Err("Mouse sensitivity: 0.05 to 20".into());
        }
        let l = &self.limits;
        if !(50..=60_000).contains(&l.max_hold_ms) {
            return Err("The longest hold: 50 to 60,000 ms".into());
        }
        if !(1..=200).contains(&l.max_actions_per_second) {
            return Err("Actions a second: 1 to 200".into());
        }
        if !(1..=5000).contains(&l.max_mouse_step) {
            return Err("The largest mouse step: 1 to 5,000 units".into());
        }
        self.storage_mb = self.storage_mb.clamp(16, 1 << 20);
        Ok(self)
    }

    pub fn permitted(&self) -> Res<Permitted> {
        Permitted::new(&self.keys, &self.buttons, self.mouse, self.mouse_scale)
    }

    /// Whether `w` is this game's window.
    pub fn matches(&self, w: &WindowInfo) -> bool {
        let title = self.window_title.to_lowercase();
        (title.is_empty() || w.title.to_lowercase().contains(&title))
            && (self.window_process.is_empty() || w.process.eq_ignore_ascii_case(&self.window_process))
    }
}

fn folder(data: &Path) -> PathBuf {
    data.join("games")
}

pub fn list(data: &Path) -> Vec<Profile> {
    let mut out: Vec<Profile> = std::fs::read_dir(folder(data))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| std::fs::read(e.path().join("profile.json")).ok())
        .filter_map(|b| serde_json::from_slice(&b).ok())
        .collect();
    out.sort_by_key(|p: &Profile| p.name.to_lowercase());
    out
}

pub fn load(data: &Path, id: &str) -> Res<Profile> {
    if id != slug(id) || id.is_empty() {
        return Err("No such game profile".into());
    }
    let b = std::fs::read(folder(data).join(id).join("profile.json"))
        .map_err(|_| "No such game profile".to_string())?;
    serde_json::from_slice(&b).map_err(|e| format!("The profile can't be read: {e}"))
}

/// Saves the profile (checked first) and returns it as saved.
pub fn save(data: &Path, p: Profile) -> Res<Profile> {
    let p = p.check()?;
    let dir = folder(data).join(&p.id);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let tmp = dir.join("profile.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&p).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, dir.join("profile.json")).map_err(|e| e.to_string())?;
    Ok(p)
}

/// Removes the profile file only; recordings and players are kept until removed themselves.
pub fn delete(data: &Path, id: &str) -> Res<()> {
    load(data, id)?;
    std::fs::remove_file(folder(data).join(id).join("profile.json")).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_are_checked_saved_and_listed() {
        let dir = std::env::temp_dir().join(format!("tools-profiles-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut p = Profile {
            name: "  Ravenfield ".into(),
            window_process: "Ravenfield.exe".into(),
            ..Profile::default()
        };
        p.keys = vec!["W".into(), "w".into(), "Shift".into()];
        let saved = save(&dir, p.clone()).unwrap();
        assert_eq!(saved.id, "ravenfield");
        assert_eq!(saved.keys, ["w", "shift"]);
        assert_eq!(list(&dir), vec![saved.clone()]);
        assert_eq!(load(&dir, "ravenfield").unwrap(), saved);
        p.keys = vec!["win".into()];
        assert!(save(&dir, p.clone()).is_err());
        p.keys = vec![];
        p.buttons = vec![];
        p.mouse = false;
        assert!(save(&dir, p).is_err());
        assert!(load(&dir, "../x").is_err());
        let w = |title: &str, process: &str| WindowInfo {
            id: 1,
            title: title.into(),
            process: process.into(),
            pid: 1,
            width: 1,
            height: 1,
        };
        assert!(saved.matches(&w("Ravenfield", "ravenfield.exe")));
        assert!(!saved.matches(&w("Ravenfield", "other.exe")));
        assert!(range_profile().matches(&w("Yendory Range", "yendory-tools")));
        delete(&dir, "ravenfield").unwrap();
        assert!(list(&dir).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
