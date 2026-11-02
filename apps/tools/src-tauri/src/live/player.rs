//! Players: Gaming Nova networks for one game, kept in Tools' data
//! (`games/<profile>/players/<id>/`):
//!
//! - `player.json`: [`PlayerInfo`]: the architecture (with the controls in order and the mouse
//!   steps), the lineage (generation, parent, method), and every training run;
//! - `weights.bin`: the accepted weights (the best measured on held-out recordings);
//! - `last.bin` and `adam.bin`: the latest weights and optimizer state, to continue training.
//!
//! Imitation (generation 0) learns only from recordings of a person (`player: person`):
//! a model's own play is never taught back as a demonstration.
use super::policy::{Adam, Arch, Policy, Size};
use super::profile::Profile;
use super::recording::{self, Split, AFTER_GAP};
use crate::Res;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const FORMAT: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lineage {
    pub generation: u32,
    /// The player this one started from (none for a new network).
    pub parent: Option<String>,
    /// How its weights came to be: `new`, then each training method in order.
    pub method: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerInfo {
    pub format: u32,
    pub id: String,
    pub name: String,
    pub profile: String,
    pub created: u64,
    pub size: Size,
    pub seed: u64,
    pub arch: Arch,
    /// The controls the heads stand for, in order (the profile's when the player was made).
    pub keys: Vec<String>,
    pub buttons: Vec<super::controls::Button>,
    pub lineage: Lineage,
    /// Training steps taken in all.
    pub steps: u64,
    /// The steps the accepted weights (`weights.bin`) were saved at.
    pub accepted_step: u64,
    pub runs: Vec<super::train::Run>,
}

pub fn folder(data: &Path, profile: &str) -> PathBuf {
    data.join("games").join(profile).join("players")
}

fn dir_of(data: &Path, profile: &str, id: &str) -> Res<PathBuf> {
    let d = folder(data, profile).join(id);
    if id.is_empty() || id != super::profile::slug(id) || !d.join("player.json").exists() {
        return Err("No such player".into());
    }
    Ok(d)
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> Res<()> {
    let tmp = dir.join(format!("{name}.tmp"));
    std::fs::write(&tmp, bytes).map_err(|e| format!("Saving the player failed: {e}"))?;
    std::fs::rename(&tmp, dir.join(name)).map_err(|e| format!("Saving the player failed: {e}"))
}

pub fn save_info(data: &Path, info: &PlayerInfo) -> Res<()> {
    let dir = folder(data, &info.profile).join(&info.id);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write(&dir, "player.json", &serde_json::to_vec_pretty(info).map_err(|e| e.to_string())?)
}

pub fn load_info(data: &Path, profile: &str, id: &str) -> Res<PlayerInfo> {
    let d = dir_of(data, profile, id)?;
    serde_json::from_slice(&std::fs::read(d.join("player.json")).map_err(|e| e.to_string())?)
        .map_err(|e| format!("The player can't be read: {e}"))
}

pub fn list(data: &Path, profile: &str) -> Vec<PlayerInfo> {
    let mut out: Vec<PlayerInfo> = std::fs::read_dir(folder(data, profile))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| std::fs::read(e.path().join("player.json")).ok())
        .filter_map(|b| serde_json::from_slice(&b).ok())
        .collect();
    out.sort_by_key(|p: &PlayerInfo| std::cmp::Reverse(p.created));
    out
}

pub fn delete(data: &Path, profile: &str, id: &str) -> Res<()> {
    std::fs::remove_dir_all(dir_of(data, profile, id)?).map_err(|e| e.to_string())
}

/// The accepted weights.
pub fn load_policy(data: &Path, info: &PlayerInfo) -> Res<Policy> {
    let d = dir_of(data, &info.profile, &info.id)?;
    Policy::from_bytes(info.arch.clone(), &std::fs::read(d.join("weights.bin")).map_err(|e| e.to_string())?)
}

pub fn save_policy(data: &Path, info: &PlayerInfo, name: &str, p: &Policy) -> Res<()> {
    write(&folder(data, &info.profile).join(&info.id), name, &p.to_bytes())
}

/// The latest weights and optimizer, to continue training (the accepted ones and a new
/// optimizer when there are none).
pub fn load_latest(data: &Path, info: &PlayerInfo) -> Res<(Policy, Adam)> {
    let d = dir_of(data, &info.profile, &info.id)?;
    let p = match std::fs::read(d.join("last.bin")) {
        Ok(b) => Policy::from_bytes(info.arch.clone(), &b)?,
        Err(_) => load_policy(data, info)?,
    };
    let adam = std::fs::read(d.join("adam.bin"))
        .ok()
        .and_then(|b| Adam::from_bytes(&b, p.parameters()))
        .unwrap_or_else(|| Adam::new(p.parameters()));
    Ok((p, adam))
}

pub fn save_latest(data: &Path, info: &PlayerInfo, p: &Policy, adam: &Adam) -> Res<()> {
    let d = folder(data, &info.profile).join(&info.id);
    write(&d, "last.bin", &p.to_bytes())?;
    write(&d, "adam.bin", &adam.to_bytes())
}

/// Mouse steps from a person's recordings: zero, and five sizes each way at the quantiles of
/// the movement they made in a tick (so the steps fit how this person moves this mouse).
pub fn mouse_bins(data: &Path, profile: &Profile) -> Vec<f32> {
    if !profile.mouse {
        return vec![];
    }
    let mut sizes: Vec<f32> = Vec::new();
    for info in recording::list(data, &profile.id)
        .into_iter()
        .filter(|r| r.player == "person" && r.split == Split::Train)
    {
        if let Ok(r) = recording::open(data, &profile.id, &info.id) {
            for t in r.ticks.iter().filter(|t| t.flags & AFTER_GAP == 0) {
                for v in [t.dx.abs(), t.dy.abs()] {
                    if v >= 0.5 {
                        sizes.push(v);
                    }
                }
            }
        }
    }
    let mut steps: Vec<f32> = if sizes.len() < 20 {
        vec![1.0, 3.0, 8.0, 20.0, 50.0]
    } else {
        sizes.sort_by(f32::total_cmp);
        [0.2, 0.45, 0.7, 0.9, 0.98]
            .iter()
            .map(|q| (sizes[((sizes.len() - 1) as f32 * q) as usize] * 4.0).round() / 4.0)
            .collect()
    };
    steps.dedup_by(|a, b| (*a - *b).abs() < 0.25);
    steps.retain(|v| *v > 0.0);
    let mut bins: Vec<f32> = steps.iter().rev().map(|v| -v).collect();
    bins.push(0.0);
    bins.extend(steps);
    bins
}

/// A new generation-0 player for the profile's game (its controls and picture as they are now).
pub fn create(data: &Path, profile: &Profile, name: &str, size: Size, seed: u64) -> Res<PlayerInfo> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Give the player a name".into());
    }
    let id = super::profile::slug(name);
    if id.is_empty() {
        return Err("The name needs a letter or a digit".into());
    }
    if folder(data, &profile.id).join(&id).join("player.json").exists() {
        return Err(format!("This game already has a player called {name}"));
    }
    let channels = if profile.capture.gray { 1 } else { 3 };
    let arch = Arch::new(
        profile.capture.width as usize,
        profile.capture.height as usize,
        channels,
        profile.keys.len(),
        profile.buttons.len(),
        mouse_bins(data, profile),
        size,
    );
    let policy = Policy::new(arch.clone(), seed)?;
    let info = PlayerInfo {
        format: FORMAT,
        id,
        name: name.to_string(),
        profile: profile.id.clone(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        size,
        seed,
        arch,
        keys: profile.keys.clone(),
        buttons: profile.buttons.clone(),
        lineage: Lineage { generation: 0, parent: None, method: "new".into() },
        steps: 0,
        accepted_step: 0,
        runs: vec![],
    };
    save_info(data, &info)?;
    save_policy(data, &info, "weights.bin", &policy)?;
    Ok(info)
}

/// Why the player can't play or learn the profile's game as it is now, if it can't.
pub fn fits(info: &PlayerInfo, profile: &Profile) -> Res<()> {
    let ch = if profile.capture.gray { 1 } else { 3 };
    if info.arch.width != profile.capture.width as usize
        || info.arch.height != profile.capture.height as usize
        || info.arch.channels != ch
    {
        return Err("The game's picture settings changed since this player was made: make a new player (or set them back)".into());
    }
    if info.keys != profile.keys
        || info.buttons != profile.buttons
        || info.arch.mouse_bins.is_empty() == profile.mouse
    {
        return Err(
            "The game's controls changed since this player was made: make a new player (or set them back)"
                .into(),
        );
    }
    Ok(())
}
