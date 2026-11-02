//! Players kept in Tools by 0.1.4 (`games/<profile>/players/<id>/`), before game skills moved
//! into the models themselves (`skills.rs`). They are only listed, moved into a Nova Neuro model
//! as its skill for the game, or deleted.
use super::policy::{Adam, Arch, Lineage, Size, Skill, SkillSpec};
use crate::Res;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
    pub keys: Vec<String>,
    pub buttons: Vec<super::controls::Button>,
    pub lineage: Lineage,
    pub steps: u64,
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

pub fn load_info(data: &Path, profile: &str, id: &str) -> Res<PlayerInfo> {
    let d = dir_of(data, profile, id)?;
    serde_json::from_slice(&std::fs::read(d.join("player.json")).map_err(|e| e.to_string())?)
        .map_err(|e| format!("The player can't be read: {e}"))
}

pub fn delete(data: &Path, profile: &str, id: &str) -> Res<()> {
    std::fs::remove_dir_all(dir_of(data, profile, id)?).map_err(|e| e.to_string())
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// The player as a game skill (its weights, optimizer, lineage and runs), to move into a model.
pub fn to_skill(data: &Path, info: &PlayerInfo) -> Res<Skill> {
    let d = dir_of(data, &info.profile, &info.id)?;
    let read = |n: &str| std::fs::read(d.join(n)).map(|b| floats(&b)).ok();
    let accepted = read("weights.bin").ok_or("The player's weights are missing")?;
    let latest = read("last.bin").unwrap_or_else(|| accepted.clone());
    let n = accepted.len();
    let adam = std::fs::read(d.join("adam.bin"))
        .ok()
        .and_then(|b| Adam::from_bytes(&b, n))
        .unwrap_or_else(|| Adam::new(n));
    let skill = Skill {
        spec: SkillSpec {
            game: info.profile.clone(),
            name: info.name.clone(),
            size: info.size,
            seed: info.seed,
            arch: info.arch.clone(),
            keys: info.keys.clone(),
            buttons: info.buttons.iter().map(|b| b.name().to_string()).collect(),
            lineage: info.lineage.clone(),
            steps: info.steps,
            accepted_step: info.accepted_step,
            created: info.created,
            notes: serde_json::json!({"runs": info.runs, "movedFrom": format!("Tools player {}", info.name)}),
        },
        accepted,
        latest,
        adam,
    };
    skill.check()?;
    Ok(skill)
}
