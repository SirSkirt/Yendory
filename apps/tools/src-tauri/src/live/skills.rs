//! Game skills of the open model: Gaming Nova's networks, kept inside the model's own file
//! (Nova Neuro or Nova Bicameral: docs/FORMAT.md on yendory-hub) and read and saved only through
//! the Yendory Network (`model.skills`, `model.skill`, `model.skill.save`, `model.skill.delete`).
//! The weights travel in parts (`model.skill.read`, uploads), so a skill can be larger than one
//! Network message.
//!
//! One skill per game: its id is the game profile's. A model made in Yendory Hub comes to Tools
//! (Open in Tools, or Open a model…), learns a game here, and carries it wherever it goes.
use super::policy::{layout_len, Adam, Arch, Lineage, Size, Skill, SkillSpec, MAX_SKILL_PARAMETERS};
use super::profile::Profile;
use super::recording::{self, Split, AFTER_GAP};
use crate::Res;
use serde_json::{json, Value};
use std::path::Path;
use tauri::AppHandle;
use yendory_client::floats;

/// The open model (game skills need one; Nova Neuro and Nova Bicameral both keep them).
pub fn open_model(app: &AppHandle) -> Res<crate::OpenModel> {
    crate::open_model_now(app)
        .ok_or_else(|| "Open a model first (Yendory Hub: Open in Tools; or Open a model… above)".into())
}

/// The open model's skills (descriptions only).
pub fn list(app: &AppHandle) -> Res<Vec<SkillSpec>> {
    let Some(m) = crate::open_model_now(app) else { return Ok(vec![]) };
    let v = crate::network_call(app, "model.skills", json!({"id": m.id}))?;
    serde_json::from_value(v).map_err(|e| format!("The Network sent skills Tools can't read: {e}"))
}

/// The open model's skill for a game, with its weights (read in parts).
pub fn get(app: &AppHandle, game: &str) -> Res<Skill> {
    let m = open_model(app)?;
    let head = crate::network_call(app, "model.skill", json!({"id": m.id, "game": game, "parts": true}))?;
    let spec: SkillSpec = serde_json::from_value(head["spec"].clone()).map_err(|e| e.to_string())?;
    let n = layout_len(&spec.arch);
    let total = head["bytes"].as_u64().unwrap_or(0) as usize;
    if total != 16 * n {
        return Err("The Network sent a skill whose size does not fit its network".into());
    }
    let data = crate::network_read_parts(app, "model.skill.read", json!({"id": m.id, "game": game}), total)?;
    let mut all = floats::from_bytes(&data)?;
    drop(data);
    let v = all.split_off(3 * n);
    let mv = all.split_off(2 * n);
    let latest = all.split_off(n);
    let skill =
        Skill { spec, accepted: all, latest, adam: Adam { m: mv, v, t: head["t"].as_u64().unwrap_or(0) } };
    skill.check()?;
    Ok(skill)
}

/// Why a skill of this many weights can't be kept in a model, if it can't (the Engine's limit
/// for one skill).
pub fn fits_network(weights: usize, profile: &Profile) -> Res<()> {
    if weights <= MAX_SKILL_PARAMETERS {
        return Ok(());
    }
    Err(format!(
        "This skill would have {:.1} million weights (the game's picture is {}×{}); a skill can have at most {:.0} million. Make the game's picture smaller on 2.1",
        weights as f64 / 1e6,
        profile.capture.width,
        profile.capture.height,
        MAX_SKILL_PARAMETERS as f64 / 1e6
    ))
}

/// Weights of a new skill of each size for the profile's game as it is now.
pub fn weights_for(data: &Path, profile: &Profile) -> Vec<(Size, usize)> {
    let channels = if profile.capture.gray { 1 } else { 3 };
    let bins = mouse_bins(data, profile);
    [Size::Small, Size::Medium]
        .into_iter()
        .map(|size| {
            let arch = Arch::new(
                profile.capture.width as usize,
                profile.capture.height as usize,
                channels,
                profile.keys.len(),
                profile.buttons.len(),
                bins.clone(),
                size,
            );
            (size, layout_len(&arch))
        })
        .collect()
}

/// Saves a skill into the open model (its file is written by the Network). The weights go up in
/// parts.
pub fn save(app: &AppHandle, model_id: &str, k: &Skill) -> Res<Value> {
    k.check()?;
    let mut data = Vec::with_capacity(16 * k.accepted.len());
    for a in [&k.accepted, &k.latest, &k.adam.m, &k.adam.v] {
        data.extend_from_slice(&floats::to_bytes(a));
    }
    let upload = crate::network_upload(app, &data)?;
    drop(data);
    crate::network_call(
        app,
        "model.skill.save",
        json!({"id": model_id, "spec": k.spec, "upload": upload, "t": k.adam.t}),
    )
}

pub fn delete(app: &AppHandle, game: &str) -> Res<()> {
    let m = open_model(app)?;
    crate::network_call(app, "model.skill.delete", json!({"id": m.id, "game": game})).map(|_| ())
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

/// A new, untrained skill for the profile's game (its controls and picture as they are now; its
/// mouse steps from the person's training recordings).
pub fn new_skill(data: &Path, profile: &Profile, size: Size, seed: u64) -> Res<Skill> {
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
    Skill::new(SkillSpec {
        game: profile.id.clone(),
        name: profile.name.clone(),
        size,
        seed,
        arch,
        keys: profile.keys.clone(),
        buttons: profile.buttons.iter().map(|b| b.name().to_string()).collect(),
        lineage: Lineage { generation: 0, parent: None, method: "new".into() },
        steps: 0,
        accepted_step: 0,
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        notes: json!({"runs": []}),
    })
}

/// Why the skill can't play or learn the profile's game as it is now, if it can't.
pub fn fits(spec: &SkillSpec, profile: &Profile) -> Res<()> {
    let ch = if profile.capture.gray { 1 } else { 3 };
    let a = &spec.arch;
    if a.width != profile.capture.width as usize
        || a.height != profile.capture.height as usize
        || a.channels != ch
    {
        return Err("The game's picture settings changed since this skill was made: set them back, or delete the skill and teach it again".into());
    }
    let buttons: Vec<&str> = profile.buttons.iter().map(|b| b.name()).collect();
    if spec.keys != profile.keys || spec.buttons != buttons || a.mouse_bins.is_empty() == profile.mouse {
        return Err("The game's controls changed since this skill was made: set them back, or delete the skill and teach it again".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skills_up_to_the_engines_limit_can_be_kept() {
        // The range's picture is small; a large picture (640x360) now fits too (before 0.1.9 a
        // skill went in one message and could not); an absurd one does not.
        let data = std::env::temp_dir().join("tools-skill-size");
        let mut p = crate::live::profile::range_profile();
        assert!(weights_for(&data, &p).iter().all(|(_, w)| fits_network(*w, &p).is_ok()));
        p.capture.width = 640;
        p.capture.height = 360;
        let (_, small) = weights_for(&data, &p)[0];
        assert!(small > 10_000_000 && fits_network(small, &p).is_ok(), "{small}");
        assert!(fits_network(MAX_SKILL_PARAMETERS + 1, &p).is_err());
    }
}
