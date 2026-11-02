//! Game skills of the open model: Gaming Nova's networks, kept inside a Nova Neuro model's own
//! file (docs/FORMAT.md on yendory-hub, format 2) and read and saved only through the Yendory
//! Network (`model.skills`, `model.skill`, `model.skill.save`, `model.skill.delete`).
//!
//! One skill per game: its id is the game profile's. A model made in Yendory Hub comes to Tools
//! (Open in Tools, or Open a model…), learns a game here, and carries it wherever it goes.
use super::policy::{Adam, Arch, Lineage, Size, Skill, SkillSpec};
use super::profile::Profile;
use super::recording::{self, Split, AFTER_GAP};
use crate::Res;
use serde_json::{json, Value};
use std::path::Path;
use tauri::AppHandle;
use yendory_client::floats::{decode, encode};

/// The open model (game skills need one, and it must be Nova Neuro).
pub fn open_model(app: &AppHandle) -> Res<crate::OpenModel> {
    let m = crate::open_model_now(app)
        .ok_or("Open a model first (Yendory Hub: Open in Tools; or Open a model… above)")?;
    if m.architecture != "neuro" {
        return Err(format!(
            "{} is a Nova Bicameral model: game skills are kept only by Nova Neuro models. Make a Neuro model in Yendory Hub and open it here.",
            m.name
        ));
    }
    Ok(m)
}

/// The open model's skills (descriptions only).
pub fn list(app: &AppHandle) -> Res<Vec<SkillSpec>> {
    let Some(m) = crate::open_model_now(app) else { return Ok(vec![]) };
    if m.architecture != "neuro" {
        return Ok(vec![]);
    }
    let v = crate::network_call(app, "model.skills", json!({"id": m.id}))?;
    serde_json::from_value(v).map_err(|e| format!("The Network sent skills Tools can't read: {e}"))
}

/// The open model's skill for a game, with its weights.
pub fn get(app: &AppHandle, game: &str) -> Res<Skill> {
    let m = open_model(app)?;
    let v = crate::network_call(app, "model.skill", json!({"id": m.id, "game": game}))?;
    let arr = |k: &str| decode(v[k].as_str().unwrap_or_default());
    let skill = Skill {
        spec: serde_json::from_value(v["spec"].clone()).map_err(|e| e.to_string())?,
        accepted: arr("accepted")?,
        latest: arr("latest")?,
        adam: Adam { m: arr("m")?, v: arr("v")?, t: v["t"].as_u64().unwrap_or(0) },
    };
    skill.check()?;
    Ok(skill)
}

/// Saves a skill into the open model (its file is written by the Network).
pub fn save(app: &AppHandle, model_id: &str, k: &Skill) -> Res<Value> {
    k.check()?;
    crate::network_call(
        app,
        "model.skill.save",
        json!({"id": model_id, "spec": k.spec, "accepted": encode(&k.accepted), "latest": encode(&k.latest),
               "m": encode(&k.adam.m), "v": encode(&k.adam.v), "t": k.adam.t}),
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
