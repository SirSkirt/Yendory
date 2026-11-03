//! Game skills (Engine 0.9.4; docs/FORMAT.md, "Nova Neuro and game skills", on yendory-hub):
//! networks a Nova Neuro model carries in its own file to play games. Yendory Tools trains them and
//! saves them into the model through the Yendory Network; Studio never changes them.
//!
//! So the model file on disk is their record. Whenever Studio writes a model, it carries the skills
//! the file holds at that moment, not the ones Studio read when it opened the model: a skill Tools
//! added or removed meanwhile (possible while Studio works without the Network, which otherwise
//! keeps Tools away from a model Studio has open) is never lost or brought back by Studio's save.
use super::*;
use std::path::Path;
pub(crate) use yendory_engine::skill::Skill;
use yendory_engine::skill::SkillSpec;

/// What meta.json lists of each skill (the same summary the Network writes), so the Hub shows
/// them without reading the checkpoint.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Summary {
    pub game: String,
    pub name: String,
    pub steps: u64,
    pub generation: u32,
    pub method: String,
    pub parameters: usize,
}

pub(crate) fn summary(skills: &[Skill]) -> Vec<Summary> {
    skills.iter().map(|k| summary_of(&k.spec)).collect()
}

fn summary_of(s: &SkillSpec) -> Summary {
    Summary {
        game: s.game.clone(),
        name: s.name.clone(),
        steps: s.steps,
        generation: s.lineage.generation,
        method: s.lineage.method.clone(),
        parameters: yendory_engine::skill::layout_len(&s.arch),
    }
}

/// What meta.json lists of the skills of the model file in `dir`; `None` when there is no readable
/// model file there.
pub(crate) fn summary_on_file(dir: &Path) -> Option<Vec<Summary>> {
    listed(&dir.join("model.nnova2")).map(|specs| specs.iter().map(summary_of).collect())
}

/// The largest header the Engine reads (a Neuro header lists every skill and its notes).
const MAX_HEADER: usize = 16 << 20;

/// The skills a model file lists in its header, without reading its weights: empty for a file
/// without skills (Bicameral, or Neuro format 1); `None` when there is no readable model file.
pub(crate) fn listed(path: &Path) -> Option<Vec<SkillSpec>> {
    #[derive(Deserialize)]
    struct Header {
        #[serde(default)]
        skills: Vec<SkillSpec>,
    }
    let mut f = std::fs::File::open(path).ok()?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).ok()?;
    match &magic {
        b"NNOVA2\0\0" => Some(Vec::new()),
        b"NNEURO1\0" => {
            let mut len = [0u8; 4];
            f.read_exact(&mut len).ok()?;
            let len = u32::from_le_bytes(len) as usize;
            if len > MAX_HEADER {
                return None;
            }
            let mut json = vec![0u8; len];
            f.read_exact(&mut json).ok()?;
            serde_json::from_slice::<Header>(&json).ok().map(|h| h.skills)
        }
        _ => None,
    }
}

/// The skills a save to `path` must carry when they differ from `model`'s: the file's own (read
/// in full only then). `None`: `model`'s are right (they match the file, there is no file yet, or
/// it cannot be read, so what Studio holds is all there is).
pub(crate) fn on_file(path: &Path, model: &Bicameral) -> Option<Vec<Skill>> {
    let specs = listed(path)?;
    if specs.iter().eq(model.skills.iter().map(|k| &k.spec)) {
        return None;
    }
    if specs.is_empty() {
        // Removed in Tools.
        return Some(Vec::new());
    }
    // Only Nova Neuro models carry skills (otherwise the file is of another model).
    model.neuro.as_ref()?;
    Bicameral::load(path).ok().map(|m| m.skills)
}

/// A skill as the Brain page shows it: what it is, how it is built, and figures measured from its
/// saved weights (no game is played here, and text never passes through a skill).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct View {
    game: String,
    name: String,
    size: yendory_engine::skill::Size,
    steps: u64,
    accepted_step: u64,
    created: u64,
    lineage: yendory_engine::skill::Lineage,
    keys: Vec<String>,
    buttons: Vec<String>,
    arch: yendory_engine::skill::Arch,
    /// (channels, height, width) entering each convolution, and leaving the last.
    shapes: Vec<(usize, usize, usize)>,
    parameters: usize,
    layers: Vec<Layer>,
    groups: Vec<GroupFigures>,
    /// The trainer's own record (Yendory Tools), as written.
    notes: serde_json::Value,
}

/// One layer's units: the root mean square of each unit's weights (the accepted weights, which
/// play) and of how far training since then has moved them (latest − accepted).
#[derive(Serialize)]
pub(crate) struct Layer {
    id: String,
    units: Vec<f32>,
    changed: Vec<f32>,
}

#[derive(Serialize)]
pub(crate) struct GroupFigures {
    group: yendory_engine::skill::Group,
    parameters: usize,
    rms: f32,
    changed: f32,
}

fn rms(values: impl Iterator<Item = f32>) -> f32 {
    let (mut sum, mut n) = (0f64, 0usize);
    for v in values {
        sum += (v as f64) * (v as f64);
        n += 1;
    }
    if n == 0 {
        0.
    } else {
        (sum / n as f64).sqrt() as f32
    }
}

pub(crate) fn view(k: &Skill) -> View {
    use yendory_engine::skill::{layout, Group, Slot};
    let slots = layout(&k.spec.arch);
    let slot = |name: &str| slots.iter().find(|s| s.name == name);
    let change: Vec<f32> = k.latest.iter().zip(&k.accepted).map(|(l, a)| l - a).collect();
    // Each output unit (a row), or each input (a column, for the goal input).
    let rows = |s: &Slot, w: &[f32]| -> Vec<f32> {
        (0..s.rows)
            .map(|r| rms(w[s.offset + r * s.cols..s.offset + (r + 1) * s.cols].iter().copied()))
            .collect()
    };
    let cols = |s: &Slot, w: &[f32]| -> Vec<f32> {
        (0..s.cols).map(|c| rms((0..s.rows).map(|r| w[s.offset + r * s.cols + c]))).collect()
    };
    let mut layers = Vec::new();
    let mut add = |id: String, weights: &str, by_column: bool| {
        if let Some(s) = slot(weights) {
            let f = if by_column { cols } else { rows };
            layers.push(Layer { id, units: f(s, &k.accepted), changed: f(s, &change) });
        }
    };
    for i in 0..k.spec.arch.convs.len() {
        add(format!("conv{i}"), &format!("conv{i}.w"), false);
    }
    add("encoder".into(), "enc.w", false);
    add("goal".into(), "goal.w", true);
    add("memory".into(), "mem.c.w", false);
    add("memoryGate".into(), "mem.z.w", false);
    add("keys".into(), "key.w", false);
    add("buttons".into(), "button.w", false);
    add("mouseX".into(), "mx.w", false);
    add("mouseY".into(), "my.w", false);
    let groups = [Group::Vision, Group::Memory, Group::Heads, Group::Interconnect]
        .into_iter()
        .filter_map(|g| {
            let mine: Vec<&Slot> = slots.iter().filter(|s| s.group == g).collect();
            let parameters: usize = mine.iter().map(|s| s.rows * s.cols).sum();
            let of = |w: &[f32]| rms(mine.iter().flat_map(|s| w[s.range()].iter().copied()));
            (parameters > 0).then(|| GroupFigures {
                group: g,
                parameters,
                rms: of(&k.accepted),
                changed: of(&change),
            })
        })
        .collect();
    let s = &k.spec;
    View {
        game: s.game.clone(),
        name: s.name.clone(),
        size: s.size,
        steps: s.steps,
        accepted_step: s.accepted_step,
        created: s.created,
        lineage: s.lineage.clone(),
        keys: s.keys.clone(),
        buttons: s.buttons.clone(),
        shapes: s.arch.shapes(),
        arch: s.arch.clone(),
        parameters: k.parameters(),
        layers,
        groups,
        notes: s.notes.clone(),
    }
}

/// The open model's game skills (Brain page).
#[tauri::command]
pub(crate) fn model_skills(studio: State<Studio>) -> Res<Vec<View>> {
    with_active(&studio, |a| Ok(a.model.skills.iter().map(view).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use yendory_engine::skill::{check_arch, Lineage, Size};

    fn skill(game: &str, steps: u64) -> Skill {
        let arch = check_arch();
        let spec = SkillSpec {
            game: game.into(),
            name: "Test range".into(),
            size: Size::Small,
            seed: 5,
            keys: (0..arch.keys).map(|i| format!("k{i}")).collect(),
            buttons: (0..arch.buttons).map(|i| format!("b{i}")).collect(),
            arch,
            lineage: Lineage { generation: 1, parent: None, method: "imitation".into() },
            steps,
            accepted_step: steps,
            created: 1,
            notes: serde_json::json!({"runs": [{"steps": steps}]}),
        };
        let mut k = Skill::new(spec).unwrap();
        k.latest[0] += 0.5;
        k.adam.t = steps;
        k
    }

    fn neuro() -> Bicameral {
        let p = yendory_engine::model::neuro::presets().into_iter().next().unwrap();
        Bicameral::with_design(p.config, Some(p.design), 3)
    }

    fn meta(id: &str) -> Meta {
        serde_json::from_value(serde_json::json!({
            "id": id, "name": "Nova", "created": 1, "updated": 1, "step": 0, "parameters": 0,
            "groups": [0, 0, 0], "words": 0, "taught": [], "origin": "",
            "hubNote": {"kept": true}
        }))
        .unwrap()
    }

    fn temp(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("yendory-skills-{tag}-{nanos:x}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_header_lists_a_files_skills() {
        let dir = temp("listed");
        let path = dir.join("model.nnova2");
        assert_eq!(listed(&path), None);
        let mut m = neuro();
        m.save(&path).unwrap();
        assert_eq!(listed(&path), Some(Vec::new()));
        m.skills = vec![skill("range", 200), skill("maze", 40)];
        m.save(&path).unwrap();
        let specs = listed(&path).unwrap();
        assert_eq!(specs, vec![m.skills[0].spec.clone(), m.skills[1].spec.clone()]);
        Bicameral::new().save(&path).unwrap();
        assert_eq!(listed(&path), Some(Vec::new()));
        std::fs::write(&path, b"not a model").unwrap();
        assert_eq!(listed(&path), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Studio opened a model, Tools added a skill to its file (or removed it), Studio trained and
    /// saved: the saved model has Studio's training and the file's skills.
    #[test]
    fn saving_keeps_the_skills_on_file() {
        let dir = temp("save");
        let path = dir.join("model.nnova2");
        let mut held = neuro();
        let m = meta("m1");
        save_in(&dir, &m, &held, &[]).unwrap();

        // Tools adds a skill to the file.
        let mut tools = Bicameral::load(&path).unwrap();
        tools.skills.push(skill("range", 200));
        tools.save(&path).unwrap();

        // Studio trains what it holds (no skills) and saves.
        held.train_text("U: Where is the key?\nN: In the box.\n", 3e-3, yendory_engine::Focus::Joint)
            .unwrap();
        save_in(&dir, &m, &held, &[]).unwrap();
        let saved = Bicameral::load(&path).unwrap();
        assert_eq!(saved.weights, held.weights, "Studio's training is saved");
        assert_eq!(saved.step, held.step);
        assert_eq!(saved.skills.len(), 1, "the skill Tools added is kept");
        assert_eq!(saved.skills[0].spec, tools.skills[0].spec);
        assert_eq!(saved.skills[0].latest, tools.skills[0].latest);
        let written: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(written["skills"][0]["game"], "range");
        assert_eq!(written["skills"][0]["steps"], 200);
        assert_eq!(written["skills"][0]["parameters"], tools.skills[0].parameters());
        assert_eq!(written["hubNote"]["kept"], true, "fields other programs wrote are kept");

        // Studio holding an older copy of the skill does not bring it back over a newer one.
        held.skills = vec![skill("range", 50)];
        save_in(&dir, &m, &held, &[]).unwrap();
        assert_eq!(Bicameral::load(&path).unwrap().skills[0].spec.steps, 200);

        // Tools removes it: Studio's save does not bring it back.
        tools.skills.clear();
        tools.save(&path).unwrap();
        held.skills = vec![skill("range", 200)];
        save_in(&dir, &m, &held, &[]).unwrap();
        assert!(Bicameral::load(&path).unwrap().skills.is_empty());
        let written: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        assert!(written.get("skills").is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A model imported with skills (no file in its folder yet) keeps them.
    #[test]
    fn a_new_model_keeps_its_own_skills() {
        let dir = temp("new");
        let mut m = neuro();
        m.skills.push(skill("range", 7));
        let bytes = m.to_bytes().unwrap();
        let imported = Bicameral::from_bytes(&bytes).unwrap();
        save_in(&dir, &meta("m2"), &imported, &[]).unwrap();
        assert_eq!(Bicameral::load(dir.join("model.nnova2")).unwrap().skills.len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_skill_is_shown_with_figures_from_its_weights() {
        let mut k = skill("range", 30);
        let v = view(&k);
        let a = &k.spec.arch;
        let layer = |id: &str| v.layers.iter().find(|l| l.id == id).unwrap();
        assert_eq!(layer("conv0").units.len(), a.convs[0].out);
        assert_eq!(layer("encoder").units.len(), a.hidden);
        assert_eq!(layer("goal").units.len(), a.goal);
        assert_eq!(layer("memory").units.len(), a.memory);
        assert_eq!(layer("keys").units.len(), a.keys);
        assert_eq!(layer("mouseX").units.len(), a.mouse_bins.len());
        // The first key's figure is its own row's root mean square.
        let s = yendory_engine::skill::layout(a).into_iter().find(|s| s.name == "key.w").unwrap();
        let row = &k.accepted[s.offset..s.offset + s.cols];
        let want = (row.iter().map(|x| x * x).sum::<f32>() / s.cols as f32).sqrt();
        assert!((layer("keys").units[0] - want).abs() < 1e-6);
        // Change since the accepted weights: only where training moved them.
        assert!(layer("keys").changed.iter().all(|&c| c == 0.));
        let first = yendory_engine::skill::layout(a).into_iter().next().unwrap();
        assert!(layer(&first.name.replace(".w", "")).changed[0] > 0.);
        assert_eq!(v.groups.iter().map(|g| g.parameters).sum::<usize>(), k.parameters());
        k.latest.clone_from(&k.accepted);
        assert!(view(&k).groups.iter().all(|g| g.changed == 0.));
    }
}
