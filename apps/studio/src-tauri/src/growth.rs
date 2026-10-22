//! Growing a model's network (docs/GROWTH.md): the Engine grows it so it computes what it
//! computed before; Studio keeps a recovery checkpoint of every model it grows, verifies the
//! grown model before replacing the original, records the growth and can roll it back.
use super::*;
use yendory_engine::model::grow::PROBES;
use yendory_engine::{compare_outputs, Growth, GrowthOption, NotOffered, OutputComparison};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GrowthMode {
    /// The network keeps its size.
    #[default]
    Off,
    /// You choose when and how it grows.
    Manual,
    /// Studio may grow it when training shows it needs capacity, within your limits.
    Automatic,
}

/// One growth of a model's network.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GrowthRecord {
    pub time: u64,
    pub step: u64,
    pub label: String,
    pub growth: Growth,
    pub parameters_before: usize,
    pub parameters_after: usize,
    /// The check made before the grown model replaced the original.
    pub check: OutputComparison,
    /// The recovery checkpoint of the model before this growth (in the model's growth folder).
    pub backup: String,
    #[serde(default)]
    pub rolled_back: bool,
    /// Grown by automatic growth (otherwise by you).
    #[serde(default)]
    pub automatic: bool,
}

/// A model's growth setting and history (kept in its meta.json).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GrowthInfo {
    pub mode: GrowthMode,
    /// Parameters before the first growth (None: never grown).
    pub original_parameters: Option<usize>,
    pub history: Vec<GrowthRecord>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OptionView {
    #[serde(flatten)]
    option: GrowthOption,
    /// Weights, gradients and both AdamW moments while training (bytes).
    state_bytes: u64,
    state_bytes_added: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GrowthView {
    mode: GrowthMode,
    original_parameters: usize,
    current_parameters: usize,
    /// Values stored per training token now (activations; 4 bytes each).
    activation_floats_per_token: usize,
    options: Vec<OptionView>,
    not_offered: Vec<NotOffered>,
    history: Vec<GrowthRecord>,
    can_roll_back: bool,
}

fn view(a: &Active) -> GrowthView {
    let (options, not_offered) = a.model.growth_options();
    let now = a.model.parameter_count();
    let state = |p: usize| 16 * p as u64;
    GrowthView {
        mode: a.meta.growth.mode,
        original_parameters: a.meta.growth.original_parameters.unwrap_or(now),
        current_parameters: now,
        activation_floats_per_token: Bicameral::activation_floats_for(
            &a.model.config,
            a.model.neuro.as_ref(),
        ),
        options: options
            .into_iter()
            .map(|o| OptionView {
                state_bytes: state(o.parameters_after),
                state_bytes_added: state(o.parameters_after) - state(o.parameters_before),
                option: o,
            })
            .collect(),
        not_offered,
        can_roll_back: a.meta.growth.history.iter().any(|r| !r.rolled_back),
        history: a.meta.growth.history.clone(),
    }
}

#[tauri::command]
pub fn growth_view(studio: State<Studio>) -> Res<GrowthView> {
    with_active(&studio, |a| Ok(view(a)))
}

#[tauri::command]
pub fn set_growth_mode(app: AppHandle, studio: State<Studio>, mode: GrowthMode) -> Res<GrowthView> {
    let mut guard = studio.active.lock().unwrap();
    let a = guard.as_mut().ok_or("Open or create a model first")?;
    a.meta.growth.mode = mode;
    let dir = model_dir(&app, &a.meta.id)?;
    write_atomic(&dir.join("meta.json"), &serde_json::to_vec_pretty(&a.meta).map_err(|e| e.to_string())?)?;
    Ok(view(a))
}

fn extension(model: &Bicameral) -> &'static str {
    if model.neuro.is_some() {
        "nneuro"
    } else {
        "nnova2"
    }
}

/// Writes `model` to the model's growth folder and checks it reads back identically.
fn keep_checkpoint(dir: &std::path::Path, file: &str, model: &Bicameral) -> Res<()> {
    let gdir = dir.join("growth");
    std::fs::create_dir_all(&gdir).map_err(|e| e.to_string())?;
    let bytes = model.to_bytes()?;
    write_atomic(&gdir.join(file), &bytes)?;
    let back = Bicameral::from_bytes(&std::fs::read(gdir.join(file)).map_err(|e| e.to_string())?)?;
    if back.weights != model.weights || back.config != model.config {
        return Err("The recovery checkpoint did not read back correctly; nothing was changed".into());
    }
    Ok(())
}

/// Grows the open model by a growth step: recovery checkpoint first, then the grown model is
/// checked against the original on probe text and only then saved (in `dir`, the model's
/// folder). `automatic`: recorded as automatic growth.
pub(crate) fn grow_active(
    dir: &std::path::Path,
    a: &mut Active,
    option: &GrowthOption,
    automatic: bool,
) -> Res<GrowthRecord> {
    let before = a.model.parameter_count();
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1) | 1;
    let grown = a.model.grow(&option.growth, seed)?;
    let check = compare_outputs(&a.model, &grown, &PROBES)?;
    if !check.preserved() {
        return Err(format!(
            "The grown network did not reproduce the original (largest output change {:.2e}, loss {:.4} -> {:.4}); nothing was changed",
            check.max_logit_change, check.loss_before, check.loss_after
        ));
    }
    let n = a.meta.growth.history.len() + 1;
    let backup = format!("{n:03}-before-{before}.{}", extension(&a.model));
    keep_checkpoint(dir, &backup, &a.model)?;
    let record = GrowthRecord {
        time: now(),
        step: a.model.step,
        label: option.label.clone(),
        growth: option.growth.clone(),
        parameters_before: before,
        parameters_after: grown.parameter_count(),
        check,
        backup,
        rolled_back: false,
        automatic,
    };
    a.meta.growth.original_parameters.get_or_insert(before);
    a.meta.growth.history.push(record.clone());
    a.model = grown;
    refresh_meta(&mut a.meta, &a.model);
    save_in(dir, &a.meta, &a.model, &a.archive)?;
    let _ = yendory_engine::learn::append_jsonl(
        &dir.join("history.jsonl"),
        &serde_json::json!({"time": now(), "grown": record.label, "parameters_before": before, "parameters_after": record.parameters_after, "automatic": automatic}),
    );
    Ok(record)
}

/// Restores the model from before its latest growth (in `dir`, the model's folder). The current
/// model (with any training since the growth) is kept as a checkpoint in the growth folder.
pub(crate) fn roll_back_active(dir: &std::path::Path, a: &mut Active) -> Res<()> {
    let i = a
        .meta
        .growth
        .history
        .iter()
        .rposition(|r| !r.rolled_back)
        .ok_or("This model has no growth to roll back")?;
    let backup = a.meta.growth.history[i].backup.clone();
    let restored = Bicameral::from_bytes(&std::fs::read(dir.join("growth").join(&backup)).map_err(|e| {
        format!("The recovery checkpoint {backup} could not be read ({e}); nothing was changed")
    })?)?;
    let kept = format!("{:03}-rolled-back-{}.{}", i + 1, a.model.parameter_count(), extension(&a.model));
    keep_checkpoint(dir, &kept, &a.model)?;
    a.meta.growth.history[i].rolled_back = true;
    a.model = restored;
    refresh_meta(&mut a.meta, &a.model);
    save_in(dir, &a.meta, &a.model, &a.archive)?;
    let _ = yendory_engine::learn::append_jsonl(
        &dir.join("history.jsonl"),
        &serde_json::json!({"time": now(), "growth_rolled_back": backup, "kept": kept}),
    );
    Ok(())
}

#[tauri::command]
pub fn expand_model(app: AppHandle, studio: State<Studio>, option: String) -> Res<GrowthView> {
    not_teaching(&studio)?;
    let mut guard = studio.active.lock().unwrap();
    let a = guard.as_mut().ok_or("Open or create a model first")?;
    if a.meta.growth.mode == GrowthMode::Off {
        return Err("Growth is off for this model; choose Manual or Automatic first".into());
    }
    let (options, _) = a.model.growth_options();
    let o = options.into_iter().find(|o| o.id == option).ok_or("Unknown growth step")?;
    let dir = model_dir(&app, &a.meta.id)?;
    grow_active(&dir, a, &o, false)?;
    Ok(view(a))
}

#[tauri::command]
pub fn roll_back_growth(app: AppHandle, studio: State<Studio>) -> Res<GrowthView> {
    not_teaching(&studio)?;
    let mut guard = studio.active.lock().unwrap();
    let a = guard.as_mut().ok_or("Open or create a model first")?;
    let dir = model_dir(&app, &a.meta.id)?;
    roll_back_active(&dir, a)?;
    Ok(view(a))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active(model: Bicameral, id: &str) -> Active {
        let mut meta = Meta {
            id: id.into(),
            name: id.into(),
            created: 0,
            updated: 0,
            step: 0,
            parameters: 0,
            groups: [0; 3],
            words: 0,
            slots: 0,
            word_limit: None,
            context: 0,
            taught: Vec::new(),
            origin: "test".into(),
            read: Vec::new(),
            architecture: bicameral(),
            design: None,
            growth: GrowthInfo { mode: GrowthMode::Manual, ..GrowthInfo::default() },
        };
        refresh_meta(&mut meta, &model);
        Active { meta, model, archive: Vec::new() }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("yendory-growth-{tag}-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// An existing trained model (the real step-15,000 checkpoint) and a Nova Neuro model grow
    /// with a verified recovery checkpoint, reload at their new size, train on, and roll back
    /// to exactly the weights they had, keeping the grown model too.
    #[test]
    fn grow_reload_train_and_roll_back() {
        let fixture =
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests/compat/fixtures/nova2-step15000.nnova2");
        let p = yendory_engine::model::neuro::presets().into_iter().next().unwrap();
        for (tag, model, option) in [
            ("bicameral", Bicameral::load(fixture).unwrap(), "balanced"),
            ("neuro", Bicameral::with_design(p.config, Some(p.design), 3), "memory"),
        ] {
            let dir = temp_dir(tag);
            let original = model.clone();
            let mut a = active(model, tag);
            save_in(&dir, &a.meta, &a.model, &a.archive).unwrap();
            let (options, _) = a.model.growth_options();
            let o = options.iter().find(|o| o.id == option).unwrap();
            let record = grow_active(&dir, &mut a, o, false).unwrap();
            assert!(record.check.preserved());
            assert_eq!(record.parameters_before, original.parameter_count());
            assert_eq!(a.meta.parameters, o.parameters_after);
            assert_eq!(a.meta.growth.original_parameters, Some(original.parameter_count()));
            // The recovery checkpoint is the original model; the saved model is the grown one.
            let backup = Bicameral::load(dir.join("growth").join(&record.backup)).unwrap();
            assert_eq!(backup.weights, original.weights);
            let saved = Bicameral::load(dir.join("model.nnova2")).unwrap();
            assert_eq!(saved.parameter_count(), o.parameters_after);
            assert_eq!(saved.neuro.is_some(), original.neuro.is_some());
            let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
            assert_eq!(meta.growth.history.len(), 1);
            // Train the grown model a little, then roll back.
            a.model.train_text("U: hello\nN: Hello!\n", 1e-3, Focus::Joint).unwrap();
            let trained_grown = a.model.clone();
            roll_back_active(&dir, &mut a).unwrap();
            assert_eq!(a.model.weights, original.weights, "{tag}: the original weights are back");
            assert_eq!(a.meta.parameters, original.parameter_count());
            assert!(a.meta.growth.history[0].rolled_back);
            let kept = std::fs::read_dir(dir.join("growth"))
                .unwrap()
                .filter_map(|e| e.ok())
                .find(|e| e.file_name().to_string_lossy().contains("rolled-back"))
                .expect("the grown model is kept");
            assert_eq!(Bicameral::load(kept.path()).unwrap().weights, trained_grown.weights);
            assert!(roll_back_active(&dir, &mut a).is_err(), "nothing left to roll back");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// A rollback whose recovery checkpoint is missing changes nothing.
    #[test]
    fn rollback_without_its_checkpoint_changes_nothing() {
        let dir = temp_dir("missing");
        let mut a = active(Bicameral::with_config(yendory_engine::Config::default(), 2), "missing");
        save_in(&dir, &a.meta, &a.model, &a.archive).unwrap();
        let (options, _) = a.model.growth_options();
        grow_active(&dir, &mut a, &options[0], false).unwrap();
        let grown = a.model.clone();
        for e in std::fs::read_dir(dir.join("growth")).unwrap() {
            std::fs::remove_file(e.unwrap().path()).unwrap();
        }
        assert!(roll_back_active(&dir, &mut a).is_err());
        assert_eq!(a.model.weights, grown.weights);
        assert!(!a.meta.growth.history[0].rolled_back);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
