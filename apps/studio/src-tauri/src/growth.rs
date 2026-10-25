//! Growing a model's network (docs/GROWTH.md): the Engine grows it so it computes what it
//! computed before; Studio keeps a recovery checkpoint of every model it grows, verifies the
//! grown model before replacing the original, records the growth and can roll it back.
use super::*;
use yendory_engine::learn::autogrow::{judge, AutoGrowthRules, Check, Verdict};
use yendory_engine::learn::memory::{self, FitEstimate, StepShape};
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
    /// Limits and thresholds of automatic growth.
    pub rules: AutoGrowthRules,
    /// Automatic growth's latest decisions (grown or not, and why), newest last.
    pub decisions: Vec<AutoDecision>,
}

/// A decision of automatic growth at the end of a reading run.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoDecision {
    pub time: u64,
    pub step: u64,
    /// The growth step taken, if the network grew.
    pub grown: Option<String>,
    pub reason: String,
}

/// Decisions kept per model.
const KEEP_DECISIONS: usize = 20;

/// How training a model of a size would fit this computer's memory, with the batches of the
/// Reading settings.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FitView {
    /// "gpu" or "cpu": where training runs now.
    device: &'static str,
    #[serde(flatten)]
    estimate: FitEstimate,
}

/// The fit of training `parameters` (with `activation` values per token) in batches of `rows`
/// rows of `tokens` tokens, on the training device and within the memory settings.
fn fit(parameters: usize, activation: usize, vocab: usize, rows: usize, tokens: usize) -> FitView {
    let s = memory::memory_settings();
    let shape = StepShape { parameters, activation_floats_per_token: activation, rows, tokens, vocab };
    if yendory_engine::learn::device() == yendory_engine::learn::Device::Cpu {
        // The CPU keeps weights, gradients, both moments and the activations in RAM.
        let need = 16 * parameters as u64 + shape.working_bytes(rows);
        let (total, _) = memory::system_ram();
        return FitView {
            device: "cpu",
            estimate: FitEstimate {
                micro_rows: rows,
                optimizer_in_ram: true,
                plan: "On the CPU, in system RAM".into(),
                gpu_bytes: 0,
                ram_bytes: need,
                fits: (total > 0).then(|| need <= total.saturating_sub(s.ram_reserve_bytes)),
            },
        };
    }
    let fixed = |ram| yendory_engine::gpu::GpuModel::fixed_bytes_for(parameters, ram);
    FitView {
        device: "gpu",
        estimate: memory::estimate(&s, &shape, fixed(false), fixed(true), s.ram_allowance()),
    }
}

/// The Reading settings' batch, when the page gives none.
const DEFAULT_ROWS: usize = 16;
const DEFAULT_TOKENS: usize = 128;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OptionView {
    #[serde(flatten)]
    option: GrowthOption,
    /// Weights, gradients and both AdamW moments while training (bytes).
    state_bytes: u64,
    state_bytes_added: u64,
    /// How training the grown model would fit memory.
    fit: FitView,
    /// Within automatic growth's parameter limit.
    within_limit: bool,
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
    /// How training the model as it is fits memory.
    fit: FitView,
    rules: AutoGrowthRules,
    decisions: Vec<AutoDecision>,
    /// The batch the fits are estimated for.
    rows: usize,
    tokens: usize,
}

fn view(a: &Active, rows: usize, tokens: usize) -> GrowthView {
    let (options, not_offered) = a.model.growth_options();
    let now = a.model.parameter_count();
    let state = |p: usize| 16 * p as u64;
    let c = &a.model.config;
    let tokens = tokens.min(c.max_tokens);
    let limit = a.meta.growth.rules.max_parameters;
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
                fit: fit(o.parameters_after, o.activation_floats_per_token, c.vocab, rows, tokens),
                within_limit: limit.is_none_or(|m| o.parameters_after <= m),
                option: o,
            })
            .collect(),
        not_offered,
        can_roll_back: a.meta.growth.history.iter().any(|r| !r.rolled_back),
        history: a.meta.growth.history.clone(),
        fit: fit(now, Bicameral::activation_floats_for(c, a.model.neuro.as_ref()), c.vocab, rows, tokens),
        rules: a.meta.growth.rules.clone(),
        decisions: a.meta.growth.decisions.clone(),
        rows,
        tokens,
    }
}

fn batch(rows: Option<usize>, tokens: Option<usize>) -> (usize, usize) {
    (rows.unwrap_or(DEFAULT_ROWS).clamp(1, 128), tokens.unwrap_or(DEFAULT_TOKENS).clamp(16, 1 << 16))
}

fn save_meta(dir: &std::path::Path, meta: &Meta) -> Res<()> {
    write_atomic(&dir.join("meta.json"), &serde_json::to_vec_pretty(meta).map_err(|e| e.to_string())?)
}

#[tauri::command]
pub fn growth_view(studio: State<Studio>, rows: Option<usize>, tokens: Option<usize>) -> Res<GrowthView> {
    let (rows, tokens) = batch(rows, tokens);
    with_active(&studio, |a| Ok(view(a, rows, tokens)))
}

#[tauri::command]
pub fn set_growth_mode(
    app: AppHandle,
    studio: State<Studio>,
    mode: GrowthMode,
    rows: Option<usize>,
    tokens: Option<usize>,
) -> Res<GrowthView> {
    let (rows, tokens) = batch(rows, tokens);
    let mut guard = studio.active.lock().unwrap();
    let a = guard.as_mut().ok_or("Open or create a model first")?;
    a.meta.growth.mode = mode;
    if mode == GrowthMode::Automatic && a.meta.growth.rules.max_parameters.is_none() {
        // A limit from the start: automatic growth up to four times the model's size now.
        a.meta.growth.rules.max_parameters = Some(4 * a.model.parameter_count());
    }
    save_meta(&model_dir(&app, &a.meta.id)?, &a.meta)?;
    Ok(view(a, rows, tokens))
}

#[tauri::command]
pub fn set_growth_rules(
    app: AppHandle,
    studio: State<Studio>,
    rules: AutoGrowthRules,
    rows: Option<usize>,
    tokens: Option<usize>,
) -> Res<GrowthView> {
    let (rows, tokens) = batch(rows, tokens);
    if rules.max_parameters.is_some_and(|m| m > yendory_engine::model::MAX_PARAMETERS)
        || !(1..=1000).contains(&rules.plateau_checks)
        || !(0. ..=0.5).contains(&rules.min_gain)
        || !(0. ..=2.).contains(&rules.max_gap)
    {
        return Err(
            "Check the limits: at most 1e9 parameters, 1-1000 checks, a gain of 0-50% and a gap of 0-200%"
                .into(),
        );
    }
    let mut guard = studio.active.lock().unwrap();
    let a = guard.as_mut().ok_or("Open or create a model first")?;
    a.meta.growth.rules = rules;
    save_meta(&model_dir(&app, &a.meta.id)?, &a.meta)?;
    Ok(view(a, rows, tokens))
}

/// Automatic growth at the end of a reading run (a safe boundary: nothing is training): judges
/// the run's held-out checks and, when the network looks short of capacity, grows it by the
/// largest step within the parameter limit and the memory limits. Every decision is recorded.
pub(crate) fn automatic_after_reading(
    dir: &std::path::Path,
    a: &mut Active,
    checks: &[Check],
    early_stop: bool,
    rows: usize,
    tokens: usize,
) -> Option<AutoDecision> {
    if a.meta.growth.mode != GrowthMode::Automatic {
        return None;
    }
    let last = a.meta.growth.history.iter().rev().find(|r| !r.rolled_back).map_or(0, |r| r.step);
    let rules = a.meta.growth.rules.clone();
    let mut decision = AutoDecision { time: now(), step: a.model.step, grown: None, reason: String::new() };
    decision.reason = match judge(checks, early_stop, a.model.step.saturating_sub(last), &rules) {
        Verdict::Wait(why) => format!("Not grown: {why}."),
        Verdict::Grow(why) => {
            let v = view(a, rows, tokens);
            let allowed: Vec<&OptionView> =
                v.options.iter().filter(|o| o.within_limit && o.fit.estimate.fits != Some(false)).collect();
            let choice = allowed
                .iter()
                .find(|o| o.option.id == "balanced")
                .or_else(|| allowed.iter().max_by_key(|o| o.option.parameters_after))
                .map(|o| o.option.clone());
            match choice {
                None => format!(
                    "Not grown, although {why}: no growth step stays within the limits ({} parameters at most; training memory as set)",
                    rules.max_parameters.map_or("no limit on".into(), |m| m.to_string())
                ),
                Some(o) => match grow_active(dir, a, &o, true) {
                    Ok(r) => {
                        decision.grown = Some(r.label.clone());
                        format!("Grown by {} ({} -> {} parameters): {why}.", r.label, r.parameters_before, r.parameters_after)
                    }
                    Err(e) => format!("Growth was due ({why}) but did not happen: {e}"),
                },
            }
        }
    };
    let d = &mut a.meta.growth.decisions;
    d.push(decision.clone());
    let excess = d.len().saturating_sub(KEEP_DECISIONS);
    d.drain(..excess);
    let _ = save_meta(dir, &a.meta);
    let _ = yendory_engine::learn::append_jsonl(
        &dir.join("history.jsonl"),
        &serde_json::json!({"time": now(), "automatic_growth": &decision}),
    );
    Some(decision)
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
pub fn expand_model(
    app: AppHandle,
    studio: State<Studio>,
    option: String,
    rows: Option<usize>,
    tokens: Option<usize>,
) -> Res<GrowthView> {
    let (rows, tokens) = batch(rows, tokens);
    not_teaching(&studio)?;
    let mut guard = studio.active.lock().unwrap();
    let a = guard.as_mut().ok_or("Open or create a model first")?;
    if a.meta.growth.mode == GrowthMode::Off {
        return Err("Growth is off for this model; choose Manual or Automatic first".into());
    }
    let (options, _) = a.model.growth_options();
    let o = options.into_iter().find(|o| o.id == option).ok_or("Unknown growth step")?;
    let f = fit(
        o.parameters_after,
        o.activation_floats_per_token,
        a.model.config.vocab,
        rows,
        tokens.min(a.model.config.max_tokens),
    );
    if f.estimate.fits == Some(false) {
        return Err(format!(
            "The grown model would not fit the training memory limits: {} with batches of {rows} x {tokens} tokens needs about {} of {} memory. Change the Training memory settings or use smaller batches first; nothing was changed",
            f.estimate.plan,
            memory::gib(if f.device == "cpu" { f.estimate.ram_bytes } else { f.estimate.gpu_bytes }),
            if f.device == "cpu" { "system" } else { "GPU" }
        ));
    }
    let dir = model_dir(&app, &a.meta.id)?;
    grow_active(&dir, a, &o, false)?;
    Ok(view(a, rows, tokens))
}

#[tauri::command]
pub fn roll_back_growth(app: AppHandle, studio: State<Studio>) -> Res<GrowthView> {
    not_teaching(&studio)?;
    let mut guard = studio.active.lock().unwrap();
    let a = guard.as_mut().ok_or("Open or create a model first")?;
    let dir = model_dir(&app, &a.meta.id)?;
    roll_back_active(&dir, a)?;
    Ok(view(a, DEFAULT_ROWS, DEFAULT_TOKENS))
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
            blocks: None,
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

    /// Automatic growth grows a plateaued, capacity-short model at the end of a reading run,
    /// within the parameter limit, records why it did or did not, and never grows a model that
    /// is still improving, memorising, over its limit or set to grow manually.
    #[test]
    fn automatic_growth_follows_the_evidence_and_the_limits() {
        let fixture =
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests/compat/fixtures/nova2-step15000.nnova2");
        let dir = temp_dir("auto");
        let mut a = active(Bicameral::load(fixture).unwrap(), "auto");
        save_in(&dir, &a.meta, &a.model, &a.archive).unwrap();
        let checks = |held: &[f32], gap: f32| -> Vec<Check> {
            held.iter().map(|&h| Check { train_loss: h * (1. - gap), held_out_loss: h }).collect()
        };
        let flat = checks(&[3.0, 2.2, 2.0, 1.999, 1.998, 1.998, 1.997], 0.02);
        // Manual models are left alone.
        assert!(automatic_after_reading(&dir, &mut a, &flat, false, 4, 64).is_none());
        a.meta.growth.mode = GrowthMode::Automatic;
        let n = a.model.parameter_count();
        // Still improving, or memorising: not grown, with the reason.
        let improving = checks(&[3.0, 2.6, 2.3, 2.1, 1.95, 1.85], 0.02);
        let d = automatic_after_reading(&dir, &mut a, &improving, false, 4, 64).unwrap();
        assert!(d.grown.is_none() && d.reason.contains("still improving"), "{}", d.reason);
        let d = automatic_after_reading(
            &dir,
            &mut a,
            &checks(&[3.0, 2.2, 2.0, 1.999, 1.998, 1.998, 1.997], 0.3),
            false,
            4,
            64,
        )
        .unwrap();
        assert!(d.grown.is_none() && d.reason.contains("memorising"), "{}", d.reason);
        // Over the parameter limit: no step allowed.
        a.meta.growth.rules.max_parameters = Some(n + 10);
        let d = automatic_after_reading(&dir, &mut a, &flat, false, 4, 64).unwrap();
        assert!(d.grown.is_none() && d.reason.contains("limits"), "{}", d.reason);
        assert_eq!(a.model.parameter_count(), n);
        // Within the limit: grown by a verified step, recorded as automatic.
        a.meta.growth.rules.max_parameters = Some(2 * n);
        let before = a.model.clone();
        let d = automatic_after_reading(&dir, &mut a, &flat, false, 4, 64).unwrap();
        println!("{}", d.reason);
        assert!(d.grown.is_some(), "{}", d.reason);
        let grown = a.model.parameter_count();
        assert!(grown > n && grown <= 2 * n);
        assert!(a.meta.growth.history.last().unwrap().automatic);
        assert!(compare_outputs(&before, &a.model, &PROBES).unwrap().preserved());
        // Right after a growth, the new capacity is trained first.
        let d = automatic_after_reading(&dir, &mut a, &flat, false, 4, 64).unwrap();
        assert!(d.grown.is_none() && d.reason.contains("since the last growth"), "{}", d.reason);
        let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta.growth.decisions.len(), 5);
        assert_eq!(meta.parameters, grown);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Growth options carry their memory fit, and a fit beyond the limits is reported.
    #[test]
    fn growth_options_report_their_memory_fit() {
        let a = active(Bicameral::with_config(yendory_engine::Config::default(), 2), "fit");
        let v = view(&a, 8, 64);
        assert!(v.options.iter().all(|o| o.fit.estimate.ram_bytes > 0 || o.fit.estimate.gpu_bytes > 0));
        let big = v.options.iter().max_by_key(|o| o.option.parameters_after).unwrap();
        assert!(big.fit.estimate.ram_bytes >= v.fit.estimate.ram_bytes);
        // A CPU fit needs the model state and activations, and respects the RAM reserve.
        let f = fit(1_000_000_000, 100_000, 50_000, 128, 4096);
        let (total, _) = memory::system_ram();
        if total > 0 {
            assert_eq!(f.estimate.fits, Some(false), "a 1e9-parameter model with large batches does not fit");
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
