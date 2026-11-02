//! The models library: the suite's models folder (Studio's `models` folder), read and written in
//! Studio's own format (`meta.json` field names and meanings as Studio's `Meta`). Every write is
//! atomic, and a change to `meta.json` keeps every field it doesn't know (docs/HANDOFF.md).
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use yendory_engine::model::neuro as engine_neuro;
use yendory_engine::model::{MAX_CONTEXT, MAX_PARAMETERS};
use yendory_engine::tokenizer::BASE;
use yendory_engine::{size_presets, Bicameral, Config, Focus};

pub type Res<T> = Result<T, String>;

/// Large enough for the largest model the Engine accepts (as in Studio).
const MAX_CHECKPOINT_BYTES: u64 = 12_200_000_000;

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Writes a file so that a crash or power loss leaves either the old file or the new one.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Res<()> {
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    drop(f);
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }
    Ok(())
}

/// A model's folder. Ids are what Studio makes: letters, digits and '-'.
pub fn model_dir(models: &Path, id: &str) -> Res<PathBuf> {
    if id.is_empty() || id.len() > 64 || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err("Invalid model id".into());
    }
    Ok(models.join(id))
}

pub fn check_name(name: &str) -> Res<String> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > 60 || name.chars().any(char::is_control) {
        return Err("Give the model a name of 1 to 60 characters".into());
    }
    Ok(name.to_string())
}

/// A new model's folder name, made the way Studio makes them.
fn new_id(models: &Path) -> String {
    loop {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let id = format!("m{nanos:x}");
        if !models.join(&id).exists() {
            return id;
        }
    }
}

fn read_meta(dir: &Path) -> Res<Value> {
    let bytes = std::fs::read(dir.join("meta.json"))
        .map_err(|_| "This model is not in the models folder".to_string())?;
    let meta: Value =
        serde_json::from_slice(&bytes).map_err(|_| "This model's meta.json could not be read".to_string())?;
    if !meta.is_object() {
        return Err("This model's meta.json could not be read".into());
    }
    Ok(meta)
}

/// A model as listed: its meta.json as written, plus the checkpoint's size on disk.
pub fn entry(dir: &Path, meta: Value) -> Value {
    let size = std::fs::metadata(dir.join("model.nnova2")).map(|m| m.len()).unwrap_or(0);
    json!({"meta": meta, "fileBytes": size})
}

/// Every model in the folder (a folder whose meta.json names another id is not one).
pub fn list(models: &Path) -> Res<Vec<Value>> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(models) else { return Ok(out) };
    for e in entries.flatten() {
        let dir = e.path();
        let Ok(meta) = read_meta(&dir) else { continue };
        if meta.get("id").and_then(Value::as_str) == Some(&*e.file_name().to_string_lossy()) {
            out.push(entry(&dir, meta));
        }
    }
    out.sort_by_key(|v| std::cmp::Reverse(v["meta"]["updated"].as_u64().unwrap_or(0)));
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// New models
// ---------------------------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SizeOption {
    id: &'static str,
    name: &'static str,
    /// Parameters with Nova-2's fixed 256 word slots.
    parameters: usize,
    /// Parameters each learned word adds when the vocabulary has a limit.
    per_word: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelOptions {
    bicameral: Vec<SizeOption>,
    neuro: Vec<SizeOption>,
    max_context: usize,
    /// Bytes a checkpoint takes per parameter (weights and the optimizer's two moments).
    bytes_per_parameter: usize,
}

fn parameters(config: &Config, design: Option<&engine_neuro::NeuroSpec>) -> usize {
    Bicameral::tensors_for(config, design).iter().map(|(_, r, c)| r * c).sum()
}

/// The sizes and designs a new model can have (the presets Studio offers).
pub fn options() -> ModelOptions {
    ModelOptions {
        bicameral: size_presets()
            .into_iter()
            .map(|p| SizeOption {
                id: p.id,
                name: p.name,
                parameters: parameters(&p.config, None),
                per_word: p.config.d + 1,
            })
            .collect(),
        neuro: engine_neuro::presets()
            .into_iter()
            .map(|p| {
                let config = Config { vocab: BASE + 256, ..p.config.clone() };
                SizeOption {
                    id: p.id,
                    name: p.name,
                    parameters: parameters(&config, Some(&p.design)),
                    per_word: p.config.d + 1,
                }
            })
            .collect(),
        max_context: MAX_CONTEXT,
        bytes_per_parameter: 12,
    }
}

/// A vocabulary limit must leave the model within the Engine's limits when every slot is used.
fn check_word_limit(config: &Config, limit: usize) -> Res<()> {
    if limit == 0 {
        return Err("Choose a vocabulary limit of at least 1 word".into());
    }
    let full = Config { vocab: BASE + limit, ..config.clone() };
    Bicameral::check_config(&full).map_err(|_| {
        let base = Bicameral::parameters_for(&Config { vocab: BASE, ..config.clone() });
        let most = MAX_PARAMETERS.saturating_sub(base) / (config.d + 1);
        format!("At this size the vocabulary can hold up to {most} words (1 billion parameters)")
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewModel {
    pub name: String,
    /// "bicameral" or "neuro".
    pub architecture: String,
    /// A size preset (Bicameral) or a design preset (Neuro).
    pub preset: String,
    /// `None`: Nova-2's fixed 256 word slots.
    #[serde(default)]
    pub word_limit: Option<usize>,
    pub context: usize,
}

/// The untrained model a choice describes, built as Studio builds it.
pub fn build(choice: &NewModel) -> Res<Bicameral> {
    let (mut config, design) = match choice.architecture.as_str() {
        "bicameral" => {
            let p = size_presets().into_iter().find(|p| p.id == choice.preset).ok_or("Unknown model size")?;
            (p.config, None)
        }
        "neuro" => {
            let p = engine_neuro::presets()
                .into_iter()
                .find(|p| p.id == choice.preset)
                .ok_or("Unknown design")?;
            (Config { vocab: BASE + 256, ..p.config }, Some(p.design))
        }
        _ => return Err("Unknown architecture".into()),
    };
    config.max_tokens = choice.context;
    Bicameral::check_config(&config).map_err(|_| format!("Choose a context of 1 to {MAX_CONTEXT} tokens"))?;
    if let Some(limit) = choice.word_limit {
        check_word_limit(&config, limit)?;
        // No slots yet: one is added for each word the model learns, up to the limit.
        config.vocab = BASE;
    }
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1) | 1;
    Ok(Bicameral::with_design(config, design, seed))
}

/// meta.json for a model just made or imported, with the fields Studio writes (`Meta`, camelCase).
pub fn new_meta(id: &str, name: &str, model: &Bicameral, word_limit: Option<usize>, origin: &str) -> Value {
    let t = now();
    let mut meta = json!({
        "id": id,
        "name": name,
        "created": t,
        "updated": t,
        "step": model.step,
        "parameters": model.parameter_count(),
        "groups": [
            model.group_parameters(Focus::Left),
            model.group_parameters(Focus::Right),
            model.group_parameters(Focus::Interconnect),
        ],
        "words": model.tokenizer.words().len(),
        "slots": model.word_capacity(),
        "wordLimit": word_limit,
        "context": model.config.max_tokens,
        "taught": [],
        "origin": origin,
        "read": [],
        "conversation": [],
        "architecture": model.architecture(),
        "blocks": model.config.blocks,
    });
    if let Some(design) = &model.neuro {
        meta["design"] = serde_json::to_value(design).unwrap_or(Value::Null);
    }
    meta
}

/// Writes a new model's folder (each file atomically) and returns its listing entry.
pub fn install(
    models: &Path,
    name: &str,
    model: &Bicameral,
    word_limit: Option<usize>,
    origin: &str,
) -> Res<Value> {
    let name = check_name(name)?;
    std::fs::create_dir_all(models).map_err(|e| e.to_string())?;
    let id = new_id(models);
    let dir = models.join(&id);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let meta = new_meta(&id, &name, model, word_limit, origin);
    let written = (|| {
        write_atomic(&dir.join("model.nnova2"), &model.to_bytes()?)?;
        write_atomic(&dir.join("archive.json"), b"[]")?;
        write_atomic(&dir.join("meta.json"), &serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?)
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(e);
    }
    Ok(entry(&dir, meta))
}

pub fn create(models: &Path, choice: &NewModel) -> Res<Value> {
    let name = check_name(&choice.name)?;
    let model = build(choice)?;
    install(models, &name, &model, choice.word_limit, "new")
}

/// Imports a checkpoint (Bicameral `.nnova2` or Nova Neuro `.nneuro`) as a new model.
pub fn import(models: &Path, path: &Path, name: Option<&str>) -> Res<Value> {
    let size = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?.len();
    if size > MAX_CHECKPOINT_BYTES {
        return Err("The file is larger than the largest model the Engine accepts".into());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let model = Bicameral::from_bytes(&bytes)?;
    let file = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    let stem = file.trim_end_matches(".nnova2").trim_end_matches(".nneuro").trim();
    let fallback: String =
        if stem.is_empty() { "Imported model".into() } else { stem.chars().take(60).collect() };
    install(models, name.filter(|n| !n.trim().is_empty()).unwrap_or(&fallback), &model, None, &file)
}

// ---------------------------------------------------------------------------------------------
// Changing models
// ---------------------------------------------------------------------------------------------

/// Renames a model. Only `name` changes in meta.json; every other field is kept as written.
pub fn rename(models: &Path, id: &str, name: &str) -> Res<Value> {
    let name = check_name(name)?;
    let dir = model_dir(models, id)?;
    let mut meta = read_meta(&dir)?;
    meta["name"] = Value::String(name);
    write_atomic(&dir.join("meta.json"), &serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?)?;
    Ok(entry(&dir, meta))
}

/// Deletes a model's folder (checkpoint, lessons archive, history, recovery points).
pub fn delete(models: &Path, id: &str) -> Res<()> {
    let dir = model_dir(models, id)?;
    read_meta(&dir)?;
    std::fs::remove_dir_all(dir).map_err(|e| e.to_string())
}

/// Writes a copy of the model's checkpoint to `path` and checks the copy.
pub fn export(models: &Path, id: &str, path: &Path) -> Res<u64> {
    let dir = model_dir(models, id)?;
    read_meta(&dir)?;
    let bytes = std::fs::read(dir.join("model.nnova2")).map_err(|e| e.to_string())?;
    // A checkpoint the Engine cannot read is not exported.
    Bicameral::from_bytes(&bytes)?;
    let mut f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(&bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    if std::fs::read(path).map_err(|e| e.to_string())? != bytes {
        return Err("The exported file does not match the model; try another location".into());
    }
    Ok(bytes.len() as u64)
}

/// Writes an open model's checkpoint after its game skills changed, and the skills' summary in
/// its meta.json (every other field kept as written): `skills` lists each skill's game, name,
/// steps, generation and method, so the Hub shows them without reading the checkpoint.
pub fn save_skills(models: &Path, id: &str, model: &Bicameral) -> Res<Value> {
    let dir = model_dir(models, id)?;
    let mut meta = read_meta(&dir)?;
    write_atomic(&dir.join("model.nnova2"), &model.to_bytes()?)?;
    meta["skills"] = Value::Array(
        model
            .skills
            .iter()
            .map(|k| {
                json!({
                    "game": k.spec.game, "name": k.spec.name, "steps": k.spec.steps,
                    "generation": k.spec.lineage.generation, "method": k.spec.lineage.method,
                    "parameters": k.parameters(),
                })
            })
            .collect(),
    );
    meta["updated"] = json!(now());
    write_atomic(&dir.join("meta.json"), &serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?)?;
    Ok(entry(&dir, meta))
}

/// Loads a model's checkpoint (the file decides its size and design).
pub fn load(models: &Path, id: &str) -> Res<(Value, Bicameral)> {
    let dir = model_dir(models, id)?;
    let meta = read_meta(&dir)?;
    let model = Bicameral::load(dir.join("model.nnova2"))?;
    Ok((meta, model))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(architecture: &str, preset: &str, word_limit: Option<usize>, context: usize) -> NewModel {
        NewModel {
            name: "Test".into(),
            architecture: architecture.into(),
            preset: preset.into(),
            word_limit,
            context,
        }
    }

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("yendory-network-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn listed_parameters_match_the_models_built() {
        let o = options();
        for s in &o.bicameral {
            assert_eq!(
                build(&choice("bicameral", s.id, None, 256)).unwrap().parameter_count(),
                s.parameters,
                "{}",
                s.id
            );
        }
        for s in &o.neuro {
            let m = build(&choice("neuro", s.id, None, 256)).unwrap();
            assert_eq!(m.parameter_count(), s.parameters, "{}", s.id);
            assert_eq!(m.architecture(), "neuro");
        }
    }

    #[test]
    fn new_models_match_studios_defaults() {
        let m = build(&choice("bicameral", "nova2", None, 256)).unwrap();
        assert_eq!(m.config, Config::default());
        let m = build(&choice("bicameral", "4m", Some(1000), 1024)).unwrap();
        assert_eq!((m.word_capacity(), m.config.max_tokens), (0, 1024));
        let meta = new_meta("m1", "Test", &m, Some(1000), "new");
        assert_eq!(
            (meta["slots"].as_u64(), meta["context"].as_u64(), meta["wordLimit"].as_u64()),
            (Some(0), Some(1024), Some(1000))
        );
        let m = build(&choice("neuro", "neuro-core-1m", None, 256)).unwrap();
        assert!(new_meta("m2", "Test", &m, None, "new")["design"].is_object());
        assert_eq!(m.word_capacity(), 256);
    }

    #[test]
    fn bad_choices_are_refused_with_a_reason() {
        assert!(build(&choice("bicameral", "nope", None, 256)).is_err());
        assert!(build(&choice("bicameral", "nova2", None, 0)).is_err());
        assert!(build(&choice("bicameral", "nova2", None, MAX_CONTEXT + 1)).is_err());
        assert!(build(&choice("bicameral", "nova2", Some(0), 256)).is_err());
        assert!(build(&choice("bicameral", "nova2", Some(usize::MAX / 4), 256)).is_err());
        assert!(check_name("  ").is_err());
        assert!(model_dir(Path::new("."), "../x").is_err());
    }

    #[test]
    fn a_model_round_trips_through_the_folder() {
        let models = temp("roundtrip");
        let made = create(&models, &choice("bicameral", "nova2", None, 256)).unwrap();
        let id = made["meta"]["id"].as_str().unwrap().to_string();
        assert_eq!(
            made["fileBytes"].as_u64(),
            Some(std::fs::metadata(models.join(&id).join("model.nnova2")).unwrap().len())
        );
        // A field the Network doesn't know survives a rename.
        let path = models.join(&id).join("meta.json");
        let mut meta: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        meta["fromStudio"] = json!({"kept": true});
        std::fs::write(&path, serde_json::to_vec(&meta).unwrap()).unwrap();
        let renamed = rename(&models, &id, "Robin").unwrap();
        assert_eq!(renamed["meta"]["name"], "Robin");
        assert_eq!(renamed["meta"]["fromStudio"]["kept"], true);
        // Export, import the export, and the two checkpoints are the same.
        let out = models.join("robin.nnova2");
        export(&models, &id, &out).unwrap();
        let imported = import(&models, &out, None).unwrap();
        assert_eq!(imported["meta"]["name"], "robin");
        assert_eq!(imported["meta"]["origin"], "robin.nnova2");
        let a = std::fs::read(models.join(&id).join("model.nnova2")).unwrap();
        let b = std::fs::read(models.join(imported["meta"]["id"].as_str().unwrap()).join("model.nnova2"))
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(list(&models).unwrap().len(), 2);
        delete(&models, &id).unwrap();
        assert_eq!(list(&models).unwrap().len(), 1);
        assert!(delete(&models, &id).is_err());
        let _ = std::fs::remove_dir_all(models);
    }
}
