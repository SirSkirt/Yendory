//! Opening a model handed over by Yendory Hub: `yendory-studio --model <id>` (the suite's
//! contract, `docs/HANDOFF.md` on the Hub's branch).
//!
//! - The id is a folder in Studio's own models folder; it must hold a `meta.json`.
//! - Studio opens it as if the owner had chosen it, after the start-up questions about unfinished
//!   training are answered, and never while training runs (the page says why it did not).
//! - A missing or unreadable model, or a bad argument, starts Studio as usual and says why.
//! - Started a second time, the new copy hands its `--model` to the running one and
//!   closes, so two copies never write the same files.
use super::*;
use std::path::Path;

/// What the command line asked for.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Asked {
    Model(String),
    /// `--model` was given but cannot be used (no id, or not an id).
    Invalid(String),
}

/// A model id as Studio names its model folders: letters, digits and `-`.
pub(crate) fn check_model_id(id: &str) -> Res<()> {
    if id.is_empty() || id.len() > 64 || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err("Invalid model id".into());
    }
    Ok(())
}

/// The first `--model <id>` (or `--model=<id>`) of a command line; other arguments are left
/// alone. `None` when there is none.
pub(crate) fn model_argument<I: IntoIterator<Item = String>>(args: I) -> Option<Asked> {
    let mut args = args.into_iter();
    while let Some(a) = args.next() {
        let value = if a == "--model" {
            match args.next() {
                Some(v) if !v.starts_with("--") => v,
                _ => return Some(Asked::Invalid("--model was given without a model id".into())),
            }
        } else if let Some(v) = a.strip_prefix("--model=") {
            v.to_string()
        } else {
            continue;
        };
        return Some(match check_model_id(&value) {
            Ok(()) => Asked::Model(value),
            Err(_) => Asked::Invalid(format!("\"{value}\" is not a model id")),
        });
    }
    None
}

/// The model handed over, as the page needs it: its id and name, or why it cannot be opened.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Handed {
    pub id: Option<String>,
    pub name: Option<String>,
    pub problem: Option<String>,
}

/// Checks what was asked against the models folder `models`.
pub(crate) fn handed_in(models: &Path, asked: Asked) -> Handed {
    let problem = |id: Option<String>, text: String| Handed { id, name: None, problem: Some(text) };
    let id = match asked {
        Asked::Invalid(why) => {
            return problem(
                None,
                format!("Studio was asked to open a model, but {why}; no model was opened."),
            )
        }
        Asked::Model(id) => id,
    };
    let dir = models.join(&id);
    let meta = dir.join("meta.json");
    if !meta.is_file() {
        return problem(
            Some(id.clone()),
            format!(
                "Studio was asked to open the model {id}, but there is no such model in its models folder ({}), so none was opened. It may have been deleted or moved.",
                models.display()
            ),
        );
    }
    let read = std::fs::read(&meta)
        .map_err(|e| e.to_string())
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).map_err(|e| e.to_string()));
    match read {
        Ok(v) => Handed {
            name: v.get("name").and_then(|n| n.as_str()).map(String::from),
            id: Some(id),
            problem: None,
        },
        Err(e) => problem(
            Some(id.clone()),
            format!("Studio was asked to open the model {id}, but its meta.json cannot be read ({e}), so none was opened."),
        ),
    }
}

/// The model the command line asked Studio to open when it started, checked; given once (a
/// page reload does not open it again).
#[tauri::command]
pub(crate) fn handed_model(app: AppHandle, studio: State<Studio>) -> Res<Option<Handed>> {
    let Some(asked) = studio.handed.lock().unwrap().take() else { return Ok(None) };
    Ok(Some(handed_in(&models_dir(&app)?, asked)))
}

/// A second copy of Studio was started: its `--model` goes to this one (as a "model-handed"
/// event), and this window comes to the front.
pub(crate) fn second_start(app: &AppHandle, argv: Vec<String>) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
    if let (Some(asked), Ok(models)) = (model_argument(argv), models_dir(app)) {
        let _ = app.emit("model-handed", handed_in(&models, asked));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_model_argument_is_read_in_both_forms() {
        let m = |id: &str| Some(Asked::Model(id.into()));
        assert_eq!(
            model_argument(args(&["yendory-studio", "--model", "m18e3ef0178564504"])),
            m("m18e3ef0178564504")
        );
        assert_eq!(
            model_argument(args(&["yendory-studio", "--model=m18e3ef0178564504"])),
            m("m18e3ef0178564504")
        );
        // Other arguments are left alone; the first --model counts.
        assert_eq!(
            model_argument(args(&["studio", "--verbose", "--model", "mab12", "--model", "mcd34"])),
            m("mab12")
        );
        // No --model: Studio starts as it always has.
        assert_eq!(model_argument(args(&["yendory-studio"])), None);
        assert_eq!(model_argument(args(&["yendory-studio", "--modelling"])), None);
    }

    #[test]
    fn a_bad_model_argument_is_reported_not_ignored() {
        for bad in [&["studio", "--model"][..], &["studio", "--model", "--other"], &["studio", "--model="]] {
            assert!(matches!(model_argument(args(bad)), Some(Asked::Invalid(_))), "{bad:?}");
        }
        // Not an id: no path can be reached through it.
        for bad in ["../outside", "m1/../../x", "a b", "m1\\x"] {
            assert!(
                matches!(model_argument(args(&["studio", "--model", bad])), Some(Asked::Invalid(_))),
                "{bad}"
            );
        }
    }

    fn models_folder(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("yendory-handoff-{tag}-{nanos:x}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_model_in_the_models_folder_is_handed_over_with_its_name() {
        let models = models_folder("found");
        std::fs::create_dir_all(models.join("m18e3ef0178564504")).unwrap();
        std::fs::write(
            models.join("m18e3ef0178564504/meta.json"),
            r#"{"id":"m18e3ef0178564504","name":"Echo","step":12}"#,
        )
        .unwrap();
        let h = handed_in(&models, Asked::Model("m18e3ef0178564504".into()));
        assert_eq!(
            h,
            Handed { id: Some("m18e3ef0178564504".into()), name: Some("Echo".into()), problem: None }
        );
        let _ = std::fs::remove_dir_all(models);
    }

    #[test]
    fn a_missing_or_unreadable_model_is_not_opened_and_says_why() {
        let models = models_folder("missing");
        let h = handed_in(&models, Asked::Model("m0000".into()));
        assert_eq!(h.id.as_deref(), Some("m0000"));
        assert!(h.problem.as_deref().unwrap().contains("no such model"), "{h:?}");
        // A folder without meta.json is not a model either.
        std::fs::create_dir_all(models.join("m0001")).unwrap();
        assert!(handed_in(&models, Asked::Model("m0001".into())).problem.is_some());
        // A damaged meta.json.
        std::fs::create_dir_all(models.join("m0002")).unwrap();
        std::fs::write(models.join("m0002/meta.json"), "{not json").unwrap();
        let h = handed_in(&models, Asked::Model("m0002".into()));
        assert!(h.problem.as_deref().unwrap().contains("cannot be read"), "{h:?}");
        // A bad argument.
        let h = handed_in(&models, Asked::Invalid("--model was given without a model id".into()));
        assert_eq!(h.id, None);
        assert!(h.problem.unwrap().contains("without a model id"));
        let _ = std::fs::remove_dir_all(models);
    }
}
