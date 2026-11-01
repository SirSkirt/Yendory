//! Opening a model handed over by Yendory Hub: `yendory-tools --model <id>` (the suite's
//! contract, docs/HANDOFF.md on yendory-hub). A second start hands its `--model` to the running
//! copy and closes.
use super::*;

/// What the command line asked for.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Asked {
    Model(String),
    /// `--model` was given but cannot be used.
    Invalid(String),
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

/// The first `--model <id>` (or `--model=<id>`) of a command line.
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
        return Some(if valid_id(&value) {
            Asked::Model(value)
        } else {
            Asked::Invalid(format!("\"{value}\" is not a model id"))
        });
    }
    None
}

/// What the window needs: the model to open, or why none was opened.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Handed {
    pub id: Option<String>,
    pub problem: Option<String>,
}

fn handed(asked: Asked) -> Handed {
    match asked {
        Asked::Model(id) => Handed { id: Some(id), problem: None },
        Asked::Invalid(why) => {
            Handed { id: None, problem: Some(format!("Tools was asked to open a model, but {why}.")) }
        }
    }
}

/// The model asked for at start (given once).
#[tauri::command]
pub(crate) fn handed_model(tools: tauri::State<'_, Tools>) -> Option<Handed> {
    tools.handed.lock().unwrap().take().map(handed)
}

/// A second copy was started: its `--model` comes here ("model-handed"), and this window comes
/// to the front.
#[cfg(desktop)]
pub(crate) fn second_start(app: &AppHandle, argv: Vec<String>) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
    if let Some(asked) = model_argument(argv) {
        let _ = app.emit("model-handed", handed(asked));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_model_argument_is_read_in_both_forms() {
        let args = |a: &[&str]| model_argument(a.iter().map(|s| s.to_string()));
        assert_eq!(args(&["tools", "--model", "m1a"]), Some(Asked::Model("m1a".into())));
        assert_eq!(args(&["tools", "--model=m1a"]), Some(Asked::Model("m1a".into())));
        assert_eq!(args(&["tools"]), None);
        assert!(matches!(args(&["tools", "--model"]), Some(Asked::Invalid(_))));
        assert!(matches!(args(&["tools", "--model", "../x"]), Some(Asked::Invalid(_))));
    }
}
