//! Per-network traces of Nova's replies for § 11 Brain (`yendory_engine::model::inspect`).
//!
//! Three ways to get one:
//! * **Recorded**: a chat with "Record brain traces" on traces each reply while it is written
//!   (exactly the reply it would write without tracing) and saves it with the chat.
//! * **Replay**: any earlier reply of a chat is read through the model again, its saved text as
//!   the reply. Said plainly on the page where the replay can differ from what happened: the
//!   weights may have changed since, Study Mode memories are not rebuilt, and the reply's
//!   tokens are its saved text encoded again.
//! * **Question**: the Brain page's own question, answered greedily (as before).
//!
//! While a trace is made, its progress is sent as "brain-trace" events (live mode). Traces are
//! files under `models/<id>/traces/<chat>/`; only recorded traces are kept.
use super::chat::{build_prompt, chats_dir, load_chat, one_line, Message};
use super::*;
use yendory_engine::model::inspect::{ConversationTrace, StepTrace, TokenInfo, TraceEvent, TraceReply};

fn traces_dir(app: &AppHandle, model_id: &str) -> Res<PathBuf> {
    Ok(model_dir(app, model_id)?.join("traces"))
}

fn check_name(name: &str) -> Res<()> {
    if name.is_empty() || name.len() > 40 || !name.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err("Invalid trace name".into());
    }
    Ok(())
}

fn chat_traces(app: &AppHandle, model_id: &str, chat: &str) -> Res<PathBuf> {
    check_name(chat)?;
    // The chat must exist under this model (its id is checked by the chat module).
    let _ = chats_dir(app, model_id)?;
    Ok(traces_dir(app, model_id)?.join(chat))
}

fn trace_path(app: &AppHandle, model_id: &str, chat: &str, name: &str) -> Res<PathBuf> {
    check_name(name)?;
    Ok(chat_traces(app, model_id, chat)?.join(format!("{name}.json")))
}

/// Saves a recorded trace; returns its name.
pub(crate) fn save(app: &AppHandle, model_id: &str, chat: &str, t: &ConversationTrace) -> Res<String> {
    let name = format!(
        "t{}",
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or_default()
    );
    let path = trace_path(app, model_id, chat, &name)?;
    std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    let bytes = serde_json::to_vec(t).map_err(|e| e.to_string())?;
    write_atomic(&path, &bytes)?;
    Ok(name)
}

/// Replies taken back: their traces go too.
pub(crate) fn forget(app: &AppHandle, model_id: &str, removed: &[Message]) {
    let Ok(dir) = traces_dir(app, model_id) else { return };
    let Ok(chats) = std::fs::read_dir(&dir) else { return };
    let names: Vec<&String> = removed.iter().filter_map(|m| m.info.as_ref()?.trace.as_ref()).collect();
    if names.is_empty() {
        return;
    }
    for chat in chats.flatten() {
        for name in &names {
            let _ = std::fs::remove_file(chat.path().join(format!("{name}.json")));
        }
    }
}

/// A chat deleted: its traces go.
pub(crate) fn forget_chat(app: &AppHandle, model_id: &str, chat: &str) {
    if let Ok(dir) = chat_traces(app, model_id, chat) {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Live<'a> {
    /// "chat" (a reply being written), "replay" or "question".
    source: &'a str,
    chat: Option<&'a str>,
    /// "read" (the prompt is read: `trace` holds it), "step" or "done".
    kind: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace: Option<&'a ConversationTrace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    step: Option<&'a StepTrace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token: Option<&'a TokenInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
}

/// Sends a trace's progress to the page (live mode).
pub(crate) fn emit(app: &AppHandle, source: &str, chat: Option<&str>, e: TraceEvent<'_>) {
    let live = match e {
        TraceEvent::Read(t) => {
            Live { source, chat, kind: "read", trace: Some(t), step: None, token: None, name: None }
        }
        TraceEvent::Step(s, t) => {
            Live { source, chat, kind: "step", trace: None, step: Some(s), token: Some(t), name: None }
        }
    };
    let _ = app.emit("brain-trace", live);
}

pub(crate) fn emit_done(app: &AppHandle, source: &str, chat: Option<&str>, name: Option<&str>) {
    let _ = app
        .emit("brain-trace", Live { source, chat, kind: "done", trace: None, step: None, token: None, name });
}

/// A turn of a chat, for choosing what to trace.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Turn {
    /// Index of Nova's reply in the chat's messages.
    index: usize,
    message: String,
    reply: String,
    time: u64,
    /// The recorded trace, if one was saved.
    trace: Option<String>,
    /// The model's step when it replied, and now.
    step: Option<u64>,
    model_step: u64,
}

#[tauri::command]
pub(crate) fn trace_turns(app: AppHandle, studio: State<Studio>, chat: String) -> Res<Vec<Turn>> {
    let (model_id, step) = with_active(&studio, |a| Ok((a.meta.id.clone(), a.model.step)))?;
    let c = load_chat(&app, &model_id, &chat)?;
    let mut out = Vec::new();
    for (i, m) in c.messages.iter().enumerate() {
        if m.role != "model" || i == 0 {
            continue;
        }
        let user = &c.messages[i - 1];
        let trace = m
            .info
            .as_ref()
            .and_then(|x| x.trace.clone())
            .filter(|name| trace_path(&app, &model_id, &chat, name).map(|p| p.exists()).unwrap_or(false));
        out.push(Turn {
            index: i,
            message: one_line(&user.text).chars().take(160).collect(),
            reply: one_line(&m.text).chars().take(160).collect(),
            time: m.time,
            trace,
            step: m.info.as_ref().map(|x| x.step),
            model_step: step,
        });
    }
    Ok(out)
}

#[tauri::command]
pub(crate) fn trace_load(
    app: AppHandle,
    studio: State<Studio>,
    chat: String,
    name: String,
) -> Res<ConversationTrace> {
    let model_id = with_active(&studio, |a| Ok(a.meta.id.clone()))?;
    let bytes = std::fs::read(trace_path(&app, &model_id, &chat, &name)?).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| format!("The trace cannot be read: {e}"))
}

/// One trace at a time (each uses every CPU thread it can).
struct Tracing<'a>(&'a Studio);
impl<'a> Tracing<'a> {
    fn start(studio: &'a Studio) -> Res<Self> {
        if studio.tracing.swap(true, Ordering::AcqRel) {
            return Err("A trace is already being made".into());
        }
        Ok(Tracing(studio))
    }
}
impl Drop for Tracing<'_> {
    fn drop(&mut self) {
        self.0.tracing.store(false, Ordering::Release);
    }
}

/// Reads an earlier reply of a chat through the current model again and traces it.
#[tauri::command]
pub(crate) async fn trace_replay(
    app: AppHandle,
    studio: State<'_, Studio>,
    chat: String,
    index: usize,
    influence: bool,
) -> Res<ConversationTrace> {
    let _busy = Tracing::start(&studio)?;
    let (model_id, model) = with_active(&studio, |a| Ok((a.meta.id.clone(), a.model.clone())))?;
    let c = load_chat(&app, &model_id, &chat)?;
    let said =
        c.messages.get(index).filter(|m| m.role == "model").ok_or("That is not one of Nova's replies")?;
    let message =
        c.messages.get(index.wrapping_sub(1)).filter(|m| m.role == "user").ok_or("No message before it")?;
    let budget = build_prompt(&model, &c.settings, &c.messages[..index - 1], message, "")?;
    let prompt = model.tokenizer.encode(&budget.prompt)?;
    let info = said.info.as_ref();
    let mut reply = model.tokenizer.encode(&yendory_engine::nexus::sanitize(&said.text))?;
    if info.is_none_or(|i| i.finished) {
        reply.push(b'\n' as TokenId);
    }
    let mut notes = vec![
        "Replay: the saved reply was read through the model as if it were writing it. The reply's tokens are its saved text encoded again, which can differ from the tokens generated (the text is the same).".to_string(),
    ];
    if let Some(i) = info {
        if i.step != model.step {
            notes.push(format!(
                "Nova has trained since this reply (step {} then, {} now): the replay shows the current weights, not those that wrote it.",
                i.step, model.step
            ));
        }
        if !i.memories.is_empty() {
            notes.push(format!(
                "{} Study Mode memories were in front of the message when Nova replied; they are not rebuilt here, so the prompt is not the same.",
                i.memories.len()
            ));
        } else if i.prompt_tokens != budget.prompt_tokens {
            notes.push(format!(
                "The rebuilt prompt is {} tokens; the reply was written after {} (the chat's settings or the model's context have changed since).",
                budget.prompt_tokens, i.prompt_tokens
            ));
        } else {
            notes.push(format!(
                "The rebuilt prompt has the same length as when Nova replied ({} tokens).",
                i.prompt_tokens
            ));
        }
    }
    let app2 = app.clone();
    let chat2 = chat.clone();
    let trace = tauri::async_runtime::spawn_blocking(move || {
        let mut t = model.trace_turn(&prompt, TraceReply::Forced(&reply), |e| {
            emit(&app2, "replay", Some(&chat2), e);
            true
        })?;
        if influence {
            model.measure_influence(&mut t)?;
        }
        t.notes.extend(notes);
        Ok::<_, String>(t)
    })
    .await
    .map_err(|e| e.to_string())??;
    emit_done(&app, "replay", Some(&chat), None);
    Ok(trace)
}

/// The Brain page's question, answered greedily by the current model and traced live.
#[tauri::command]
pub(crate) async fn trace_question(
    app: AppHandle,
    studio: State<'_, Studio>,
    prompt: String,
    influence: bool,
) -> Res<ConversationTrace> {
    let _busy = Tracing::start(&studio)?;
    let message = clean_message(&prompt)?;
    let model = with_active(&studio, |a| Ok(a.model.clone()))?;
    let ids = model.tokenizer.encode(&format!("U: {message}\nN: "))?;
    let app2 = app.clone();
    let trace = tauri::async_runtime::spawn_blocking(move || {
        // As long as the Brain page's thinking tree writes.
        let limit = yendory_engine::model::ThinkOptions::default().max_tokens;
        let mut choose = |p: &[f32], _: &[TokenId]| yendory_engine::model::argmax(p);
        let mut t =
            model.trace_turn(&ids, TraceReply::Generate { max_tokens: limit, choose: &mut choose }, |e| {
                emit(&app2, "question", None, e);
                true
            })?;
        if influence {
            model.measure_influence(&mut t)?;
        }
        t.notes.push(
            "A question on its own (no earlier turns), answered with the most likely token each time.".into(),
        );
        Ok::<_, String>(t)
    })
    .await
    .map_err(|e| e.to_string())??;
    emit_done(&app, "question", None, None);
    Ok(trace)
}

/// Measures each network's influence on a recorded trace (with the weights that wrote it) and
/// saves it.
#[tauri::command]
pub(crate) async fn trace_influence(
    app: AppHandle,
    studio: State<'_, Studio>,
    chat: String,
    name: String,
) -> Res<ConversationTrace> {
    let _busy = Tracing::start(&studio)?;
    let (model_id, model) = with_active(&studio, |a| Ok((a.meta.id.clone(), a.model.clone())))?;
    let path = trace_path(&app, &model_id, &chat, &name)?;
    let mut t: ConversationTrace = serde_json::from_slice(&std::fs::read(&path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if t.model_step != model.step {
        return Err(format!(
            "Nova has trained since this trace was recorded (step {} then, {} now). Influence must be measured with the weights that wrote the reply; replay the turn with influence instead.",
            t.model_step, model.step
        ));
    }
    let t = tauri::async_runtime::spawn_blocking(move || {
        model.measure_influence(&mut t)?;
        let bytes = serde_json::to_vec(&t).map_err(|e| e.to_string())?;
        write_atomic(&path, &bytes)?;
        Ok::<_, String>(t)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(t)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TraceStorage {
    traces: usize,
    bytes: u64,
}

/// Saved traces of the open model: how many and their size on disk.
#[tauri::command]
pub(crate) fn trace_storage(app: AppHandle, studio: State<Studio>) -> Res<TraceStorage> {
    let model_id = with_active(&studio, |a| Ok(a.meta.id.clone()))?;
    let mut out = TraceStorage { traces: 0, bytes: 0 };
    if let Ok(chats) = std::fs::read_dir(traces_dir(&app, &model_id)?) {
        for chat in chats.flatten() {
            for f in std::fs::read_dir(chat.path()).into_iter().flatten().flatten() {
                out.traces += 1;
                out.bytes += f.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    Ok(out)
}

/// Deletes one saved trace, or every saved trace of the open model (`name` and `chat` absent).
#[tauri::command]
pub(crate) fn trace_delete(
    app: AppHandle,
    studio: State<Studio>,
    chat: Option<String>,
    name: Option<String>,
) -> Res<()> {
    let model_id = with_active(&studio, |a| Ok(a.meta.id.clone()))?;
    match (chat, name) {
        (Some(chat), Some(name)) => {
            let path = trace_path(&app, &model_id, &chat, &name)?;
            if path.exists() {
                std::fs::remove_file(path).map_err(|e| e.to_string())?;
            }
        }
        _ => {
            let dir = traces_dir(&app, &model_id)?;
            if dir.exists() {
                std::fs::remove_dir_all(dir).map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}
