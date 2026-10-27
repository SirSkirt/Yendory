//! The Conversation stage, after the Classroom: Nova practises real conversations between two
//! people (and the replies the teacher marked in its own chats). The training itself is
//! `yendory_engine::learn::conversation::converse`; this is the page's backend.
use super::*;
use crate::reading::{corpora, corpus_path, load_info, CorpusInfo};
use crate::sources::{conversation_sources as dataset_sources, Source};
use yendory_engine::learn::conversation::{
    converse as practise, parse_dialogue, ConversationConfig, ConversationEvent, ConversationSet,
};
use yendory_engine::learn::corpus::{normalize_ascii, Category};

/// A source of conversations, as recorded with the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationSource {
    name: String,
    sha256: String,
    license: String,
    attribution: String,
}

/// A Conversation stage run, kept in the model's metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationRecord {
    sources: Vec<ConversationSource>,
    /// Replies from the teacher's chats that were trained (marked consistent).
    chat_replies: usize,
    tokens: u64,
    baseline_loss: f32,
    best_loss: f32,
    accepted: bool,
    stopped: bool,
    baseline_classroom: usize,
    classroom_total: usize,
    settings: ConversationConfig,
    time: u64,
}

/// What can be practised with the open model.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Material {
    /// Conversation corpora on this device.
    corpora: Vec<CorpusInfo>,
    /// Replies marked consistent in the model's chats.
    chat_replies: usize,
    /// Taught lesson units (mixed in) and held-out Classroom questions (checked).
    lesson_units: usize,
    classroom_questions: usize,
}

#[tauri::command]
pub fn conversation_sources() -> Vec<Source> {
    dataset_sources()
}

#[tauri::command]
pub fn conversation_material(app: AppHandle, studio: State<Studio>) -> Res<Material> {
    let corpora =
        corpora(app.clone())?.into_iter().filter(|c| c.category == Category::Conversation).collect();
    let (chat_replies, lesson_units, classroom_questions) =
        match with_active(&studio, |a| Ok((a.meta.id.clone(), a.archive.clone()))) {
            Ok((id, archive)) => (
                chat::marked_dialogues(&app, &id, "Nova").1,
                archive.iter().filter(|i| !i.held_out).count(),
                archive
                    .iter()
                    .filter(|i| {
                        i.held_out && i.kind != Kind::Prose && (i.answer.is_some() || !i.replies.is_empty())
                    })
                    .count(),
            ),
            Err(_) => (0, 0, 0),
        };
    Ok(Material { corpora, chat_replies, lesson_units, classroom_questions })
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseOptions {
    corpus_ids: Vec<String>,
    /// Also practise the replies marked consistent in the model's chats.
    chats: bool,
    tokens: u64,
    batch_size: usize,
    learning_rate: f32,
    both_sides: bool,
    own_name: String,
    situations: bool,
    /// Percent of training rows from earlier lessons.
    earlier_percent: f32,
    guard_questions: usize,
    /// Percent of the Classroom answers a checkpoint may lose.
    guard_drop_percent: f32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConverseFinished {
    model: Option<Meta>,
    accepted: bool,
    baseline_loss: f32,
    best_loss: f32,
    tokens: u64,
    stopped: bool,
    early_stop: bool,
    error: Option<String>,
}

/// Nova's own name in practice conversations: letters and spaces only (or empty to keep the
/// speakers' names).
fn check_name(name: &str) -> Res<String> {
    let n = name.split_whitespace().collect::<Vec<_>>().join(" ");
    if n.len() > 32 || !n.chars().all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '-') {
        return Err(
            "Nova's name in conversations may have only letters, spaces and hyphens (at most 32)".into()
        );
    }
    Ok(n)
}

#[tauri::command]
pub fn converse(app: AppHandle, studio: State<Studio>, options: ConverseOptions) -> Res<()> {
    if !(10_000..=2_000_000_000).contains(&options.tokens)
        || !(1..=128).contains(&options.batch_size)
        || !(options.learning_rate > 0. && options.learning_rate <= 0.01)
    {
        return Err(
            "Choose 10,000 or more tokens, a batch size of 1-128 and a learning rate up to 0.01".into()
        );
    }
    if !(0. ..=80.).contains(&options.earlier_percent) || !(0. ..=100.).contains(&options.guard_drop_percent)
    {
        return Err("Earlier lessons may be 0-80% of the rows, and the allowed drop 0-100%".into());
    }
    if options.corpus_ids.is_empty() || options.corpus_ids.len() > 8 {
        return Err(
            "Choose 1 to 8 conversation datasets (your chats alone are too few to check progress)".into()
        );
    }
    let own_name = check_name(&options.own_name)?;
    let infos = options.corpus_ids.iter().map(|id| load_info(&app, id)).collect::<Res<Vec<_>>>()?;
    if let Some(i) = infos.iter().find(|i| i.category != Category::Conversation) {
        return Err(format!("{} is not a conversation dataset", i.name));
    }
    let (meta, model, archive) =
        with_active(&studio, |a| Ok((a.meta.clone(), a.model.clone(), a.archive.clone())))?;
    no_waiting_checkpoints(&app, &meta.id)?;
    if studio.teaching.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("Already teaching or reading".into());
    }
    studio.stop.store(false, Ordering::Release);
    let cfg = ConversationConfig {
        tokens: options.tokens,
        batch_size: options.batch_size,
        learning_rate: options.learning_rate,
        both_sides: options.both_sides,
        own_name,
        situations: options.situations,
        earlier_share: options.earlier_percent / 100.,
        guard_questions: options.guard_questions.min(2000),
        guard_drop: options.guard_drop_percent / 100.,
        seed: now(),
        ..ConversationConfig::default()
    };
    let sources: Vec<ConversationSource> = infos
        .iter()
        .map(|i| ConversationSource {
            name: i.name.clone(),
            sha256: i.sha256.clone(),
            license: i.license.clone(),
            attribution: i.attribution.clone(),
        })
        .collect();
    append_history(
        &app,
        &meta.id,
        serde_json::json!({"type": "conversation_started", "time": now(), "sources": sources, "chats": options.chats, "settings": cfg}),
    );
    let stop = studio.stop.clone();
    let chats = options.chats;
    std::thread::spawn(move || {
        let studio = app.state::<Studio>();
        let id = meta.id.clone();
        let current = Arc::new(Mutex::new(meta.clone()));
        let run = || -> Res<ConverseFinished> {
            let mut sets = Vec::new();
            for info in &infos {
                let text = std::fs::read_to_string(corpus_path(&app, &info.id, "txt")?)
                    .map_err(|e| e.to_string())?;
                if sha256_hex(text.as_bytes()) != info.sha256 {
                    return Err(format!(
                        "{}: the stored dataset does not match its fingerprint; download it again",
                        info.name
                    ));
                }
                let (ascii, _) = normalize_ascii(&text);
                let dialogues: Vec<_> = yendory_engine::learn::corpus::split_documents(&ascii)
                    .iter()
                    .filter_map(|d| parse_dialogue(d))
                    .collect();
                if dialogues.is_empty() {
                    return Err(format!("{}: no two-person conversations found in it", info.name));
                }
                sets.push(ConversationSet { name: info.name.clone(), dialogues, nova: None, hold_out: true });
            }
            let mut chat_replies = 0;
            if chats {
                // Nova's speaker carries the name it is given, so nothing in its replies is renamed.
                let label = if cfg.own_name.is_empty() { "Nova".to_string() } else { cfg.own_name.clone() };
                let (dialogues, marked) = chat::marked_dialogues(&app, &id, &label);
                chat_replies = marked;
                if !dialogues.is_empty() {
                    sets.push(ConversationSet {
                        name: "your chats".into(),
                        dialogues,
                        nova: Some(label),
                        hold_out: false,
                    });
                }
            }
            let mut observer = |e: &ConversationEvent| {
                let _ = app.emit("converse-event", e);
                if !matches!(e, ConversationEvent::Batch { .. }) {
                    append_history(&app, &id, serde_json::json!({"time": now(), "conversation": e}));
                }
            };
            let mut on_accepted = |m: &Bicameral| -> Res<()> {
                let mut meta = current.lock().unwrap();
                refresh_meta(&mut meta, m);
                save_all(&app, &meta, m, &archive)?;
                if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == meta.id) {
                    a.model = m.clone();
                    a.meta = meta.clone();
                }
                Ok(())
            };
            let r = practise(&model, &sets, &archive, &cfg, &mut observer, &mut on_accepted, &stop)?;
            drop(sets);
            let mut meta = current.lock().unwrap().clone();
            meta.conversation.push(ConversationRecord {
                sources: sources.clone(),
                chat_replies,
                tokens: r.tokens,
                baseline_loss: r.baseline_loss,
                best_loss: r.best_loss,
                accepted: r.accepted,
                stopped: r.stopped,
                baseline_classroom: r.baseline_classroom,
                classroom_total: r.classroom_total,
                settings: cfg.clone(),
                time: now(),
            });
            write_atomic(
                &model_dir(&app, &meta.id)?.join("meta.json"),
                &serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?,
            )?;
            if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == meta.id) {
                a.meta = meta.clone();
            }
            Ok(ConverseFinished {
                model: Some(meta),
                accepted: r.accepted,
                baseline_loss: r.baseline_loss,
                best_loss: r.best_loss,
                tokens: r.tokens,
                stopped: r.stopped,
                early_stop: r.early_stop,
                error: None,
            })
        };
        let finished = guarded(run).unwrap_or_else(|e| ConverseFinished {
            model: None,
            accepted: false,
            baseline_loss: 0.,
            best_loss: 0.,
            tokens: 0,
            stopped: false,
            early_stop: false,
            error: Some(e),
        });
        append_history(
            &app,
            &id,
            serde_json::json!({"type": "conversation_finished", "time": now(), "accepted": finished.accepted,
                "tokens": finished.tokens, "stopped": finished.stopped, "error": finished.error}),
        );
        studio.teaching.store(false, Ordering::Release);
        if let Some(n) = yendory_engine::learn::take_device_notice() {
            let _ = app.emit("device-notice", n);
        }
        let _ = app.emit("converse-finished", &finished);
    });
    Ok(())
}
