//! The Conversation stage, after the Classroom: Nova practises real conversations between two
//! people (and the replies the teacher marked in its own chats). The training itself is
//! `yendory_engine::learn::conversation::converse`; this is the page's backend, and the run the
//! Planner calls for a conversation step.
use super::*;
use crate::reading::{corpora, corpus_path, load_info, CorpusInfo};
use crate::sources::{conversation_sources as dataset_sources, Source};
use yendory_engine::learn::conversation::{
    converse as practise, parse_dialogue, ConversationCheck, ConversationConfig, ConversationEvent,
    ConversationSet, UnderstandingScore,
};
use yendory_engine::learn::corpus::{normalize_ascii, split_documents, Category};
use yendory_engine::learn::listening::ContextScore;

/// Read text rehearsed and guarded during practice, at most (documents, bytes).
const READING_DOCS: usize = 3000;
const READING_BYTES: usize = 3 << 20;
/// Conversation datasets practised in one run (downloads and your own together).
const MAX_DATASETS: usize = 32;

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
    /// The curriculum stage reached (0-based) and why the run stopped if Nova was stuck.
    #[serde(default)]
    stage: usize,
    #[serde(default)]
    stuck: Option<String>,
    /// Context tests before practice, and the check of the checkpoint kept.
    #[serde(default)]
    baseline_context: Option<ContextScore>,
    #[serde(default)]
    best_check: Option<ConversationCheck>,
    /// Understanding tests before practice (the kept checkpoint's are in `best_check`).
    #[serde(default)]
    baseline_understanding: Option<UnderstandingScore>,
    /// Read text used for rehearsal and its guard (documents).
    #[serde(default)]
    reading_documents: usize,
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
    /// Corpora the model read that are on this device (rehearsed and guarded).
    reading: Vec<String>,
}

#[tauri::command]
pub fn conversation_sources() -> Vec<Source> {
    dataset_sources()
}

/// Corpora the model read that are still on this device (conversation corpora aside).
fn read_corpora(app: &AppHandle, meta: &Meta) -> Vec<CorpusInfo> {
    let read: Vec<&str> = meta.read.iter().flat_map(|r| r.sha256s()).collect();
    corpora(app.clone())
        .unwrap_or_default()
        .into_iter()
        .filter(|c| read.contains(&c.sha256.as_str()) && c.category != Category::Conversation)
        .collect()
}

#[tauri::command]
pub fn conversation_material(app: AppHandle, studio: State<Studio>) -> Res<Material> {
    let corpora =
        corpora(app.clone())?.into_iter().filter(|c| c.category == Category::Conversation).collect();
    let Ok((meta, archive)) = with_active(&studio, |a| Ok((a.meta.clone(), a.archive.clone()))) else {
        return Ok(Material {
            corpora,
            chat_replies: 0,
            lesson_units: 0,
            classroom_questions: 0,
            reading: Vec::new(),
        });
    };
    Ok(Material {
        corpora,
        chat_replies: chat::marked_dialogues(&app, &meta.id, "Nova").1,
        lesson_units: archive.iter().filter(|i| !i.held_out).count(),
        classroom_questions: archive
            .iter()
            .filter(|i| i.held_out && i.kind != Kind::Prose && (i.answer.is_some() || !i.replies.is_empty()))
            .count(),
        reading: read_corpora(&app, &meta).into_iter().map(|c| c.name).collect(),
    })
}

fn yes() -> bool {
    true
}

/// The Conversation page's settings (also kept in a plan's conversation step).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
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
    /// Percent of the Classroom answers (right before practice) a checkpoint may get wrong.
    guard_drop_percent: f32,
    // How Nova is taught.
    #[serde(default = "yes")]
    curriculum: bool,
    competency_loss: f32,
    competency_gain_percent: f32,
    advance_when_stuck: bool,
    review_percent: f32,
    #[serde(default = "yes")]
    targeted: bool,
    context_weight: f32,
    listening_percent: f32,
    context_tests: usize,
    #[serde(default = "yes")]
    reply_checks: bool,
    reading_percent: f32,
    reading_rise_percent: f32,
    /// Leave out conversations where a speaker says it is an AI or an assistant.
    #[serde(default = "yes")]
    skip_ai_voice: bool,
    /// Experimental: also keep the best balanced checkpoint (saved as a model of its own).
    #[serde(default)]
    keep_balanced: bool,
    /// Percent of training rows from the built-in example conversations.
    examples_percent: f32,
    /// Listening practice also practises the understanding skills.
    #[serde(default = "yes")]
    understanding_practice: bool,
    /// Understanding test pairs per kind at each check (teacher-forced and free-running).
    understanding_tests: usize,
}

impl Default for ConverseOptions {
    fn default() -> Self {
        let d = ConversationConfig::default();
        ConverseOptions {
            corpus_ids: Vec::new(),
            chats: true,
            tokens: d.tokens,
            batch_size: d.batch_size,
            learning_rate: d.learning_rate,
            both_sides: d.both_sides,
            own_name: d.own_name,
            situations: d.situations,
            earlier_percent: 100. * d.earlier_share,
            guard_questions: d.guard_questions,
            guard_drop_percent: 100. * d.guard_drop,
            curriculum: d.curriculum,
            competency_loss: d.competency_loss,
            competency_gain_percent: 100. * d.competency_gain,
            advance_when_stuck: d.advance_when_stuck,
            review_percent: 100. * d.review_share,
            targeted: d.targeted,
            context_weight: d.context_weight,
            listening_percent: 100. * d.context_practice,
            context_tests: d.context_tests,
            reply_checks: d.reply_checks,
            reading_percent: 100. * d.reading_share,
            reading_rise_percent: 100. * d.reading_rise,
            skip_ai_voice: d.skip_ai_voice,
            keep_balanced: d.keep_balanced,
            examples_percent: 100. * d.examples_share,
            understanding_practice: d.understanding_practice,
            understanding_tests: d.understanding_tests,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ConverseFinished {
    pub(crate) model: Option<Meta>,
    pub(crate) accepted: bool,
    pub(crate) baseline_loss: f32,
    pub(crate) best_loss: f32,
    pub(crate) tokens: u64,
    pub(crate) stopped: bool,
    pub(crate) early_stop: bool,
    pub(crate) error: Option<String>,
    /// Why the run stopped when Nova was stuck in a stage.
    pub(crate) stuck: Option<String>,
    /// `keep_balanced`: the best balanced checkpoint, saved as a model of its own (its name),
    /// when it is not the checkpoint kept.
    #[serde(default)]
    pub(crate) balanced: Option<String>,
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

/// A conversation run checked and ready.
pub(crate) struct ConversationJob {
    infos: Vec<CorpusInfo>,
    cfg: ConversationConfig,
    chats: bool,
}

/// Checks the options (and that the datasets are on this device).
pub(crate) fn conversation_job(app: &AppHandle, o: &ConverseOptions) -> Res<ConversationJob> {
    if !(10_000..=2_000_000_000).contains(&o.tokens)
        || !(1..=128).contains(&o.batch_size)
        || !(o.learning_rate > 0. && o.learning_rate <= 0.01)
    {
        return Err(
            "Choose 10,000 or more tokens, a batch size of 1-128 and a learning rate up to 0.01".into()
        );
    }
    let percent = |x: f32, hi: f32| (0. ..=hi).contains(&x);
    if !percent(o.earlier_percent, 80.)
        || !percent(o.guard_drop_percent, 100.)
        || !percent(o.review_percent, 50.)
        || !percent(o.listening_percent, 50.)
        || !percent(o.reading_percent, 50.)
        || !percent(o.reading_rise_percent, 100.)
        || !percent(o.competency_gain_percent, 90.)
        || !percent(o.examples_percent, 50.)
    {
        return Err("A percentage setting is out of range (earlier lessons up to 80%; review, listening practice, examples and reading up to 50% each)".into());
    }
    if o.earlier_percent + o.listening_percent + o.reading_percent + o.examples_percent >= 90. {
        return Err(
            "Earlier lessons, listening practice, examples and reading together must leave most rows to conversations"
                .into(),
        );
    }
    if !(o.competency_loss > 0. && o.competency_loss.is_finite()) || !(1. ..=10.).contains(&o.context_weight)
    {
        return Err(
            "Mastery needs a loss above 0, and the weight of replies that use earlier turns is 1 to 10"
                .into(),
        );
    }
    if o.corpus_ids.is_empty() {
        return Err(
            "Tick at least one conversation dataset (your chats alone are too few to check progress)".into(),
        );
    }
    if o.corpus_ids.len() > MAX_DATASETS {
        return Err(format!(
            "{} conversation datasets are ticked; practise with at most {MAX_DATASETS} at a time",
            o.corpus_ids.len()
        ));
    }
    let infos = o.corpus_ids.iter().map(|id| load_info(app, id)).collect::<Res<Vec<_>>>()?;
    if let Some(i) = infos.iter().find(|i| i.category != Category::Conversation) {
        return Err(format!("{} is not a conversation dataset", i.name));
    }
    let cfg = ConversationConfig {
        tokens: o.tokens,
        batch_size: o.batch_size,
        learning_rate: o.learning_rate,
        both_sides: o.both_sides,
        own_name: check_name(&o.own_name)?,
        situations: o.situations,
        earlier_share: o.earlier_percent / 100.,
        guard_questions: o.guard_questions.min(2000),
        guard_drop: o.guard_drop_percent / 100.,
        seed: now(),
        curriculum: o.curriculum,
        competency_loss: o.competency_loss,
        competency_gain: o.competency_gain_percent / 100.,
        advance_when_stuck: o.advance_when_stuck,
        review_share: o.review_percent / 100.,
        targeted: o.targeted,
        context_weight: o.context_weight,
        context_practice: o.listening_percent / 100.,
        context_tests: o.context_tests.min(50),
        reply_checks: o.reply_checks,
        reading_share: o.reading_percent / 100.,
        reading_rise: o.reading_rise_percent / 100.,
        skip_ai_voice: o.skip_ai_voice,
        keep_balanced: o.keep_balanced,
        examples_share: o.examples_percent / 100.,
        understanding_practice: o.understanding_practice,
        understanding_tests: o.understanding_tests.min(50),
        ..ConversationConfig::default()
    };
    Ok(ConversationJob { infos, cfg, chats: o.chats })
}

/// A short description of a conversation run, for plans.
pub(crate) fn conversation_label(app: &AppHandle, o: &ConverseOptions) -> String {
    let names: Vec<String> = o
        .corpus_ids
        .iter()
        .map(|id| load_info(app, id).map(|i| i.name).unwrap_or_else(|_| "(missing dataset)".into()))
        .collect();
    format!("Conversation: {} · {} tokens", names.join(" + "), tokens_text(o.tokens))
}

/// Documents of the text the model read (evenly spread, within the limits).
fn reading_documents(app: &AppHandle, meta: &Meta) -> Vec<String> {
    let mut docs = Vec::new();
    for info in read_corpora(app, meta) {
        let Ok(path) = corpus_path(app, &info.id, "txt") else { continue };
        let Ok(text) = std::fs::read_to_string(path) else { continue };
        docs.extend(split_documents(&normalize_ascii(&text).0));
    }
    let picks: Vec<String> = if docs.len() > READING_DOCS {
        (0..READING_DOCS).map(|i| docs[i * docs.len() / READING_DOCS].clone()).collect()
    } else {
        docs
    };
    let mut bytes = 0;
    picks
        .into_iter()
        .take_while(|d| {
            bytes += d.len();
            bytes <= READING_BYTES
        })
        .collect()
}

/// Runs conversation practice on the open model (on the calling thread), saving every accepted
/// checkpoint and recording the run. `emit` receives every event.
pub(crate) fn run_conversation(
    app: &AppHandle,
    job: &ConversationJob,
    meta: Meta,
    model: Bicameral,
    archive: Vec<Item>,
    stop: &AtomicBool,
    emit: &dyn Fn(&ConversationEvent),
) -> Res<ConverseFinished> {
    let studio = app.state::<Studio>();
    let id = meta.id.clone();
    let cfg = &job.cfg;
    let sources: Vec<ConversationSource> = job
        .infos
        .iter()
        .map(|i| ConversationSource {
            name: i.name.clone(),
            sha256: i.sha256.clone(),
            license: i.license.clone(),
            attribution: i.attribution.clone(),
        })
        .collect();
    append_history(
        app,
        &id,
        serde_json::json!({"type": "conversation_started", "time": now(), "sources": sources, "chats": job.chats, "settings": cfg}),
    );
    let current = Arc::new(Mutex::new(meta.clone()));
    let mut sets = Vec::new();
    for info in &job.infos {
        let text = std::fs::read_to_string(corpus_path(app, &info.id, "txt")?).map_err(|e| e.to_string())?;
        if sha256_hex(text.as_bytes()) != info.sha256 {
            return Err(format!(
                "{}: the stored dataset does not match its fingerprint; download it again",
                info.name
            ));
        }
        let (ascii, _) = normalize_ascii(&text);
        let dialogues: Vec<_> = split_documents(&ascii).iter().filter_map(|d| parse_dialogue(d)).collect();
        if dialogues.is_empty() {
            return Err(format!("{}: no two-person conversations found in it", info.name));
        }
        sets.push(ConversationSet { name: info.name.clone(), dialogues, nova: None, hold_out: true });
    }
    let mut chat_replies = 0;
    if job.chats {
        // Nova's speaker carries the name it is given, so nothing in its replies is renamed.
        let label = if cfg.own_name.is_empty() { "Nova".to_string() } else { cfg.own_name.clone() };
        let (dialogues, marked) = chat::marked_dialogues(app, &id, &label);
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
    let reading = reading_documents(app, &meta);
    let mut observer = |e: &ConversationEvent| {
        emit(e);
        if !matches!(e, ConversationEvent::Batch { .. }) {
            append_history(app, &id, serde_json::json!({"time": now(), "conversation": e}));
        }
    };
    let mut on_accepted = |m: &Bicameral| -> Res<()> {
        let mut meta = current.lock().unwrap();
        refresh_meta(&mut meta, m);
        save_all(app, &meta, m, &archive)?;
        if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == meta.id) {
            a.model = m.clone();
            a.meta = meta.clone();
        }
        Ok(())
    };
    let r = practise(&model, &sets, &archive, &reading, cfg, &mut observer, &mut on_accepted, stop)?;
    drop(sets);
    let mut meta = current.lock().unwrap().clone();
    // The best balanced checkpoint, when it differs from the one kept: a model of its own beside
    // this one (not opened), so the two can be compared.
    let mut balanced = None;
    if let Some((bm, bc)) = r.balanced.as_ref().filter(|(bm, _)| bm.step != r.model.step) {
        let name: String = format!("{} (balanced, step {})", meta.name, bc.step).chars().take(60).collect();
        let mut copy = meta.clone();
        copy.id = new_id();
        copy.name = name.clone();
        copy.created = now();
        copy.origin = format!("best balanced checkpoint of {}'s conversation practice", meta.name);
        refresh_meta(&mut copy, bm);
        save_all(app, &copy, bm, &archive)?;
        append_history(
            app,
            &copy.id,
            serde_json::json!({"type": "balanced_checkpoint", "time": now(), "from": meta.id, "check": bc}),
        );
        balanced = Some(name);
    }
    meta.conversation.push(ConversationRecord {
        sources,
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
        stage: r.stage,
        stuck: r.stuck.clone(),
        baseline_context: r.baseline_context.clone(),
        best_check: r.best_check.clone(),
        baseline_understanding: r.baseline_understanding.clone(),
        reading_documents: reading.len(),
    });
    write_atomic(
        &model_dir(app, &meta.id)?.join("meta.json"),
        &serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?,
    )?;
    if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == meta.id) {
        a.meta = meta.clone();
    }
    append_history(
        app,
        &id,
        serde_json::json!({"type": "conversation_finished", "time": now(), "accepted": r.accepted,
            "tokens": r.tokens, "stopped": r.stopped, "stuck": r.stuck}),
    );
    Ok(ConverseFinished {
        model: Some(meta),
        accepted: r.accepted,
        baseline_loss: r.baseline_loss,
        best_loss: r.best_loss,
        tokens: r.tokens,
        stopped: r.stopped,
        early_stop: r.early_stop,
        error: None,
        stuck: r.stuck,
        balanced,
    })
}

#[tauri::command]
pub fn converse(app: AppHandle, studio: State<Studio>, options: ConverseOptions) -> Res<()> {
    let job = conversation_job(&app, &options)?;
    let (meta, model, archive) =
        with_active(&studio, |a| Ok((a.meta.clone(), a.model.clone(), a.archive.clone())))?;
    no_waiting_checkpoints(&app, &meta.id)?;
    if studio.teaching.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("Already teaching or reading".into());
    }
    studio.stop.store(false, Ordering::Release);
    let stop = studio.stop.clone();
    std::thread::spawn(move || {
        let studio = app.state::<Studio>();
        let id = meta.id.clone();
        let emit = |e: &ConversationEvent| {
            let _ = app.emit("converse-event", e);
        };
        let finished = guarded(|| run_conversation(&app, &job, meta, model, archive, &stop, &emit))
            .unwrap_or_else(|e| {
                append_history(
                    &app,
                    &id,
                    serde_json::json!({"type": "conversation_finished", "time": now(), "error": e}),
                );
                ConverseFinished {
                    model: None,
                    accepted: false,
                    baseline_loss: 0.,
                    best_loss: 0.,
                    tokens: 0,
                    stopped: false,
                    early_stop: false,
                    error: Some(e),
                    stuck: None,
                    balanced: None,
                }
            });
        studio.teaching.store(false, Ordering::Release);
        if let Some(n) = yendory_engine::learn::take_device_notice() {
            let _ = app.emit("device-notice", n);
        }
        let _ = app.emit("converse-finished", &finished);
    });
    Ok(())
}
