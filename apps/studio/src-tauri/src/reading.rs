//! Reading (corpus pre-training): corpora on this device, downloads and conversion, practice
//! generated on this device, and the reading thread. Corpora are stored as
//! <app data>/corpora/<id>.{txt,json}; the training itself is
//! `yendory_engine::learn::corpus::pretrain_mix`.
use super::*;
use crate::sources::{conversation_sources, gutenberg_book, sources, Convert, Fetch, Source};
use std::time::Duration;
use yendory_engine::learn::autogrow::Check;
use yendory_engine::learn::corpus::{
    pretrain_mix, Category, Corpus, MixPart, PretrainConfig, PretrainEvent, END_OF_TEXT,
};
use yendory_engine::learn::generate::{generate, Family, GenerateConfig};

/// Largest corpus kept on the device (downloads stop here; larger files are refused).
pub const MAX_CORPUS_BYTES: u64 = 400 << 20;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusInfo {
    pub(crate) id: String,
    pub(crate) name: String,
    /// Download URL or the imported file's name.
    pub(crate) origin: String,
    pub(crate) license: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
    pub(crate) documents: usize,
    pub(crate) held_out: usize,
    pub(crate) dropped_chars: usize,
    /// The download stopped at the size limit (the file continues beyond it).
    pub(crate) truncated: bool,
    pub(crate) added: u64,
    /// What kind of text it is (for mixing and bicameral reading).
    #[serde(default)]
    pub(crate) category: Category,
    /// How to credit the source.
    #[serde(default)]
    pub(crate) attribution: String,
}

/// One corpus of a reading, as recorded.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadPartRecord {
    name: String,
    sha256: String,
    category: Category,
    share: f32,
    stage: Option<EducationStage>,
    license: String,
    attribution: String,
}

/// What a model has read, kept in its metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadRecord {
    corpus: String,
    sha256: String,
    /// The corpora of a mixed reading, with their shares, stages and licences.
    #[serde(default)]
    parts: Vec<ReadPartRecord>,
    /// "joint" or "bicameral".
    #[serde(default)]
    mode: String,
    tokens: u64,
    baseline_loss: f32,
    best_loss: f32,
    accepted: bool,
    new_words: usize,
    stopped: bool,
    time: u64,
}

fn corpora_dir(app: &AppHandle) -> Res<PathBuf> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join("corpora");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

pub(crate) fn corpus_path(app: &AppHandle, id: &str, ext: &str) -> Res<PathBuf> {
    if id.len() != 16 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Invalid corpus id".into());
    }
    Ok(corpora_dir(app)?.join(format!("{id}.{ext}")))
}

/// Checks and stores a corpus; the same text is stored once (its id is its fingerprint).
struct Provenance<'a> {
    origin: &'a str,
    license: &'a str,
    attribution: &'a str,
    category: Category,
}

fn store(app: &AppHandle, name: &str, from: &Provenance, bytes: Vec<u8>, truncated: bool) -> Res<CorpusInfo> {
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let corpus = Corpus::prepare(&text);
    if corpus.train.is_empty() || corpus.held_out.is_empty() {
        return Err(format!(
            "{name}: too little text ({} documents). A corpus needs at least a few dozen documents \
             (stories or paragraphs separated by blank lines).",
            corpus.train.len() + corpus.held_out.len()
        ));
    }
    let sha256 = sha256_hex(text.as_bytes());
    let info = CorpusInfo {
        id: sha256[..16].to_string(),
        name: name.to_string(),
        origin: from.origin.to_string(),
        license: from.license.to_string(),
        attribution: from.attribution.to_string(),
        category: from.category,
        bytes: text.len() as u64,
        sha256,
        documents: corpus.train.len() + corpus.held_out.len(),
        held_out: corpus.held_out.len(),
        dropped_chars: corpus.dropped_chars,
        truncated,
        added: now(),
    };
    write_atomic(&corpus_path(app, &info.id, "txt")?, text.as_bytes())?;
    write_atomic(
        &corpus_path(app, &info.id, "json")?,
        &serde_json::to_vec_pretty(&info).map_err(|e| e.to_string())?,
    )?;
    Ok(info)
}

pub(crate) fn load_info(app: &AppHandle, id: &str) -> Res<CorpusInfo> {
    let bytes = std::fs::read(corpus_path(app, id, "json")?).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

/// Streams `url` into memory, at most `limit` bytes. A download cut at the limit ends at the last
/// complete story (`<|endoftext|>`) or paragraph, so no document is torn in half.
pub fn fetch(
    agent: &ureq::Agent,
    url: &str,
    limit: u64,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> Res<(Vec<u8>, bool)> {
    let resp = agent.get(url).call().map_err(|e| format!("Download failed: {e}"))?;
    let total = resp.header("Content-Length").and_then(|v| v.parse::<u64>().ok());
    let mut reader = resp.into_reader();
    let mut data = Vec::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut truncated = false;
    let mut reported = 0;
    loop {
        if cancel.load(Ordering::Acquire) {
            return Err("Download cancelled".into());
        }
        let n = reader.read(&mut buf).map_err(|e| format!("Download failed: {e}"))?;
        if n == 0 {
            break;
        }
        let room = (limit - data.len() as u64) as usize;
        data.extend_from_slice(&buf[..n.min(room)]);
        if data.len() as u64 - reported >= 1 << 20 {
            reported = data.len() as u64;
            progress(reported, total);
        }
        if data.len() as u64 >= limit {
            truncated = total.is_none_or(|t| t > limit);
            break;
        }
    }
    progress(data.len() as u64, total);
    if truncated {
        let cut = find_last(&data, END_OF_TEXT.as_bytes())
            .map(|i| i + END_OF_TEXT.len())
            .or_else(|| find_last(&data, b"\n\n"))
            .unwrap_or(data.len());
        data.truncate(cut);
    }
    Ok((data, truncated))
}

fn find_last(data: &[u8], needle: &[u8]) -> Option<usize> {
    data.windows(needle.len()).rposition(|w| w == needle)
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .try_proxy_from_env(true)
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(120))
        .user_agent(concat!("YendoryStudio/", env!("CARGO_PKG_VERSION")))
        .build()
}

/// Paces requests to one server: a gap between requests that widens when the server says
/// "too many requests" (429), so a long download slows down instead of failing.
pub struct Pacer {
    gap: Duration,
    last: Option<std::time::Instant>,
}
impl Pacer {
    pub fn new() -> Self {
        Pacer { gap: Duration::from_millis(1000), last: None }
    }
}

/// Sleeps in short slices so a cancel is noticed. False if cancelled.
fn wait_for(time: Duration, cancel: &AtomicBool) -> bool {
    let until = std::time::Instant::now() + time;
    while std::time::Instant::now() < until {
        if cancel.load(Ordering::Acquire) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(200).min(until - std::time::Instant::now()));
    }
    !cancel.load(Ordering::Acquire)
}

/// Rate limited: the server kept answering 429 after every wait.
const RATE_LIMITED: &str = "the server is limiting downloads (too many requests)";

/// GETs JSON at the pacer's speed. When the server answers 429 it waits as long as the server
/// asks (Retry-After) or 30 s, doubling up to 5 minutes, for up to about 20 minutes, and
/// slows later requests down. Busy servers (5xx) and dropped connections are retried too.
/// `note` reports waiting (and `None` when downloading again).
fn get_json(
    agent: &ureq::Agent,
    url: &str,
    cancel: &AtomicBool,
    pacer: &mut Pacer,
    note: &mut dyn FnMut(Option<String>),
) -> Res<serde_json::Value> {
    let (mut limited_wait, mut limited_total) = (Duration::from_secs(30), Duration::ZERO);
    let (mut busy_wait, mut busy_tries) = (Duration::from_secs(2), 0);
    let mut waited = false;
    loop {
        if let Some(last) = pacer.last {
            let next = last + pacer.gap;
            let now = std::time::Instant::now();
            if next > now && !wait_for(next - now, cancel) {
                return Err("Download cancelled".into());
            }
        }
        if cancel.load(Ordering::Acquire) {
            return Err("Download cancelled".into());
        }
        pacer.last = Some(std::time::Instant::now());
        let mut request = agent.get(url);
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if url.contains("huggingface.co") && !token.trim().is_empty() {
                request = request.set("Authorization", &format!("Bearer {}", token.trim()));
            }
        }
        let wait = match request.call() {
            Ok(r) => {
                if waited {
                    note(None);
                }
                let text = r.into_string().map_err(|e| format!("Download failed: {e}"))?;
                return serde_json::from_str(&text).map_err(|e| format!("Download failed: {e}"));
            }
            Err(ureq::Error::Status(429, r)) => {
                if limited_total >= Duration::from_secs(20 * 60) {
                    return Err(format!("Download stopped: {RATE_LIMITED}. Try again later."));
                }
                let asked = r.header("Retry-After").and_then(|v| v.trim().parse::<u64>().ok());
                let wait =
                    asked.map(Duration::from_secs).unwrap_or(limited_wait).min(Duration::from_secs(600));
                limited_wait = (limited_wait * 2).min(Duration::from_secs(300));
                limited_total += wait;
                pacer.gap = (pacer.gap * 2).clamp(Duration::from_secs(2), Duration::from_secs(10));
                note(Some(format!(
                    "The server asked to slow down (too many requests). Waiting {} s, then continuing more slowly.",
                    wait.as_secs()
                )));
                wait
            }
            Err(ureq::Error::Status(code, _)) if code < 500 => {
                return Err(format!("Download failed: the server answered {code} for {url}"))
            }
            Err(e) => {
                busy_tries += 1;
                if busy_tries > 5 {
                    return Err(format!("Download failed: {e}"));
                }
                note(Some(format!(
                    "The server did not answer ({e}). Trying again in {} s.",
                    busy_wait.as_secs()
                )));
                let w = busy_wait;
                busy_wait *= 2;
                w
            }
        };
        waited = true;
        if !wait_for(wait, cancel) {
            return Err("Download cancelled".into());
        }
    }
}

/// Documents from a Hugging Face dataset, page by page (100 rows), until `limit` bytes. The
/// limit is shared evenly between the configs (e.g. the 20 bAbI tasks). If the server stops
/// answering partway, what was downloaded is kept (marked as cut short).
#[allow(clippy::too_many_arguments)]
pub fn hf_documents(
    agent: &ureq::Agent,
    dataset: &str,
    configs: &[&str],
    split: &str,
    convert: Convert,
    limit: u64,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, Option<u64>),
    note: &mut dyn FnMut(Option<String>),
) -> Res<(Vec<String>, bool)> {
    let mut docs = Vec::new();
    let mut truncated = false;
    let per_config = (limit / configs.len().max(1) as u64).max(1);
    let mut pacer = Pacer::new();
    for config in configs {
        let (mut used, mut offset) = (0u64, 0u64);
        loop {
            let url = format!(
                "https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length=100"
            );
            let page = match get_json(agent, &url, cancel, &mut pacer, note) {
                Ok(p) => p,
                Err(e) if !docs.is_empty() && !cancel.load(Ordering::Acquire) => {
                    note(Some(format!("{e} Keeping the {} documents downloaded so far.", docs.len())));
                    return Ok((docs, true));
                }
                Err(e) => return Err(e),
            };
            let rows = page.get("rows").and_then(|r| r.as_array()).cloned().unwrap_or_default();
            if rows.is_empty() {
                break;
            }
            for r in &rows {
                for d in convert(&r["row"]) {
                    used += d.len() as u64 + 16;
                    docs.push(d);
                }
            }
            offset += rows.len() as u64;
            progress(docs.iter().map(|d| d.len() as u64 + 16).sum::<u64>().min(limit), Some(limit));
            let rows_total = page.get("num_rows_total").and_then(|n| n.as_u64()).unwrap_or(0);
            if offset >= rows_total {
                break;
            }
            if used >= per_config {
                truncated = true;
                break;
            }
        }
    }
    Ok((docs, truncated))
}

/// Documents from a JSON-lines file.
fn jsonl_documents(
    agent: &ureq::Agent,
    url: &str,
    convert: Convert,
    limit: u64,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> Res<(Vec<String>, bool)> {
    let (bytes, truncated) = fetch(agent, url, limit, cancel, progress)?;
    let text = String::from_utf8_lossy(&bytes);
    let mut docs = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            docs.extend(convert(&v));
        }
    }
    Ok((docs, truncated))
}

/// Documents from Project Gutenberg books, one book after another, until `limit` bytes.
fn gutenberg_documents(
    agent: &ureq::Agent,
    books: &[(u32, &str)],
    limit: u64,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> Res<(Vec<String>, bool)> {
    let mut docs = Vec::new();
    let mut used = 0u64;
    for (i, (id, _)) in books.iter().enumerate() {
        if used >= limit {
            return Ok((docs, true));
        }
        if i > 0 {
            std::thread::sleep(Duration::from_millis(500)); // be gentle with gutenberg.org
        }
        let url = format!("https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt");
        let (bytes, _) = fetch(agent, &url, 8 << 20, cancel, &mut |_, _| {})?;
        for d in gutenberg_book(&String::from_utf8_lossy(&bytes)) {
            used += d.len() as u64 + 16;
            docs.push(d);
        }
        progress(used.min(limit), Some(limit));
    }
    Ok((docs, false))
}

/// Downloads a source and converts it into documents (plain text sources are kept as they are).
pub fn source_text(
    agent: &ureq::Agent,
    src: &Source,
    limit: u64,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, Option<u64>),
    note: &mut dyn FnMut(Option<String>),
) -> Res<(Vec<u8>, bool)> {
    let (docs, truncated) = match src.fetch {
        Fetch::Text(url) => return fetch(agent, url, limit, cancel, progress),
        Fetch::HfRows { dataset, configs, split, convert } => {
            hf_documents(agent, dataset, configs, split, convert, limit, cancel, progress, note)?
        }
        Fetch::JsonLines(url, convert) => jsonl_documents(agent, url, convert, limit, cancel, progress)?,
        Fetch::Gutenberg(books) => gutenberg_documents(agent, books, limit, cancel, progress)?,
    };
    if docs.is_empty() {
        return Err(format!("{}: nothing usable was downloaded", src.name));
    }
    let mut text = docs.join(&format!("\n{END_OF_TEXT}\n"));
    text.push('\n');
    Ok((text.into_bytes(), truncated))
}

#[tauri::command]
pub fn corpus_sources() -> Vec<Source> {
    sources()
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FamilyInfo {
    id: Family,
    name: &'static str,
}

#[tauri::command]
pub fn practice_families() -> Vec<FamilyInfo> {
    Family::ALL.iter().map(|&f| FamilyInfo { id: f, name: f.name() }).collect()
}

/// Generates practice text on this device and stores it as a Reasoning corpus.
#[tauri::command]
pub async fn generate_corpus(
    app: AppHandle,
    families: Vec<Family>,
    difficulty: u8,
    mb: f32,
) -> Res<CorpusInfo> {
    if !(0.1..=200.).contains(&mb) || !(1..=3).contains(&difficulty) {
        return Err("Choose 0.1 to 200 MB and a difficulty of 1 to 3".into());
    }
    tauri::async_runtime::spawn_blocking(move || {
        let cfg = GenerateConfig {
            families: families.clone(),
            difficulty,
            bytes: (mb * 1048576.) as usize,
            seed: now(),
        };
        let text = generate(&cfg)?;
        let kinds = if families.len() == Family::ALL.len() {
            "all kinds".to_string()
        } else {
            families.iter().map(|f| f.name()).collect::<Vec<_>>().join(", ")
        };
        let name = format!("Practice ({kinds}, difficulty {difficulty}, {mb} MB)");
        store(
            &app,
            &name,
            &Provenance {
                origin: "generated on this device",
                license: "Generated on this device (no restrictions)",
                attribution: "Yendory practice generator",
                category: Category::Reasoning,
            },
            text.into_bytes(),
            false,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn corpora(app: AppHandle) -> Res<Vec<CorpusInfo>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(corpora_dir(&app)?).map_err(|e| e.to_string())? {
        let name = entry.map_err(|e| e.to_string())?.file_name().to_string_lossy().to_string();
        if let Some(id) = name.strip_suffix(".json") {
            if let Ok(info) = load_info(&app, id) {
                out.push(info);
            }
        }
    }
    out.sort_by_key(|c| std::cmp::Reverse(c.added));
    Ok(out)
}

#[tauri::command]
pub fn delete_corpus(app: AppHandle, studio: State<Studio>, id: String) -> Res<Vec<CorpusInfo>> {
    not_teaching(&studio)?;
    for ext in ["txt", "json"] {
        let p = corpus_path(&app, &id, ext)?;
        if p.exists() {
            std::fs::remove_file(p).map_err(|e| e.to_string())?;
        }
    }
    corpora(app)
}

#[tauri::command]
pub async fn import_corpus(app: AppHandle, category: Option<Category>) -> Res<Option<CorpusInfo>> {
    let Some(path) = app.dialog().file().add_filter("Plain text (.txt)", &["txt"]).blocking_pick_file()
    else {
        return Ok(None);
    };
    let name = display_name(&path);
    let bytes = read_picked(&app, &path, MAX_CORPUS_BYTES as usize)?;
    tauri::async_runtime::spawn_blocking(move || {
        let from = Provenance {
            origin: &name,
            license: "as supplied by you",
            attribution: "",
            category: category.unwrap_or_default(),
        };
        store(&app, &name, &from, bytes, false)
    })
    .await
    .map_err(|e| e.to_string())?
    .map(Some)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DownloadEvent {
    source: String,
    received: u64,
    total: Option<u64>,
    done: bool,
    error: Option<String>,
    corpus: Option<CorpusInfo>,
    /// Why the download is waiting (rate limit, busy server), or a note on the result.
    note: Option<String>,
}

#[tauri::command]
pub fn download_corpus(app: AppHandle, studio: State<Studio>, source: String, max_mb: u64) -> Res<()> {
    let src = sources()
        .into_iter()
        .chain(conversation_sources())
        .find(|s| s.id == source)
        .ok_or("Unknown corpus source")?;
    let limit = (max_mb.max(1) << 20).min(MAX_CORPUS_BYTES);
    if studio.downloading.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("A download is already running".into());
    }
    studio.cancel_download.store(false, Ordering::Release);
    let cancel = studio.cancel_download.clone();
    std::thread::spawn(move || {
        let emit = |e: DownloadEvent| {
            let _ = app.emit("corpus-download", e);
        };
        let last_note = std::cell::RefCell::new(None::<String>);
        let mut progress = |received, total| {
            emit(DownloadEvent {
                source: source.clone(),
                received,
                total,
                done: false,
                error: None,
                corpus: None,
                note: None,
            })
        };
        let mut note = |n: Option<String>| {
            last_note.replace(n.clone());
            emit(DownloadEvent {
                source: source.clone(),
                received: 0,
                total: None,
                done: false,
                error: None,
                corpus: None,
                note: Some(n.unwrap_or_default()),
            })
        };
        let result = source_text(&agent(), &src, limit, &cancel, &mut progress, &mut note).and_then(
            |(bytes, truncated)| {
                let mb = bytes.len() as f64 / (1 << 20) as f64;
                let name = if truncated {
                    format!("{} · first {mb:.0} MB", src.name)
                } else {
                    src.name.to_string()
                };
                let from = Provenance {
                    origin: src.page,
                    license: src.license,
                    attribution: src.attribution,
                    category: src.category,
                };
                store(&app, &name, &from, bytes, truncated)
            },
        );
        let (corpus, error) = match result {
            Ok(c) => (Some(c), None),
            Err(e) => (None, Some(e)),
        };
        app.state::<Studio>().downloading.store(false, Ordering::Release);
        let note = last_note.into_inner();
        emit(DownloadEvent { source, received: 0, total: None, done: true, error, corpus, note });
    });
    Ok(())
}

#[tauri::command]
pub fn cancel_download(studio: State<Studio>) {
    studio.cancel_download.store(true, Ordering::Release);
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadOptions {
    /// The corpora to read and their shares of the batches.
    parts: Vec<ReadPart>,
    tokens: u64,
    batch_size: usize,
    learning_rate: f32,
    window: usize,
    grow_vocabulary: bool,
    max_words: usize,
    /// "joint" (the whole brain reads everything) or "bicameral" (each category in its stage).
    #[serde(default)]
    mode: String,
    /// Bicameral reading: the stage of each category ("dense", "transformer", "cooperative",
    /// "off"); categories not listed use the defaults (knowledge → Dense, conversation →
    /// Transformer, reasoning → Co-op, general → Transformer).
    #[serde(default)]
    stages: std::collections::BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadPart {
    corpus_id: String,
    share: f32,
}

fn category_key(c: Category) -> &'static str {
    match c {
        Category::General => "general",
        Category::Knowledge => "knowledge",
        Category::Conversation => "conversation",
        Category::Reasoning => "reasoning",
    }
}

/// The stage a category is read in, in bicameral reading.
fn stage_for(options: &ReadOptions, c: Category) -> Res<Option<EducationStage>> {
    if options.mode != "bicameral" {
        return Ok(None);
    }
    let default = match c {
        Category::Knowledge => "dense",
        Category::Reasoning => "cooperative",
        Category::General | Category::Conversation => "transformer",
    };
    education_stage(Some(options.stages.get(category_key(c)).map(String::as_str).unwrap_or(default)))
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReadFinished {
    model: Option<Meta>,
    accepted: bool,
    baseline_loss: f32,
    best_loss: f32,
    tokens: u64,
    stopped: bool,
    early_stop: bool,
    error: Option<String>,
    /// Automatic growth's decision after this run (models set to grow automatically).
    growth: Option<growth::AutoDecision>,
}

#[tauri::command]
pub fn read_corpus(app: AppHandle, studio: State<Studio>, options: ReadOptions) -> Res<()> {
    if !(10_000..=2_000_000_000).contains(&options.tokens)
        || !(1..=128).contains(&options.batch_size)
        || !(options.learning_rate > 0. && options.learning_rate <= 0.01)
        || !(16..=yendory_engine::model::MAX_CONTEXT).contains(&options.window)
    {
        return Err("Choose 10,000 or more tokens, a batch size of 1-128, a learning rate up to 0.01 \
                    and a window of at least 16 tokens (it is cut to the model's context)"
            .into());
    }
    if options.parts.is_empty() || options.parts.len() > 12 {
        return Err("Choose 1 to 12 corpora to read".into());
    }
    if options.parts.iter().any(|p| !(p.share.is_finite() && p.share > 0.)) {
        return Err("Give every corpus a share above 0".into());
    }
    let mut infos = Vec::new();
    for p in &options.parts {
        let info = load_info(&app, &p.corpus_id)?;
        let stage = stage_for(&options, info.category)?;
        infos.push((info, p.share, stage));
    }
    let (meta, model) = with_active(&studio, |a| Ok((a.meta.clone(), a.model.clone())))?;
    if studio.teaching.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("Already teaching or reading".into());
    }
    studio.stop.store(false, Ordering::Release);
    let cfg = PretrainConfig {
        tokens: options.tokens,
        batch_size: options.batch_size,
        learning_rate: options.learning_rate,
        window: options.window,
        grow_vocabulary: options.grow_vocabulary,
        max_words: options.max_words,
        word_limit: meta.word_limit,
        seed: now(),
        ..PretrainConfig::default()
    };
    let mode = if options.mode == "bicameral" { "bicameral" } else { "joint" };
    let records: Vec<ReadPartRecord> = infos
        .iter()
        .map(|(i, share, stage)| ReadPartRecord {
            name: i.name.clone(),
            sha256: i.sha256.clone(),
            category: i.category,
            share: *share,
            stage: *stage,
            license: i.license.clone(),
            attribution: i.attribution.clone(),
        })
        .collect();
    let corpora: Vec<&CorpusInfo> = infos.iter().map(|(i, _, _)| i).collect();
    append_history(
        &app,
        &meta.id,
        serde_json::json!({"type": "reading_started", "time": now(), "corpus": corpora.first(), "corpora": corpora,
            "parts": records, "mode": mode, "options": cfg}),
    );
    let stop = studio.stop.clone();
    std::thread::spawn(move || {
        let studio = app.state::<Studio>();
        let id = meta.id.clone();
        let current = Arc::new(Mutex::new(meta.clone()));
        let run = || -> Res<ReadFinished> {
            let mut parts = Vec::new();
            for (info, share, stage) in &infos {
                let path = corpus_path(&app, &info.id, "txt")?;
                let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
                if sha256_hex(text.as_bytes()) != info.sha256 {
                    return Err(format!(
                        "{}: the stored corpus does not match its fingerprint; download or import it again",
                        info.name
                    ));
                }
                parts.push(MixPart {
                    name: info.name.clone(),
                    category: info.category,
                    corpus: Corpus::prepare(&text),
                    share: *share,
                    stage: *stage,
                });
            }
            let mut observer = |e: &PretrainEvent| {
                let _ = app.emit("read-event", e);
                if !matches!(e, PretrainEvent::Batch { .. }) {
                    append_history(&app, &id, serde_json::json!({"time": now(), "reading": e}));
                }
            };
            let mut on_accepted = |m: &Bicameral| -> Res<()> {
                let mut meta = current.lock().unwrap();
                refresh_meta(&mut meta, m);
                let archive = with_active(&studio, |a| Ok(a.archive.clone())).unwrap_or_default();
                save_all(&app, &meta, m, &archive)?;
                if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == meta.id) {
                    a.model = m.clone();
                    a.meta = meta.clone();
                }
                Ok(())
            };
            let r = pretrain_mix(&model, &parts, &cfg, &mut observer, &mut on_accepted, &stop)?;
            drop(parts);
            let mut meta = current.lock().unwrap().clone();
            meta.read.push(ReadRecord {
                corpus: infos.iter().map(|(i, _, _)| i.name.as_str()).collect::<Vec<_>>().join(" + "),
                sha256: infos[0].0.sha256.clone(),
                parts: records.clone(),
                mode: mode.into(),
                tokens: r.tokens,
                baseline_loss: r.baseline_loss,
                best_loss: r.best_loss,
                accepted: r.accepted,
                new_words: r.new_words,
                stopped: r.stopped,
                time: now(),
            });
            write_atomic(
                &model_dir(&app, &meta.id)?.join("meta.json"),
                &serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?,
            )?;
            let mut decision = None;
            if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == meta.id) {
                a.meta = meta.clone();
                // A safe boundary: the run is over and nothing else trains until it is reported.
                if !r.stopped {
                    let checks: Vec<Check> = r
                        .evaluations
                        .iter()
                        .map(|e| Check { train_loss: e.train_loss, held_out_loss: e.held_out_loss })
                        .collect();
                    let dir = model_dir(&app, &meta.id)?;
                    decision = growth::automatic_after_reading(
                        &dir,
                        a,
                        &checks,
                        r.early_stop,
                        cfg.batch_size,
                        cfg.window,
                    );
                    meta = a.meta.clone();
                }
            }
            Ok(ReadFinished {
                growth: decision,
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
        let finished = guarded(run).unwrap_or_else(|e| ReadFinished {
            model: None,
            accepted: false,
            baseline_loss: 0.,
            best_loss: 0.,
            tokens: 0,
            stopped: false,
            early_stop: false,
            error: Some(e),
            growth: None,
        });
        append_history(
            &app,
            &id,
            serde_json::json!({"type": "reading_finished", "time": now(), "accepted": finished.accepted,
                "tokens": finished.tokens, "stopped": finished.stopped, "error": finished.error}),
        );
        studio.teaching.store(false, Ordering::Release);
        if let Some(n) = yendory_engine::learn::take_device_notice() {
            let _ = app.emit("device-notice", n);
        }
        let _ = app.emit("read-finished", &finished);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// Serves `body` once over plain HTTP on localhost.
    fn serve(body: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut req = [0u8; 1024];
            let _ = s.read(&mut req);
            let _ =
                write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            let _ = s.write_all(&body);
        });
        format!("http://{addr}/stories.txt")
    }

    /// A server that answers 429 (asking to wait 1 s) `limited` times, then serves JSON.
    fn rate_limited(limited: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for i in 0..=limited {
                let (mut s, _) = listener.accept().unwrap();
                let mut req = [0u8; 1024];
                let _ = s.read(&mut req);
                let _ = if i < limited {
                    write!(s, "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                } else {
                    write!(
                        s,
                        "HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{{\"rows\":[]}}"
                    )
                };
            }
        });
        format!("http://{addr}/rows")
    }

    #[test]
    fn waits_out_rate_limits_and_slows_down() {
        let local = ureq::AgentBuilder::new().build();
        let mut pacer = Pacer::new();
        let mut notes = Vec::new();
        let started = std::time::Instant::now();
        let page =
            get_json(&local, &rate_limited(2), &AtomicBool::new(false), &mut pacer, &mut |n| notes.push(n))
                .unwrap();
        assert_eq!(page["rows"], serde_json::json!([]));
        assert!(started.elapsed() >= Duration::from_secs(2), "it waited as the server asked");
        assert_eq!(notes.len(), 3, "{notes:?}");
        assert!(notes[0].as_deref().unwrap().contains("Waiting 1 s"));
        assert_eq!(notes[2], None, "the note clears when downloading continues");
        assert!(pacer.gap >= Duration::from_secs(2), "later requests are spaced further apart");
        // Cancelling while waiting stops at once.
        let cancel = AtomicBool::new(true);
        assert!(get_json(&local, &rate_limited(5), &cancel, &mut Pacer::new(), &mut |_| {}).is_err());
    }

    fn stories(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| format!("Story {i}: a cat found a hat.\n<|endoftext|>\n"))
            .collect::<String>()
            .into_bytes()
    }

    #[test]
    fn downloads_whole_files_and_cuts_partial_ones_at_a_story_boundary() {
        let local = ureq::AgentBuilder::new().build();
        let body = stories(100);
        let (all, truncated) =
            fetch(&local, &serve(body.clone()), 1 << 20, &AtomicBool::new(false), &mut |_, _| {}).unwrap();
        assert_eq!((all, truncated), (body.clone(), false));
        let (part, truncated) =
            fetch(&local, &serve(body.clone()), 1000, &AtomicBool::new(false), &mut |_, _| {}).unwrap();
        assert!(truncated && part.len() <= 1000 && part.ends_with(END_OF_TEXT.as_bytes()));
        assert!(body.starts_with(&part));
        let cancelled = fetch(&local, &serve(body), 1 << 20, &AtomicBool::new(true), &mut |_, _| {});
        assert!(cancelled.is_err());
    }

    /// Downloads a little of every source through the real code path and shows what the model
    /// will read (network; run in CI with `--ignored`).
    #[test]
    #[ignore]
    fn sources_are_reachable_and_convert() {
        let mut failed = Vec::new();
        for src in sources().into_iter().chain(conversation_sources()) {
            let limit = match src.fetch {
                Fetch::Gutenberg(_) => 200_000,
                _ => 300_000,
            };
            match source_text(&agent(), &src, limit, &AtomicBool::new(false), &mut |_, _| {}, &mut |n| {
                if let Some(n) = n {
                    println!("{n}");
                }
            }) {
                Ok((bytes, truncated)) => {
                    let text = String::from_utf8_lossy(&bytes);
                    let corpus = Corpus::prepare(&text);
                    let first = corpus.train.first().cloned().unwrap_or_default();
                    println!(
                        "OK {}: {} bytes, {} documents ({} held out), truncated {truncated}, {} dropped characters\n--- first document ---\n{}\n",
                        src.id,
                        bytes.len(),
                        corpus.train.len() + corpus.held_out.len(),
                        corpus.held_out.len(),
                        corpus.dropped_chars,
                        &first[..first.len().min(700)]
                    );
                    if corpus.train.is_empty() || corpus.held_out.is_empty() {
                        failed.push(format!("{}: too few documents", src.id));
                    }
                    if conversation_sources().iter().any(|c| c.id == src.id) {
                        let parsed = corpus
                            .train
                            .iter()
                            .filter(|d| yendory_engine::learn::conversation::parse_dialogue(d).is_some())
                            .count();
                        println!(
                            "{}: {parsed} of {} documents are two-speaker dialogues",
                            src.id,
                            corpus.train.len()
                        );
                        if parsed * 2 < corpus.train.len() {
                            failed.push(format!("{}: only {parsed} dialogues", src.id));
                        }
                    }
                }
                Err(e) => failed.push(format!("{}: {e}", src.id)),
            }
        }
        assert!(failed.is_empty(), "{failed:#?}");
    }

    /// Checks the real TinyStories download (network; run in CI with `--ignored`).
    #[test]
    #[ignore]
    fn tinystories_is_reachable() {
        let src = &sources()[0];
        let (bytes, truncated) =
            source_text(&agent(), src, 1 << 20, &AtomicBool::new(false), &mut |_, _| {}, &mut |_| {})
                .unwrap();
        assert!(truncated);
        let corpus = Corpus::prepare(&String::from_utf8_lossy(&bytes));
        println!(
            "{} documents, {} held out, {} dropped characters",
            corpus.train.len() + corpus.held_out.len(),
            corpus.held_out.len(),
            corpus.dropped_chars
        );
        assert!(corpus.train.len() > 500 && !corpus.held_out.is_empty());
        let card = agent()
            .get("https://huggingface.co/datasets/roneneldan/TinyStories/raw/main/README.md")
            .call()
            .unwrap()
            .into_string()
            .unwrap();
        let license =
            card.lines().find(|l| l.trim_start().starts_with("license")).unwrap_or("license: (not stated)");
        println!("dataset card: {license}");
    }
}
