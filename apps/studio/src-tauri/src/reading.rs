//! Reading (corpus pre-training): corpora on this device, downloads, and the reading thread.
//! Corpora are stored as <app data>/corpora/<id>.{txt,json}; the training itself is
//! `yendory_engine::learn::corpus::pretrain`.
use super::*;
use std::time::Duration;
use yendory_engine::learn::corpus::{pretrain, Corpus, PretrainConfig, PretrainEvent, END_OF_TEXT};

/// Largest corpus kept on the device (downloads stop here; larger files are refused).
pub const MAX_CORPUS_BYTES: u64 = 200 << 20;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    url: &'static str,
    page: &'static str,
    license: &'static str,
    /// Full size of the file in MB (approximate); downloads can stop earlier.
    size_mb: u64,
    default_mb: u64,
}

pub fn sources() -> Vec<Source> {
    const PAGE: &str = "https://huggingface.co/datasets/roneneldan/TinyStories";
    const LICENSE: &str = "CDLA-Sharing-1.0 (see the dataset page)";
    vec![
        Source {
            id: "tinystories-valid",
            name: "TinyStories (validation file)",
            description: "About 22,000 short stories in simple English, written for training very small language models.",
            url: "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
            page: PAGE,
            license: LICENSE,
            size_mb: 19,
            default_mb: 19,
        },
        Source {
            id: "tinystories-train",
            name: "TinyStories (training file)",
            description: "The same kind of stories, about 2.1 million of them. Download only the first part you need.",
            url: "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
            page: PAGE,
            license: LICENSE,
            size_mb: 1900,
            default_mb: 50,
        },
    ]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusInfo {
    id: String,
    name: String,
    /// Download URL or the imported file's name.
    origin: String,
    license: String,
    bytes: u64,
    sha256: String,
    documents: usize,
    held_out: usize,
    dropped_chars: usize,
    /// The download stopped at the size limit (the file continues beyond it).
    truncated: bool,
    added: u64,
}

/// What a model has read, kept in its metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadRecord {
    corpus: String,
    sha256: String,
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

fn corpus_path(app: &AppHandle, id: &str, ext: &str) -> Res<PathBuf> {
    if id.len() != 16 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Invalid corpus id".into());
    }
    Ok(corpora_dir(app)?.join(format!("{id}.{ext}")))
}

/// Checks and stores a corpus; the same text is stored once (its id is its fingerprint).
fn store(
    app: &AppHandle,
    name: &str,
    origin: &str,
    license: &str,
    bytes: Vec<u8>,
    truncated: bool,
) -> Res<CorpusInfo> {
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
        origin: origin.to_string(),
        license: license.to_string(),
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

fn load_info(app: &AppHandle, id: &str) -> Res<CorpusInfo> {
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

#[tauri::command]
pub fn corpus_sources() -> Vec<Source> {
    sources()
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
pub async fn import_corpus(app: AppHandle) -> Res<Option<CorpusInfo>> {
    let Some(path) = app.dialog().file().add_filter("Plain text (.txt)", &["txt"]).blocking_pick_file()
    else {
        return Ok(None);
    };
    let name = display_name(&path);
    let bytes = read_picked(&app, &path, MAX_CORPUS_BYTES as usize)?;
    tauri::async_runtime::spawn_blocking(move || {
        store(&app, &name, &name, "as supplied by you", bytes, false)
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
}

#[tauri::command]
pub fn download_corpus(app: AppHandle, studio: State<Studio>, source: String, max_mb: u64) -> Res<()> {
    let src = sources().into_iter().find(|s| s.id == source).ok_or("Unknown corpus source")?;
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
        let mut progress = |received, total| {
            emit(DownloadEvent {
                source: source.clone(),
                received,
                total,
                done: false,
                error: None,
                corpus: None,
            })
        };
        let result =
            fetch(&agent(), src.url, limit, &cancel, &mut progress).and_then(|(bytes, truncated)| {
                let mb = bytes.len() as f64 / (1 << 20) as f64;
                let name = if truncated {
                    format!("{} · first {mb:.0} MB", src.name)
                } else {
                    src.name.to_string()
                };
                store(&app, &name, src.url, src.license, bytes, truncated)
            });
        let (corpus, error) = match result {
            Ok(c) => (Some(c), None),
            Err(e) => (None, Some(e)),
        };
        app.state::<Studio>().downloading.store(false, Ordering::Release);
        emit(DownloadEvent { source, received: 0, total: None, done: true, error, corpus });
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
    corpus_id: String,
    tokens: u64,
    batch_size: usize,
    learning_rate: f32,
    window: usize,
    grow_vocabulary: bool,
    max_words: usize,
    /// Experimental bicameral education stage ("dense", "transformer", "cooperative").
    #[serde(default)]
    education: Option<String>,
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
    let info = load_info(&app, &options.corpus_id)?;
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
        education: education_stage(options.education.as_deref())?,
        seed: now(),
        ..PretrainConfig::default()
    };
    append_history(
        &app,
        &meta.id,
        serde_json::json!({"type": "reading_started", "time": now(), "corpus": info, "options": cfg}),
    );
    let stop = studio.stop.clone();
    std::thread::spawn(move || {
        let studio = app.state::<Studio>();
        let id = meta.id.clone();
        let current = Arc::new(Mutex::new(meta.clone()));
        let run = || -> Res<ReadFinished> {
            let path = corpus_path(&app, &info.id, "txt")?;
            let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            if sha256_hex(text.as_bytes()) != info.sha256 {
                return Err(
                    "The stored corpus does not match its fingerprint; download or import it again".into()
                );
            }
            let corpus = Corpus::prepare(&text);
            drop(text);
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
            let r = pretrain(&model, &corpus, &cfg, &mut observer, &mut on_accepted, &stop)?;
            let mut meta = current.lock().unwrap().clone();
            meta.read.push(ReadRecord {
                corpus: info.name.clone(),
                sha256: info.sha256.clone(),
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
            if let Some(a) = studio.active.lock().unwrap().as_mut().filter(|a| a.meta.id == meta.id) {
                a.meta = meta.clone();
            }
            Ok(ReadFinished {
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
        let finished = run().unwrap_or_else(|e| ReadFinished {
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
            serde_json::json!({"type": "reading_finished", "time": now(), "accepted": finished.accepted,
                "tokens": finished.tokens, "stopped": finished.stopped, "error": finished.error}),
        );
        studio.teaching.store(false, Ordering::Release);
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

    /// Checks the real TinyStories download (network; run in CI with `--ignored`).
    #[test]
    #[ignore]
    fn tinystories_is_reachable() {
        let src = &sources()[0];
        let (bytes, truncated) =
            fetch(&agent(), src.url, 1 << 20, &AtomicBool::new(false), &mut |_, _| {}).unwrap();
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
