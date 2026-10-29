//! Importing datasets by hand (0.8.8): from a Hugging Face dataset (its link or `owner/name`, at a
//! chosen revision), a file's link, or files on this device (also inside a ZIP). Formats: Parquet,
//! JSON, JSON lines, CSV / TSV and plain text, each also gzip-compressed (`.gz`).
//!
//! 1. **Inspect**: the files (with their split, format and size), the dataset's licence, and
//!    Hugging Face's Parquet conversion when it has one. Python loading scripts are listed and
//!    never run: Studio does not execute downloaded code.
//! 2. **Sample**: the chosen files' columns (with their kinds and an example) and a few rows, and
//!    a suggested mapping.
//! 3. **Map and preview**: what becomes a document: text columns for reading; for conversations a
//!    list of role/content messages, a prompt and a response column, a list of turns, or one row
//!    per turn grouped by a conversation column; roles can be renamed or left out. The preview
//!    shows the first converted documents and what the check left out.
//! 4. **Import**: every record is converted and checked (missing fields, unmapped roles, empty
//!    messages, malformed records, broken turn order, more than two speakers, oversized
//!    documents, duplicates); the report and the source (repository, revision, files with their
//!    SHA-256, mapping) are kept with the dataset. Train, validation and test files are imported
//!    separately, never mixed.
//!
//! Large files are read record by record (a JSON array file is read whole, so it is limited to
//! 512 MB); ZIP entries are extracted only by their own safe names (no `..` or absolute paths),
//! at most 2,000 entries and 4 GB, and an entry claiming to be much smaller than it unpacks to is
//! refused.
use super::*;
use crate::reading::{CorpusInfo, MAX_CORPUS_BYTES};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use yendory_engine::learn::corpus::{Category, END_OF_TEXT};

/// A file larger than this is not imported.
const MAX_FILE: u64 = 4 << 30;
/// A JSON file (an array, read whole) larger than this is refused: JSON lines read record by record.
const MAX_JSON: u64 = 512 << 20;
const ZIP_ENTRIES: usize = 2000;
const ZIP_TOTAL: u64 = 4 << 30;
/// An entry unpacking to more than this many times its packed size is refused (a "zip bomb").
const ZIP_RATIO: u64 = 200;
/// A document longer than this many characters is left out ("oversized").
const MAX_DOC: usize = 20_000;
/// Records the preview reads.
const PREVIEW_RECORDS: usize = 2000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Parquet,
    Json,
    Jsonl,
    Csv,
    Tsv,
    Text,
    Zip,
    /// A Python loading script: listed, never run.
    Script,
    Other,
}

/// The format a file name says, and whether it is gzip-compressed.
pub fn format_of(name: &str) -> (Format, bool) {
    let lower = name.to_ascii_lowercase();
    let (base, gz) = match lower.strip_suffix(".gz") {
        Some(b) => (b.to_string(), true),
        None => (lower, false),
    };
    let f = match base.rsplit('.').next().unwrap_or("") {
        "parquet" => Format::Parquet,
        "json" => Format::Json,
        "jsonl" | "ndjson" => Format::Jsonl,
        "csv" => Format::Csv,
        "tsv" => Format::Tsv,
        "txt" | "text" | "md" => Format::Text,
        "zip" if !gz => Format::Zip,
        "py" if !gz => Format::Script,
        _ => Format::Other,
    };
    (f, gz && !matches!(f, Format::Other | Format::Parquet))
}

/// train / validation / test, from a file's path (None: not said).
pub fn split_of(path: &str) -> Option<String> {
    let lower = path.to_ascii_lowercase();
    let parts: Vec<&str> = lower.split(['/', '\\', '-', '_', '.']).collect();
    let has = |w: &[&str]| parts.iter().any(|p| w.contains(p));
    if has(&["test", "testing"]) {
        Some("test".into())
    } else if has(&["validation", "valid", "val", "dev", "eval"]) {
        Some("validation".into())
    } else if has(&["train", "training"]) {
        Some("train".into())
    } else {
        None
    }
}

/// One file of a source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportFile {
    /// Its name within the source (path in the repository, file name, or ZIP entry).
    pub name: String,
    /// Where it is: an https link or a path on this device.
    pub location: String,
    /// The entry inside the ZIP at `location`.
    #[serde(default)]
    pub zip_entry: Option<String>,
    pub size: u64,
    pub format: Format,
    #[serde(default)]
    pub gz: bool,
    pub split: Option<String>,
    /// Hugging Face's Parquet conversion (config of the dataset).
    #[serde(default)]
    pub converted: Option<String>,
}

impl ImportFile {
    fn supported(&self) -> bool {
        !matches!(self.format, Format::Script | Format::Other)
    }
}

/// What a source holds.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Inspection {
    /// "huggingface", "url" or "local".
    pub kind: String,
    pub source: String,
    pub name: String,
    pub revision: Option<String>,
    pub license: Option<String>,
    pub files: Vec<ImportFile>,
    pub notes: Vec<String>,
}

fn hf_get(agent: &ureq::Agent, url: &str) -> Res<ureq::Response> {
    let mut r = agent.get(url);
    if let Some(t) = crate::reading::hf_token() {
        r = r.set("Authorization", &format!("Bearer {}", t.trim()));
    }
    r.call().map_err(|e| match e {
        ureq::Error::Status(401, _) => format!(
            "Hugging Face answered 401 for {url}: the dataset is private or gated (accept its conditions on its page while signed in, and connect your account on § 4)"
        ),
        ureq::Error::Status(403, _) => format!("Hugging Face refused access (403) for {url}: accept the dataset's conditions on its page while signed in"),
        ureq::Error::Status(404, _) => format!("Not found (404): {url}"),
        e => format!("{url}: {e}"),
    })
}

/// `owner/name` and the revision from a Hugging Face dataset link or id.
pub fn parse_hub(source: &str, revision: Option<&str>) -> Option<(String, String)> {
    let s = source.trim().trim_end_matches('/');
    let rest = s
        .strip_prefix("https://huggingface.co/datasets/")
        .or_else(|| s.strip_prefix("huggingface.co/datasets/"))
        .or_else(|| s.strip_prefix("hf://datasets/"))
        .unwrap_or(s);
    let parts: Vec<&str> = rest.split('/').collect();
    if parts.len() < 2 {
        return None;
    }
    let ok = |p: &str| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c));
    if !ok(parts[0]) || !ok(parts[1]) {
        return None;
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    // .../tree/<rev> or .../blob/<rev>/...
    let in_link =
        (parts.len() >= 4 && matches!(parts[2], "tree" | "blob" | "resolve")).then(|| parts[3].to_string());
    let rev = revision
        .map(str::trim)
        .filter(|r| !r.is_empty())
        .map(str::to_string)
        .or(in_link)
        .unwrap_or_else(|| "main".into());
    if !rev.chars().all(|c| c.is_ascii_alphanumeric() || "-_.%".contains(c)) {
        return None;
    }
    Some((repo, rev))
}

fn inspect_hub(agent: &ureq::Agent, repo: &str, rev: &str) -> Res<Inspection> {
    let mut notes = Vec::new();
    let info: Value = hf_get(agent, &format!("https://huggingface.co/api/datasets/{repo}/revision/{rev}"))
        .or_else(|_| hf_get(agent, &format!("https://huggingface.co/api/datasets/{repo}")))?
        .into_string()
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    let license = info
        .get("cardData")
        .and_then(|c| c.get("license"))
        .map(|l| match l {
            Value::Array(a) => a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", "),
            v => v.as_str().unwrap_or("").to_string(),
        })
        .filter(|l| !l.is_empty());
    if info.get("gated").is_some_and(|g| g.as_bool() != Some(false)) {
        notes.push("This dataset is gated: accept its conditions on its Hugging Face page while signed in, and connect your account (§ 4).".into());
    }
    // Every file of the revision (the listing comes in pages).
    let mut files = Vec::new();
    let mut url =
        Some(format!("https://huggingface.co/api/datasets/{repo}/tree/{rev}?recursive=true&expand=false"));
    let mut pages = 0;
    while let Some(u) = url.take() {
        pages += 1;
        let r = hf_get(agent, &u)?;
        let next = r
            .header("Link")
            .and_then(|l| l.split(',').find(|p| p.contains("rel=\"next\"")))
            .and_then(|p| p.split(['<', '>']).nth(1))
            .map(str::to_string);
        let list: Vec<Value> =
            r.into_string().ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        for f in list.iter().filter(|f| f.get("type").and_then(Value::as_str) == Some("file")) {
            let Some(path) = f.get("path").and_then(Value::as_str) else { continue };
            let (format, gz) = format_of(path);
            if path.starts_with('.') || path.eq_ignore_ascii_case("README.md") {
                continue;
            }
            files.push(ImportFile {
                name: path.into(),
                location: format!("https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"),
                zip_entry: None,
                size: f.get("size").and_then(Value::as_u64).unwrap_or(0),
                format,
                gz,
                split: split_of(path),
                converted: None,
            });
        }
        if pages < 20 {
            url = next;
        }
    }
    if let Some(script) = files.iter().find(|f| f.format == Format::Script).cloned() {
        notes.push(format!(
            "{} is a Python loading script. Studio never runs downloaded code; the data files it names are offered below when they can be downloaded, and Hugging Face's Parquet conversion when there is one.",
            script.name
        ));
        // The script is read as text (never run) for the links to its data files.
        if let Ok(text) =
            hf_get(agent, &script.location).and_then(|r| r.into_string().map_err(|e| e.to_string()))
        {
            let (found, plain) = script_links(&text);
            for url in found {
                let name = url.rsplit('/').next().unwrap_or("data").to_string();
                let (format, gz) = format_of(&name);
                files.push(ImportFile {
                    name: format!("Named by {}: {url}", script.name),
                    location: url.clone(),
                    zip_entry: None,
                    size: 0,
                    format,
                    gz,
                    split: split_of(&name),
                    converted: None,
                });
            }
            for url in plain {
                notes.push(format!("The script names {url}, which is not https: Studio does not download it (an unencrypted link can be altered on the way)."));
            }
        }
    }
    // Hugging Face's own conversion to Parquet (also for datasets built by a script).
    let conv = crate::reading::get_json(
        agent,
        &format!("https://datasets-server.huggingface.co/parquet?dataset={repo}"),
        &AtomicBool::new(false),
        &mut crate::reading::Pacer::new(),
        &mut |_| {},
    )
    .ok();
    for f in
        conv.as_ref().and_then(|c| c.get("parquet_files")).and_then(Value::as_array).into_iter().flatten()
    {
        let (Some(url), Some(config), Some(split), Some(name)) = (
            f.get("url").and_then(Value::as_str),
            f.get("config").and_then(Value::as_str),
            f.get("split").and_then(Value::as_str),
            f.get("filename").and_then(Value::as_str),
        ) else {
            continue;
        };
        if !url.starts_with("https://huggingface.co/") {
            continue;
        }
        files.push(ImportFile {
            name: format!("Parquet conversion: {config}/{split}/{name}"),
            location: url.into(),
            zip_entry: None,
            size: f.get("size").and_then(Value::as_u64).unwrap_or(0),
            format: Format::Parquet,
            gz: false,
            split: Some(split.to_string()),
            converted: Some(config.to_string()),
        });
    }
    if files.iter().all(|f| !f.supported()) {
        notes.push(
            "No file Studio can read (Parquet, JSON, JSON lines, CSV, TSV, text, ZIP) was found.".into(),
        );
    }
    Ok(Inspection {
        kind: "huggingface".into(),
        source: format!("https://huggingface.co/datasets/{repo}"),
        name: repo.rsplit('/').next().unwrap_or(repo).into(),
        revision: Some(rev.into()),
        license,
        files,
        notes,
    })
}

/// Links to data files in a loading script's text (https ones, and the others apart).
pub fn script_links(text: &str) -> (Vec<String>, Vec<String>) {
    let mut https = Vec::new();
    let mut other = Vec::new();
    for start in text.match_indices("http").map(|(i, _)| i) {
        let url: String =
            text[start..].chars().take_while(|c| !c.is_whitespace() && !"\"'`)]},".contains(*c)).collect();
        let (format, _) = format_of(url.split('?').next().unwrap_or(&url));
        if matches!(format, Format::Other | Format::Script) || url.contains("{") {
            continue;
        }
        let list = if url.starts_with("https://") {
            &mut https
        } else if url.starts_with("http://") {
            &mut other
        } else {
            continue;
        };
        if !list.contains(&url) {
            list.push(url);
        }
    }
    (https, other)
}

/// The entries of a ZIP that can be imported (checked: safe names, counts and sizes).
pub fn zip_entries(path: &Path) -> Res<Vec<(String, u64)>> {
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| format!("Not a readable ZIP: {e}"))?;
    if z.len() > ZIP_ENTRIES {
        return Err(format!("The ZIP has {} entries; at most {ZIP_ENTRIES} are read", z.len()));
    }
    let mut out = Vec::new();
    let mut total = 0u64;
    for i in 0..z.len() {
        let e = z.by_index(i).map_err(|e| e.to_string())?;
        if e.is_dir() {
            continue;
        }
        let Some(name) = e.enclosed_name().map(|p| p.to_string_lossy().replace('\\', "/")) else {
            return Err(format!(
                "The ZIP has an entry with an unsafe path ({}); nothing was extracted",
                e.name()
            ));
        };
        total += e.size();
        if total > ZIP_TOTAL {
            return Err("The ZIP unpacks to more than 4 GB".into());
        }
        if e.compressed_size() > 0 && e.size() / e.compressed_size().max(1) > ZIP_RATIO {
            return Err(format!("{name} unpacks to over {ZIP_RATIO} times its packed size; refused"));
        }
        out.push((name, e.size()));
    }
    Ok(out)
}

/// Extracts one entry (by its safe name) into `dir`, reading no more than it declares.
fn extract(zip_path: &Path, entry: &str, dir: &Path) -> Res<PathBuf> {
    let f = std::fs::File::open(zip_path).map_err(|e| e.to_string())?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| e.to_string())?;
    let mut e = z.by_name(entry).map_err(|e| format!("{entry}: {e}"))?;
    if e.enclosed_name().is_none() {
        return Err(format!("{entry}: unsafe path"));
    }
    let declared = e.size();
    let safe: String =
        entry.chars().map(|c| if c.is_ascii_alphanumeric() || ".-_".contains(c) { c } else { '_' }).collect();
    let out = dir.join(format!("zip-{safe}"));
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut w = std::fs::File::create(&out).map_err(|e| e.to_string())?;
    let copied = std::io::copy(&mut (&mut e).take(declared + 1), &mut w).map_err(|e| e.to_string())?;
    if copied > declared {
        let _ = std::fs::remove_file(&out);
        return Err(format!("{entry} unpacks to more than it declares; refused"));
    }
    Ok(out)
}

fn cache_dir() -> PathBuf {
    std::env::temp_dir().join("yendory-import")
}

fn cache_path(location: &str) -> PathBuf {
    let h = sha256_hex(location.as_bytes());
    let name: String = location
        .rsplit('/')
        .next()
        .unwrap_or("file")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || ".-_".contains(c) { c } else { '_' })
        .take(80)
        .collect();
    cache_dir().join(&h[..16]).join(name)
}

/// The file on this device (downloaded once to a cache, resumable; a ZIP entry extracted).
fn materialize(
    agent: &ureq::Agent,
    file: &ImportFile,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, u64),
) -> Res<PathBuf> {
    let base = if file.location.starts_with("https://") {
        let path = cache_path(&file.location);
        let size = if file.zip_entry.is_some() { 0 } else { file.size };
        if !(path.exists() && (size == 0 || std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) == size)) {
            crate::hfparquet::download_to(agent, &file.location, &path, size, cancel, &|| false, &mut |b| {
                progress(b, size)
            })?;
        }
        path
    } else if file.location.starts_with("http://") {
        return Err("Only https links are downloaded".into());
    } else {
        PathBuf::from(&file.location)
    };
    let len = std::fs::metadata(&base).map(|m| m.len()).map_err(|e| format!("{}: {e}", file.name))?;
    if len > MAX_FILE {
        return Err(format!("{} is larger than 4 GB", file.name));
    }
    match &file.zip_entry {
        Some(entry) => {
            zip_entries(&base)?;
            extract(&base, entry, &cache_dir().join(&sha256_hex(file.location.as_bytes())[..16]))
        }
        None => Ok(base),
    }
}

/// Calls `each` with every record of a file (Err: a record that could not be read) until it
/// returns false.
pub fn for_each_record(
    file: &ImportFile,
    path: &Path,
    each: &mut dyn FnMut(Result<Value, String>) -> bool,
) -> Res<()> {
    let open = || -> Res<Box<dyn std::io::Read>> {
        let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
        Ok(if file.gz { Box::new(flate2::read::MultiGzDecoder::new(f)) } else { Box::new(f) })
    };
    match file.format {
        Format::Parquet => {
            let mut go = true;
            crate::hfparquet::for_each_row(path, &mut |r| {
                go = each(Ok(r.clone()));
                go
            })?;
        }
        Format::Jsonl => {
            for line in std::io::BufReader::new(open()?).split(b'\n') {
                let line = line.map_err(|e| e.to_string())?;
                let text = String::from_utf8_lossy(&line);
                if text.trim().is_empty() {
                    continue;
                }
                let rec = match std::str::from_utf8(&line) {
                    Ok(t) => serde_json::from_str(t.trim()).map_err(|e| format!("a line is not JSON ({e})")),
                    Err(_) => Err("a line is not UTF-8 text".into()),
                };
                if !each(rec) {
                    break;
                }
            }
        }
        Format::Json => {
            if !file.gz && std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) > MAX_JSON {
                return Err(format!(
                    "{} is a JSON file over 512 MB; JSON lines (one record per line) can be any size",
                    file.name
                ));
            }
            let mut text = String::new();
            open()?
                .take(MAX_JSON + 1)
                .read_to_string(&mut text)
                .map_err(|e| format!("{}: {e}", file.name))?;
            let v: Value =
                serde_json::from_str(&text).map_err(|e| format!("{} is not JSON: {e}", file.name))?;
            let list = match v {
                Value::Array(a) => a,
                Value::Object(o) => {
                    // The first list of records in it ("data", "rows", "train", ...), else itself.
                    match o.iter().find_map(|(_, v)| {
                        v.as_array().filter(|a| a.first().is_some_and(Value::is_object)).cloned()
                    }) {
                        Some(a) => a,
                        None => vec![Value::Object(o)],
                    }
                }
                other => vec![other],
            };
            for r in list {
                if !each(Ok(r)) {
                    break;
                }
            }
        }
        Format::Csv | Format::Tsv => {
            let mut r = csv::ReaderBuilder::new()
                .delimiter(if file.format == Format::Tsv { b'\t' } else { b',' })
                .flexible(true)
                .from_reader(open()?);
            let headers: Vec<String> =
                r.headers().map_err(|e| e.to_string())?.iter().map(str::to_string).collect();
            for rec in r.records() {
                let v = rec.map_err(|e| format!("a row cannot be read ({e})")).map(|rec| {
                    Value::Object(
                        headers
                            .iter()
                            .cloned()
                            .zip(rec.iter().map(|x| Value::String(x.to_string())))
                            .collect(),
                    )
                });
                if !each(v) {
                    break;
                }
            }
        }
        Format::Text => {
            let mut bytes = Vec::new();
            open()?.take(MAX_FILE).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
            let text = String::from_utf8_lossy(&bytes);
            // Documents end at <|endoftext|> when the file uses it, else at a blank line.
            let docs: Vec<&str> = if text.contains(END_OF_TEXT) {
                text.split(END_OF_TEXT).collect()
            } else {
                text.split("\n\n").collect()
            };
            for d in docs.into_iter().map(str::trim).filter(|d| !d.is_empty()) {
                if !each(Ok(serde_json::json!({ "text": d }))) {
                    break;
                }
            }
        }
        Format::Zip => return Err("A ZIP is opened by its entries".into()),
        Format::Script => {
            return Err(format!("{} is a Python script; Studio never runs downloaded code", file.name))
        }
        Format::Other => return Err(format!("{}: Studio cannot read this kind of file", file.name)),
    }
    Ok(())
}

/// A column of the records.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Column {
    pub name: String,
    /// text, number, true/false, list, list of records, record, empty.
    pub kind: String,
    pub example: String,
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "text",
        Value::Number(_) => "number",
        Value::Bool(_) => "true/false",
        Value::Array(a) if a.first().is_some_and(Value::is_object) => "list of records",
        Value::Array(_) => "list",
        Value::Object(_) => "record",
        Value::Null => "empty",
    }
}

fn short(v: &Value) -> String {
    let t = match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let t: String = t.chars().take(160).collect();
    t.replace('\n', " ⏎ ")
}

/// How records become documents.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "layout", rename_all = "snake_case", rename_all_fields = "camelCase")]
pub enum Layout {
    /// Reading: these columns' text (joined by a blank line). Conversation: a column holding a
    /// whole conversation as `Name: text` lines (or `<|im_start|>` turns).
    Text { columns: Vec<String> },
    /// A list of messages, each with a role and a text.
    Messages { column: String, role_field: String, content_field: String },
    /// A prompt and its response (two turns).
    Pair { prompt: String, response: String },
    /// A list of turns (the two speakers alternate).
    Turns { column: String },
    /// One record per turn, grouped by a conversation column (its records together, in order).
    Rows { conversation: String, speaker: String, text: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Mapping {
    /// What the dataset is: general, knowledge, reasoning (reading) or conversation.
    pub category: Category,
    #[serde(flatten)]
    pub layout: Layout,
    /// Role (as written in the data) → the speaker's name; an empty name leaves that role's
    /// turns out. Empty map: roles that are names are kept, others get names.
    #[serde(default)]
    pub roles: BTreeMap<String, String>,
    /// The two speakers' names for prompt/response and turn lists (empty: names are chosen).
    #[serde(default)]
    pub names: Vec<String>,
    /// A column with the scene (`Situation:`), when there is one.
    #[serde(default)]
    pub situation: Option<String>,
}

/// Why a record was left out at import.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ImportReport {
    pub examined: usize,
    pub accepted: usize,
    pub unreadable: usize,
    pub missing_field: usize,
    pub invalid_role: usize,
    pub empty_message: usize,
    pub malformed: usize,
    pub broken_order: usize,
    pub too_many_speakers: usize,
    pub oversized: usize,
    pub duplicates: usize,
    pub split_across: usize,
    /// Roles found (for the role mapping) and how often.
    pub roles: BTreeMap<String, usize>,
    /// Stopped at the size asked for.
    pub truncated: bool,
}

impl ImportReport {
    pub fn summary(&self) -> String {
        let parts: Vec<String> = [
            (self.unreadable, "unreadable"),
            (self.missing_field, "missing a field"),
            (self.invalid_role, "with a role not mapped"),
            (self.empty_message, "with an empty message"),
            (self.malformed, "malformed"),
            (self.broken_order, "with the same speaker twice in a row"),
            (self.too_many_speakers, "with more than two speakers"),
            (self.oversized, "over 20,000 characters"),
            (self.duplicates, "duplicates"),
            (self.split_across, "split across the file"),
        ]
        .iter()
        .filter(|(n, _)| *n > 0)
        .map(|(n, w)| format!("{n} {w}"))
        .collect();
        format!(
            "{} of {} records imported{}{}",
            self.accepted,
            self.examined,
            if parts.is_empty() { String::new() } else { format!(" (left out: {})", parts.join(", ")) },
            if self.truncated { "; stopped at the size asked for" } else { "" }
        )
    }
}

#[derive(Debug, PartialEq)]
enum Reject {
    Unreadable,
    Missing,
    Role,
    Empty,
    Malformed,
    Order,
    Speakers,
    Oversized,
    Duplicate,
    Split,
}

/// A conversation being gathered from one record per turn: its id, its turns (speaker, text) and
/// its scene.
type Gathering = (String, Vec<(Option<String>, String)>, Option<String>);

/// Turns records into documents, checking each.
pub struct Converter<'a> {
    m: &'a Mapping,
    pub report: ImportReport,
    seen: HashSet<u64>,
    /// Layout::Rows: the conversation being gathered, and the ones finished.
    open: Option<Gathering>,
    closed: HashSet<String>,
}

fn text_of(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

impl<'a> Converter<'a> {
    pub fn new(m: &'a Mapping) -> Self {
        Converter {
            m,
            report: ImportReport::default(),
            seen: HashSet::new(),
            open: None,
            closed: HashSet::new(),
        }
    }

    fn reject(&mut self, r: Reject) {
        let rep = &mut self.report;
        *match r {
            Reject::Unreadable => &mut rep.unreadable,
            Reject::Missing => &mut rep.missing_field,
            Reject::Role => &mut rep.invalid_role,
            Reject::Empty => &mut rep.empty_message,
            Reject::Malformed => &mut rep.malformed,
            Reject::Order => &mut rep.broken_order,
            Reject::Speakers => &mut rep.too_many_speakers,
            Reject::Oversized => &mut rep.oversized,
            Reject::Duplicate => &mut rep.duplicates,
            Reject::Split => &mut rep.split_across,
        } += 1;
    }

    /// Checks a finished document (size, duplicates) and hands it on.
    fn emit(&mut self, doc: String, out: &mut dyn FnMut(String)) {
        if doc.chars().count() > MAX_DOC {
            return self.reject(Reject::Oversized);
        }
        let key: String = doc.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&key, &mut h);
        if !self.seen.insert(std::hash::Hasher::finish(&h)) {
            return self.reject(Reject::Duplicate);
        }
        self.report.accepted += 1;
        out(doc);
    }

    /// A role → its speaker name (None: the turn is left out; Err: the role is not mapped).
    fn speaker(&mut self, role: Option<String>) -> Result<Option<Option<String>>, Reject> {
        let Some(role) = role else { return Ok(Some(None)) };
        *self.report.roles.entry(role.clone()).or_default() += 1;
        if self.m.roles.is_empty() {
            return Ok((!role.eq_ignore_ascii_case("system")).then_some(Some(role)));
        }
        match self.m.roles.get(&role) {
            Some(name) if name.trim().is_empty() => Ok(None),
            Some(name) => Ok(Some(Some(name.trim().to_string()))),
            None => Err(Reject::Role),
        }
    }

    /// A conversation's turns as a document.
    fn conversation(
        &mut self,
        turns: Vec<(Option<String>, String)>,
        situation: Option<String>,
        out: &mut dyn FnMut(String),
    ) {
        if turns.iter().any(|(_, t)| t.is_empty()) {
            return self.reject(Reject::Empty);
        }
        if turns.len() < 2 {
            return self.reject(Reject::Malformed);
        }
        if turns.iter().all(|(r, _)| r.is_some()) {
            let labels: HashSet<&String> = turns.iter().filter_map(|(r, _)| r.as_ref()).collect();
            if labels.len() > 2 {
                return self.reject(Reject::Speakers);
            }
            if turns.windows(2).any(|w| w[0].0 == w[1].0) {
                return self.reject(Reject::Order);
            }
        }
        // Unlabelled turns alternate; named ones keep their names.
        let names = &self.m.names;
        let turns: Vec<(Option<String>, String)> = turns
            .into_iter()
            .enumerate()
            .map(|(i, (r, t))| (r.or_else(|| names.get(i % 2).filter(|n| !n.trim().is_empty()).cloned()), t))
            .collect();
        match crate::sources::dialogue_doc(&turns, situation.as_deref()) {
            Some(doc) => self.emit(doc, out),
            None => self.reject(Reject::Speakers),
        }
    }

    /// One record (Err: it could not be read).
    pub fn feed(&mut self, rec: Result<Value, String>, out: &mut dyn FnMut(String)) {
        let Ok(rec) = rec else {
            self.report.examined += 1;
            return self.reject(Reject::Unreadable);
        };
        let situation = self.m.situation.as_ref().and_then(|c| text_of(rec.get(c))).filter(|s| !s.is_empty());
        let layout = self.m.layout.clone();
        if let Layout::Rows { conversation, speaker, text } = &layout {
            return self.feed_row(&rec, conversation, speaker, text, situation, out);
        }
        self.report.examined += 1;
        match &layout {
            Layout::Text { columns } => {
                let mut parts = Vec::new();
                for c in columns {
                    match rec.get(c) {
                        None => return self.reject(Reject::Missing),
                        Some(v) => match text_of(Some(v)) {
                            Some(t) if !t.is_empty() => parts.push(t),
                            Some(_) | None if v.is_null() || v.as_str().is_some() => {}
                            _ => return self.reject(Reject::Malformed),
                        },
                    }
                }
                if parts.is_empty() {
                    return self.reject(Reject::Empty);
                }
                let text = parts.join("\n\n");
                if self.m.category == Category::Conversation {
                    match crate::sources::whole_dialogue(
                        &serde_json::json!({ "text": text, "situation": situation }),
                    )
                    .into_iter()
                    .next()
                    .or_else(|| {
                        yendory_engine::learn::conversation::parse_dialogue(&text).map(|_| text.clone())
                    }) {
                        Some(doc) => self.emit(doc, out),
                        None => self.reject(Reject::Malformed),
                    }
                } else {
                    self.emit(text, out)
                }
            }
            Layout::Messages { column, role_field, content_field } => {
                let Some(list) = rec.get(column) else { return self.reject(Reject::Missing) };
                let Some(list) = list.as_array() else { return self.reject(Reject::Malformed) };
                let mut turns = Vec::new();
                for m in list {
                    let Some(content) = m.get(content_field) else { return self.reject(Reject::Missing) };
                    let Some(content) = text_of(Some(content)) else { return self.reject(Reject::Malformed) };
                    let role = text_of(m.get(role_field));
                    if role.is_none() && m.get(role_field).is_some() {
                        return self.reject(Reject::Malformed);
                    }
                    match self.speaker(role) {
                        Ok(Some(who)) => turns.push((who, content)),
                        Ok(None) => {}
                        Err(r) => return self.reject(r),
                    }
                }
                self.conversation(turns, situation, out)
            }
            Layout::Pair { prompt, response } => {
                let (Some(p), Some(r)) = (rec.get(prompt), rec.get(response)) else {
                    return self.reject(Reject::Missing);
                };
                let (Some(p), Some(r)) = (text_of(Some(p)), text_of(Some(r))) else {
                    return self.reject(Reject::Malformed);
                };
                let names = &self.m.names;
                let name = |i: usize| names.get(i).filter(|n| !n.trim().is_empty()).cloned();
                self.conversation(vec![(name(0), p), (name(1), r)], situation, out)
            }
            Layout::Turns { column } => {
                let Some(list) = rec.get(column) else { return self.reject(Reject::Missing) };
                let Some(list) = list.as_array() else { return self.reject(Reject::Malformed) };
                let mut turns = Vec::new();
                for t in list {
                    match text_of(Some(t)) {
                        Some(t) => turns.push((None, t)),
                        None => return self.reject(Reject::Malformed),
                    }
                }
                self.conversation(turns, situation, out)
            }
            Layout::Rows { .. } => unreachable!(),
        }
    }

    fn feed_row(
        &mut self,
        rec: &Value,
        conversation: &str,
        speaker: &str,
        text: &str,
        situation: Option<String>,
        out: &mut dyn FnMut(String),
    ) {
        let (Some(id), Some(t)) = (text_of(rec.get(conversation)), text_of(rec.get(text))) else {
            self.report.examined += 1;
            return self.reject(Reject::Missing);
        };
        let who = match self.speaker(text_of(rec.get(speaker))) {
            Ok(w) => w,
            Err(r) => {
                // The whole conversation is left out.
                self.report.examined += 1;
                self.reject(r);
                if let Some((open_id, ..)) = &self.open {
                    if *open_id == id {
                        self.open = Some((id.clone(), vec![(None, String::new())], None));
                    }
                }
                return;
            }
        };
        if self.open.as_ref().is_some_and(|(o, ..)| *o != id) {
            self.flush(out);
        }
        if self.open.is_none() {
            if self.closed.contains(&id) {
                // Its records are not together: the earlier part was taken as a conversation.
                self.report.examined += 1;
                return self.reject(Reject::Split);
            }
            self.open = Some((id, Vec::new(), situation));
        }
        if let (Some(w), Some((_, turns, _))) = (who, self.open.as_mut()) {
            turns.push((w, t));
        }
    }

    fn flush(&mut self, out: &mut dyn FnMut(String)) {
        if let Some((id, turns, situation)) = self.open.take() {
            self.report.examined += 1;
            self.closed.insert(id);
            self.conversation(turns, situation, out);
        }
    }

    /// The end of the records.
    pub fn finish(&mut self, out: &mut dyn FnMut(String)) {
        self.flush(out);
    }
}

/// A few records of the chosen files, their columns, and a suggested mapping.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sample {
    pub columns: Vec<Column>,
    pub rows: Vec<Value>,
    pub suggested: Option<Mapping>,
    /// The files read (with ZIPs opened into their entries).
    pub files: Vec<ImportFile>,
}

/// A mapping that fits the columns (None: nothing obvious).
pub fn suggest(columns: &[Column], rows: &[Value]) -> Option<Mapping> {
    let has = |n: &str| columns.iter().find(|c| c.name.eq_ignore_ascii_case(n)).map(|c| c.name.clone());
    let situation = ["situation", "narrative", "context", "scenario"].iter().find_map(|n| has(n));
    let conv = |layout| Mapping {
        category: Category::Conversation,
        layout,
        roles: BTreeMap::new(),
        names: Vec::new(),
        situation: situation.clone(),
    };
    // A list of messages with a role and a text.
    for c in columns.iter().filter(|c| c.kind == "list of records") {
        let first =
            rows.iter().find_map(|r| r.get(&c.name)?.as_array()?.first().cloned()).unwrap_or_default();
        let pick = |keys: &[&str]| keys.iter().find(|k| first.get(**k).is_some()).map(|k| k.to_string());
        if let (Some(role), Some(content)) = (
            pick(&["role", "from", "speaker", "author"]),
            pick(&["content", "value", "text", "utterance", "message"]),
        ) {
            return Some(conv(Layout::Messages {
                column: c.name.clone(),
                role_field: role,
                content_field: content,
            }));
        }
    }
    // A list of turns: one named like it first (a "speakers" list is not the conversation).
    let named = ["dialogue", "dialog", "turns", "utterances", "conversation", "conversations"];
    if let Some(c) = columns.iter().filter(|c| c.kind == "list").min_by_key(|c| {
        (
            !named.contains(&c.name.to_ascii_lowercase().as_str()),
            c.name.to_ascii_lowercase().contains("speaker"),
        )
    }) {
        return Some(conv(Layout::Turns { column: c.name.clone() }));
    }
    for (p, r) in [
        ("prompt", "response"),
        ("prompt", "completion"),
        ("instruction", "output"),
        ("question", "answer"),
        ("input", "output"),
        ("user", "assistant"),
    ] {
        if let (Some(p), Some(r)) = (has(p), has(r)) {
            return Some(conv(Layout::Pair { prompt: p, response: r }));
        }
    }
    if let (Some(id), Some(sp), Some(t)) = (
        has("conversation_id").or(has("dialogue_id")).or(has("dialog_id")),
        has("speaker").or(has("role")),
        has("text").or(has("utterance")).or(has("content")),
    ) {
        return Some(conv(Layout::Rows { conversation: id, speaker: sp, text: t }));
    }
    let text = has("text").or_else(|| {
        columns.iter().filter(|c| c.kind == "text").max_by_key(|c| c.example.len()).map(|c| c.name.clone())
    })?;
    Some(Mapping {
        category: Category::General,
        layout: Layout::Text { columns: vec![text] },
        roles: BTreeMap::new(),
        names: Vec::new(),
        situation: None,
    })
}

/// ZIPs opened into their importable entries (each keeps the ZIP's split unless its own path says).
fn expand(
    agent: &ureq::Agent,
    files: &[ImportFile],
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, u64),
) -> Res<Vec<ImportFile>> {
    let mut out = Vec::new();
    for f in files {
        if f.format != Format::Zip || f.zip_entry.is_some() {
            out.push(f.clone());
            continue;
        }
        let path = materialize(agent, f, cancel, progress)?;
        for (entry, size) in zip_entries(&path)? {
            let (format, gz) = format_of(&entry);
            if matches!(format, Format::Other | Format::Script | Format::Zip) {
                continue;
            }
            out.push(ImportFile {
                name: format!("{} › {entry}", f.name),
                location: f.location.clone(),
                zip_entry: Some(entry.clone()),
                size,
                format,
                gz,
                split: split_of(&entry).or_else(|| f.split.clone()),
                converted: None,
            });
        }
    }
    Ok(out)
}

fn emit_progress(app: &AppHandle, what: &str, done: u64, total: u64) {
    let _ = app.emit("import-progress", serde_json::json!({ "what": what, "done": done, "total": total }));
}

static CANCEL: AtomicBool = AtomicBool::new(false);

fn one_split(files: &[ImportFile]) -> Res<()> {
    let splits: HashSet<Option<&String>> = files.iter().map(|f| f.split.as_ref()).collect();
    if splits.len() > 1 {
        return Err("These files are from different splits (train, validation, test): import each split as its own dataset, so they are never mixed".into());
    }
    if files.iter().any(|f| !f.supported()) {
        return Err(
            "Some chosen files cannot be read (a Python script or another kind of file); untick them".into(),
        );
    }
    Ok(())
}

// ---- commands -------------------------------------------------------------------------------

/// What a Hugging Face dataset (its link or `owner/name`) or a file's link holds.
#[tauri::command]
pub async fn import_inspect(source: String, revision: Option<String>) -> Res<Inspection> {
    tauri::async_runtime::spawn_blocking(move || {
        let agent = crate::reading::agent();
        if let Some((repo, rev)) = parse_hub(&source, revision.as_deref()) {
            if !source.trim().starts_with("https://") || source.contains("huggingface.co/datasets/") {
                return inspect_hub(&agent, &repo, &rev);
            }
        }
        let url = source.trim();
        if !url.starts_with("https://") {
            return Err(
                "Give a Hugging Face dataset (its link, or owner/name) or an https link to a file".into()
            );
        }
        let name = url.rsplit('/').next().unwrap_or("file").split('?').next().unwrap_or("file").to_string();
        let (format, gz) = format_of(&name);
        let size = agent
            .head(url)
            .call()
            .ok()
            .and_then(|r| r.header("Content-Length").and_then(|v| v.parse().ok()))
            .unwrap_or(0);
        Ok(Inspection {
            kind: "url".into(),
            source: url.into(),
            name: name.clone(),
            revision: None,
            license: None,
            files: vec![ImportFile {
                name: name.clone(),
                location: url.into(),
                zip_entry: None,
                size,
                format,
                gz,
                split: split_of(&name),
                converted: None,
            }],
            notes: if matches!(format, Format::Other) {
                vec![format!("{name}: Studio cannot tell its format from its name")]
            } else {
                Vec::new()
            },
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Files on this device (ZIPs are listed by their entries).
#[tauri::command]
pub async fn import_pick_local(app: AppHandle) -> Res<Option<Inspection>> {
    let Some(paths) = app
        .dialog()
        .file()
        .add_filter("Datasets", &["parquet", "json", "jsonl", "ndjson", "csv", "tsv", "txt", "zip", "gz"])
        .blocking_pick_files()
    else {
        return Ok(None);
    };
    let mut files = Vec::new();
    let mut notes = Vec::new();
    for p in paths {
        let name = crate::display_name(&p);
        // Desktop paths are read in place; elsewhere (Android) the file is copied to the cache.
        let path = match p.clone().into_path() {
            Ok(path) => path,
            Err(_) => {
                let bytes = crate::read_picked(&app, &p, MAX_FILE as usize)?;
                let dest = cache_path(&format!("local/{name}"));
                std::fs::create_dir_all(dest.parent().unwrap_or(Path::new(".")))
                    .map_err(|e| e.to_string())?;
                std::fs::write(&dest, bytes).map_err(|e| e.to_string())?;
                dest
            }
        };
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let (format, gz) = format_of(&name);
        let file = ImportFile {
            name: name.clone(),
            location: path.to_string_lossy().into(),
            zip_entry: None,
            size,
            format,
            gz,
            split: split_of(&name),
            converted: None,
        };
        if format == Format::Zip {
            match expand(&crate::reading::agent(), &[file], &AtomicBool::new(false), &mut |_, _| {}) {
                Ok(entries) => files.extend(entries),
                Err(e) => notes.push(format!("{name}: {e}")),
            }
        } else {
            files.push(file);
        }
    }
    let name = files.first().map(|f| f.name.clone()).unwrap_or_default();
    Ok(Some(Inspection {
        kind: "local".into(),
        source: "files on this device".into(),
        name,
        revision: None,
        license: None,
        files,
        notes,
    }))
}

/// The chosen files' columns, a few records, and a suggested mapping.
#[tauri::command]
pub async fn import_sample(app: AppHandle, files: Vec<ImportFile>) -> Res<Sample> {
    tauri::async_runtime::spawn_blocking(move || {
        CANCEL.store(false, Ordering::Release);
        let agent = crate::reading::agent();
        let files = expand(&agent, &files, &CANCEL, &mut |d, t| emit_progress(&app, "Downloading", d, t))?;
        one_split(&files)?;
        let mut rows = Vec::new();
        let mut seen: Vec<(String, Value)> = Vec::new();
        'files: for f in &files {
            let path = materialize(&agent, f, &CANCEL, &mut |d, t| {
                emit_progress(&app, &format!("Downloading {}", f.name), d, t)
            })?;
            let mut n = 0;
            for_each_record(f, &path, &mut |r| {
                if let Ok(Value::Object(o)) = &r {
                    for (k, v) in o {
                        match seen.iter_mut().find(|(c, _)| c == k) {
                            Some((_, old)) if old.is_null() => *old = v.clone(),
                            Some(_) => {}
                            None => seen.push((k.clone(), v.clone())),
                        }
                    }
                    if rows.len() < 5 {
                        rows.push(Value::Object(o.clone()));
                    }
                }
                n += 1;
                n < 200
            })?;
            if rows.len() >= 5 {
                break 'files;
            }
        }
        let columns: Vec<Column> = seen
            .iter()
            .map(|(k, v)| Column { name: k.clone(), kind: kind_of(v).into(), example: short(v) })
            .collect();
        let suggested = suggest(&columns, &rows);
        let rows = rows
            .into_iter()
            .map(|r| match r {
                Value::Object(o) => {
                    Value::Object(o.into_iter().map(|(k, v)| (k, Value::String(short(&v)))).collect())
                }
                v => v,
            })
            .collect();
        Ok(Sample { columns, rows, suggested, files })
    })
    .await
    .map_err(|e| e.to_string())?
}

/// The first documents the mapping makes, and the check's counts over the first records.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Preview {
    pub docs: Vec<String>,
    pub report: ImportReport,
    pub summary: String,
}

#[tauri::command]
pub async fn import_preview(app: AppHandle, files: Vec<ImportFile>, mapping: Mapping) -> Res<Preview> {
    tauri::async_runtime::spawn_blocking(move || {
        CANCEL.store(false, Ordering::Release);
        one_split(&files)?;
        let agent = crate::reading::agent();
        let mut conv = Converter::new(&mapping);
        let mut docs = Vec::new();
        let mut read = 0;
        for f in &files {
            let path = materialize(&agent, f, &CANCEL, &mut |d, t| {
                emit_progress(&app, &format!("Downloading {}", f.name), d, t)
            })?;
            for_each_record(f, &path, &mut |r| {
                conv.feed(r, &mut |d| docs.push(d));
                read += 1;
                read < PREVIEW_RECORDS
            })?;
            if read >= PREVIEW_RECORDS {
                break;
            }
        }
        conv.finish(&mut |d| docs.push(d));
        docs.truncate(5);
        let report = conv.report.clone();
        Ok(Preview {
            docs,
            summary: format!("First {} records: {}", report.examined, report.summary()),
            report,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Where an imported dataset came from, kept with it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportProvenance {
    /// "huggingface", "url" or "local".
    pub kind: String,
    pub source: String,
    pub revision: Option<String>,
    pub split: Option<String>,
    /// The original files: name, SHA-256 and size.
    pub files: Vec<(String, String, u64)>,
    pub imported: u64,
    pub mapping: Mapping,
    pub report: ImportReport,
    /// Conversation datasets: the quality filter's switches (§ 4) apply to it at practice.
    pub filtered_at_practice: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportRequest {
    pub kind: String,
    pub source: String,
    pub revision: Option<String>,
    pub files: Vec<ImportFile>,
    pub mapping: Mapping,
    pub name: String,
    pub license: String,
    pub filter_at_practice: bool,
    pub max_mb: u64,
}

/// Imports the chosen files with the mapping (in the background; progress as "import-progress").
#[tauri::command]
pub async fn import_run(app: AppHandle, request: ImportRequest) -> Res<CorpusInfo> {
    tauri::async_runtime::spawn_blocking(move || {
        CANCEL.store(false, Ordering::Release);
        let r = request;
        one_split(&r.files)?;
        let name = check_name(&r.name)?;
        if !(1..=400).contains(&r.max_mb) {
            return Err("Choose 1 to 400 MB to import".into());
        }
        let cap = (r.max_mb << 20).min(MAX_CORPUS_BYTES);
        let agent = crate::reading::agent();
        let mut conv = Converter::new(&r.mapping);
        let mut docs: Vec<String> = Vec::new();
        let mut bytes = 0u64;
        let mut originals = Vec::new();
        let mut full = false;
        for f in &r.files {
            if full {
                break;
            }
            let path = materialize(&agent, f, &CANCEL, &mut |d, t| {
                emit_progress(&app, &format!("Downloading {}", f.name), d, t)
            })?;
            let data = std::fs::read(&path).map_err(|e| e.to_string())?;
            originals.push((f.name.clone(), sha256_hex(&data), data.len() as u64));
            drop(data);
            let mut n = 0u64;
            for_each_record(f, &path, &mut |rec| {
                conv.feed(rec, &mut |d| {
                    bytes += d.len() as u64 + 16;
                    docs.push(d);
                });
                n += 1;
                if n.is_multiple_of(2000) {
                    emit_progress(
                        &app,
                        &format!("Converting {} ({} records)", f.name, n),
                        bytes.min(cap),
                        cap,
                    );
                }
                if bytes >= cap {
                    full = true;
                }
                !full && !CANCEL.load(Ordering::Acquire)
            })?;
            if CANCEL.load(Ordering::Acquire) {
                return Err("Import cancelled".into());
            }
        }
        conv.finish(&mut |d| docs.push(d));
        conv.report.truncated = full;
        if docs.is_empty() {
            return Err(format!("Nothing to import: {}", conv.report.summary()));
        }
        let category = r.mapping.category;
        // Conversations: the quality filter's counts, when its switches are to apply at practice.
        let filter = (category == Category::Conversation && r.filter_at_practice)
            .then(|| crate::filter::select(docs.iter().map(String::as_str), [], &Default::default(), None).1);
        let mut text = docs.join(&format!("\n{END_OF_TEXT}\n"));
        text.push('\n');
        let provenance = ImportProvenance {
            kind: r.kind.clone(),
            source: r.source.clone(),
            revision: r.revision.clone(),
            split: r.files.first().and_then(|f| f.split.clone()),
            files: originals,
            imported: now(),
            mapping: r.mapping.clone(),
            report: conv.report.clone(),
            filtered_at_practice: filter.is_some(),
        };
        let license = if r.license.trim().is_empty() {
            "as given by its source".to_string()
        } else {
            r.license.trim().to_string()
        };
        let info = crate::reading::store_imported(
            &app,
            &name,
            &r.source,
            &license,
            category,
            text.into_bytes(),
            full,
            filter,
            provenance,
        )?;
        let _ = std::fs::remove_dir_all(cache_dir());
        Ok(info)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Stops a download or an import being read.
#[tauri::command]
pub fn import_cancel() {
    CANCEL.store(true, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp() -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir().join(format!("yendory-imp-{nanos:x}"));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn file(path: &Path) -> ImportFile {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let (format, gz) = format_of(&name);
        ImportFile {
            name: name.clone(),
            location: path.to_string_lossy().into(),
            zip_entry: None,
            size: 0,
            format,
            gz,
            split: split_of(&name),
            converted: None,
        }
    }

    fn records(path: &Path) -> Vec<Result<Value, String>> {
        let mut out = Vec::new();
        for_each_record(&file(path), path, &mut |r| {
            out.push(r);
            true
        })
        .unwrap();
        out
    }

    fn convert(m: &Mapping, recs: Vec<Result<Value, String>>) -> (Vec<String>, ImportReport) {
        let mut c = Converter::new(m);
        let mut docs = Vec::new();
        for r in recs {
            c.feed(r, &mut |d| docs.push(d));
        }
        c.finish(&mut |d| docs.push(d));
        (docs, c.report)
    }

    #[test]
    fn names_say_format_split_and_hub_links() {
        assert_eq!(format_of("data/train-00000.parquet"), (Format::Parquet, false));
        assert_eq!(format_of("x.jsonl.gz"), (Format::Jsonl, true));
        assert_eq!(format_of("loader.py"), (Format::Script, false));
        assert_eq!(split_of("data/validation-0001.jsonl").as_deref(), Some("validation"));
        assert_eq!(split_of("test.csv").as_deref(), Some("test"));
        assert_eq!(split_of("conversations.json"), None);
        assert_eq!(
            parse_hub("lmsys/lmsys-chat-1m", None),
            Some(("lmsys/lmsys-chat-1m".into(), "main".into()))
        );
        assert_eq!(
            parse_hub("https://huggingface.co/datasets/allenai/soda/tree/refs%2Fconvert%2Fparquet", None)
                .map(|x| x.1),
            Some("refs%2Fconvert%2Fparquet".into())
        );
        assert_eq!(
            parse_hub("https://huggingface.co/datasets/a/b", Some("v1.0")).map(|x| x.1),
            Some("v1.0".into())
        );
        assert_eq!(parse_hub("not a dataset", None), None);
        assert_eq!(parse_hub("a/b/../../c", None).map(|x| x.0), Some("a/b".into()));
    }

    /// Every format reads to the same records; gzip is opened; a bad line is counted, not fatal.
    #[test]
    fn every_format_reads_records() {
        let d = temp();
        std::fs::write(
            d.join("a.jsonl"),
            "{\"q\":\"hi\",\"a\":\"hello\"}\nnot json\n\n{\"q\":\"bye\",\"a\":\"see you\"}\n",
        )
        .unwrap();
        std::fs::write(d.join("a.json"), r#"{"data": [{"q":"hi","a":"hello"},{"q":"bye","a":"see you"}]}"#)
            .unwrap();
        std::fs::write(d.join("a.csv"), "q,a\nhi,hello\n\"bye, then\",see you\n").unwrap();
        std::fs::write(d.join("a.tsv"), "q\ta\nhi\thello\n").unwrap();
        std::fs::write(d.join("a.txt"), "First story.\n\nSecond story.\n").unwrap();
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gz.write_all(b"{\"q\":\"hi\",\"a\":\"hello\"}\n").unwrap();
        std::fs::write(d.join("b.jsonl.gz"), gz.finish().unwrap()).unwrap();
        let jl = records(&d.join("a.jsonl"));
        assert_eq!(jl.len(), 3);
        assert!(jl[1].is_err());
        assert_eq!(records(&d.join("a.json")).len(), 2);
        let csv = records(&d.join("a.csv"));
        assert_eq!(csv[1].as_ref().unwrap()["q"], "bye, then");
        assert_eq!(records(&d.join("a.tsv"))[0].as_ref().unwrap()["a"], "hello");
        assert_eq!(records(&d.join("a.txt"))[1].as_ref().unwrap()["text"], "Second story.");
        assert_eq!(records(&d.join("b.jsonl.gz"))[0].as_ref().unwrap()["q"], "hi");
        let _ = std::fs::remove_dir_all(d);
    }

    /// Role/content messages, prompt/response pairs, turn lists and one-row-per-turn layouts all
    /// become `Name: text` conversations; roles can be renamed or left out; each problem is
    /// counted under its reason.
    #[test]
    fn layouts_become_conversations_and_problems_are_counted() {
        let msgs = |list: Value| Ok(serde_json::json!({ "messages": list }));
        let m = Mapping {
            category: Category::Conversation,
            layout: Layout::Messages {
                column: "messages".into(),
                role_field: "role".into(),
                content_field: "content".into(),
            },
            roles: [("user", "Mia"), ("assistant", "Leo"), ("system", "")]
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            names: vec![],
            situation: None,
        };
        let (docs, r) = convert(
            &m,
            vec![
                msgs(
                    serde_json::json!([{"role":"system","content":"Be nice."},{"role":"user","content":"Hi there!"},{"role":"assistant","content":"Hello, Mia."}]),
                ),
                msgs(
                    serde_json::json!([{"role":"user","content":"Hi there!"},{"role":"assistant","content":"Hello, Mia."}]),
                ),
                msgs(serde_json::json!([{"role":"user","content":"Hi"},{"role":"tool","content":"x"}])),
                msgs(serde_json::json!([{"role":"user","content":"Hi"},{"role":"assistant","content":""}])),
                msgs(serde_json::json!([{"role":"user","content":"One"},{"role":"user","content":"Two"}])),
                Ok(serde_json::json!({"other": 1})),
                msgs(serde_json::json!("not a list")),
                Err("bad line".into()),
                msgs(
                    serde_json::json!([{"role":"user","content":"x".repeat(20_001)},{"role":"assistant","content":"ok"}]),
                ),
            ],
        );
        assert_eq!(docs, ["Mia: Hi there!\nLeo: Hello, Mia."]);
        assert_eq!(
            (
                r.examined,
                r.accepted,
                r.duplicates,
                r.invalid_role,
                r.empty_message,
                r.broken_order,
                r.missing_field,
                r.malformed,
                r.unreadable,
                r.oversized
            ),
            (9, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        );
        assert_eq!(r.roles["tool"], 1);
        // A prompt and a response.
        let pair = Mapping {
            layout: Layout::Pair { prompt: "q".into(), response: "a".into() },
            roles: BTreeMap::new(),
            names: vec!["Ann".into(), "Ben".into()],
            ..m.clone()
        };
        let (docs, _) = convert(&pair, vec![Ok(serde_json::json!({"q":"How are you?","a":"Fine, thanks."}))]);
        assert_eq!(docs, ["Ann: How are you?\nBen: Fine, thanks."]);
        // A list of turns (names chosen when none are given).
        let turns = Mapping {
            layout: Layout::Turns { column: "dialog".into() },
            roles: BTreeMap::new(),
            names: vec![],
            ..m.clone()
        };
        let (docs, _) =
            convert(&turns, vec![Ok(serde_json::json!({"dialog":["Hi!","Hello!","How was it?"]}))]);
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].lines().count(), 3);
        // One row per turn, grouped; a conversation whose rows are not together is left out.
        let rows = Mapping {
            layout: Layout::Rows { conversation: "id".into(), speaker: "who".into(), text: "t".into() },
            roles: BTreeMap::new(),
            names: vec![],
            ..m
        };
        let row = |id: &str, who: &str, t: &str| Ok(serde_json::json!({"id": id, "who": who, "t": t}));
        let (docs, r) = convert(
            &rows,
            vec![
                row("1", "Ann", "Hi"),
                row("1", "Ben", "Hello"),
                row("2", "Ann", "Yo"),
                row("2", "Ben", "Hey"),
                row("1", "Ann", "Again"),
            ],
        );
        assert_eq!(docs, ["Ann: Hi\nBen: Hello", "Ann: Yo\nBen: Hey"]);
        assert_eq!(r.split_across, 1);
    }

    /// A ZIP is read by its entries; an entry with an unsafe path refuses the whole archive.
    #[test]
    fn zip_entries_are_safe_or_refused() {
        let d = temp();
        let make = |name: &str, entries: &[(&str, &[u8])]| {
            let p = d.join(name);
            let mut z = zip::ZipWriter::new(std::fs::File::create(&p).unwrap());
            for (n, b) in entries {
                z.start_file(*n, zip::write::SimpleFileOptions::default()).unwrap();
                z.write_all(b).unwrap();
            }
            z.finish().unwrap();
            p
        };
        let good = make(
            "good.zip",
            &[("data/train.jsonl", b"{\"text\":\"hello there\"}\n"), ("notes.py", b"print(1)")],
        );
        assert_eq!(zip_entries(&good).unwrap().len(), 2);
        let files = expand(&crate::reading::agent(), &[file(&good)], &AtomicBool::new(false), &mut |_, _| {})
            .unwrap();
        assert_eq!(files.len(), 1, "the script is not offered");
        assert_eq!(files[0].split.as_deref(), Some("train"));
        let path = materialize(&crate::reading::agent(), &files[0], &AtomicBool::new(false), &mut |_, _| {})
            .unwrap();
        assert!(path.starts_with(cache_dir()));
        assert_eq!(records(&path).len(), 1);
        let evil = make("evil.zip", &[("../../outside.txt", b"x")]);
        assert!(zip_entries(&evil).unwrap_err().contains("unsafe path"));
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn script_links_are_found_without_running_it() {
        let script = r#"
_URL = "https://example.org/data/dialogues.zip"
_TRAIN = "http://example.org/train.json"
_DOC = "https://example.org/about.html"
_FMT = "https://example.org/{split}.jsonl"
"#;
        let (https, plain) = script_links(script);
        assert_eq!(https, ["https://example.org/data/dialogues.zip"]);
        assert_eq!(plain, ["http://example.org/train.json"]);
    }

    #[test]
    fn mappings_are_suggested_from_columns() {
        let col = |n: &str, k: &str| Column { name: n.into(), kind: k.into(), example: String::new() };
        let rows = vec![serde_json::json!({"messages": [{"role": "user", "content": "hi"}]})];
        let m = suggest(&[col("messages", "list of records")], &rows).unwrap();
        assert!(matches!(m.layout, Layout::Messages { ref role_field, .. } if role_field == "role"));
        let m = suggest(&[col("instruction", "text"), col("output", "text")], &[]).unwrap();
        assert!(matches!(m.layout, Layout::Pair { .. }));
        let m = suggest(&[col("text", "text")], &[]).unwrap();
        assert_eq!(m.category, Category::General);
    }

    /// Real Hugging Face datasets (network; `--ignored`): files and the Parquet conversion are
    /// listed, a loading script is noted (not run), and records map to conversations.
    #[test]
    #[ignore]
    fn hub_datasets_are_inspected_and_mapped() {
        let agent = crate::reading::agent();
        for repo in ["HuggingFaceH4/ultrachat_200k", "li2017dailydialog/daily_dialog", "allenai/soda"] {
            let i = inspect_hub(&agent, repo, "main").unwrap();
            println!("{repo}: licence {:?}, {} files, notes {:?}", i.license, i.files.len(), i.notes);
            for f in i.files.iter().take(6) {
                println!("  {} {:?} {:?} {} MB", f.name, f.format, f.split, f.size >> 20);
            }
        }
        // A dataset built by a script: noted, never run; its http link is named but not offered.
        let i = inspect_hub(&agent, "li2017dailydialog/daily_dialog", "main").unwrap();
        assert!(i.notes.iter().any(|n| n.contains("never runs downloaded code")), "{:?}", i.notes);
        assert!(i.notes.iter().any(|n| n.contains("not https")), "{:?}", i.notes);
        // A small file read and mapped (SODA's test split, Hugging Face's Parquet conversion).
        let soda = inspect_hub(&agent, "allenai/soda", "main").unwrap();
        let f = soda
            .files
            .iter()
            .find(|f| f.converted.is_some() && f.split.as_deref() == Some("test"))
            .unwrap()
            .clone();
        let path = materialize(&agent, &f, &AtomicBool::new(false), &mut |_, _| {}).unwrap();
        let mut rows = Vec::new();
        for_each_record(&f, &path, &mut |r| {
            rows.push(r.unwrap());
            rows.len() < 50
        })
        .unwrap();
        let cols: Vec<Column> = rows[0]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| Column { name: k.clone(), kind: kind_of(v).into(), example: short(v) })
            .collect();
        println!("columns: {:?}", cols.iter().map(|c| (&c.name, &c.kind)).collect::<Vec<_>>());
        let m = suggest(&cols, &rows).unwrap();
        println!("suggested: {m:?}");
        let (docs, r) = convert(&m, rows.into_iter().map(Ok).collect());
        println!("{}\n{}", r.summary(), docs.first().cloned().unwrap_or_default());
        assert!(r.accepted > 30, "{}", r.summary());
        let _ = std::fs::remove_dir_all(cache_dir());
    }

    /// The mapping as the page sends it (camelCase, the layout as a tag).
    #[test]
    fn the_page_mapping_reads() {
        let m: Mapping = serde_json::from_value(serde_json::json!({
            "category": "conversation", "layout": "messages", "column": "conversation",
            "roleField": "role", "contentField": "content", "roles": {"user": "Mia", "system": ""},
            "names": [], "situation": null
        }))
        .unwrap();
        assert_eq!(
            m.layout,
            Layout::Messages {
                column: "conversation".into(),
                role_field: "role".into(),
                content_field: "content".into()
            }
        );
        assert_eq!(m.roles["system"], "");
        let back = serde_json::to_value(&m).unwrap();
        assert_eq!(back["roleField"], "role");
        let t: Mapping = serde_json::from_value(
            serde_json::json!({"category": "general", "layout": "text", "columns": ["text"]}),
        )
        .unwrap();
        assert_eq!(t.layout, Layout::Text { columns: vec!["text".into()] });
    }

    /// Files of different splits are not imported together.
    #[test]
    fn splits_are_never_mixed() {
        let f = |n: &str| ImportFile {
            name: n.into(),
            location: n.into(),
            zip_entry: None,
            size: 0,
            format: Format::Jsonl,
            gz: false,
            split: split_of(n),
            converted: None,
        };
        assert!(one_split(&[f("train-1.jsonl"), f("train-2.jsonl")]).is_ok());
        assert!(one_split(&[f("train.jsonl"), f("test.jsonl")]).unwrap_err().contains("different splits"));
    }
}
