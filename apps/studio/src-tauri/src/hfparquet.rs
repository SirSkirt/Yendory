//! Hugging Face datasets read from their Parquet files (0.8.3).
//!
//! The dataset viewer's `/rows` API answers 100 rows a request, and Studio leaves a second
//! between requests: about 65 rows a second, so a million-row dataset takes hours (measured:
//! UltraChat's 207,865 training rows would take about 53 minutes; LMSYS-Chat-1M over 4 hours).
//! Hugging Face also publishes every viewer-supported dataset as Parquet files
//! (`refs/convert/parquet`): one 244 MB UltraChat file downloaded here in 10 s and its 69,289
//! rows were read in 1.3 s. Each row comes out as the same JSON the viewer gives, so the same
//! conversion and quality filter apply.
//!
//! Files are downloaded one at a time into a temporary folder (resumed with an HTTP range request
//! after an interruption, restarted when the server does not honour the range), read row by row
//! and deleted. The access token is sent only to huggingface.co; ureq never forwards it on a
//! redirect (to the file servers, whose links are signed).
use super::*;
use crate::reading::{get_json, hf_token, Pacer};
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// One Parquet file of a dataset split.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParquetFile {
    pub config: String,
    pub url: String,
    pub size: u64,
    pub filename: String,
}

/// The Parquet files of `split` in `configs` (in that order, then by file name), or None when the
/// dataset has none (or the viewer does not answer).
pub(crate) fn parquet_files(
    agent: &ureq::Agent,
    dataset: &str,
    configs: &[&str],
    split: &str,
    cancel: &AtomicBool,
    note: &mut dyn FnMut(Option<String>),
) -> Option<Vec<ParquetFile>> {
    let url = format!("https://datasets-server.huggingface.co/parquet?dataset={dataset}");
    let page = get_json(agent, &url, cancel, &mut Pacer::new(), note).ok()?;
    let files = files_of(&page, configs, split);
    (!files.is_empty()).then_some(files)
}

fn files_of(page: &serde_json::Value, configs: &[&str], split: &str) -> Vec<ParquetFile> {
    let mut files: Vec<ParquetFile> = page
        .get("parquet_files")
        .and_then(|f| f.as_array())
        .into_iter()
        .flatten()
        .filter(|f| f.get("split").and_then(|s| s.as_str()) == Some(split))
        .filter_map(|f| {
            let config = f.get("config")?.as_str()?;
            if !configs.contains(&config) {
                return None;
            }
            Some(ParquetFile {
                config: config.into(),
                url: f.get("url")?.as_str()?.into(),
                size: f.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
                filename: f.get("filename").and_then(|s| s.as_str()).unwrap_or("data.parquet").into(),
            })
        })
        .filter(|f| f.url.starts_with("https://huggingface.co/"))
        .collect();
    files.sort_by_key(|f| (configs.iter().position(|c| *c == f.config), f.filename.clone()));
    files
}

/// The rows of a dataset split, as the viewer counts them (for progress), when it says.
pub(crate) fn split_rows(
    agent: &ureq::Agent,
    dataset: &str,
    configs: &[&str],
    split: &str,
    cancel: &AtomicBool,
) -> Option<u64> {
    let url = format!("https://datasets-server.huggingface.co/size?dataset={dataset}");
    let page = get_json(agent, &url, cancel, &mut Pacer::new(), &mut |_| {}).ok()?;
    let n = page
        .get("size")?
        .get("splits")?
        .as_array()?
        .iter()
        .filter(|s| {
            s.get("split").and_then(|x| x.as_str()) == Some(split)
                && s.get("config").and_then(|x| x.as_str()).is_some_and(|c| configs.contains(&c))
        })
        .filter_map(|s| s.get("num_rows").and_then(|x| x.as_u64()))
        .sum::<u64>();
    (n > 0).then_some(n)
}

/// Where a file is downloaded to (the same place every time, so an interrupted download resumes).
pub(crate) fn temp_path(dataset: &str, file: &ParquetFile, split: &str) -> PathBuf {
    let safe = |s: &str| {
        s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect::<String>()
    };
    std::env::temp_dir().join("yendory-downloads").join(safe(dataset)).join(format!(
        "{}-{}-{}",
        safe(&file.config),
        safe(split),
        safe(&file.filename)
    ))
}

/// Downloads `url` to `dest`, continuing a partial file with a range request. `size` is the
/// expected size (0: unknown). `on_bytes` gets the bytes on disk. A server that answers a range
/// request with the whole file starts it again (never appended to). Transient failures are
/// retried (5 times, waiting longer each time); a cancel or `stop` ends it, keeping the partial
/// file for the next attempt.
pub(crate) fn download_to(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    size: u64,
    cancel: &AtomicBool,
    stop: &dyn Fn() -> bool,
    on_bytes: &mut dyn FnMut(u64),
) -> Res<()> {
    std::fs::create_dir_all(dest.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?;
    if let Some(free) = recovery::free_space(dest.parent().unwrap_or(Path::new("."))) {
        let have = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
        if size > have && free < size - have + (512 << 20) {
            return Err(format!(
                "Not enough free disk space to download it ({} MB needed and 512 MB to spare; {} MB free)",
                (size - have) >> 20,
                free >> 20
            ));
        }
    }
    let mut failures = 0;
    loop {
        let have = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
        if size > 0 && have == size {
            return Ok(());
        }
        if size > 0 && have > size {
            let _ = std::fs::remove_file(dest);
            continue;
        }
        let mut request = agent.get(url);
        if let Some(token) = hf_token().filter(|_| url.starts_with("https://huggingface.co/")) {
            request = request.set("Authorization", &format!("Bearer {}", token.trim()));
        }
        if have > 0 {
            request = request.set("Range", &format!("bytes={have}-"));
        }
        let outcome = match request.call() {
            Ok(r) => {
                // 206: the rest of the file. 200: the whole file (range not honoured): start again.
                let append = have > 0 && r.status() == 206;
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .append(append)
                    .truncate(!append)
                    .open(dest)
                    .map_err(|e| e.to_string())?;
                let mut on_disk = if append { have } else { 0 };
                let mut reader = r.into_reader();
                let mut buf = vec![0u8; 1 << 16];
                let mut last = 0;
                loop {
                    if cancel.load(Ordering::Acquire) || stop() {
                        return Err("Download cancelled".into());
                    }
                    match reader.read(&mut buf) {
                        Ok(0) => break Ok(()),
                        Ok(n) => {
                            file.write_all(&buf[..n])
                                .map_err(|e| format!("Writing the download failed: {e}"))?;
                            on_disk += n as u64;
                            if on_disk - last >= 1 << 20 {
                                last = on_disk;
                                on_bytes(on_disk);
                            }
                        }
                        Err(e) => break Err(format!("the connection broke ({e})")),
                    }
                }
            }
            Err(ureq::Error::Status(416, _)) => {
                // The range is past the end: the partial file is not this file. Start again.
                let _ = std::fs::remove_file(dest);
                Err("the partial download did not match the file".into())
            }
            Err(ureq::Error::Status(code, _)) if code < 500 && code != 429 => {
                return Err(format!("Download failed: the server answered {code} for {url}"));
            }
            Err(e) => Err(format!("{e}")),
        };
        match outcome {
            Ok(()) => {
                let have = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
                on_bytes(have);
                if size == 0 || have == size {
                    return Ok(());
                }
                // Short: continue it (a range request) on the next turn.
                failures += 1;
            }
            Err(_) => failures += 1,
        }
        if failures > 5 {
            return Err(format!("Download failed after {failures} attempts: {url}"));
        }
        let wait = Duration::from_secs(2u64 << failures.min(5));
        let until = std::time::Instant::now() + wait;
        while std::time::Instant::now() < until {
            if cancel.load(Ordering::Acquire) || stop() {
                return Err("Download cancelled".into());
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

/// The rows a Parquet file says it has.
#[cfg(test)]
fn row_count(path: &Path) -> Res<u64> {
    let r = SerializedFileReader::new(std::fs::File::open(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("Not a readable Parquet file: {e}"))?;
    Ok(r.metadata().file_metadata().num_rows().max(0) as u64)
}

/// Calls `each` with every row of a Parquet file as JSON (as the dataset viewer gives rows) until
/// it returns false. Returns the rows read.
pub(crate) fn for_each_row(path: &Path, each: &mut dyn FnMut(&serde_json::Value) -> bool) -> Res<u64> {
    let r = SerializedFileReader::new(std::fs::File::open(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("Not a readable Parquet file: {e}"))?;
    let mut n = 0;
    for row in r.get_row_iter(None).map_err(|e| e.to_string())? {
        let row = row.map_err(|e| format!("A row of the Parquet file cannot be read: {e}"))?;
        n += 1;
        if !each(&row.to_json_value()) {
            break;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn files_are_chosen_by_config_and_split_in_order() {
        let page = serde_json::json!({"parquet_files": [
            {"config": "b", "split": "train", "url": "https://huggingface.co/x/b/0000.parquet", "filename": "0000.parquet", "size": 5},
            {"config": "a", "split": "train", "url": "https://huggingface.co/x/a/0001.parquet", "filename": "0001.parquet", "size": 3},
            {"config": "a", "split": "test", "url": "https://huggingface.co/x/a/t.parquet", "filename": "0000.parquet", "size": 3},
            {"config": "a", "split": "train", "url": "https://huggingface.co/x/a/0000.parquet", "filename": "0000.parquet", "size": 4},
            {"config": "c", "split": "train", "url": "https://huggingface.co/x/c/0000.parquet", "filename": "0000.parquet", "size": 4},
            {"config": "a", "split": "train", "url": "https://elsewhere.example/a.parquet", "filename": "0002.parquet", "size": 4}
        ]});
        let f = files_of(&page, &["a", "b"], "train");
        let urls: Vec<&str> = f.iter().map(|f| f.url.as_str()).collect();
        assert_eq!(
            urls,
            [
                "https://huggingface.co/x/a/0000.parquet",
                "https://huggingface.co/x/a/0001.parquet",
                "https://huggingface.co/x/b/0000.parquet"
            ],
            "the test split, other configs and other hosts are left out"
        );
    }

    /// A server that serves `body`, honouring range requests or not, and breaking the first
    /// response after `cut` bytes.
    fn server(
        body: Vec<u8>,
        ranges: bool,
        cut: Option<usize>,
        requests: usize,
    ) -> (String, std::sync::mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for i in 0..requests {
                let Ok((mut s, _)) = listener.accept() else { return };
                let mut req = [0u8; 2048];
                let n = s.read(&mut req).unwrap_or(0);
                let text = String::from_utf8_lossy(&req[..n]).to_string();
                let from = text
                    .lines()
                    .find_map(|l| {
                        l.to_ascii_lowercase()
                            .strip_prefix("range: bytes=")
                            .map(|r| r.trim_end_matches('-').parse::<usize>().unwrap_or(0))
                    })
                    .filter(|_| ranges);
                let _ = tx.send(text);
                let (status, part) = match from {
                    Some(f) => ("206 Partial Content", &body[f..]),
                    None => ("200 OK", &body[..]),
                };
                let send = if i == 0 { cut.map_or(part, |c| &part[..c.min(part.len())]) } else { part };
                let _ = write!(
                    s,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    part.len()
                );
                let _ = s.write_all(send);
            }
        });
        (format!("http://{addr}/file.parquet"), rx)
    }

    fn temp() -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("yendory-dl-test-{nanos:x}")).join("file.parquet")
    }

    /// A download broken part-way continues with a range request and ends with the whole file;
    /// a server that ignores ranges sends the whole file, which replaces the partial one (never
    /// appended to it).
    #[test]
    fn broken_downloads_continue_or_start_again() {
        let body: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        let agent = ureq::AgentBuilder::new().build();
        for ranges in [true, false] {
            let (url, rx) = server(body.clone(), ranges, Some(100_000), 2);
            let dest = temp();
            download_to(
                &agent,
                &url,
                &dest,
                body.len() as u64,
                &AtomicBool::new(false),
                &|| false,
                &mut |_| {},
            )
            .unwrap();
            assert_eq!(std::fs::read(&dest).unwrap(), body, "ranges honoured: {ranges}");
            let second = rx.iter().nth(1).unwrap().to_ascii_lowercase();
            assert!(second.contains("range: bytes=100000-"), "the second request continues: {second}");
            let _ = std::fs::remove_dir_all(dest.parent().unwrap());
        }
    }

    /// Rows come out as the JSON the dataset viewer gives (lists of objects stay lists of
    /// objects), so the same conversion applies.
    #[test]
    fn parquet_rows_read_as_json() {
        use parquet::data_type::{ByteArray, ByteArrayType, Int64Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;
        let schema = std::sync::Arc::new(
            parse_message_type("message m { required binary text (UTF8); required int64 n; }").unwrap(),
        );
        let path = temp();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let file = std::fs::File::create(&path).unwrap();
        let mut w =
            SerializedFileWriter::new(file, schema, std::sync::Arc::new(WriterProperties::builder().build()))
                .unwrap();
        let mut g = w.next_row_group().unwrap();
        let mut c = g.next_column().unwrap().unwrap();
        let texts: Vec<ByteArray> =
            ["Hi, Ann.", "Hello, Ben.", "Bye."].iter().map(|t| ByteArray::from(*t)).collect();
        c.typed::<ByteArrayType>().write_batch(&texts, None, None).unwrap();
        c.close().unwrap();
        let mut c = g.next_column().unwrap().unwrap();
        c.typed::<Int64Type>().write_batch(&[1, 2, 3], None, None).unwrap();
        c.close().unwrap();
        g.close().unwrap();
        w.close().unwrap();
        assert_eq!(row_count(&path).unwrap(), 3);
        let mut rows = Vec::new();
        let n = for_each_row(&path, &mut |r| {
            rows.push(r.clone());
            rows.len() < 2
        })
        .unwrap();
        assert_eq!(n, 2, "stops when asked");
        assert_eq!(rows[0], serde_json::json!({"text": "Hi, Ann.", "n": 1}));
        assert!(for_each_row(&path.with_extension("missing"), &mut |_| true).is_err());
        std::fs::write(&path, b"not parquet").unwrap();
        assert!(row_count(&path).unwrap_err().contains("Parquet"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
