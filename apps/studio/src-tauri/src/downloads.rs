//! The download queue (0.7.9; saved and controllable since 0.8.7).
//!
//! Downloads run one at a time, in order. The queue is saved to `download_queue.json` after every
//! change, so it survives closing Studio, a crash or an update. When Studio starts, the queue comes
//! back paused (the download that was running is first in line, marked as interrupted) unless the
//! owner chose "Resume the download queue when Studio starts". Each item can be moved up or down,
//! removed, retried (failed or cancelled ones) or cleared once finished; the queue can be paused
//! (the running download stops; a Parquet file being downloaded continues from where it stopped)
//! and resumed.
use super::*;
use std::collections::VecDeque;
use std::path::Path;

/// The queue file's layout.
const QUEUE_VERSION: u32 = 1;
/// Finished items kept in the list (newest first).
const KEEP_FINISHED: usize = 50;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemStatus {
    #[default]
    Waiting,
    Running,
    /// Stopped by Pause, or interrupted when Studio closed: continues when the queue resumes.
    Paused,
    Done,
    Failed,
    Cancelled,
}

/// A download asked for.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueuedDownload {
    pub id: String,
    pub source: String,
    pub name: String,
    pub max_mb: u64,
    #[serde(default)]
    pub status: ItemStatus,
    /// Why it failed or stopped.
    #[serde(default)]
    pub note: Option<String>,
    #[serde(default)]
    pub retries: u32,
    #[serde(default)]
    pub added: u64,
    #[serde(default)]
    pub finished: Option<u64>,
    /// The dataset it saved.
    #[serde(default)]
    pub saved: Option<String>,
}

/// The download running, the ones waiting (in order), the finished ones, and whether the queue is
/// paused.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadQueue {
    #[serde(default)]
    pub version: u32,
    pub current: Option<QueuedDownload>,
    pub waiting: VecDeque<QueuedDownload>,
    #[serde(default)]
    pub finished: Vec<QueuedDownload>,
    #[serde(default)]
    pub paused: bool,
}

impl DownloadQueue {
    /// The queue as saved, when Studio starts: the download that was running goes first in line,
    /// marked as interrupted, and nothing runs until the queue is resumed (unless `resume`).
    pub fn restored(mut self, resume: bool) -> DownloadQueue {
        if let Some(mut c) = self.current.take() {
            c.status = ItemStatus::Paused;
            c.note = Some("Interrupted when Studio closed; it continues when the queue resumes.".into());
            self.waiting.push_front(c);
        }
        for w in &mut self.waiting {
            if w.status == ItemStatus::Running {
                w.status = ItemStatus::Paused;
            }
        }
        self.paused = !self.waiting.is_empty() && !resume;
        self.version = QUEUE_VERSION;
        self
    }

    fn contains(&self, source: &str) -> bool {
        self.current.as_ref().is_some_and(|c| c.source == source)
            || self.waiting.iter().any(|w| w.source == source)
    }

    /// Takes the next one to run (when nothing runs and the queue is not paused).
    fn take_next(&mut self) -> Option<QueuedDownload> {
        if self.current.is_some() || self.paused {
            return None;
        }
        let mut next = self.waiting.pop_front()?;
        next.status = ItemStatus::Running;
        next.note = None;
        self.current = Some(next.clone());
        Some(next)
    }

    /// The running download ended: finished (`Ok(saved)`), failed, cancelled, or paused (back
    /// to the front of the line).
    fn end_current(&mut self, outcome: Result<String, String>, cancelled: bool, paused: bool) {
        let Some(mut c) = self.current.take() else { return };
        if paused {
            c.status = ItemStatus::Paused;
            c.note = Some("Paused; it continues when the queue resumes.".into());
            self.waiting.push_front(c);
            return;
        }
        c.finished = Some(now());
        match outcome {
            Ok(saved) => {
                c.status = ItemStatus::Done;
                c.saved = Some(saved);
                c.note = None;
            }
            Err(_) if cancelled => {
                c.status = ItemStatus::Cancelled;
                c.note = Some("Cancelled (nothing kept).".into());
            }
            Err(e) => {
                c.status = ItemStatus::Failed;
                c.note = Some(e);
            }
        }
        self.finished.insert(0, c);
        self.finished.truncate(KEEP_FINISHED);
    }

    /// Moves a waiting item up (`delta` < 0) or down.
    fn move_item(&mut self, id: &str, delta: i32) -> Res<()> {
        let i = self.waiting.iter().position(|w| w.id == id).ok_or("That download is not waiting")?;
        let j = (i as i64 + delta as i64).clamp(0, self.waiting.len() as i64 - 1) as usize;
        let item = self.waiting.remove(i).ok_or("That download is not waiting")?;
        self.waiting.insert(j, item);
        Ok(())
    }

    /// A failed or cancelled download back in line (at the end).
    fn retry(&mut self, id: &str) -> Res<()> {
        let i = self
            .finished
            .iter()
            .position(|f| f.id == id && matches!(f.status, ItemStatus::Failed | ItemStatus::Cancelled))
            .ok_or("Only a failed or cancelled download can be retried")?;
        if self.contains(&self.finished[i].source) {
            return Err(format!("{} is already in the queue", self.finished[i].name));
        }
        let mut item = self.finished.remove(i);
        item.status = ItemStatus::Waiting;
        item.retries += 1;
        item.finished = None;
        item.note = None;
        self.waiting.push_back(item);
        Ok(())
    }
}

fn queue_file(app: &AppHandle) -> Res<std::path::PathBuf> {
    Ok(app.path().app_data_dir().map_err(|e| e.to_string())?.join("download_queue.json"))
}

pub(crate) fn save_in(path: &Path, q: &DownloadQueue) -> Res<()> {
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?;
    write_atomic(path, &serde_json::to_vec_pretty(q).map_err(|e| e.to_string())?)
}

pub(crate) fn load_in(path: &Path) -> DownloadQueue {
    match std::fs::read(path) {
        Ok(b) => serde_json::from_slice(&b).unwrap_or_else(|_| {
            // Kept for the owner; the queue starts empty.
            let _ = std::fs::rename(path, path.with_extension(format!("damaged-{}.json", now())));
            DownloadQueue::default()
        }),
        Err(_) => DownloadQueue::default(),
    }
}

/// Shows the queue on the pages and saves it.
fn changed(app: &AppHandle, q: &DownloadQueue) {
    let _ = app.emit("download-queue", q.clone());
    if let Ok(p) = queue_file(app) {
        if let Err(e) = save_in(&p, q) {
            let _ = app.emit("device-notice", format!("The download queue could not be saved: {e}"));
        }
    }
}

/// Pause asked for: the running download stops and goes back to the front of the line.
static PAUSING: AtomicBool = AtomicBool::new(false);
pub(crate) fn pausing() -> bool {
    PAUSING.load(Ordering::Acquire)
}

fn resume_on_start(app: &AppHandle) -> bool {
    read_settings(app).get("resumeDownloadsOnStart").and_then(|v| v.as_bool()).unwrap_or(false)
}

/// At start-up: the saved queue, restored (paused unless the owner chose otherwise).
pub(crate) fn start(app: &AppHandle) {
    crate::hfparquet::clean_old_downloads();
    let Ok(path) = queue_file(app) else { return };
    let q = load_in(&path).restored(resume_on_start(app));
    let studio = app.state::<Studio>();
    *studio.downloads.lock().unwrap_or_else(|e| e.into_inner()) = q;
    let q = studio.downloads.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let _ = save_in(&path, &q);
    run_queue(app.clone());
}

/// Runs the queue on its own thread until it is empty or paused (nothing happens when a download
/// is already running).
fn run_queue(app: AppHandle) {
    let studio = app.state::<Studio>();
    let first = {
        let mut q = studio.downloads.lock().unwrap_or_else(|e| e.into_inner());
        let next = q.take_next();
        if next.is_some() {
            changed(&app, &q);
        }
        next
    };
    let Some(first) = first else { return };
    let cancel = studio.cancel_download.clone();
    std::thread::spawn(move || {
        let mut next = Some(first);
        while let Some(item) = next {
            cancel.store(false, Ordering::Release);
            PAUSING.store(false, Ordering::Release);
            let outcome = crate::reading::run_download(&app, &item.source, item.max_mb, &cancel);
            let studio = app.state::<Studio>();
            let mut q = studio.downloads.lock().unwrap_or_else(|e| e.into_inner());
            let paused = PAUSING.swap(false, Ordering::AcqRel);
            q.end_current(outcome, cancel.load(Ordering::Acquire) && !paused, paused);
            // The next one (taken under the lock, so a download asked for now is either taken
            // here or starts its own run).
            next = q.take_next();
            changed(&app, &q);
        }
    });
}

/// What Download did: started at once, or queued behind others (`position` 1 = next).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadStart {
    queued: bool,
    position: usize,
    /// The queue is paused: it starts when the queue is resumed.
    paused: bool,
}

/// Downloads a source, or queues it behind the downloads already asked for.
#[tauri::command]
pub fn download_corpus(
    app: AppHandle,
    studio: State<Studio>,
    source: String,
    max_mb: u64,
) -> Res<DownloadStart> {
    let src = crate::sources::sources()
        .into_iter()
        .chain(crate::sources::conversation_sources())
        .find(|s| s.id == source)
        .ok_or("Unknown corpus source")?;
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let item = QueuedDownload {
        id: format!("d{nanos:x}"),
        source,
        name: src.name.to_string(),
        max_mb: max_mb.max(1),
        status: ItemStatus::Waiting,
        note: None,
        retries: 0,
        added: now(),
        finished: None,
        saved: None,
    };
    let start = {
        let mut q = studio.downloads.lock().unwrap_or_else(|e| e.into_inner());
        if q.contains(&item.source) {
            return Err(format!("{} is already downloading or waiting its turn", item.name));
        }
        q.waiting.push_back(item);
        let start = DownloadStart {
            queued: q.current.is_some() || q.paused || q.waiting.len() > 1,
            position: q.waiting.len(),
            paused: q.paused,
        };
        changed(&app, &q);
        start
    };
    run_queue(app);
    Ok(start)
}

/// The download running, the ones waiting and the finished ones.
#[tauri::command]
pub fn download_queue(studio: State<Studio>) -> DownloadQueue {
    studio.downloads.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Cancels the download running (nothing of it is kept; the next one then starts).
#[tauri::command]
pub fn cancel_download(studio: State<Studio>) {
    studio.cancel_download.store(true, Ordering::Release);
}

/// Pauses the queue: the running download stops and waits first in line.
#[tauri::command]
pub fn pause_downloads(app: AppHandle, studio: State<Studio>) -> DownloadQueue {
    let mut q = studio.downloads.lock().unwrap_or_else(|e| e.into_inner());
    q.paused = true;
    if q.current.is_some() {
        PAUSING.store(true, Ordering::Release);
        studio.cancel_download.store(true, Ordering::Release);
    }
    changed(&app, &q);
    q.clone()
}

/// Resumes the queue.
#[tauri::command]
pub fn resume_downloads(app: AppHandle, studio: State<Studio>) -> DownloadQueue {
    {
        let mut q = studio.downloads.lock().unwrap_or_else(|e| e.into_inner());
        q.paused = false;
        changed(&app, &q);
    }
    run_queue(app.clone());
    studio.downloads.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Takes a waiting download out of the queue (nothing downloaded is deleted).
#[tauri::command]
pub fn remove_queued_download(app: AppHandle, studio: State<Studio>, id: String) -> DownloadQueue {
    let mut q = studio.downloads.lock().unwrap_or_else(|e| e.into_inner());
    q.waiting.retain(|w| w.id != id);
    changed(&app, &q);
    q.clone()
}

/// Moves a waiting download up (`delta` -1) or down (1).
#[tauri::command]
pub fn move_queued_download(
    app: AppHandle,
    studio: State<Studio>,
    id: String,
    delta: i32,
) -> Res<DownloadQueue> {
    let mut q = studio.downloads.lock().unwrap_or_else(|e| e.into_inner());
    q.move_item(&id, delta)?;
    changed(&app, &q);
    Ok(q.clone())
}

/// Puts a failed or cancelled download back in line.
#[tauri::command]
pub fn retry_download(app: AppHandle, studio: State<Studio>, id: String) -> Res<DownloadQueue> {
    {
        let mut q = studio.downloads.lock().unwrap_or_else(|e| e.into_inner());
        q.retry(&id)?;
        changed(&app, &q);
    }
    run_queue(app.clone());
    Ok(studio.downloads.lock().unwrap_or_else(|e| e.into_inner()).clone())
}

/// Clears the finished downloads from the list (the datasets they saved stay).
#[tauri::command]
pub fn clear_finished_downloads(app: AppHandle, studio: State<Studio>) -> DownloadQueue {
    let mut q = studio.downloads.lock().unwrap_or_else(|e| e.into_inner());
    q.finished.clear();
    changed(&app, &q);
    q.clone()
}

/// Whether the queue resumes by itself when Studio starts (off by default).
#[tauri::command]
pub fn download_resume_on_start(app: AppHandle) -> bool {
    resume_on_start(&app)
}

#[tauri::command]
pub fn set_download_resume_on_start(app: AppHandle, on: bool) -> Res<bool> {
    write_setting(&app, "resumeDownloadsOnStart", on.into())?;
    Ok(on)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str) -> QueuedDownload {
        QueuedDownload {
            id: id.into(),
            source: format!("src-{id}"),
            name: id.into(),
            max_mb: 10,
            status: ItemStatus::Waiting,
            note: None,
            retries: 0,
            added: 1,
            finished: None,
            saved: None,
        }
    }

    /// The queue as Studio saved it comes back with the interrupted download first in line, and
    /// paused (nothing starts by itself) unless asked; the file round-trips.
    #[test]
    fn a_saved_queue_comes_back_paused_with_the_interrupted_one_first() {
        let mut q = DownloadQueue::default();
        q.waiting.extend([item("a"), item("b"), item("c")]);
        assert_eq!(q.take_next().unwrap().id, "a");
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("yendory-queue-{nanos:x}")).join("download_queue.json");
        save_in(&path, &q).unwrap();
        let back = load_in(&path).restored(false);
        let ids: Vec<&str> = back.waiting.iter().map(|w| w.id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
        assert_eq!(back.waiting[0].status, ItemStatus::Paused);
        assert!(back.paused && back.current.is_none());
        let mut back = back;
        assert!(back.take_next().is_none(), "nothing starts while paused");
        let mut on = load_in(&path).restored(true);
        assert!(!on.paused && on.take_next().unwrap().id == "a", "resumes when the owner chose that");
        // A damaged file is kept aside; the queue starts empty.
        std::fs::write(&path, b"{oops").unwrap();
        assert!(load_in(&path).waiting.is_empty());
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn pause_reorder_retry_and_outcomes() {
        let mut q = DownloadQueue::default();
        q.waiting.extend([item("a"), item("b"), item("c")]);
        q.take_next();
        // Paused: back to the front of the line.
        q.end_current(Err("Download cancelled".into()), false, true);
        assert_eq!((q.waiting[0].id.as_str(), q.waiting[0].status), ("a", ItemStatus::Paused));
        // Reorder.
        q.move_item("c", -1).unwrap();
        q.move_item("c", -5).unwrap();
        let ids: Vec<&str> = q.waiting.iter().map(|w| w.id.as_str()).collect();
        assert_eq!(ids, ["c", "a", "b"]);
        // Failed, cancelled, done.
        q.take_next();
        q.end_current(Err("Download failed: 404".into()), false, false);
        q.take_next();
        q.end_current(Err("Download cancelled".into()), true, false);
        q.take_next();
        q.end_current(Ok("Dataset b".into()), false, false);
        let st: Vec<(&str, ItemStatus)> = q.finished.iter().map(|f| (f.id.as_str(), f.status)).collect();
        assert_eq!(st, [("b", ItemStatus::Done), ("a", ItemStatus::Cancelled), ("c", ItemStatus::Failed)]);
        // Retry a failed one; a finished one cannot be retried.
        q.retry("c").unwrap();
        assert_eq!((q.waiting[0].id.as_str(), q.waiting[0].retries), ("c", 1));
        assert!(q.retry("b").is_err());
        assert!(q.retry("c").is_err(), "already waiting");
    }
}
