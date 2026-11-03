//! The debug log: what the backend did (Network calls with their size and time, failures,
//! panics), kept in memory (the latest lines) and sent to the window's debug console as it
//! happens (`debug-log`). Nothing is written to disk.
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};

const KEEP: usize = 2000;

#[derive(Clone, Debug, Serialize)]
pub struct Line {
    /// Milliseconds since 1970.
    pub t: u64,
    /// `info`, `warn` or `error`.
    pub level: &'static str,
    pub text: String,
}

static LINES: Mutex<VecDeque<Line>> = Mutex::new(VecDeque::new());
static APP: OnceLock<AppHandle> = OnceLock::new();

/// Starts sending lines to the window, and catches panics into the log.
pub fn start(app: &AppHandle) {
    let _ = APP.set(app.clone());
    let before = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |p| {
        let place = p.location().map(|l| format!(" at {}:{}", l.file(), l.line())).unwrap_or_default();
        let what = p
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| p.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "(no message)".into());
        error(format!("The backend failed (panic){place}: {what}"));
        before(p);
    }));
    info(format!("Yendory Tools {} started", crate::VERSION));
}

pub fn log(level: &'static str, text: String) {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let line = Line { t, level, text };
    if let Ok(mut lines) = LINES.lock() {
        if lines.len() >= KEEP {
            lines.pop_front();
        }
        lines.push_back(line.clone());
    }
    if let Some(app) = APP.get() {
        let _ = app.emit("debug-log", line);
    }
}

pub fn info(text: impl Into<String>) {
    log("info", text.into());
}

pub fn warn(text: impl Into<String>) {
    log("warn", text.into());
}

pub fn error(text: impl Into<String>) {
    log("error", text.into());
}

/// Sizes for people: bytes as KB or MB.
pub fn size(bytes: usize) -> String {
    if bytes >= 1 << 20 {
        format!("{:.1} MB", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    }
}

/// The lines kept so far (the window asks when it opens).
#[tauri::command]
pub fn debug_lines() -> Vec<Line> {
    LINES.lock().map(|l| l.iter().cloned().collect()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_kept_up_to_a_limit() {
        for i in 0..KEEP + 5 {
            info(format!("line {i}"));
        }
        let lines = debug_lines();
        assert_eq!(lines.len(), KEEP);
        assert_eq!(lines.last().unwrap().text, format!("line {}", KEEP + 4));
        assert_eq!(size(2048), "2.0 KB");
        assert_eq!(size(3 << 20), "3.0 MB");
    }
}
