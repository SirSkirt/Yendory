//! What Tools needs from the operating system for real games: the list of windows, which one is
//! in front, its picture, and virtual keyboard and mouse input. Windows (GDI, SendInput) and
//! Linux on X11 (XTEST); Wayland does not let ordinary programs do either.
use super::controls::{Button, KeyInfo};
use crate::Res;
use serde::Serialize;

#[cfg(target_os = "linux")]
mod x11;
#[cfg(target_os = "linux")]
pub use x11::{listen, open, stop_key_watch};

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::{listen, open, stop_key_watch};

#[cfg(not(any(windows, target_os = "linux")))]
pub fn open() -> Res<Box<dyn Desktop>> {
    Err("Real games need Windows or Linux (X11)".into())
}
#[cfg(not(any(windows, target_os = "linux")))]
pub fn stop_key_watch(_on_stop: Box<dyn Fn() + Send>) -> Res<()> {
    Err("Real games need Windows or Linux (X11)".into())
}
#[cfg(not(any(windows, target_os = "linux")))]
pub fn listen(_accept_virtual: bool, _tx: std::sync::mpsc::Sender<Heard>) -> Res<Listening> {
    Err("Real games need Windows or Linux (X11)".into())
}

/// A person's input, as the keyboard and mouse report it (before any program sees it).
#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    /// A key from the controls table (others are not reported).
    Key {
        key: &'static str,
        down: bool,
    },
    Button {
        button: Button,
        down: bool,
    },
    /// Relative mouse movement, in the mouse's own units.
    Motion {
        dx: f64,
        dy: f64,
    },
}

/// An input and when it arrived.
pub type Heard = (std::time::Instant, Input);

/// Listening to the keyboard and mouse; stops when dropped.
pub struct Listening {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    wake: Option<Box<dyn Fn() + Send>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Listening {
    pub(crate) fn new(
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        wake: Option<Box<dyn Fn() + Send>>,
        thread: std::thread::JoinHandle<()>,
    ) -> Listening {
        Listening { stop, wake, thread: Some(thread) }
    }
}

impl Drop for Listening {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(w) = &self.wake {
            w();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WindowInfo {
    pub id: u64,
    pub title: String,
    /// The program's file name (`Ravenfield.exe`, `yendory-tools`).
    pub process: String,
    pub pid: u32,
    pub width: u32,
    pub height: u32,
}

/// A rectangle on the screen, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// One connection to the desktop. Used by one thread at a time.
pub trait Desktop: Send {
    fn windows(&mut self) -> Res<Vec<WindowInfo>>;
    /// Whether `window` is the window keyboard input goes to.
    fn in_front(&mut self, window: u64) -> bool;
    fn bring_to_front(&mut self, window: u64) -> Res<()>;
    /// The window's drawing area on the screen.
    fn area(&mut self, window: u64) -> Res<Rect>;
    fn key(&mut self, key: &KeyInfo, down: bool) -> Res<()>;
    fn button(&mut self, button: Button, down: bool) -> Res<()>;
    /// Moves the mouse by `dx`, `dy` (relative, as a mouse does).
    fn motion(&mut self, dx: i32, dy: i32) -> Res<()>;
    /// The window's picture shrunk to `width` x `height`, three bytes (RGB) a pixel.
    fn grab(&mut self, window: u64, width: u32, height: u32) -> Res<Vec<u8>>;
}

/// Shrinks a picture by averaging the source pixels under each target pixel. `pixel` gives the
/// RGB of source pixel (x, y).
pub fn shrink(
    src_w: usize,
    src_h: usize,
    width: usize,
    height: usize,
    pixel: impl Fn(usize, usize) -> [u8; 3],
) -> Vec<u8> {
    let mut out = vec![0u8; width * height * 3];
    if src_w == 0 || src_h == 0 {
        return out;
    }
    for ty in 0..height {
        let (y0, y1) =
            (ty * src_h / height, ((ty + 1) * src_h / height).max(ty * src_h / height + 1).min(src_h));
        for tx in 0..width {
            let (x0, x1) =
                (tx * src_w / width, ((tx + 1) * src_w / width).max(tx * src_w / width + 1).min(src_w));
            let mut sum = [0u32; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = pixel(x, y);
                    for c in 0..3 {
                        sum[c] += p[c] as u32;
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)).max(1) as u32;
            let at = (ty * width + tx) * 3;
            for c in 0..3 {
                out[at + c] = ((sum[c] + n / 2) / n) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrinking_averages_blocks() {
        // 4x2 source: left half black, right half white; to 2x1.
        let out = shrink(4, 2, 2, 1, |x, _| if x < 2 { [0, 0, 0] } else { [255, 255, 255] });
        assert_eq!(out, vec![0, 0, 0, 255, 255, 255]);
        // Growing repeats pixels.
        let out = shrink(1, 1, 2, 2, |_, _| [10, 20, 30]);
        assert_eq!(out, [10, 20, 30].repeat(4));
    }
}
