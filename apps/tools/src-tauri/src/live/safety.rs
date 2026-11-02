//! The safety layer: every virtual press, release and mouse movement passes through a
//! [`Controller`], which sends it only if the profile permits that control, the combination
//! cannot reach the system, the game window is in front, the emergency stop has not been
//! pressed, and the action rate is within the limit. It releases a control held longer than the
//! longest hold, releases everything when the game leaves the front (and pauses), and releases
//! everything when it is dropped. Every action and every refusal is logged.
use super::controls::{key, refused_combination, Button, Control, KeyInfo};
use super::sys::Desktop;
use crate::Res;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Bumped by the emergency stop: a controller made before it refuses everything after it.
pub static STOPS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Limits {
    /// A control held longer than this is released.
    pub max_hold_ms: u64,
    /// Presses and mouse movements a second (releases are never limited).
    pub max_actions_per_second: u32,
    /// The largest mouse movement in one action, in units after the profile's sensitivity.
    pub max_mouse_step: i32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_hold_ms: 5000, max_actions_per_second: 60, max_mouse_step: 400 }
    }
}

/// What a profile lets the network (or a test script) use.
#[derive(Clone, Debug, Default)]
pub struct Permitted {
    pub keys: Vec<&'static KeyInfo>,
    pub buttons: Vec<Button>,
    pub mouse: bool,
    /// Mouse units sent per unit asked for (sensitivity).
    pub mouse_scale: f32,
}

impl Permitted {
    pub fn new(keys: &[String], buttons: &[Button], mouse: bool, mouse_scale: f32) -> Res<Permitted> {
        let mut out = Permitted { mouse, mouse_scale, buttons: buttons.to_vec(), ..Permitted::default() };
        for k in keys {
            out.keys.push(key(k).ok_or_else(|| format!("{k} is not a key Tools can press"))?);
        }
        Ok(out)
    }

    fn allows(&self, c: Control) -> bool {
        match c {
            Control::Key(k) => self.keys.iter().any(|i| i.name == k),
            Control::Button(b) => self.buttons.contains(&b),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LogEntry {
    /// Milliseconds since the controller began.
    pub ms: u64,
    pub action: String,
    /// Empty when it was sent; otherwise why it was not (or what the safety layer did).
    pub note: String,
    pub sent: bool,
}

pub struct Controller {
    desktop: Box<dyn Desktop>,
    window: u64,
    permitted: Permitted,
    limits: Limits,
    held: BTreeMap<Control, Instant>,
    recent: VecDeque<Instant>,
    remainder: (f32, f32),
    started: Instant,
    stop_source: &'static AtomicU64,
    stops: u64,
    /// Why input is paused (the game left the front), until [`Controller::resume`].
    pub paused: Option<String>,
    pub log: Vec<LogEntry>,
}

impl Controller {
    pub fn new(desktop: Box<dyn Desktop>, window: u64, permitted: Permitted, limits: Limits) -> Controller {
        Controller {
            desktop,
            window,
            permitted,
            limits,
            held: BTreeMap::new(),
            recent: VecDeque::new(),
            remainder: (0.0, 0.0),
            started: Instant::now(),
            stop_source: &STOPS,
            stops: STOPS.load(Ordering::Acquire),
            paused: None,
            log: Vec::new(),
        }
    }

    pub fn stopped(&self) -> bool {
        self.stop_source.load(Ordering::Acquire) != self.stops
    }

    pub fn held(&self) -> Vec<&'static str> {
        self.held.keys().map(|c| c.name()).collect()
    }

    fn note(&mut self, action: String, sent: bool, note: impl Into<String>) {
        let ms = self.started.elapsed().as_millis() as u64;
        self.log.push(LogEntry { ms, action, note: note.into(), sent });
        if self.log.len() > 10_000 {
            self.log.drain(..1000);
        }
    }

    /// Whether anything may be sent now; refuses (and logs) when not.
    fn may_send(&mut self, action: &str) -> bool {
        let why = if self.stopped() {
            Some("the emergency stop was pressed".to_string())
        } else if let Some(p) = &self.paused {
            Some(p.clone())
        } else if !self.desktop.in_front(self.window) {
            self.release_all("the game is not in front");
            self.paused = Some("paused: the game is not in front".into());
            Some("the game is not in front".to_string())
        } else {
            let now = Instant::now();
            while self.recent.front().is_some_and(|t| now.duration_since(*t) > Duration::from_secs(1)) {
                self.recent.pop_front();
            }
            if self.recent.len() as u32 >= self.limits.max_actions_per_second {
                Some(format!("more than {} actions a second", self.limits.max_actions_per_second))
            } else {
                self.recent.push_back(now);
                None
            }
        };
        match why {
            Some(w) => {
                self.note(action.to_string(), false, w);
                false
            }
            None => true,
        }
    }

    fn send(&mut self, c: Control, down: bool) -> Res<()> {
        match c {
            Control::Key(k) => self.desktop.key(key(k).ok_or("unknown key")?, down),
            Control::Button(b) => self.desktop.button(b, down),
        }
    }

    /// Presses (and holds) a control. Pressing one already held does nothing.
    pub fn press(&mut self, c: Control) -> bool {
        let action = format!("press {}", c.name());
        if !self.permitted.allows(c) {
            self.note(action, false, format!("{} is not permitted in this profile", c.name()));
            return false;
        }
        if self.held.contains_key(&c) {
            return true;
        }
        let held = self.held();
        if let Some(why) = refused_combination(&held, c.name()) {
            self.note(action, false, why);
            return false;
        }
        if !self.may_send(&action) {
            return false;
        }
        match self.send(c, true) {
            Ok(()) => {
                self.held.insert(c, Instant::now());
                self.note(action, true, "");
                true
            }
            Err(e) => {
                self.note(action, false, e);
                false
            }
        }
    }

    /// Releases a held control. Never refused: letting go is always safe.
    pub fn release(&mut self, c: Control, why: &str) -> bool {
        if self.held.remove(&c).is_none() {
            return true;
        }
        let r = self.send(c, false);
        let ok = r.is_ok();
        self.note(format!("release {}", c.name()), ok, r.err().unwrap_or_else(|| why.to_string()));
        ok
    }

    pub fn release_all(&mut self, why: &str) {
        let held: Vec<Control> = self.held.keys().copied().collect();
        for c in held {
            self.release(c, why);
        }
    }

    /// Moves the mouse by `dx`, `dy` units (times the profile's sensitivity, each axis at most
    /// the largest step). Returns the units sent.
    pub fn motion(&mut self, dx: f32, dy: f32) -> Option<(i32, i32)> {
        let action = format!("move {dx} {dy}");
        if !self.permitted.mouse {
            self.note(action, false, "the mouse is not permitted in this profile");
            return None;
        }
        if !self.may_send(&action) {
            return None;
        }
        let max = self.limits.max_mouse_step as f32;
        let want = (
            (dx * self.permitted.mouse_scale).clamp(-max, max) + self.remainder.0,
            (dy * self.permitted.mouse_scale).clamp(-max, max) + self.remainder.1,
        );
        let sent = (want.0.round() as i32, want.1.round() as i32);
        self.remainder = (want.0 - sent.0 as f32, want.1 - sent.1 as f32);
        if sent == (0, 0) {
            return Some(sent);
        }
        match self.desktop.motion(sent.0, sent.1) {
            Ok(()) => {
                let note = if (dx * self.permitted.mouse_scale).abs() > max
                    || (dy * self.permitted.mouse_scale).abs() > max
                {
                    format!("limited to {max} units")
                } else {
                    String::new()
                };
                self.note(format!("move {} {}", sent.0, sent.1), true, note);
                Some(sent)
            }
            Err(e) => {
                self.note(action, false, e);
                None
            }
        }
    }

    /// Releases controls held too long, and everything when the game has left the front or
    /// the emergency stop was pressed. Call it often while anything is held.
    pub fn tick(&mut self) {
        if self.held.is_empty() {
            return;
        }
        if self.stopped() {
            self.release_all("the emergency stop was pressed");
            return;
        }
        if !self.desktop.in_front(self.window) {
            self.release_all("the game is not in front");
            self.paused = Some("paused: the game is not in front".into());
            return;
        }
        let limit = Duration::from_millis(self.limits.max_hold_ms);
        let long: Vec<Control> =
            self.held.iter().filter(|(_, t)| t.elapsed() > limit).map(|(c, _)| *c).collect();
        for c in long {
            self.release(c, &format!("held longer than {} ms", self.limits.max_hold_ms));
        }
    }

    /// Lets input through again after a pause (the person's choice).
    pub fn resume(&mut self) {
        self.paused = None;
    }

    pub fn window(&self) -> u64 {
        self.window
    }

    pub fn desktop(&mut self) -> &mut dyn Desktop {
        self.desktop.as_mut()
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        self.release_all("input ended");
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::live::sys::{Rect, WindowInfo};
    use std::sync::{Arc, Mutex};

    /// A desktop that records what was sent; `front` says whether the game is in front.
    #[derive(Clone, Default)]
    pub struct Fake {
        pub sent: Arc<Mutex<Vec<String>>>,
        pub front: Arc<Mutex<bool>>,
    }

    impl Desktop for Fake {
        fn windows(&mut self) -> Res<Vec<WindowInfo>> {
            Ok(vec![])
        }
        fn in_front(&mut self, _: u64) -> bool {
            *self.front.lock().unwrap()
        }
        fn bring_to_front(&mut self, _: u64) -> Res<()> {
            Ok(())
        }
        fn area(&mut self, _: u64) -> Res<Rect> {
            Ok(Rect { x: 0, y: 0, width: 10, height: 10 })
        }
        fn key(&mut self, key: &KeyInfo, down: bool) -> Res<()> {
            self.sent.lock().unwrap().push(format!("{}{}", if down { "+" } else { "-" }, key.name));
            Ok(())
        }
        fn button(&mut self, b: Button, down: bool) -> Res<()> {
            self.sent.lock().unwrap().push(format!("{}{}", if down { "+" } else { "-" }, b.name()));
            Ok(())
        }
        fn motion(&mut self, dx: i32, dy: i32) -> Res<()> {
            self.sent.lock().unwrap().push(format!("m{dx},{dy}"));
            Ok(())
        }
        fn grab(&mut self, _: u64, w: u32, h: u32) -> Res<Vec<u8>> {
            Ok(vec![0; (w * h * 3) as usize])
        }
    }

    pub fn controller(keys: &[&str], limits: Limits) -> (Controller, Fake) {
        let fake = Fake::default();
        *fake.front.lock().unwrap() = true;
        let keys: Vec<String> = keys.iter().map(|s| s.to_string()).collect();
        let p = Permitted::new(&keys, &[Button::Left, Button::Right], true, 1.0).unwrap();
        (Controller::new(Box::new(fake.clone()), 1, p, limits), fake)
    }

    fn sent(f: &Fake) -> Vec<String> {
        f.sent.lock().unwrap().clone()
    }

    #[test]
    fn only_permitted_controls_and_safe_combinations_are_sent() {
        let (mut c, f) = controller(&["w", "alt", "tab", "ctrl"], Limits::default());
        assert!(c.press(Control::Key("w")));
        assert!(!c.press(Control::Key("e")));
        assert!(!c.press(Control::Button(Button::Middle)));
        assert!(c.press(Control::Key("alt")));
        assert!(!c.press(Control::Key("tab")));
        assert!(!c.press(Control::Key("ctrl")));
        assert!(c.press(Control::Button(Button::Left)));
        assert_eq!(sent(&f), ["+w", "+alt", "+left"]);
        assert_eq!(c.log.iter().filter(|e| !e.sent).count(), 4);
        drop(c);
        // Dropping the controller lets go of everything.
        let s = sent(&f);
        assert_eq!(s[3..].iter().filter(|x| x.starts_with('-')).count(), 3);
    }

    #[test]
    fn leaving_the_front_releases_everything_and_pauses() {
        let (mut c, f) = controller(&["w", "shift"], Limits::default());
        c.press(Control::Key("w"));
        c.press(Control::Key("shift"));
        *f.front.lock().unwrap() = false;
        c.tick();
        assert!(c.held().is_empty());
        assert!(c.paused.is_some());
        *f.front.lock().unwrap() = true;
        assert!(!c.press(Control::Key("w")), "paused until resumed");
        c.resume();
        assert!(c.press(Control::Key("w")));
        assert_eq!(sent(&f), ["+w", "+shift", "-shift", "-w", "+w"]);
    }

    #[test]
    fn long_holds_are_released_and_the_rate_is_limited() {
        let (mut c, f) = controller(
            &["w", "a"],
            Limits { max_hold_ms: 30, max_actions_per_second: 3, max_mouse_step: 400 },
        );
        c.press(Control::Key("w"));
        std::thread::sleep(Duration::from_millis(50));
        c.tick();
        assert!(c.held().is_empty());
        assert!(c.press(Control::Key("a")));
        c.release(Control::Key("a"), "");
        assert!(c.press(Control::Key("w")));
        assert!(!c.press(Control::Key("a")), "a fourth action within a second");
        // Releases are never limited.
        assert!(c.release(Control::Key("w"), ""));
        assert_eq!(sent(&f), ["+w", "-w", "+a", "-a", "+w", "-w"]);
    }

    #[test]
    fn mouse_steps_are_scaled_limited_and_keep_fractions() {
        let (mut c, f) = controller(&[], Limits { max_mouse_step: 50, ..Limits::default() });
        c.permitted.mouse_scale = 0.5;
        assert_eq!(c.motion(15.0, 0.0), Some((8, 0)));
        assert_eq!(c.motion(15.0, 0.0), Some((7, 0)));
        assert_eq!(c.motion(1000.0, -1000.0), Some((50, -50)));
        assert_eq!(sent(&f), ["m8,0", "m7,0", "m50,-50"]);
    }

    #[test]
    fn the_emergency_stop_refuses_and_releases() {
        let (mut c, f) = controller(&["w"], Limits::default());
        // Its own stop counter, so other tests' controllers are not stopped.
        let source: &'static AtomicU64 = Box::leak(Box::new(AtomicU64::new(0)));
        c.stop_source = source;
        c.stops = 0;
        c.press(Control::Key("w"));
        source.fetch_add(1, Ordering::AcqRel);
        assert!(!c.press(Control::Key("w")) || c.held().contains(&"w"));
        c.tick();
        assert!(c.held().is_empty());
        assert!(c.motion(5.0, 0.0).is_none());
        assert_eq!(sent(&f), ["+w", "-w"]);
    }
}
