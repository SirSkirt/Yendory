//! Recording a person's play: at the profile's rate, the game's picture and the controls active
//! since the previous picture, while the game is in front (docs/GAME_TRAINING.md, phase 3).
//!
//! Only the profile's controls are recorded (other keys are never written down, so typing
//! elsewhere is not kept), and only while the game is in front: switching away pauses the
//! recording, and those moments are left out. It ends on the Stop button, the emergency stop
//! keys (the recording is kept), the game's window closing, or the disk refusing to take more.
//! There is no length limit: a recording runs as long as the person plays.
use super::profile::Profile;
use super::recording::{self, control_line, held_line, motion_line, Info, Split, Tick, Writer, AFTER_GAP};
use super::safety::STOPS;
use super::sys::{Desktop, Heard, Input};
use crate::Res;
use serde::Serialize;
use std::collections::{BTreeSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub recording: bool,
    pub in_front: bool,
    pub ms: u64,
    pub ticks: u64,
    /// Pictures a second over the last second.
    pub fps: f64,
    pub bytes: u64,
    pub held: Vec<String>,
}

/// A new recording's id: its start time, readable and sortable.
pub fn new_id() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}-{:03}", now.as_secs(), now.subsec_millis())
}

pub struct Setup<'a> {
    pub data: &'a Path,
    pub profile: &'a Profile,
    pub window: u64,
    pub split: Split,
    pub notes: String,
}

/// Records until stopped; returns the finished recording's info.
pub fn record(
    s: Setup,
    desk: &mut dyn Desktop,
    heard: Receiver<Heard>,
    stop: &AtomicBool,
    mut status: impl FnMut(&Status),
) -> Res<Info> {
    let p = s.profile;
    let channels: u8 = if p.capture.gray { 1 } else { 3 };
    let info = Info {
        format: recording::FORMAT,
        id: new_id(),
        profile: p.id.clone(),
        game: p.name.clone(),
        player: "person".into(),
        split: s.split,
        notes: s.notes,
        started: SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        width: p.capture.width,
        height: p.capture.height,
        channels,
        fps: p.capture.fps,
        keys: p.keys.clone(),
        buttons: p.buttons.clone(),
        mouse: p.mouse,
        ticks: 0,
        duration_ms: 0,
        paused_ms: 0,
        dropped_ticks: 0,
        bytes: 0,
        complete: false,
        ended: String::new(),
    };
    let mut w = Writer::create(s.data, info)?;
    let stops = STOPS.load(Ordering::Acquire);
    let start = Instant::now();
    let period = Duration::from_micros(1_000_000 / p.capture.fps.max(1) as u64);
    let us = |t: Instant| t.saturating_duration_since(start).as_micros() as u64;

    let mut held: BTreeSet<String> = BTreeSet::new();
    let mut active: BTreeSet<String> = BTreeSet::new();
    let (mut dx, mut dy) = (0f64, 0f64);
    let mut pending: VecDeque<Heard> = VecDeque::new();
    // The time of the last tick written: an input heard after it but from before it is late.
    let mut written_us: Option<u64> = None;
    let mut was_in_front = false;
    let mut paused_since: Option<Instant> = Some(start);
    let mut paused = Duration::ZERO;
    let mut dropped = 0u64;
    let mut failures = 0u32;
    let mut recent: VecDeque<Instant> = VecDeque::new();
    let mut last_status = start;
    let mut k: u32 = 0;
    let ended = loop {
        // The next tick, skipping any that are already late.
        k += 1;
        let mut due = start + period * k;
        let now = Instant::now();
        if now > due + period {
            let behind = ((now - due).as_micros() / period.as_micros().max(1)) as u32;
            dropped += behind as u64;
            k += behind;
            due = start + period * k;
        }
        while Instant::now() < due {
            if stop.load(Ordering::Acquire) || STOPS.load(Ordering::Acquire) != stops {
                break;
            }
            std::thread::sleep((due - Instant::now()).min(Duration::from_millis(5)));
        }
        if stop.load(Ordering::Acquire) {
            break "stopped".to_string();
        }
        if STOPS.load(Ordering::Acquire) != stops {
            break "ended with the emergency stop keys".to_string();
        }

        let now = Instant::now();
        pending.extend(heard.try_iter());
        let in_front = desk.in_front(s.window);
        // Every input heard up to now changes what is held; while the game is in front it is
        // also written down and counts for this tick.
        while pending.front().is_some_and(|(t, _)| *t <= now) {
            let (t, input) = pending.pop_front().unwrap();
            // Heard only after its tick was written (the system was slow to pass it on): it
            // counts in this tick, and its line says so: the time it counts at, and when it
            // happened (`late`), so checking rebuilds every tick as recorded.
            let happened = us(t);
            let at = match written_us {
                Some(w) if happened <= w => w + 1,
                _ => happened,
            };
            let line = match input {
                Input::Key { key, down } if p.keys.iter().any(|k| k == key) => {
                    update(&mut held, &mut active, key, down);
                    Some(control_line(at, "k", key, down))
                }
                Input::Button { button, down } if p.buttons.contains(&button) => {
                    update(&mut held, &mut active, button.name(), down);
                    Some(control_line(at, "b", button.name(), down))
                }
                Input::Motion { dx: x, dy: y } if p.mouse => {
                    dx += x;
                    dy += y;
                    Some(motion_line(at, x, y))
                }
                _ => None,
            };
            if let (Some(mut line), true) = (line, in_front && was_in_front) {
                if at != happened {
                    line["late"] = happened.into();
                }
                w.event(&line)?;
            }
        }
        if !in_front {
            if was_in_front {
                paused_since = Some(now);
            }
            was_in_front = false;
            if last_status.elapsed() >= Duration::from_millis(500) {
                last_status = Instant::now();
                status(&Status {
                    recording: true,
                    in_front: false,
                    ms: us(now) / 1000,
                    ticks: w.info.ticks,
                    fps: 0.0,
                    bytes: w.bytes(),
                    held: held.iter().cloned().collect(),
                });
            }
            continue;
        }
        let mut flags = 0;
        if !was_in_front {
            // (Re)starting: what is held now begins the next stretch; what happened while away
            // is not kept.
            if let Some(since) = paused_since.take() {
                paused += now - since;
            }
            active = held.clone();
            dx = 0.0;
            dy = 0.0;
            w.event(&held_line(us(now), &held))?;
            flags = AFTER_GAP;
            was_in_front = true;
        }
        let pixels = match desk.grab(s.window, p.capture.width, p.capture.height) {
            Ok(rgb) => {
                failures = 0;
                if p.capture.gray {
                    super::to_gray(&rgb)
                } else {
                    rgb
                }
            }
            Err(e) => {
                failures += 1;
                dropped += 1;
                if failures >= 40 {
                    break format!("the game's picture could not be taken ({e})");
                }
                // The controls carry over to the next tick that has a picture.
                continue;
            }
        };
        let mut keys = 0u64;
        let mut buttons = 0u8;
        for name in &active {
            if let Some(b) = p.keys.iter().position(|k| k == name) {
                keys |= 1 << b;
            } else if let Some(b) = p.buttons.iter().position(|x| x.name() == name) {
                buttons |= 1 << b;
            }
        }
        if let Err(e) = w.tick(
            Tick { t_us: us(now), keys, buttons, dx: dx as f32, dy: dy as f32, flags, ..Tick::default() },
            &pixels,
        ) {
            // Usually a full disk: what was written so far is kept.
            break format!("the recording could not be written any further ({e})");
        }
        written_us = Some(us(now));
        active = held.clone();
        dx = 0.0;
        dy = 0.0;
        recent.push_back(now);
        while recent.front().is_some_and(|t| now.duration_since(*t) > Duration::from_secs(1)) {
            recent.pop_front();
        }
        if last_status.elapsed() >= Duration::from_millis(500) {
            last_status = Instant::now();
            status(&Status {
                recording: true,
                in_front: true,
                ms: us(now) / 1000,
                ticks: w.info.ticks,
                fps: recent.len() as f64,
                bytes: w.bytes(),
                held: held.iter().cloned().collect(),
            });
        }
    };
    let now = Instant::now();
    if let Some(since) = paused_since {
        paused += now - since;
    }
    w.info.duration_ms = us(now) / 1000;
    w.info.paused_ms = paused.as_millis() as u64;
    w.info.dropped_ticks = dropped;
    let info = w.finish(&ended)?;
    status(&Status {
        recording: false,
        in_front: false,
        ms: info.duration_ms,
        ticks: info.ticks,
        fps: 0.0,
        bytes: info.bytes,
        held: vec![],
    });
    Ok(info)
}

fn update(held: &mut BTreeSet<String>, active: &mut BTreeSet<String>, name: &str, down: bool) {
    if down {
        held.insert(name.to_string());
        active.insert(name.to_string());
    } else {
        held.remove(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::controls::Button;
    use crate::live::recording::{open, validate};
    use crate::live::safety::tests::Fake;
    use std::sync::mpsc;

    #[test]
    fn a_session_is_recorded_in_sync_and_checks_out() {
        let data = std::env::temp_dir().join(format!("tools-record-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        let mut profile = crate::live::profile::range_profile();
        profile.capture.width = 16;
        profile.capture.height = 9;
        profile.capture.fps = 50;
        let fake = Fake::default();
        *fake.front.lock().unwrap() = true;
        let (tx, rx) = mpsc::channel();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let (stop2, front) = (stop.clone(), fake.front.clone());
        // The "person": W for 200 ms, a quick click, a mouse move; E (not permitted); a pause
        // (another window in front) with S pressed meanwhile; then stop.
        let person = std::thread::spawn(move || {
            let at = |ms| std::thread::sleep(Duration::from_millis(ms));
            let send = |i: Input| tx.send((Instant::now(), i)).unwrap();
            at(60);
            send(Input::Key { key: "w", down: true });
            at(200);
            send(Input::Key { key: "w", down: false });
            send(Input::Button { button: Button::Left, down: true });
            at(5);
            send(Input::Button { button: Button::Left, down: false });
            send(Input::Motion { dx: 7.0, dy: -2.0 });
            send(Input::Key { key: "e", down: true });
            at(60);
            *front.lock().unwrap() = false;
            at(60);
            send(Input::Key { key: "s", down: true });
            at(60);
            *front.lock().unwrap() = true;
            at(100);
            send(Input::Key { key: "s", down: false });
            at(60);
            stop2.store(true, Ordering::Release);
        });
        let mut desk = fake.clone();
        let setup =
            Setup { data: &data, profile: &profile, window: 1, split: Split::Train, notes: "t".into() };
        let info = record(setup, &mut desk, rx, &stop, |_| {}).unwrap();
        person.join().unwrap();
        assert!(info.complete);
        assert_eq!(info.ended, "stopped");
        assert!(info.paused_ms >= 100, "{info:?}");
        let v = validate(&data, &profile.id, &info.id).unwrap();
        assert!(v.ok, "{v:?}");
        let count = |n: &str| v.active_ticks.iter().find(|x| x.0 == n).unwrap().1;
        // W for 200 ms at 50 a second: about 10 ticks; the 5 ms click: one tick.
        assert!((9..=12).contains(&count("w")), "{v:?}");
        // (Two when it straddles a tick: held at one tick's end, so active in the next.)
        assert!((1..=2).contains(&count("left")), "{v:?}");
        assert!((v.mouse_x - 7.0).abs() < 1e-9 && (v.mouse_y + 2.0).abs() < 1e-9);
        // S was pressed during the pause: it is held after it (it began the next stretch).
        assert!(count("s") >= 3);
        let r = open(&data, &profile.id, &info.id).unwrap();
        assert_eq!(r.ticks.iter().filter(|t| t.flags & AFTER_GAP != 0).count(), 2);
        let events = std::fs::read_to_string(r.events_path()).unwrap();
        assert!(!events.contains("\"e\""), "a key outside the profile was written down");
        let _ = std::fs::remove_dir_all(&data);
    }

    #[test]
    fn an_input_heard_after_its_tick_counts_in_the_next_and_checks_out() {
        let data = std::env::temp_dir().join(format!("tools-record-late-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        let mut profile = crate::live::profile::range_profile();
        profile.capture.width = 16;
        profile.capture.height = 9;
        profile.capture.fps = 50;
        let fake = Fake::default();
        *fake.front.lock().unwrap() = true;
        let (tx, rx) = mpsc::channel();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        // Inputs that happened 100 ms (five ticks) before they are passed on: a mouse movement,
        // then a click (down and up), as a slow system delivers them.
        let person = std::thread::spawn(move || {
            let at = |ms| std::thread::sleep(Duration::from_millis(ms));
            let late = |i: Input| tx.send((Instant::now() - Duration::from_millis(100), i)).unwrap();
            at(200);
            late(Input::Motion { dx: 30.0, dy: 0.0 });
            at(100);
            late(Input::Button { button: Button::Left, down: true });
            late(Input::Button { button: Button::Left, down: false });
            at(100);
            stop2.store(true, Ordering::Release);
        });
        let mut desk = fake.clone();
        let setup =
            Setup { data: &data, profile: &profile, window: 1, split: Split::Train, notes: "t".into() };
        let info = record(setup, &mut desk, rx, &stop, |_| {}).unwrap();
        person.join().unwrap();
        let v = validate(&data, &profile.id, &info.id).unwrap();
        assert!(v.ok, "{v:?}");
        // Nothing is lost: the movement and the click are each in exactly one tick.
        assert!((v.mouse_x - 30.0).abs() < 1e-9, "{v:?}");
        assert_eq!(v.active_ticks.iter().find(|x| x.0 == "left").unwrap().1, 1, "{v:?}");
        // Their lines keep when they happened.
        let r = open(&data, &profile.id, &info.id).unwrap();
        let events = std::fs::read_to_string(r.events_path()).unwrap();
        let lines: Vec<serde_json::Value> =
            events.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let late: Vec<&serde_json::Value> = lines.iter().filter(|l| l.get("late").is_some()).collect();
        assert_eq!(late.len(), 3, "{events}");
        for l in late {
            assert!(l["late"].as_u64().unwrap() < l["t"].as_u64().unwrap(), "{l}");
        }
        let _ = std::fs::remove_dir_all(&data);
    }
}
