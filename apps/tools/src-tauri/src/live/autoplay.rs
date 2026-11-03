//! A network playing a game by itself: every tick the game's picture goes in, the network's
//! decision comes out, and the safety layer turns it into presses, releases and mouse movement
//! (docs/GAME_TRAINING.md, section 4). Its play can be recorded like a person's (marked as the
//! model's, never used as a person's demonstration).
//!
//! The deciders: the network ([`Brain::Net`]), and for comparison the **frequency baseline**
//! ([`Brain::Frequencies`]), which uses each control as often as the person did but never looks
//! at the picture. Nothing here knows any game.
use super::policy::{encode, sigmoid, softmax, Policy, Rng};
use super::profile::Profile;
use super::recording::{self, control_line, held_line, motion_line, Info, Split, Tick, Writer, AFTER_GAP};
use super::safety::Controller;
use super::sys::Desktop;
use super::train::Frequencies;
use crate::Res;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// How a decision is taken from the network's probabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Decide {
    /// The most likely: a control is used when its probability is above one half; the most
    /// likely mouse step.
    Likely,
    /// Drawn from the probabilities (more varied).
    Sampled,
}

/// One tick's decision.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Action {
    pub keys: u64,
    pub buttons: u8,
    pub dx: f32,
    pub dy: f32,
}

pub enum Brain {
    /// A skill's network; with `cortex`, a linked skill playing with its model's core (the
    /// core reads what the skill sees every tick and its state is the goal input).
    Net {
        policy: Policy,
        memory: Vec<f32>,
        pictures: VecDeque<Vec<u8>>,
        cortex: Option<Box<super::cortex::Cortex>>,
    },
    Frequencies {
        freq: Frequencies,
        bins: Vec<f32>,
    },
}

impl Brain {
    /// A skill playing alone (a linked skill with its core cut: the goal input stays zero).
    pub fn net(policy: Policy) -> Brain {
        Brain::Net { memory: vec![0.0; policy.arch.memory], policy, pictures: VecDeque::new(), cortex: None }
    }

    /// A linked skill playing with its model's core.
    pub fn with_core(policy: Policy, cortex: super::cortex::Cortex) -> Res<Brain> {
        yendory_engine::cortex::fits(&policy, &cortex.core)?;
        Ok(Brain::Net {
            memory: vec![0.0; policy.arch.memory],
            policy,
            pictures: VecDeque::new(),
            cortex: Some(Box::new(cortex)),
        })
    }

    /// Forgets what was seen (a new episode).
    pub fn reset(&mut self) {
        if let Brain::Net { policy, memory, pictures, cortex } = self {
            *memory = vec![0.0; policy.arch.memory];
            pictures.clear();
            if let Some(c) = cortex {
                c.reset();
            }
        }
    }

    pub fn act(&mut self, picture: &[u8], decide: Decide, rng: &mut Rng) -> Action {
        let on = |p: f32, rng: &mut Rng| match decide {
            Decide::Likely => p > 0.5,
            Decide::Sampled => rng.unit() < p,
        };
        let pick = |p: &[f32], rng: &mut Rng| match decide {
            Decide::Likely => {
                p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0).unwrap_or(0)
            }
            Decide::Sampled => {
                let mut r = rng.unit();
                for (i, v) in p.iter().enumerate() {
                    r -= v;
                    if r < 0.0 {
                        return i;
                    }
                }
                p.len().saturating_sub(1)
            }
        };
        match self {
            Brain::Net { policy, memory, pictures, cortex } => {
                let k = policy.arch.frames;
                if pictures.is_empty() {
                    for _ in 0..k {
                        pictures.push_back(picture.to_vec());
                    }
                } else {
                    pictures.push_back(picture.to_vec());
                    while pictures.len() > k {
                        pictures.pop_front();
                    }
                }
                let stack: Vec<&[u8]> = pictures.iter().map(|p| p.as_slice()).collect();
                let seen = policy.see(&encode(&policy.arch, &stack));
                let goal = match cortex {
                    // If the core fails (it shouldn't), the skill plays on alone.
                    Some(c) => {
                        c.goal(policy, policy.sight(&seen)).unwrap_or_else(|_| vec![0f32; policy.arch.goal])
                    }
                    None => vec![0f32; policy.arch.goal],
                };
                let cache = policy.forward_seen(seen, memory, &goal);
                *memory = cache.m;
                let l = cache.logits;
                let mut a = Action::default();
                for (i, &z) in l.keys.iter().enumerate() {
                    if on(sigmoid(z), rng) {
                        a.keys |= 1 << i;
                    }
                }
                for (i, &z) in l.buttons.iter().enumerate() {
                    if on(sigmoid(z), rng) {
                        a.buttons |= 1 << i;
                    }
                }
                let bins = &policy.arch.mouse_bins;
                if !bins.is_empty() {
                    a.dx = bins[pick(&softmax(&l.mx), rng)];
                    a.dy = bins[pick(&softmax(&l.my), rng)];
                }
                a
            }
            Brain::Frequencies { freq, bins } => {
                // Ignores the picture: each control as often as the person used it.
                let mut a = Action::default();
                for (i, &p) in freq.keys.iter().enumerate() {
                    if rng.unit() < p as f32 {
                        a.keys |= 1 << i;
                    }
                }
                for (i, &p) in freq.buttons.iter().enumerate() {
                    if rng.unit() < p as f32 {
                        a.buttons |= 1 << i;
                    }
                }
                if !bins.is_empty() {
                    let mx: Vec<f32> = freq.mx.iter().map(|&v| v as f32).collect();
                    let my: Vec<f32> = freq.my.iter().map(|&v| v as f32).collect();
                    let draw = |p: &[f32], rng: &mut Rng| {
                        let mut r = rng.unit();
                        for (i, v) in p.iter().enumerate() {
                            r -= v;
                            if r < 0.0 {
                                return i;
                            }
                        }
                        p.len().saturating_sub(1)
                    };
                    a.dx = bins[draw(&mx, rng)];
                    a.dy = bins[draw(&my, rng)];
                }
                a
            }
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub ms: u64,
    pub ticks: u64,
    pub fps: f64,
    pub held: Vec<String>,
    pub mouse: (f32, f32),
    /// Decisions the safety layer refused so far.
    pub refused: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Played {
    pub ticks: u64,
    pub ms: u64,
    pub ended: String,
    /// The recording of this play, if it was recorded.
    pub recording: Option<String>,
    pub refused: u64,
}

pub struct Setup<'a> {
    pub data: &'a Path,
    pub profile: &'a Profile,
    pub window: u64,
    pub decide: Decide,
    /// Record the play under this player's name (None: not recorded).
    pub record_as: Option<String>,
    /// Stop after this long (None: until stopped).
    pub seconds: Option<f64>,
    pub seed: u64,
}

/// Plays until stopped, the time is up, or the safety layer pauses (the game left the front).
pub fn play(
    s: Setup,
    brain: &mut Brain,
    desk: &mut dyn Desktop,
    controller: &Mutex<Controller>,
    stop: &AtomicBool,
    mut status: impl FnMut(&Status),
) -> Res<Played> {
    let p = s.profile;
    let period = Duration::from_micros(1_000_000 / p.capture.fps.max(1) as u64);
    let mut rng = Rng::new(s.seed);
    let mut writer = match &s.record_as {
        Some(who) => Some(Writer::create(
            s.data,
            Info {
                format: recording::FORMAT,
                id: super::record::new_id(),
                profile: p.id.clone(),
                game: p.name.clone(),
                player: who.clone(),
                split: Split::Test,
                notes: format!("played by {who}"),
                started: SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
                width: p.capture.width,
                height: p.capture.height,
                channels: if p.capture.gray { 1 } else { 3 },
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
            },
        )?),
        None => None,
    };
    let start = Instant::now();
    let us = |t: Instant| t.saturating_duration_since(start).as_micros() as u64;
    if let Some(w) = writer.as_mut() {
        w.event(&held_line(0, &BTreeSet::new()))?;
    }
    // What was held and used, for the recording (as a person's recording counts it).
    let mut held: BTreeSet<String> = BTreeSet::new();
    let mut active: BTreeSet<String> = BTreeSet::new();
    let (mut mdx, mut mdy) = (0f32, 0f32);
    let mut ticks = 0u64;
    let mut refused_before = 0usize;
    let mut recent: VecDeque<Instant> = VecDeque::new();
    let mut last_status = start;
    let mut k: u32 = 0;
    let mut failures = 0;
    let mut last_mouse = (0f32, 0f32);
    let ended = loop {
        k += 1;
        let mut due = start + period * k;
        if Instant::now() > due + period {
            k = (Instant::now().duration_since(start).as_micros() / period.as_micros().max(1)) as u32 + 1;
            due = start + period * k;
        }
        while Instant::now() < due {
            if stop.load(Ordering::Acquire) {
                break;
            }
            {
                let mut c = controller.lock().unwrap();
                c.tick();
            }
            std::thread::sleep((due - Instant::now()).min(Duration::from_millis(5)));
        }
        if stop.load(Ordering::Acquire) {
            break "stopped".to_string();
        }
        if s.seconds.is_some_and(|secs| start.elapsed().as_secs_f64() >= secs) {
            break "the time was up".to_string();
        }
        {
            let c = controller.lock().unwrap();
            if c.stopped() {
                break "ended with the emergency stop".to_string();
            }
            if let Some(why) = &c.paused {
                break format!("the safety layer paused it ({why})");
            }
        }
        let picture = match desk.grab(s.window, p.capture.width, p.capture.height) {
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
                if failures > 40 {
                    break format!("the game's picture could not be taken ({e})");
                }
                continue;
            }
        };
        let now = Instant::now();
        if let Some(w) = writer.as_mut() {
            let (keys, buttons) = bits(p, &active);
            let flags = if ticks == 0 { AFTER_GAP } else { 0 };
            w.tick(
                Tick { t_us: us(now), keys, buttons, dx: mdx, dy: mdy, flags, ..Tick::default() },
                &picture,
            )?;
        }
        active = held.clone();
        mdx = 0.0;
        mdy = 0.0;
        ticks += 1;

        let action = brain.act(&picture, s.decide, &mut rng);
        let mut c = controller.lock().unwrap();
        let mut lines = Vec::new();
        let want = |name: &str| -> bool {
            if let Some(i) = p.keys.iter().position(|k| k == name) {
                return (action.keys >> i) & 1 == 1;
            }
            if let Some(i) = p.buttons.iter().position(|b| b.name() == name) {
                return (action.buttons >> i) & 1 == 1;
            }
            false
        };
        let names: Vec<(String, super::controls::Control, &str)> = p
            .keys
            .iter()
            .filter_map(|k| super::controls::Control::parse(k).map(|c| (k.clone(), c, "k")))
            .chain(
                p.buttons.iter().map(|b| (b.name().to_string(), super::controls::Control::Button(*b), "b")),
            )
            .collect();
        for (name, control, kind) in &names {
            let before = c.held().contains(&control.name());
            if want(name) && !before {
                c.press(*control);
            } else if !want(name) && before {
                c.release(*control, "");
            }
            let after = c.held().contains(&control.name());
            if after != before {
                let t = us(Instant::now());
                lines.push(control_line(t, kind, name, after));
                if after {
                    held.insert(name.clone());
                    active.insert(name.clone());
                } else {
                    held.remove(name);
                }
            }
        }
        if (action.dx != 0.0 || action.dy != 0.0) && p.mouse {
            if let Some((sx, sy)) = c.motion(action.dx, action.dy) {
                if sx != 0 || sy != 0 {
                    lines.push(motion_line(us(Instant::now()), sx as f64, sy as f64));
                    mdx += sx as f32;
                    mdy += sy as f32;
                }
                last_mouse = (sx as f32, sy as f32);
            }
        }
        // Releases by the safety layer (a hold too long) count too.
        let truly: BTreeSet<String> = c.held().iter().map(|s| s.to_string()).collect();
        for gone in held.difference(&truly).cloned().collect::<Vec<_>>() {
            let kind = if p.keys.contains(&gone) { "k" } else { "b" };
            lines.push(control_line(us(Instant::now()), kind, &gone, false));
            held.remove(&gone);
        }
        let refused = c.log.iter().filter(|e| !e.sent).count();
        drop(c);
        if let Some(w) = writer.as_mut() {
            for l in &lines {
                w.event(l)?;
            }
        }
        recent.push_back(now);
        while recent.front().is_some_and(|t| now.duration_since(*t) > Duration::from_secs(1)) {
            recent.pop_front();
        }
        if last_status.elapsed() >= Duration::from_millis(500) {
            last_status = Instant::now();
            refused_before = refused;
            status(&Status {
                ms: us(now) / 1000,
                ticks,
                fps: recent.len() as f64,
                held: held.iter().cloned().collect(),
                mouse: last_mouse,
                refused: refused as u64,
            });
        }
    };
    let refused = {
        let mut c = controller.lock().unwrap();
        c.release_all("play ended");
        c.log.iter().filter(|e| !e.sent).count().max(refused_before)
    };
    let ms = start.elapsed().as_millis() as u64;
    let recording = match writer {
        Some(mut w) => {
            w.info.duration_ms = ms;
            Some(w.finish(&ended)?.id)
        }
        None => None,
    };
    Ok(Played { ticks, ms, ended, recording, refused: refused as u64 })
}

fn bits(p: &Profile, names: &BTreeSet<String>) -> (u64, u8) {
    let mut keys = 0u64;
    let mut buttons = 0u8;
    for n in names {
        if let Some(i) = p.keys.iter().position(|k| k == n) {
            keys |= 1 << i;
        } else if let Some(i) = p.buttons.iter().position(|b| b.name() == n) {
            buttons |= 1 << i;
        }
    }
    (keys, buttons)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::policy::{Arch, Size};
    use crate::live::recording::validate;
    use crate::live::safety::tests::Fake;
    use crate::live::safety::Limits;

    #[test]
    fn a_network_plays_through_the_safety_layer_and_its_play_is_recorded() {
        let data = std::env::temp_dir().join(format!("tools-autoplay-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        let mut profile = crate::live::profile::range_profile();
        profile.capture.width = 32;
        profile.capture.height = 18;
        profile.capture.fps = 40;
        let arch = Arch::new(
            32,
            18,
            3,
            profile.keys.len(),
            profile.buttons.len(),
            vec![-4.0, 0.0, 4.0],
            Size::Small,
        );
        // A network whose key biases are large: it holds every key, every tick.
        let mut policy = Policy::new(arch, 1).unwrap();
        let kb = policy.slot("key.b").unwrap().range();
        policy.w[kb].fill(8.0);
        let mut brain = Brain::net(policy);
        let fake = Fake::default();
        *fake.front.lock().unwrap() = true;
        let controller = Mutex::new(Controller::new(
            Box::new(fake.clone()),
            1,
            profile.permitted().unwrap(),
            Limits::default(),
        ));
        let stop = AtomicBool::new(false);
        let mut desk = fake.clone();
        let setup = Setup {
            data: &data,
            profile: &profile,
            window: 1,
            decide: Decide::Likely,
            record_as: Some("nova".into()),
            seconds: Some(0.5),
            seed: 1,
        };
        let played = play(setup, &mut brain, &mut desk, &controller, &stop, |_| {}).unwrap();
        assert_eq!(played.ended, "the time was up");
        assert!(played.ticks >= 15, "{played:?}");
        let sent = fake.sent.lock().unwrap().clone();
        // Every permitted key was pressed once and held; all released at the end.
        for k in &profile.keys {
            assert_eq!(sent.iter().filter(|s| *s == &format!("+{k}")).count(), 1, "{sent:?}");
            assert_eq!(sent.iter().filter(|s| *s == &format!("-{k}")).count(), 1, "{sent:?}");
        }
        assert!(controller.lock().unwrap().held().is_empty());
        // The play's recording checks out, and it is the model's, not a person's.
        let rec = played.recording.unwrap();
        let v = validate(&data, &profile.id, &rec).unwrap();
        assert!(v.ok, "{v:?}");
        let info =
            crate::live::recording::list(&data, &profile.id).into_iter().find(|r| r.id == rec).unwrap();
        assert_eq!(info.player, "nova");
        let w = v.active_ticks.iter().find(|x| x.0 == "w").unwrap().1;
        assert!(w + 1 >= v.ticks, "{v:?}");
        let _ = std::fs::remove_dir_all(&data);
    }

    #[test]
    fn the_safety_layer_still_decides() {
        let mut profile = crate::live::profile::range_profile();
        profile.capture.width = 32;
        profile.capture.height = 18;
        let arch = Arch::new(32, 18, 3, profile.keys.len(), profile.buttons.len(), vec![], Size::Small);
        let mut policy = Policy::new(arch, 1).unwrap();
        let kb = policy.slot("key.b").unwrap().range();
        policy.w[kb].fill(8.0);
        let mut brain = Brain::net(policy);
        let fake = Fake::default();
        // The game is not in front: nothing may be sent, and play ends.
        *fake.front.lock().unwrap() = false;
        let controller = Mutex::new(Controller::new(
            Box::new(fake.clone()),
            1,
            profile.permitted().unwrap(),
            Limits::default(),
        ));
        let mut desk = fake.clone();
        let data = std::env::temp_dir();
        let setup = Setup {
            data: &data,
            profile: &profile,
            window: 1,
            decide: Decide::Likely,
            record_as: None,
            seconds: Some(2.0),
            seed: 1,
        };
        let played =
            play(setup, &mut brain, &mut desk, &controller, &AtomicBool::new(false), |_| {}).unwrap();
        assert!(played.ended.contains("paused"), "{}", played.ended);
        assert!(fake.sent.lock().unwrap().iter().all(|s| !s.starts_with('+')));
    }
}
