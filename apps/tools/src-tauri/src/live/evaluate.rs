//! Measuring play on the test range: each decider plays the same episodes (the same layouts, on
//! seeds no person's recording used), and the range's own count of shots and hits is the score.
//! A network has learned something only if it beats the deciders that can't have: the untrained
//! network (the same architecture, never taught) and the frequency baseline (the person's use of
//! each control, ignoring the picture).
use super::autoplay::{play, Brain, Decide, Setup};
use super::profile::Profile;
use super::safety::Controller;
use super::sys;
use super::RangeProcess;
use crate::Res;
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Layout seeds for measuring: far above those a person reaches by playing (the range starts
/// at 1 and moves on one a minute).
pub const SEEDS_FROM: u64 = 100_000;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Episode {
    pub seed: u64,
    pub hits: u64,
    pub shots: u64,
    /// Presses the range received, and mouse units across (how active the decider was).
    pub presses: u64,
    pub mouse_x: f64,
    /// Straight-line distance from the start (cells).
    pub moved: f64,
    /// The recording of this episode, if it was recorded (as the decider's play).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recording: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Row {
    pub who: String,
    pub episodes: Vec<Episode>,
    pub hits_per_minute: f64,
    pub shots_per_minute: f64,
    /// Hits per shot.
    pub accuracy: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalProgress {
    pub who: String,
    pub episode: usize,
    pub episodes: usize,
    pub hits: u64,
    pub shots: u64,
}

fn num(v: &serde_json::Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or(0.0)
}

pub struct EvalSetup<'a> {
    pub data: &'a Path,
    pub profile: &'a Profile,
    pub window: u64,
    pub episodes: usize,
    pub seconds: f64,
    pub decide: Decide,
    /// The first episode's layout (the others follow it).
    pub first_seed: u64,
    /// Record every episode as the decider's play (never as a person's).
    pub record: bool,
}

/// Plays every episode with every decider (episode by episode, so all meet the same layouts in
/// the same order) and scores them by the range's report.
pub fn evaluate(
    s: EvalSetup,
    range: &mut RangeProcess,
    brains: &mut [(String, Brain)],
    stop: &AtomicBool,
    mut progress: impl FnMut(&EvalProgress),
) -> Res<Vec<Row>> {
    let mut rows: Vec<Row> = brains
        .iter()
        .map(|(who, _)| Row {
            who: who.clone(),
            episodes: vec![],
            hits_per_minute: 0.0,
            shots_per_minute: 0.0,
            accuracy: 0.0,
        })
        .collect();
    'all: for e in 0..s.episodes {
        let seed = s.first_seed + e as u64;
        for (bi, (who, brain)) in brains.iter_mut().enumerate() {
            if stop.load(Ordering::Acquire) {
                break 'all;
            }
            let mut desk = sys::open()?;
            desk.bring_to_front(s.window)?;
            for _ in 0..50 {
                if desk.in_front(s.window) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(30));
            }
            range.reset(seed)?;
            std::thread::sleep(Duration::from_millis(250));
            let before = range.state()?;
            brain.reset();
            let controller = Mutex::new(Controller::new(
                sys::open()?,
                s.window,
                s.profile.permitted()?,
                s.profile.limits.clone(),
            ));
            let setup = Setup {
                data: s.data,
                profile: s.profile,
                window: s.window,
                decide: s.decide,
                record_as: s.record.then(|| who.clone()),
                seconds: Some(s.seconds),
                seed: seed ^ bi as u64,
            };
            let played = play(setup, brain, desk.as_mut(), &controller, stop, |_| {})?;
            drop(controller);
            std::thread::sleep(Duration::from_millis(150));
            let after = range.state()?;
            if played.ended != "the time was up" && !stop.load(Ordering::Acquire) {
                return Err(format!("An episode ended early: {}", played.ended));
            }
            let ep = Episode {
                seed,
                hits: (num(&after, "hits") - num(&before, "hits")).max(0.0) as u64,
                shots: (num(&after, "shots") - num(&before, "shots")).max(0.0) as u64,
                presses: (num(&after, "presses") - num(&before, "presses")).max(0.0) as u64,
                mouse_x: num(&after, "mouseX") - num(&before, "mouseX"),
                moved: (num(&after, "x") - num(&before, "x")).hypot(num(&after, "y") - num(&before, "y")),
                recording: played.recording.clone(),
            };
            progress(&EvalProgress {
                who: who.clone(),
                episode: e + 1,
                episodes: s.episodes,
                hits: ep.hits,
                shots: ep.shots,
            });
            rows[bi].episodes.push(ep);
        }
    }
    for r in &mut rows {
        let minutes = (r.episodes.len() as f64 * s.seconds / 60.0).max(1e-9);
        let hits: u64 = r.episodes.iter().map(|e| e.hits).sum();
        let shots: u64 = r.episodes.iter().map(|e| e.shots).sum();
        r.hits_per_minute = hits as f64 / minutes;
        r.shots_per_minute = shots as f64 / minutes;
        r.accuracy = if shots == 0 { 0.0 } else { hits as f64 / shots as f64 };
    }
    Ok(rows)
}
