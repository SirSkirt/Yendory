//! Generations: Gaming Nova learning from its own play, after the person's demonstrations taught
//! it the basics (docs/GAME_TRAINING.md, section 5).
//!
//! Each generation:
//!
//! 1. **Candidates**: the champion (the skill's accepted weights) and copies of it with random
//!    changes to their weights, each a different size (candidate 1 is the champion unchanged).
//! 2. **Play**: every candidate plays the same new layouts, drawing its decisions from its
//!    probabilities; its play is recorded as the model's (never as a person's).
//! 3. **Selection**: the game's score picks the best episodes of all of them (on the range: its
//!    count of hits). The score only chooses which play to learn from; it is never an input to
//!    the network.
//! 4. **Learning**: the best candidate learns from those episodes (what it did after each
//!    picture), mixed with the person's recordings so the basics aren't forgotten.
//! 5. **Test**: the new network and the champion play the same layouts nobody has played, taking
//!    their most likely decisions. Only a better score replaces the champion (it is protected:
//!    a generation that does worse changes nothing).
//!
//! The champion's lineage counts the generations it won; every attempt is kept in the skill's
//! notes (`generations`). Nothing here knows any game: the score comes from outside (the range
//! reports its own hits), so a real game needs its score read from the screen (section 5,
//! phase 5) before it can use this.
use super::autoplay::{Brain, Decide};
use super::evaluate::{evaluate, Episode, EvalSetup, Row};
use super::policy::{Adam, Policy, Rng, Skill};
use super::profile::Profile;
use super::recording::{self, Split};
use super::train::{self, Clip};
use super::RangeProcess;
use crate::Res;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Layouts for the candidates' play, and for the tests, by attempt (far from the person's and the
/// measuring page's).
pub const PLAY_SEEDS_FROM: u64 = 200_000;
pub const TEST_SEEDS_FROM: u64 = 400_000;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct GenSettings {
    /// Generations to run (it can be stopped sooner).
    pub generations: u32,
    /// Players in each generation, the champion unchanged among them.
    pub candidates: usize,
    /// Episodes each candidate plays, and their length (seconds).
    pub episodes: usize,
    pub seconds: f64,
    /// The largest random change to the weights, as a share of each layer's typical weight.
    pub noise: f32,
    /// The share of all episodes (the best) learned from.
    pub elite: f64,
    /// Learning steps from those episodes.
    pub steps: u32,
    pub learning_rate: f32,
    /// The share of each learning batch from the person's recordings (the rest from the episodes).
    pub demonstrations: f64,
    /// Episodes in the test of the new network against the champion.
    pub test_episodes: usize,
}

impl Default for GenSettings {
    fn default() -> Self {
        GenSettings {
            generations: 10,
            candidates: 4,
            episodes: 2,
            seconds: 30.0,
            noise: 0.05,
            elite: 0.25,
            steps: 300,
            learning_rate: 3e-4,
            demonstrations: 0.5,
            test_episodes: 4,
        }
    }
}

impl GenSettings {
    pub fn check(&self) -> Res<()> {
        if !(1..=1000).contains(&self.generations)
            || !(1..=8).contains(&self.candidates)
            || !(1..=10).contains(&self.episodes)
            || !(5.0..=55.0).contains(&self.seconds)
            || !(0.0..=1.0).contains(&self.noise)
            || !(0.01..=1.0).contains(&self.elite)
            || !(1..=100_000).contains(&self.steps)
            || !(1e-6..=1e-1).contains(&self.learning_rate)
            || !(0.0..=0.95).contains(&self.demonstrations)
            || !(1..=20).contains(&self.test_episodes)
        {
            return Err("A generation setting is out of range".into());
        }
        Ok(())
    }

    /// Seconds one generation takes in play (learning comes on top).
    pub fn play_seconds(&self) -> f64 {
        (self.candidates * self.episodes + 2 * self.test_episodes) as f64 * (self.seconds + 0.5)
    }
}

/// One generation's attempt, as kept in the skill's notes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Attempt {
    /// Counts every attempt (it picks the layouts).
    pub attempt: u64,
    /// The champion's generation it competed for.
    pub generation: u32,
    pub started: u64,
    pub settings: GenSettings,
    /// Each candidate's mean score per episode (candidate 1: the champion unchanged).
    pub candidates: Vec<f64>,
    /// The episodes learned from: their recordings and scores.
    pub elite: Vec<(String, u64)>,
    pub learned_steps: u32,
    /// The test: hits a minute of the champion and of the new network.
    pub champion_score: f64,
    pub child_score: f64,
    pub accepted: bool,
    /// Why it ended without a test, if it did.
    pub note: String,
}

/// The attempts recorded in a skill's notes.
pub fn attempts(skill: &Skill) -> Vec<Attempt> {
    skill
        .spec
        .notes
        .get("generations")
        .and_then(|r| serde_json::from_value(r.clone()).ok())
        .unwrap_or_default()
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenProgress {
    /// Which generation of this run (1-based) of how many.
    pub run: u32,
    pub of: u32,
    /// `playing`, `learning`, `testing`, `done`.
    pub phase: String,
    pub detail: String,
    /// The finished attempt (with `done`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attempt: Option<Attempt>,
}

/// A copy of the weights, each layer changed by random amounts of `scale` times its typical
/// weight (layers still all zero are left).
pub fn perturb(p: &Policy, scale: f32, rng: &mut Rng) -> Policy {
    let mut q = p.clone();
    if scale <= 0.0 {
        return q;
    }
    for s in &p.slots {
        let r = s.range();
        let n = r.len().max(1) as f32;
        let rms = (p.w[r.clone()].iter().map(|v| v * v).sum::<f32>() / n).sqrt();
        if rms == 0.0 {
            continue;
        }
        for v in &mut q.w[r] {
            *v += rng.normal() * rms * scale;
        }
    }
    q
}

/// The best `share` of the episodes (by score; only those that scored), best first.
pub fn select(episodes: &[Episode], share: f64) -> Vec<&Episode> {
    let mut scored: Vec<&Episode> = episodes.iter().filter(|e| e.hits > 0).collect();
    scored.sort_by(|a, b| b.hits.cmp(&a.hits).then(a.shots.cmp(&b.shots)));
    let keep = ((episodes.len() as f64 * share).ceil() as usize).max(1);
    scored.truncate(keep);
    scored
}

/// What runs the game while generations play: the range (or later a game with a score).
pub struct Arena<'a> {
    pub data: &'a Path,
    pub profile: &'a Profile,
    pub window: u64,
    pub range: &'a Mutex<Option<RangeProcess>>,
    /// The name the play is recorded under.
    pub name: String,
}

fn with_range<T>(arena: &Arena, f: impl FnOnce(&mut RangeProcess) -> Res<T>) -> Res<T> {
    let mut slot = arena.range.lock().unwrap();
    match super::alive(&mut slot) {
        Some(r) => f(r),
        None => Err("The range closed".into()),
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Runs generations on the skill; every attempt is saved (with `save`) as it ends. Returns the
/// skill with the champion that stood at the end.
#[allow(clippy::too_many_arguments)]
pub fn run(
    arena: &Arena,
    mut skill: Skill,
    settings: GenSettings,
    core: Option<yendory_engine::Bicameral>,
    stop: &AtomicBool,
    mut progress: impl FnMut(&GenProgress),
    mut save: impl FnMut(&Skill) -> Res<()>,
) -> Res<Skill> {
    settings.check()?;
    super::skills::fits(&skill.spec, arena.profile)?;
    let demos = train::load_clips(arena.data, &skill.spec, Split::Train)?;
    // The person's offset (from their last training run), so their recordings mean the same.
    let demo_offset = train::runs(&skill.spec).last().map(|r| r.settings.offset).unwrap_or(1);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 16);
    // A linked skill plays and learns with its model's core (frozen here: generations change
    // the skill and its link), in the stretches it was trained with.
    let stretch = train::runs(&skill.spec).last().map(|r| r.settings.length).unwrap_or(8);
    let learn_settings = train::Settings {
        learning_rate: settings.learning_rate,
        length: if core.is_some() { stretch } else { train::Settings::default().length },
        ..train::Settings::default()
    };
    if let Some(c) = &core {
        yendory_engine::cortex::fits(&skill.policy()?, c)?;
    }
    let brain = |p: Policy| -> Res<Brain> {
        match &core {
            Some(c) => Brain::with_core(p, super::cortex::Cortex::new(c.clone(), stretch)?),
            None => Ok(Brain::net(p)),
        }
    };
    let mut history = attempts(&skill);
    for g in 1..=settings.generations {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let attempt_no = history.iter().map(|a| a.attempt + 1).max().unwrap_or(0);
        let mut rng = Rng::new(skill.spec.seed ^ attempt_no.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut attempt = Attempt {
            attempt: attempt_no,
            generation: skill.spec.lineage.generation + 1,
            started: now(),
            settings: settings.clone(),
            ..Attempt::default()
        };
        let say = |phase: &str, detail: String, attempt: Option<Attempt>| GenProgress {
            run: g,
            of: settings.generations,
            phase: phase.into(),
            detail,
            attempt,
        };

        // 1-2. Candidates play.
        let champion = skill.policy()?;
        let k = settings.candidates;
        let candidates: Vec<Policy> = (0..k)
            .map(|i| {
                let scale = if k > 1 { settings.noise * i as f32 / (k - 1) as f32 } else { 0.0 };
                perturb(&champion, scale, &mut rng)
            })
            .collect();
        let mut brains: Vec<(String, Brain)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let who = format!("{} (generation {}, candidate {})", arena.name, attempt.generation, i + 1);
                brain(c.clone()).map(|b| (who, b))
            })
            .collect::<Res<_>>()?;
        progress(&say("playing", format!("{k} players, {} episodes each", settings.episodes), None));
        let setup = EvalSetup {
            data: arena.data,
            profile: arena.profile,
            window: arena.window,
            episodes: settings.episodes,
            seconds: settings.seconds,
            decide: Decide::Sampled,
            first_seed: PLAY_SEEDS_FROM + attempt_no * 100,
            record: true,
        };
        let rows: Vec<Row> = with_range(arena, |r| {
            evaluate(setup, r, &mut brains, stop, |p| {
                progress(&say(
                    "playing",
                    format!("{}: episode {} of {}, {} hits", p.who, p.episode, p.episodes, p.hits),
                    None,
                ))
            })
        })?;
        let all: Vec<Episode> = rows.iter().flat_map(|r| r.episodes.iter().cloned()).collect();
        attempt.candidates = rows
            .iter()
            .map(|r| r.episodes.iter().map(|e| e.hits as f64).sum::<f64>() / r.episodes.len().max(1) as f64)
            .collect();

        // 3. Selection; the play not learned from is removed.
        let elite: Vec<Episode> = select(&all, settings.elite).into_iter().cloned().collect();
        attempt.elite = elite.iter().filter_map(|e| e.recording.clone().map(|r| (r, e.hits))).collect();
        for e in &all {
            if let Some(id) = &e.recording {
                if !attempt.elite.iter().any(|x| &x.0 == id) {
                    let _ = recording::delete(arena.data, &arena.profile.id, id);
                }
            }
        }
        if stop.load(Ordering::Acquire) {
            attempt.note = "stopped while playing".into();
        } else if elite.is_empty() {
            attempt.note = "no episode scored: nothing to learn from (the champion stays)".into();
        }
        if !attempt.note.is_empty() {
            finish(&mut skill, &mut history, attempt, &mut progress, &say, &mut save)?;
            continue;
        }

        // 4. The best candidate learns from the best episodes (and the person's recordings).
        let best = attempt
            .candidates
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
            .map(|x| x.0)
            .unwrap_or(0);
        let ids: Vec<String> = attempt.elite.iter().map(|x| x.0.clone()).collect();
        let own = train::load_recordings(arena.data, &skill.spec, &ids)?;
        let mut child = candidates[best].clone();
        progress(&say("learning", format!("from {} episodes (candidate {})", own.len(), best + 1), None));
        attempt.learned_steps = learn(
            &mut child,
            &skill,
            &own,
            &demos,
            demo_offset,
            &settings,
            &learn_settings,
            threads,
            &mut rng,
            core.as_ref(),
            stop,
            |step| progress(&say("learning", format!("step {step} of {}", settings.steps), None)),
        )?;
        if stop.load(Ordering::Acquire) {
            attempt.note = "stopped while learning".into();
            finish(&mut skill, &mut history, attempt, &mut progress, &say, &mut save)?;
            continue;
        }

        // 5. The test against the champion, on layouts nobody has played.
        progress(&say("testing", "the new network against the champion".into(), None));
        let mut pair = vec![
            ("champion".to_string(), brain(champion)?),
            ("new network".to_string(), brain(child.clone())?),
        ];
        let setup = EvalSetup {
            data: arena.data,
            profile: arena.profile,
            window: arena.window,
            episodes: settings.test_episodes,
            seconds: settings.seconds,
            decide: Decide::Likely,
            first_seed: TEST_SEEDS_FROM + attempt_no * 100,
            record: false,
        };
        let rows = with_range(arena, |r| {
            evaluate(setup, r, &mut pair, stop, |p| {
                progress(&say(
                    "testing",
                    format!("{}: episode {} of {}, {} hits", p.who, p.episode, p.episodes, p.hits),
                    None,
                ))
            })
        })?;
        if stop.load(Ordering::Acquire) {
            attempt.note = "stopped while testing".into();
            finish(&mut skill, &mut history, attempt, &mut progress, &say, &mut save)?;
            continue;
        }
        attempt.champion_score = rows[0].hits_per_minute;
        attempt.child_score = rows[1].hits_per_minute;
        attempt.accepted = attempt.child_score > attempt.champion_score;
        skill.spec.steps += attempt.learned_steps as u64;
        if attempt.accepted {
            let lineage = &mut skill.spec.lineage;
            lineage.parent = Some(format!("{} generation {}", arena.name, lineage.generation));
            lineage.generation += 1;
            if !lineage.method.contains("generations") {
                lineage.method = format!("{}, generations", lineage.method);
            }
            skill.accepted.clone_from(&child.w);
            // Imitation continues from the champion with a new optimizer.
            skill.latest.clone_from(&child.w);
            skill.adam = Adam::new(child.parameters());
            skill.spec.accepted_step = skill.spec.steps;
        }
        finish(&mut skill, &mut history, attempt, &mut progress, &say, &mut save)?;
    }
    Ok(skill)
}

/// Records an attempt in the skill's notes and saves it.
fn finish(
    skill: &mut Skill,
    history: &mut Vec<Attempt>,
    attempt: Attempt,
    progress: &mut impl FnMut(&GenProgress),
    say: &impl Fn(&str, String, Option<Attempt>) -> GenProgress,
    save: &mut impl FnMut(&Skill) -> Res<()>,
) -> Res<()> {
    history.push(attempt.clone());
    if !skill.spec.notes.is_object() {
        skill.spec.notes = serde_json::json!({});
    }
    skill.spec.notes["generations"] = serde_json::to_value(&*history).map_err(|e| e.to_string())?;
    save(skill)?;
    let detail = if !attempt.note.is_empty() {
        attempt.note.clone()
    } else if attempt.accepted {
        format!(
            "the new network won ({:.1} hits a minute against {:.1}): it is the champion, generation {}",
            attempt.child_score, attempt.champion_score, skill.spec.lineage.generation
        )
    } else {
        format!(
            "the champion stays ({:.1} hits a minute against the new network's {:.1})",
            attempt.champion_score, attempt.child_score
        )
    };
    progress(&say("done", detail, Some(attempt)));
    Ok(())
}

/// Learning from the model's own best play, each batch mixed with the person's recordings.
/// Returns the steps taken.
#[allow(clippy::too_many_arguments)]
fn learn(
    policy: &mut Policy,
    skill: &Skill,
    own: &[Clip],
    demos: &[Clip],
    demo_offset: usize,
    settings: &GenSettings,
    ls: &train::Settings,
    threads: usize,
    rng: &mut Rng,
    core: Option<&yendory_engine::Bicameral>,
    stop: &AtomicBool,
    mut step_done: impl FnMut(u32),
) -> Res<u32> {
    let mut frozen = core.map(|c| train::CoreTraining { model: c.clone(), train: false, learning_rate: 0.0 });
    // The model's own decision after a picture is in the next tick: offset 1.
    let own_starts = train::sequence_starts(own, ls.length, 1);
    let demo_starts = train::sequence_starts(demos, ls.length, demo_offset);
    if own_starts.is_empty() {
        return Ok(0);
    }
    let share = if demo_starts.is_empty() { 0.0 } else { settings.demonstrations };
    let mut adam = Adam::new(policy.parameters());
    for step in 1..=settings.steps {
        let batch: Vec<(&Clip, usize, usize)> = (0..ls.batch)
            .map(|_| {
                if (rng.unit() as f64) < share {
                    let (c, t) = demo_starts[rng.below(demo_starts.len())];
                    (&demos[c], t, demo_offset)
                } else {
                    let (c, t) = own_starts[rng.below(own_starts.len())];
                    (&own[c], t, 1)
                }
            })
            .collect();
        train::learn_step(policy, &mut adam, &skill.spec, &batch, ls, threads, frozen.as_mut())?;
        if step % 25 == 0 {
            step_done(step);
        }
        if stop.load(Ordering::Acquire) {
            return Ok(step);
        }
    }
    Ok(settings.steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::policy::{Arch, Size};

    fn ep(hits: u64, shots: u64, id: &str) -> Episode {
        Episode { seed: 0, hits, shots, presses: 0, mouse_x: 0.0, moved: 0.0, recording: Some(id.into()) }
    }

    #[test]
    fn selection_keeps_the_best_that_scored() {
        let eps =
            vec![ep(0, 3, "a"), ep(4, 9, "b"), ep(4, 5, "c"), ep(1, 1, "d"), ep(2, 2, "e"), ep(0, 0, "f")];
        let ids = |v: Vec<&Episode>| v.iter().map(|e| e.recording.clone().unwrap()).collect::<Vec<_>>();
        // A quarter of six: two; equal hits with fewer shots first.
        assert_eq!(ids(select(&eps, 0.25)), ["c", "b"]);
        // All of them: only those that scored.
        assert_eq!(ids(select(&eps, 1.0)), ["c", "b", "e", "d"]);
        assert!(select(&[ep(0, 4, "x")], 1.0).is_empty());
    }

    #[test]
    fn perturbing_changes_each_layer_in_proportion() {
        let arch = Arch::new(64, 36, 1, 4, 1, vec![-3.0, 0.0, 3.0], Size::Small);
        let p = Policy::new(arch, 3).unwrap();
        let same = perturb(&p, 0.0, &mut Rng::new(1));
        assert_eq!(same.w, p.w);
        let q = perturb(&p, 0.1, &mut Rng::new(1));
        assert_ne!(q.w, p.w);
        for s in &p.slots {
            let r = s.range();
            let n = r.len() as f32;
            let rms = (p.w[r.clone()].iter().map(|v| v * v).sum::<f32>() / n).sqrt();
            let d = (r.clone().map(|i| (q.w[i] - p.w[i]).powi(2)).sum::<f32>() / n).sqrt();
            if rms == 0.0 {
                // Biases still at zero stay there.
                assert_eq!(d, 0.0, "{}", s.name);
            } else if r.len() >= 100 {
                assert!((d / rms - 0.1).abs() < 0.03, "{}: {}", s.name, d / rms);
            }
        }
    }

    #[test]
    fn settings_are_checked() {
        assert!(GenSettings::default().check().is_ok());
        assert!(GenSettings { candidates: 0, ..GenSettings::default() }.check().is_err());
        assert!(GenSettings { demonstrations: 1.0, ..GenSettings::default() }.check().is_err());
    }
}
