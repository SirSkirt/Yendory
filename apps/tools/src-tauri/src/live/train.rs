//! Generation 0: imitation. The network learns, from a person's recordings, the controls they
//! used after each picture (docs/GAME_TRAINING.md, section 4).
//!
//! - Pictures: the last `frames` of a stretch (the first repeated at a stretch's start); the
//!   target: the controls of the tick `offset` ticks after the picture (people react in a few
//!   hundred milliseconds; 1 is the very next tick).
//! - Sequences of `length` ticks are trained with the memory carried through them.
//! - Held out: recordings marked for testing are only measured, with the memory carried through
//!   each one as in play; every measurement is shown beside the **frequency baseline**, what
//!   predicting the person's average use of each control (no picture at all) would score, so a
//!   model only counts as having learned when it beats that.
//! - The accepted weights are the best measured on held-out recordings; training can stop and
//!   continue later from where it was.
use super::policy::{encode, sigmoid, softmax, Adam, Loss, Policy, Rng, Skill, SkillSpec, Target};
use super::profile::Profile;
use super::recording::{self, Split, AFTER_GAP};
use crate::Res;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use yendory_engine::model::{Ablation, Focus};
use yendory_engine::Bicameral;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub steps: u32,
    /// Sequences in each step.
    pub batch: usize,
    /// Ticks in each sequence.
    pub length: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    /// Ticks between a picture and the controls learned for it.
    pub offset: usize,
    /// Held-out measurement every this many steps.
    pub measure_every: u32,
    pub seed: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            steps: 1500,
            batch: 16,
            length: 8,
            learning_rate: 5e-4,
            weight_decay: 1e-4,
            offset: 1,
            measure_every: 100,
            seed: 1,
        }
    }
}

impl Settings {
    pub fn check(&self) -> Res<()> {
        if !(1..=1_000_000).contains(&self.steps)
            || !(1..=256).contains(&self.batch)
            || !(1..=64).contains(&self.length)
            || !(1e-6..=1e-1).contains(&self.learning_rate)
            || !(0.0..=1.0).contains(&self.weight_decay)
            || self.offset > 20
            || self.measure_every == 0
        {
            return Err("A training setting is out of range".into());
        }
        Ok(())
    }
}

/// One control's held-out score.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ControlScore {
    pub name: String,
    /// How often the person used it in these ticks.
    pub used: f64,
    /// How often the model would (probability above one half).
    pub predicted: f64,
    /// F1 of "used" (precision and recall together); the baseline's is 0 for a control used
    /// less than half the time (it never predicts it).
    pub f1: f64,
}

/// A held-out measurement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Measure {
    pub step: u64,
    pub ticks: u64,
    /// Average loss a tick (keys + buttons + mouse), the model's and the frequency baseline's.
    pub loss: f64,
    pub baseline_loss: f64,
    pub keys_loss: f64,
    pub baseline_keys_loss: f64,
    pub mouse_loss: f64,
    pub baseline_mouse_loss: f64,
    /// How often the model's most likely mouse step is the person's, and the baseline's (always
    /// the person's most common step).
    pub mouse_accuracy: f64,
    pub baseline_mouse_accuracy: f64,
    pub controls: Vec<ControlScore>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Run {
    pub started: u64,
    pub settings: Settings,
    pub recordings: Vec<String>,
    pub held_out: Vec<String>,
    pub steps_done: u32,
    pub stopped: bool,
    /// Training loss over the last steps.
    pub train_loss: f64,
    /// The measurement before training (the untrained network) and the best one.
    pub first: Option<Measure>,
    pub best: Option<Measure>,
    /// With the model's core (a linked skill): whether the core learned too, and how fast.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub core: Option<CoreRun>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CoreRun {
    pub trained: bool,
    pub learning_rate: f32,
}

/// The model's core in a linked skill's training: it plays its part every tick; with `train`
/// its weights learn too (the model's own optimizer), at `learning_rate`.
pub struct CoreTraining {
    pub model: Bicameral,
    pub train: bool,
    pub learning_rate: f32,
}

/// One recording, ready to learn from or measure on. Its pictures stay on disk and are read as
/// training needs them, so there is no limit on how much can be learned from at once.
pub struct Clip {
    pub id: String,
    pub frames: recording::Frames,
    pub targets: Vec<Target>,
    /// Where each stretch (between pauses) begins.
    pub starts: Vec<bool>,
}

/// The nearest mouse step's index.
pub fn bin_of(bins: &[f32], v: f32) -> usize {
    bins.iter()
        .enumerate()
        .min_by(|a, b| (a.1 - v).abs().total_cmp(&(b.1 - v).abs()))
        .map(|x| x.0)
        .unwrap_or(0)
}

/// Loads the person's recordings of the game for `split` (and checks they fit the player).
pub fn load_clips(data: &Path, info: &SkillSpec, split: Split) -> Res<Vec<Clip>> {
    let mut clips = Vec::new();
    for r in
        recording::list(data, &info.game).into_iter().filter(|r| r.player == "person" && r.split == split)
    {
        if let Some(c) = clip_of(data, info, &r)? {
            clips.push(c);
        }
    }
    Ok(clips)
}

/// Loads the given recordings (of anyone, the model's own play included) that fit the player.
pub fn load_recordings(data: &Path, info: &SkillSpec, ids: &[String]) -> Res<Vec<Clip>> {
    let mut clips = Vec::new();
    for r in recording::list(data, &info.game).into_iter().filter(|r| ids.contains(&r.id)) {
        if let Some(c) = clip_of(data, info, &r)? {
            clips.push(c);
        }
    }
    Ok(clips)
}

/// A recording as a clip, if its picture and controls fit the player.
fn clip_of(data: &Path, info: &SkillSpec, r: &recording::Info) -> Res<Option<Clip>> {
    let a = &info.arch;
    if r.width as usize != a.width
        || r.height as usize != a.height
        || r.channels as usize != a.channels
        || r.keys != info.keys
        || r.buttons.iter().map(|b| b.name()).ne(info.buttons.iter().map(String::as_str))
    {
        return Ok(None);
    }
    let reader = recording::open(data, &info.game, &r.id)?;
    let mut clip = Clip { id: r.id.clone(), frames: reader.frames()?, targets: vec![], starts: vec![] };
    let bins = &a.mouse_bins;
    for (i, t) in reader.ticks.iter().enumerate() {
        clip.targets.push(Target {
            keys: t.keys,
            buttons: t.buttons,
            mx: if bins.is_empty() { 0 } else { bin_of(bins, t.dx) },
            my: if bins.is_empty() { 0 } else { bin_of(bins, t.dy) },
        });
        clip.starts.push(i == 0 || t.flags & AFTER_GAP != 0);
    }
    Ok(Some(clip))
}

/// The input for the picture at `t`: the last `frames` pictures of its stretch.
pub(crate) fn input_at(info: &SkillSpec, clip: &Clip, t: usize) -> Res<Vec<f32>> {
    let k = info.arch.frames;
    let mut idx = vec![t; k];
    let mut cur = t;
    for slot in (0..k).rev() {
        idx[slot] = cur;
        if cur > 0 && !clip.starts[cur] {
            cur -= 1;
        }
    }
    let mut pics: Vec<(usize, Vec<u8>)> = Vec::with_capacity(k);
    for &i in &idx {
        let pic = match pics.iter().find(|p| p.0 == i) {
            Some(p) => p.1.clone(),
            None => clip.frames.get(i)?,
        };
        pics.push((i, pic));
    }
    let refs: Vec<&[u8]> = pics.iter().map(|p| p.1.as_slice()).collect();
    Ok(encode(&info.arch, &refs))
}

/// Positions whose target (`offset` ticks on) is in the same stretch.
fn valid(clip: &Clip, t: usize, offset: usize) -> bool {
    t + offset < clip.targets.len() && (t + 1..=t + offset).all(|i| !clip.starts[i])
}

/// How often the person used each control, and each mouse step, in the training recordings.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Frequencies {
    pub keys: Vec<f64>,
    pub buttons: Vec<f64>,
    pub mx: Vec<f64>,
    pub my: Vec<f64>,
}

pub fn frequencies(info: &SkillSpec, clips: &[Clip]) -> Frequencies {
    let a = &info.arch;
    let nb = a.mouse_bins.len();
    let mut f = Frequencies {
        keys: vec![0.0; a.keys],
        buttons: vec![0.0; a.buttons],
        mx: vec![1.0; nb],
        my: vec![1.0; nb],
    };
    let mut n = 0f64;
    for c in clips {
        for t in &c.targets {
            n += 1.0;
            for (k, v) in f.keys.iter_mut().enumerate() {
                *v += ((t.keys >> k) & 1) as f64;
            }
            for (b, v) in f.buttons.iter_mut().enumerate() {
                *v += ((t.buttons >> b) & 1) as f64;
            }
            if nb > 0 {
                f.mx[t.mx] += 1.0;
                f.my[t.my] += 1.0;
            }
        }
    }
    let n = n.max(1.0);
    for v in f.keys.iter_mut().chain(f.buttons.iter_mut()) {
        *v = (*v / n).clamp(1e-4, 1.0 - 1e-4);
    }
    let total = n + nb as f64;
    for v in f.mx.iter_mut().chain(f.my.iter_mut()) {
        *v /= total;
    }
    f
}

/// Measures the network on held-out clips, the memory carried through each stretch as in play.
#[allow(clippy::too_many_arguments)]
pub fn measure(
    p: &Policy,
    info: &SkillSpec,
    clips: &[Clip],
    freq: &Frequencies,
    offset: usize,
    step: u64,
    mut cortex: Option<&mut super::cortex::Cortex>,
) -> Res<Measure> {
    let a = &p.arch;
    let zero = vec![0f32; a.goal];
    let names: Vec<String> = info.keys.iter().chain(info.buttons.iter()).cloned().collect();
    let nc = names.len();
    let (mut tp, mut fp, mut fneg, mut used, mut pred) =
        (vec![0f64; nc], vec![0f64; nc], vec![0f64; nc], vec![0f64; nc], vec![0f64; nc]);
    let (mut loss, mut base, mut kl, mut bkl, mut ml, mut bml) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let (mut macc, mut bmacc, mut n) = (0f64, 0f64, 0f64);
    let common_mx = freq.mx.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0).unwrap_or(0);
    let common_my = freq.my.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0).unwrap_or(0);
    for clip in clips {
        let mut m = vec![0f32; a.memory];
        for t in 0..clip.targets.len() {
            if clip.starts[t] {
                m = vec![0f32; a.memory];
                if let Some(c) = cortex.as_deref_mut() {
                    c.reset();
                }
            }
            let seen = p.see(&input_at(info, clip, t)?);
            let goal = match cortex.as_deref_mut() {
                Some(c) => c.goal(p, p.sight(&seen))?,
                None => zero.clone(),
            };
            let cache = p.forward_seen(seen, &m, &goal);
            m = cache.m.clone();
            if !valid(clip, t, offset) {
                continue;
            }
            let target = clip.targets[t + offset];
            let l = &cache.logits;
            n += 1.0;
            let mut tick = 0f64;
            let mut tick_base = 0f64;
            let mut key_part = (0f64, 0f64);
            for (i, (&z, &q)) in
                l.keys.iter().chain(&l.buttons).zip(freq.keys.iter().chain(&freq.buttons)).enumerate()
            {
                let y = if i < a.keys {
                    (target.keys >> i) & 1
                } else {
                    ((target.buttons >> (i - a.keys)) & 1) as u64
                } as f64;
                let e = super::policy::bce(z, y as f32) as f64;
                let eb = -(y * q.ln() + (1.0 - y) * (1.0 - q).ln());
                tick += e;
                tick_base += eb;
                if i < a.keys {
                    key_part.0 += e;
                    key_part.1 += eb;
                }
                let on = sigmoid(z) > 0.5;
                used[i] += y;
                pred[i] += on as u8 as f64;
                match (on, y > 0.5) {
                    (true, true) => tp[i] += 1.0,
                    (true, false) => fp[i] += 1.0,
                    (false, true) => fneg[i] += 1.0,
                    _ => {}
                }
            }
            kl += key_part.0;
            bkl += key_part.1;
            if !a.mouse_bins.is_empty() {
                for (zs, want, f, common) in
                    [(&l.mx, target.mx, &freq.mx, common_mx), (&l.my, target.my, &freq.my, common_my)]
                {
                    let pr = softmax(zs);
                    let e = -(pr[want].max(1e-12) as f64).ln();
                    let eb = -f[want].max(1e-12).ln();
                    tick += e;
                    tick_base += eb;
                    ml += e;
                    bml += eb;
                    let best =
                        pr.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0).unwrap_or(0);
                    macc += (best == want) as u8 as f64 / 2.0;
                    bmacc += (common == want) as u8 as f64 / 2.0;
                }
            }
            loss += tick;
            base += tick_base;
        }
    }
    let d = n.max(1.0);
    Ok(Measure {
        step,
        ticks: n as u64,
        loss: loss / d,
        baseline_loss: base / d,
        keys_loss: kl / d,
        baseline_keys_loss: bkl / d,
        mouse_loss: ml / d,
        baseline_mouse_loss: bml / d,
        mouse_accuracy: macc / d,
        baseline_mouse_accuracy: bmacc / d,
        controls: (0..nc)
            .map(|i| ControlScore {
                name: names[i].clone(),
                used: used[i] / d,
                predicted: pred[i] / d,
                f1: if tp[i] == 0.0 { 0.0 } else { 2.0 * tp[i] / (2.0 * tp[i] + fp[i] + fneg[i]) },
            })
            .collect(),
    })
}

/// Progress of a training run.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub step: u32,
    pub steps: u32,
    pub train_loss: f64,
    pub measure: Option<Measure>,
    /// Whether this measurement was the best yet (and its weights were kept).
    pub accepted: bool,
}

/// Where sequences can start: every position with `length` usable positions after it in its
/// stretch (or, when no stretch is that long, every usable position).
pub(crate) fn sequence_starts(clips: &[Clip], length: usize, offset: usize) -> Vec<(usize, usize)> {
    let mut starts: Vec<(usize, usize)> = Vec::new();
    for (ci, c) in clips.iter().enumerate() {
        for t in 0..c.targets.len() {
            let ok = (t..t + length)
                .all(|u| u < c.targets.len() && valid(c, u, offset) && (u == t || !c.starts[u]));
            if ok {
                starts.push((ci, t));
            }
        }
    }
    if starts.is_empty() {
        for (ci, c) in clips.iter().enumerate() {
            for t in 0..c.targets.len() {
                if valid(c, t, offset) {
                    starts.push((ci, t));
                }
            }
        }
    }
    starts
}

/// Positions usable for learning in these clips.
pub(crate) fn usable(clips: &[Clip], offset: usize) -> usize {
    clips.iter().map(|c| (0..c.targets.len()).filter(|&t| valid(c, t, offset)).count()).sum()
}

/// One learning step on a batch of sequences (clip, start, offset), shared out over `threads`;
/// returns the batch's mean loss per sequence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn learn_step(
    policy: &mut Policy,
    adam: &mut Adam,
    info: &SkillSpec,
    batch: &[(&Clip, usize, usize)],
    settings: &Settings,
    threads: usize,
    core: Option<&mut CoreTraining>,
) -> Res<f64> {
    let n = policy.parameters();
    let scale = 1.0 / batch.len().max(1) as f32;
    let length = settings.length;
    let core_ref = core.as_deref().map(|c| &c.model);
    let cn = core_ref.map(|c| c.weights.len()).unwrap_or(0);
    type Part = (Vec<f32>, Vec<f32>, Loss);
    let parts: Vec<Res<Part>> = std::thread::scope(|s| {
        let handles: Vec<_> = batch
            .chunks(batch.len().div_ceil(threads.max(1)).max(1))
            .map(|chunk| {
                let policy = &*policy;
                s.spawn(move || -> Res<Part> {
                    let mut g = vec![0f32; n];
                    let mut cg = vec![0f32; cn];
                    let mut total = Loss::default();
                    for &(c, t0, offset) in chunk {
                        let mut inputs = Vec::new();
                        let mut targets = Vec::new();
                        for t in t0..t0 + length {
                            if t >= c.targets.len() || !valid(c, t, offset) || (t > t0 && c.starts[t]) {
                                break;
                            }
                            inputs.push(input_at(info, c, t)?);
                            targets.push(c.targets[t + offset]);
                        }
                        total += match core_ref {
                            Some(c) => yendory_engine::cortex::sequence_loss(
                                policy,
                                c,
                                &inputs,
                                &targets,
                                &mut g,
                                &mut cg,
                                scale,
                                Ablation::default(),
                            )?,
                            None => policy.sequence_loss(&inputs, &targets, &mut g, scale),
                        };
                    }
                    Ok((g, cg, total))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("A training thread failed".into())))
            .collect()
    });
    let mut g = vec![0f32; n];
    let mut cg = vec![0f32; cn];
    let mut loss = Loss::default();
    for part in parts {
        let (pg, pcg, pl) = part?;
        for (a, b) in g.iter_mut().zip(&pg) {
            *a += b;
        }
        for (a, b) in cg.iter_mut().zip(&pcg) {
            *a += b;
        }
        loss += pl;
    }
    // The core learns at its own rate with the model's own optimizer (its gradient clipped
    // on its own).
    if let Some(c) = core {
        if c.train {
            c.model.adam_step(&cg, c.learning_rate, Focus::Joint)?;
        }
    }
    // Clip the gradient's length at 1.
    let norm = g.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
    if !norm.is_finite() {
        return Err("Training went wrong (the gradient is not a number); try a lower learning rate".into());
    }
    if norm > 1.0 {
        let k = (1.0 / norm) as f32;
        g.iter_mut().for_each(|v| *v *= k);
    }
    adam.step(policy, &g, settings.learning_rate, settings.weight_decay);
    Ok(loss.total / batch.len().max(1) as f64)
}

/// The training runs recorded in a skill's notes.
pub fn runs(spec: &SkillSpec) -> Vec<Run> {
    spec.notes.get("runs").and_then(|r| serde_json::from_value(r.clone()).ok()).unwrap_or_default()
}

/// Trains a game skill; returns it with the run recorded. `fresh`: start again from the
/// accepted weights with a new optimizer instead of continuing from the latest. `save` keeps
/// the skill (in its model, through the Network) at every measurement and at the end.
#[allow(clippy::too_many_arguments)]
pub fn train(
    data: &Path,
    profile: &Profile,
    mut skill: Skill,
    settings: Settings,
    fresh: bool,
    mut core: Option<CoreTraining>,
    stop: &AtomicBool,
    mut progress: impl FnMut(&Progress),
    mut save: impl FnMut(&Skill, Option<&Bicameral>) -> Res<()>,
) -> Res<Skill> {
    settings.check()?;
    super::skills::fits(&skill.spec, profile)?;
    match (&skill.spec.arch.cortex, &core) {
        (Some(_), None) => {
            return Err("This skill is linked to its model's core: load the core to train it".into())
        }
        (None, Some(_)) => return Err("This skill has no link to its model's core".into()),
        (Some(_), Some(c)) => {
            yendory_engine::cortex::fits(&skill.policy()?, &c.model)?;
            if settings.length > c.model.config.max_tokens {
                return Err(format!(
                    "With the core, a stretch can be at most {} ticks (the model's context)",
                    c.model.config.max_tokens
                ));
            }
        }
        (None, None) => {}
    }
    let clips = load_clips(data, &skill.spec, Split::Train)?;
    let held = load_clips(data, &skill.spec, Split::Test)?;
    if usable(&clips, settings.offset) < settings.length * 4 {
        return Err("There is not enough of your play recorded for training (record some on 2.3, marked for training)".into());
    }
    let freq = frequencies(&skill.spec, &clips);
    let (mut policy, mut adam) = if fresh {
        let p = skill.policy()?;
        let n = p.parameters();
        (p, Adam::new(n))
    } else {
        skill.latest_policy()?
    };
    let mut run = Run {
        started: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        settings: settings.clone(),
        recordings: clips.iter().map(|c| c.id.clone()).collect(),
        held_out: held.iter().map(|c| c.id.clone()).collect(),
        steps_done: 0,
        stopped: false,
        train_loss: 0.0,
        first: None,
        best: None,
        core: core.as_ref().map(|c| CoreRun { trained: c.train, learning_rate: c.learning_rate }),
    };
    let fixed = skill.spec.clone();
    let measure_now = |p: &Policy, step: u64, core: Option<&CoreTraining>| -> Res<Option<Measure>> {
        if held.is_empty() {
            return Ok(None);
        }
        let mut cortex = match core {
            Some(c) => Some(super::cortex::Cortex::new(c.model.clone(), settings.length)?),
            None => None,
        };
        measure(p, &fixed, &held, &freq, settings.offset, step, cortex.as_mut()).map(Some)
    };
    let first = measure_now(&policy, skill.spec.steps, core.as_ref())?;
    // The accepted weights stand until a measurement beats them (on this run's held-out set).
    let mut best_loss = first.as_ref().map(|m| m.loss).unwrap_or(f64::INFINITY);
    run.first = first.clone();
    progress(&Progress { step: 0, steps: settings.steps, train_loss: 0.0, measure: first, accepted: false });

    let starts = sequence_starts(&clips, settings.length, settings.offset);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 16);
    let mut rng = Rng::new(settings.seed ^ skill.spec.steps);
    let mut recent = std::collections::VecDeque::new();
    for step in 1..=settings.steps {
        let batch: Vec<(&Clip, usize, usize)> = (0..settings.batch)
            .map(|_| {
                let (ci, t) = starts[rng.below(starts.len())];
                (&clips[ci], t, settings.offset)
            })
            .collect();
        let loss = learn_step(&mut policy, &mut adam, &fixed, &batch, &settings, threads, core.as_mut())?;
        skill.spec.steps += 1;
        run.steps_done = step;
        recent.push_back(loss);
        if recent.len() > 50 {
            recent.pop_front();
        }
        run.train_loss = recent.iter().sum::<f64>() / recent.len() as f64;
        let stopping = stop.load(Ordering::Acquire);
        let last = step == settings.steps || stopping;
        if step % settings.measure_every == 0 || last {
            let m = measure_now(&policy, skill.spec.steps, core.as_ref())?;
            let accepted = match &m {
                Some(m) if m.loss < best_loss => {
                    best_loss = m.loss;
                    true
                }
                Some(_) => false,
                // Nothing held out: the latest weights are accepted (and the page says so).
                None => true,
            };
            if accepted {
                skill.accepted.clone_from(&policy.w);
                skill.spec.accepted_step = skill.spec.steps;
                if m.is_some() {
                    run.best = m.clone();
                }
            }
            match core.as_ref().filter(|c| c.train) {
                // The core learned too: the model is saved only as a pair measured together,
                // the accepted skill with the core it was measured with.
                Some(c) => {
                    if accepted {
                        skill.latest.clone_from(&policy.w);
                        skill.adam = adam.clone();
                        save(&skill, Some(&c.model))?;
                    }
                }
                None => {
                    skill.latest.clone_from(&policy.w);
                    skill.adam = adam.clone();
                    save(&skill, None)?;
                }
            }
            progress(&Progress {
                step,
                steps: settings.steps,
                train_loss: run.train_loss,
                measure: m,
                accepted,
            });
        } else if step % 10 == 0 {
            progress(&Progress {
                step,
                steps: settings.steps,
                train_loss: run.train_loss,
                measure: None,
                accepted: false,
            });
        }
        if stopping {
            run.stopped = true;
            break;
        }
    }
    let lineage = &mut skill.spec.lineage;
    if !lineage.method.contains("imitation") {
        lineage.method = if lineage.method == "new" {
            "imitation".into()
        } else {
            format!("{}, imitation", lineage.method)
        };
    }
    let mut all = runs(&skill.spec);
    all.push(run);
    if !skill.spec.notes.is_object() {
        skill.spec.notes = serde_json::json!({});
    }
    skill.spec.notes["runs"] = serde_json::to_value(&all).map_err(|e| e.to_string())?;
    if core.as_ref().is_some_and(|c| c.train) {
        // The latest weights go with the core last saved: the accepted pair.
        skill.latest.clone_from(&skill.accepted);
    } else {
        skill.latest.clone_from(&policy.w);
        skill.adam = adam;
    }
    save(&skill, None)?;
    Ok(skill)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::controls::Button;
    use crate::live::policy::Size;
    use crate::live::recording::{Info, Tick, Writer, FORMAT};

    #[test]
    fn bins_are_nearest() {
        let bins = [-5.0, -1.0, 0.0, 1.0, 5.0];
        assert_eq!(bin_of(&bins, 0.2), 2);
        assert_eq!(bin_of(&bins, -3.5), 0);
        assert_eq!(bin_of(&bins, 2.9), 3);
    }

    /// A small game and recordings where W follows a bright picture, written directly (tick k's
    /// controls are what was used after picture k-1): two for training, one held out.
    fn synthetic(name: &str) -> (std::path::PathBuf, Profile) {
        let data = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        let mut profile = crate::live::profile::range_profile();
        profile.capture.width = 32;
        profile.capture.height = 18;
        profile.mouse = false;
        profile.keys = vec!["w".into()];
        profile.buttons = vec![Button::Left];
        let profile = crate::live::profile::save(&data, profile).unwrap();
        // Recordings where W follows a bright picture, written directly (tick k's controls are
        // what was used after picture k-1).
        for (id, split, seed) in [("a", Split::Train, 1), ("b", Split::Train, 2), ("c", Split::Test, 3)] {
            let info = Info {
                format: FORMAT,
                id: id.into(),
                profile: profile.id.clone(),
                game: profile.name.clone(),
                player: "person".into(),
                split,
                notes: String::new(),
                started: seed,
                width: 32,
                height: 18,
                channels: 3,
                fps: 20,
                keys: profile.keys.clone(),
                buttons: profile.buttons.clone(),
                mouse: false,
                ticks: 0,
                duration_ms: 0,
                paused_ms: 0,
                dropped_ticks: 0,
                bytes: 0,
                complete: false,
                ended: String::new(),
            };
            let mut wr = Writer::create(&data, info).unwrap();
            let mut rng = Rng::new(seed);
            let mut bright = false;
            let mut was_bright = false;
            for i in 0..500u64 {
                if rng.below(5) == 0 {
                    bright = !bright;
                }
                let px = vec![if bright { 220u8 } else { 30 }; 32 * 18 * 3];
                let keys = (i > 0 && was_bright) as u64;
                wr.tick(
                    Tick {
                        t_us: i * 50_000,
                        keys,
                        flags: if i == 0 { AFTER_GAP } else { 0 },
                        ..Tick::default()
                    },
                    &px,
                )
                .unwrap();
                was_bright = bright;
            }
            wr.finish("stopped").unwrap();
        }
        (data, profile)
    }

    #[test]
    fn the_network_learns_what_the_picture_says_and_beats_the_baseline() {
        let (data, profile) = synthetic("tools-train");
        let skill = crate::live::skills::new_skill(&data, &profile, Size::Small, 3).unwrap();
        let settings = Settings {
            steps: 150,
            batch: 8,
            length: 4,
            learning_rate: 2e-3,
            measure_every: 50,
            ..Settings::default()
        };
        let stop = AtomicBool::new(false);
        let mut seen = Vec::new();
        let mut saved: Vec<Skill> = Vec::new();
        let done = train(
            &data,
            &profile,
            skill,
            settings,
            false,
            None,
            &stop,
            |p| seen.push(p.clone()),
            |k, _| {
                saved.push(k.clone());
                Ok(())
            },
        )
        .unwrap();
        let all = runs(&done.spec);
        let run = all.last().unwrap();
        let first = run.first.as_ref().unwrap();
        let best = run.best.as_ref().unwrap();
        assert_eq!(run.recordings.len(), 2);
        assert_eq!(run.held_out, vec!["c".to_string()]);
        assert!(
            best.loss < best.baseline_loss * 0.5,
            "held-out {} vs baseline {}",
            best.loss,
            best.baseline_loss
        );
        assert!(best.loss < first.loss);
        let w = best.controls.iter().find(|c| c.name == "w").unwrap();
        assert!(w.f1 > 0.9, "{w:?}");
        assert_eq!(done.spec.lineage.method, "imitation");
        assert!(seen.iter().any(|p| p.accepted));
        // Saved at every measurement (3) and at the end; the accepted weights are what plays.
        assert_eq!(saved.len(), 4);
        assert_eq!(done.policy().unwrap().w, done.accepted);
        // Continuing works from the latest weights and optimizer.
        let more = train(
            &data,
            &profile,
            done,
            Settings { steps: 20, measure_every: 10, ..Settings::default() },
            false,
            None,
            &stop,
            |_| {},
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(runs(&more.spec).len(), 2);
        assert_eq!(more.spec.steps, 170);
        assert_eq!(more.adam.t, 170);
        let _ = std::fs::remove_dir_all(&data);
    }

    fn cores() -> Vec<Bicameral> {
        use yendory_engine::model::brain::{design_for, BrainChoice};
        use yendory_engine::model::Config;
        let bica = Bicameral::with_config(
            Config {
                vocab: 140,
                d: 32,
                heads: 2,
                ff: 48,
                blocks: 2,
                split: 1,
                loops: 1,
                window_chars: 2,
                r1: 16,
                r2: 16,
                max_tokens: 16,
                ..Config::default()
            },
            5,
        );
        let (mut c, n) = design_for(
            &BrainChoice {
                width: 64,
                transformer: Some(60_000),
                memory: Some(10_000),
                controller: Some(2_000),
                ..BrainChoice::default()
            },
            140,
        )
        .unwrap();
        c.max_tokens = 16;
        vec![bica, Bicameral::with_design(c, Some(n), 7)]
    }

    #[test]
    fn a_skill_linked_to_its_core_learns_and_the_core_learns_only_when_asked() {
        let (data, profile) = synthetic("tools-train-core");
        for core in cores() {
            let (width, right) = core.sense_widths();
            let skill = crate::live::skills::new_skill(&data, &profile, Size::Small, 3)
                .unwrap()
                .with_cortex(Some(yendory_engine::skill::CortexLink { width, right }))
                .unwrap();
            let settings = Settings {
                steps: 60,
                batch: 8,
                length: 4,
                learning_rate: 2e-3,
                measure_every: 30,
                ..Settings::default()
            };
            let stop = AtomicBool::new(false);
            // Without its core it can't train (and a skill without a link can't take one).
            assert!(train(
                &data,
                &profile,
                skill.clone(),
                settings.clone(),
                false,
                None,
                &stop,
                |_| {},
                |_, _| Ok(())
            )
            .is_err());
            // The core plays its part but stays as it is: only the skill is saved.
            let mut core_saves = 0;
            let frozen = train(
                &data,
                &profile,
                skill.clone(),
                settings.clone(),
                false,
                Some(CoreTraining { model: core.clone(), train: false, learning_rate: 1e-3 }),
                &stop,
                |_| {},
                |_, c| {
                    core_saves += c.is_some() as usize;
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(core_saves, 0);
            let run = runs(&frozen.spec).pop().unwrap();
            assert_eq!(run.core, Some(CoreRun { trained: false, learning_rate: 1e-3 }));
            let best = run.best.unwrap();
            assert!(best.loss < run.first.unwrap().loss, "the linked skill learns");
            assert!(best.loss < best.baseline_loss, "{} vs {}", best.loss, best.baseline_loss);
            // The core learns too: the model is saved as a pair (the accepted skill with its core).
            let mut saved: Vec<(Skill, Vec<f32>)> = Vec::new();
            let trained = train(
                &data,
                &profile,
                skill,
                settings,
                false,
                Some(CoreTraining { model: core.clone(), train: true, learning_rate: 1e-3 }),
                &stop,
                |_| {},
                |k, c| {
                    if let Some(c) = c {
                        saved.push((k.clone(), c.weights.clone()));
                    }
                    Ok(())
                },
            )
            .unwrap();
            assert!(!saved.is_empty());
            for (k, w) in &saved {
                assert_eq!(k.latest, k.accepted, "saved with the core it was measured with");
                assert_ne!(*w, core.weights, "the core's weights learned");
            }
            assert_eq!(trained.latest, trained.accepted);
            // The skill's link has learned to carry something back.
            let p = trained.policy().unwrap();
            let out = p.slot("link.out.w").unwrap().range();
            assert!(p.w[out].iter().any(|v| *v != 0.0));
        }
        let _ = std::fs::remove_dir_all(&data);
    }
}
