//! Reflection: Nova revisits material she has read (Study Mode memories, or the sentences of
//! Reading blocks when Study Mode is off) and writes her own observations about how pieces of it
//! connect.
//!
//! Three things are kept apart:
//!
//! * **What Yendory does**: proposes pairs of statements that share a concept word and measures,
//!   with Nova's own network, how much one statement helps her predict the other (her loss on
//!   the second read alone minus her loss on it with the first in front, nats per token). This
//!   is a measurement of the network, not a conclusion.
//! * **What Nova does**: given a pair, the network writes an observation in its own words
//!   (greedy generation). Yendory never writes or edits it.
//! * **What the teacher does**: marks each observation, may write a correction, and approves an
//!   optional training pass. Targets keep their origin (Nova's observation, confirmed; or the
//!   teacher's correction), as in practice.
//!
//! The training pass is an ordinary teaching session: a quarter of the approved observations
//! (every fourth) is held out to judge it, and Nova's loss on the material she reflected on is
//! measured before and after, without memory.
use super::lessons::{qa, Item, Kind, Segment};
use super::practice::{target, Mark, PracticeItem, TargetOrigin};
use super::session::{teach_with, Event, SessionConfig, SessionResult};
use crate::memory::{terms, MEMORY};
use crate::model::{Ablation, Bicameral};
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;

/// The question Nova is asked about a pair (the same in every reflection, so she can learn it).
pub const CUE: &str = "What do these tell you together?";
/// Longest observation, in tokens.
pub const OBSERVATION_TOKENS: usize = 32;

/// A statement reflected on: its text and, when it is a Study Mode memory, the memory's id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Statement {
    pub text: String,
    #[serde(default)]
    pub memory: Option<u64>,
    /// Where it was read.
    #[serde(default)]
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Connection {
    pub a: Statement,
    pub b: Statement,
    /// Concept words the two share.
    pub shared: Vec<String>,
    /// Measured: how much one statement lowers Nova's loss on the other (the larger direction),
    /// nats per token. Positive = reading one helps her predict the other.
    pub help: f32,
    /// The direction measured: true when `a` helps predict `b`.
    pub a_helps_b: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub connection: Connection,
    /// Nova's own observation (written by the network).
    pub observation: String,
    #[serde(default)]
    pub mark: Option<Mark>,
    #[serde(default)]
    pub correction: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub target: Option<(String, TargetOrigin)>,
    /// Held out to judge the training pass (never trained).
    #[serde(default)]
    pub held_out: bool,
    /// Nova's observation after the training pass.
    #[serde(default)]
    pub after: Option<String>,
}

/// Nova's mean loss per token on `text` with `context` in front (only `text` is scored).
pub fn conditional_loss(model: &Bicameral, context: &str, text: &str) -> Result<f32, String> {
    let ctx = model.tokenizer.encode(context)?;
    let txt = model.tokenizer.encode(&format!("{text}\n"))?;
    let mut ids: Vec<_> = ctx.iter().chain(txt.iter()).copied().collect();
    let mut weights: Vec<f32> =
        std::iter::repeat(0.).take(ctx.len()).chain(std::iter::repeat(1.).take(txt.len())).collect();
    let max = model.config.max_tokens + 1;
    if ids.len() > max {
        let cut = ids.len() - max;
        ids.drain(..cut);
        weights.drain(..cut);
    }
    if ids.len() < 2 {
        return Ok(0.);
    }
    let targets = ids[1..].to_vec();
    let mask = weights[1..].to_vec();
    ids.pop();
    model.loss(&ids, &targets, &mask, Ablation::default())
}

/// The context lines a pair is shown in: memory lines when they are memories, plain lines else.
fn context(c: &Connection, memory_mode: bool) -> String {
    if memory_mode {
        format!("{MEMORY}{}\n{MEMORY}{}\n", c.a.text, c.b.text)
    } else {
        format!("{}\n{}\n", c.a.text, c.b.text)
    }
}

/// The full prompt Nova answers in a reflection.
pub fn prompt(c: &Connection, memory_mode: bool) -> String {
    format!("{}U: {CUE}\nN: ", context(c, memory_mode))
}

/// Pairs of statements sharing a concept word, measured by how much one helps Nova predict the
/// other, best first (at most `max`). Pairs made of the same text are skipped.
pub fn connections(
    model: &Bicameral,
    statements: &[Statement],
    max: usize,
) -> Result<Vec<Connection>, String> {
    let t: Vec<Vec<String>> = statements.iter().map(|s| terms(&s.text)).collect();
    let alone: Vec<f32> =
        statements.iter().map(|s| conditional_loss(model, "", &s.text)).collect::<Result<_, _>>()?;
    let mut pairs = Vec::new();
    for i in 0..statements.len() {
        for j in i + 1..statements.len() {
            let shared: Vec<String> = t[i].iter().filter(|w| t[j].contains(w)).cloned().collect();
            if shared.is_empty() || statements[i].text == statements[j].text {
                continue;
            }
            pairs.push((i, j, shared));
        }
    }
    // Cheap first ranking by shared words, then the measurement on the most promising pairs.
    pairs.sort_by(|a, b| b.2.len().cmp(&a.2.len()).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    pairs.truncate(max * 4);
    let mut out = Vec::new();
    for (i, j, shared) in pairs {
        let ab =
            alone[j] - conditional_loss(model, &format!("{}\n", statements[i].text), &statements[j].text)?;
        let ba =
            alone[i] - conditional_loss(model, &format!("{}\n", statements[j].text), &statements[i].text)?;
        out.push(Connection {
            a: statements[i].clone(),
            b: statements[j].clone(),
            shared,
            help: ab.max(ba),
            a_helps_b: ab >= ba,
        });
    }
    out.sort_by(|x, y| y.help.total_cmp(&x.help));
    out.truncate(max);
    Ok(out)
}

/// Nova writes an observation about each connection.
pub fn reflect(
    model: &Bicameral,
    conns: Vec<Connection>,
    memory_mode: bool,
) -> Result<Vec<Observation>, String> {
    conns
        .into_iter()
        .map(|c| {
            let ids = model.tokenizer.encode(&prompt(&c, memory_mode))?;
            let out = model.generate_cached_ids(&ids, OBSERVATION_TOKENS, Ablation::default())?;
            let text = model.tokenizer.decode(&out)?;
            let observation = crate::nexus::readable(text.lines().next().unwrap_or("")).trim().to_string();
            Ok(Observation {
                connection: c,
                observation,
                mark: None,
                correction: String::new(),
                note: String::new(),
                target: None,
                held_out: false,
                after: None,
            })
        })
        .collect()
}

/// Nova writes her observations again (after training).
pub fn reflect_again(model: &Bicameral, obs: &mut [Observation], memory_mode: bool) -> Result<(), String> {
    let again = reflect(model, obs.iter().map(|o| o.connection.clone()).collect(), memory_mode)?;
    for (o, a) in obs.iter_mut().zip(again) {
        o.after = Some(a.observation);
    }
    Ok(())
}

/// The training item of an approved observation: the pair in front (never a target), the cue,
/// and the target.
fn item(o: &Observation, t: &str, memory_mode: bool, held_out: bool) -> Item {
    let mut segments = vec![Segment { text: context(&o.connection, memory_mode), weight: 0. }];
    segments.extend(qa(CUE, t));
    let ctx = context(&o.connection, memory_mode);
    Item {
        kind: Kind::Dialogue,
        source: "reflection".into(),
        held_out,
        segments,
        prompt: Some(CUE.into()),
        answer: Some(t.to_string()),
        contradicted: None,
        users: Vec::new(),
        replies: Vec::new(),
        key: None,
        reading: None,
        context: Some(ctx),
        spelling: false,
    }
}

/// Nova's mean loss on the reflected material read without memory (each statement alone).
pub fn material_loss(model: &Bicameral, obs: &[Observation]) -> Result<f32, String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut total = 0.;
    let mut n = 0;
    for o in obs {
        for s in [&o.connection.a, &o.connection.b] {
            if seen.insert(s.text.clone()) {
                total += conditional_loss(model, "", &s.text)?;
                n += 1;
            }
        }
    }
    Ok(if n == 0 { 0. } else { total / n as f32 })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReflectionTraining {
    pub learned: bool,
    pub baseline: super::session::Score,
    pub final_score: super::session::Score,
    pub trained: usize,
    pub held_out: usize,
    /// Nova's loss on the reflected statements read alone, before and after.
    pub material_before: f32,
    pub material_after: f32,
}

/// The approved training pass. Needs at least three approved observations (a quarter are held
/// out to judge it). Returns the new model with the record.
pub fn train(
    model: &Bicameral,
    obs: &mut [Observation],
    memory_mode: bool,
    archive: &[Item],
    cfg: &SessionConfig,
    observer: &mut dyn FnMut(&Event),
    stop: &AtomicBool,
) -> Result<(SessionResult, ReflectionTraining), String> {
    let mut approved = Vec::new();
    for (k, o) in obs.iter_mut().enumerate() {
        let as_practice = PracticeItem {
            question: CUE.into(),
            expected: None,
            answer: o.observation.clone(),
            memories: Vec::new(),
            mark: o.mark,
            correction: o.correction.clone(),
            note: String::new(),
            target: None,
            after: None,
            grade: None,
            auto: None,
        };
        o.target = target(&as_practice);
        if o.target.is_some() {
            approved.push(k);
        }
    }
    if approved.len() < 3 {
        return Err("Approve at least three observations (right, or wrong with your correction): a quarter of them judge the training".into());
    }
    let mut lesson = Vec::new();
    for (n, &k) in approved.iter().enumerate() {
        let held = n % 4 == 3 || (n == approved.len() - 1 && approved.len() < 4);
        obs[k].held_out = held;
        let (t, _) = obs[k].target.clone().unwrap();
        lesson.push(item(&obs[k], &t, memory_mode, held));
    }
    let material_before = material_loss(model, obs)?;
    let r = teach_with(model, &lesson, None, archive, cfg, observer, stop)?;
    let material_after = material_loss(&r.model, obs)?;
    let rec = ReflectionTraining {
        learned: r.learned,
        baseline: r.baseline.clone(),
        final_score: r.final_score.clone(),
        trained: lesson.iter().filter(|i| !i.held_out).count(),
        held_out: lesson.iter().filter(|i| i.held_out).count(),
        material_before,
        material_after,
    };
    Ok((r, rec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Focus};

    #[test]
    fn connections_are_measured_by_the_network() {
        let mut model = Bicameral::with_config(Config::default(), 4);
        // Train the pair together so that one predicts the other.
        for _ in 0..200 {
            model.train_text("Max is a dog.\nMax likes the park.\n", 3e-3, Focus::Joint).unwrap();
        }
        let st = |t: &str| Statement { text: t.into(), memory: None, source: "t".into() };
        let conns =
            connections(&model, &[st("Max is a dog."), st("Max likes the park."), st("Rain is wet.")], 4)
                .unwrap();
        assert_eq!(conns.len(), 1, "only the pair sharing a word");
        assert_eq!(conns[0].shared, ["max"]);
        assert!(conns[0].help > 0.5, "reading one helps predict the other: {}", conns[0].help);
        let obs = reflect(&model, conns, false).unwrap();
        assert_eq!(obs.len(), 1);
        assert!(obs[0].mark.is_none() && obs[0].target.is_none(), "nothing is approved by itself");
        assert_eq!(
            prompt(&obs[0].connection, true),
            "\x03Max is a dog.\n\x03Max likes the park.\nU: What do these tell you together?\nN: "
        );
    }
}
