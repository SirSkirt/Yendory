//! Conversation training: the stage after the Classroom, where Nova practises talking with real
//! conversations between two people.
//!
//! * **Peers, not assistants.** The material is everyday conversation between two people.
//!   Dialogues in an assistant's voice ("as an AI", "how can I assist you") are left out, and no
//!   persona or instruction is put in front of a conversation.
//! * **Only Nova's lines are trained.** Nova plays one speaker: that speaker's turns are its
//!   replies (`N:`, full weight) and the other speaker's turns are the conversation it answers
//!   (`U:`, weight `user_weight`, 0 by default).
//! * **Both sides.** Every dialogue is practised twice, once as each speaker, so Nova learns to
//!   start conversations and ask as well as answer.
//! * **Its own name.** When a dialogue names its speakers, Nova's speaker's name becomes Nova's
//!   own (`own_name`), so it does not learn to call itself by a hundred different names.
//! * **Progress is checked like reading.** Conversations have no single right reply, so a
//!   checkpoint is judged by the loss of Nova's lines in held-out conversations (about one in
//!   ten, never trained on), and it must keep the Classroom answers it had: a sample of the
//!   model's held-out Classroom questions may lose at most `guard_drop` of its correct answers.
//! * **Earlier material.** A share of the training rows (`earlier_share`) comes from the
//!   lessons the model was taught, so the Classroom is not forgotten.
use super::eval::{evaluate, reply};
use super::lessons::{is_held_out, Item, Kind};
use super::{batch_loss, encode_item, lr_schedule, Rng, Row};
use crate::model::{Ablation, Bicameral, Focus};
use crate::tokenizer::TokenId;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

/// A conversation between two speakers, turn by turn.
#[derive(Clone, Debug, PartialEq)]
pub struct Dialogue {
    /// The scene, when the source gives one (not trained on; put in front only on request).
    pub situation: String,
    /// The speaker of each turn (exactly two different names in a dialogue).
    pub speakers: Vec<String>,
    pub turns: Vec<String>,
    /// Which turns may be trained as Nova's replies (all of them for a dataset; only the replies
    /// the teacher marked for a chat).
    pub trainable: Vec<bool>,
}

impl Dialogue {
    /// The two speakers, in the order they first speak.
    pub fn pair(&self) -> (String, String) {
        let a = self.speakers[0].clone();
        let b = self.speakers.iter().find(|s| **s != a).cloned().unwrap_or_default();
        (a, b)
    }
    fn text(&self) -> String {
        self.speakers.iter().zip(&self.turns).map(|(s, t)| format!("{s}: {t}\n")).collect()
    }
}

/// A speaker's name at the start of a line: one to three words of letters (and `.'-`), the
/// first capitalised, followed by a colon.
fn speaker_of(line: &str) -> Option<(&str, &str)> {
    let (name, rest) = line.split_once(": ")?;
    let words: Vec<&str> = name.split(' ').collect();
    let ok = !name.is_empty()
        && name.len() <= 32
        && words.len() <= 3
        && words
            .iter()
            .all(|w| !w.is_empty() && w.chars().all(|c| c.is_ascii_alphabetic() || ".'-".contains(c)))
        && name.starts_with(|c: char| c.is_ascii_uppercase());
    ok.then_some((name, rest))
}

/// A document of `Name: text` lines (or `U:` / `N:` turns) as a dialogue. A first line starting
/// `Situation: ` (or any lines before the first turn) is the situation. Documents with fewer
/// than two turns or other than two speakers are not dialogues.
pub fn parse_dialogue(doc: &str) -> Option<Dialogue> {
    let mut situation = Vec::new();
    let (mut speakers, mut turns) = (Vec::new(), Vec::new());
    for line in doc.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if let Some(s) = line.strip_prefix("Situation: ") {
            situation.push(s.to_string());
            continue;
        }
        match speaker_of(line) {
            Some((name, said)) if !said.trim().is_empty() => {
                speakers.push(name.to_string());
                turns.push(crate::nexus::sanitize(said).replace(['\n', '\r', '\t'], " ").trim().to_string());
            }
            Some(_) => {}
            None if turns.is_empty() => situation.push(line.to_string()),
            None => {}
        }
    }
    let mut names = speakers.clone();
    names.sort();
    names.dedup();
    if turns.len() < 2 || names.len() != 2 {
        return None;
    }
    let trainable = vec![true; turns.len()];
    Some(Dialogue { situation: situation.join(" "), speakers, turns, trainable })
}

/// Phrases of an assistant's voice: dialogues containing one are left out.
const ASSISTANT_VOICE: [&str; 14] = [
    "as an ai",
    "i am an ai",
    "i'm an ai",
    "im an ai",
    "just an ai",
    "an ai language model",
    "a language model",
    "ai assistant",
    "virtual assistant",
    "digital assistant",
    "how can i assist you",
    "how may i assist you",
    "i don't have personal opinions",
    "i do not have personal opinions",
];

/// True when a dialogue speaks in an assistant's voice.
pub fn assistant_voice(d: &Dialogue) -> bool {
    d.turns.iter().any(|t| {
        let t = t.to_ascii_lowercase();
        ASSISTANT_VOICE.iter().any(|p| t.contains(p))
    })
}

/// `text` with every whole-word `name` replaced by `with`.
pub fn rename(text: &str, name: &str, with: &str) -> String {
    if name.is_empty() || with.is_empty() || name == with {
        return text.to_string();
    }
    let word = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find(name) {
        let before = rest[..i].chars().next_back().or_else(|| out.chars().next_back());
        let after = rest[i + name.len()..].chars().next();
        out.push_str(&rest[..i]);
        if word(before) || word(after) {
            out.push_str(name);
        } else {
            out.push_str(with);
        }
        rest = &rest[i + name.len()..];
    }
    out.push_str(rest);
    out
}

/// Conversations from one source.
#[derive(Clone, Debug)]
pub struct ConversationSet {
    pub name: String,
    pub dialogues: Vec<Dialogue>,
    /// Nova always plays this speaker (a chat: Nova's own replies); otherwise either speaker.
    pub nova: Option<String>,
    /// About one dialogue in ten is held out to check progress (not for a teacher's chats,
    /// which are too few and are not a test).
    pub hold_out: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConversationConfig {
    /// Tokens to train on (the run may stop earlier).
    pub tokens: u64,
    pub batch_size: usize,
    pub learning_rate: f32,
    /// Held-out checks over the run (at least).
    pub evaluations: usize,
    /// Held-out conversations scored at each check.
    pub eval_rows: usize,
    /// A checkpoint must lower the best held-out loss by at least this much.
    pub min_loss_improvement: f32,
    /// Stop after this many checks in a row without an accepted checkpoint.
    pub patience: usize,
    /// Practise every dialogue as both speakers.
    pub both_sides: bool,
    /// Nova's speaker's name becomes this name ("" keeps the names).
    pub own_name: String,
    /// Put a dialogue's situation in front of it (context, not trained on).
    pub situations: bool,
    /// Loss weight of the other speaker's lines (0: only Nova's lines are trained).
    pub user_weight: f32,
    /// Share of training rows from the lessons the model was taught.
    pub earlier_share: f32,
    /// Held-out Classroom questions checked at each evaluation (0: no check).
    pub guard_questions: usize,
    /// Share of those correct answers a checkpoint may lose (0: none).
    pub guard_drop: f32,
    pub seed: u64,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        ConversationConfig {
            tokens: 2_000_000,
            batch_size: 16,
            learning_rate: 3e-4,
            evaluations: 20,
            eval_rows: 256,
            min_loss_improvement: 0.002,
            patience: 6,
            both_sides: true,
            own_name: "Nova".into(),
            situations: false,
            user_weight: 0.,
            earlier_share: 0.2,
            guard_questions: 200,
            guard_drop: 0.02,
            seed: 1,
        }
    }
}

/// One reply written by the model at a check, next to the reply in the conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub prompt: String,
    pub reply: String,
    /// What the person in the held-out conversation said.
    pub original: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversationCheck {
    pub step: u64,
    pub tokens: u64,
    pub train_loss: f32,
    /// Loss of Nova's lines in the held-out conversations.
    pub held_out_loss: f32,
    pub best_loss: f32,
    /// Correct answers to the sampled held-out Classroom questions, and their number.
    pub classroom_correct: usize,
    pub classroom_total: usize,
    pub accepted: bool,
    pub reason: Option<String>,
    pub samples: Vec<Sample>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationEvent {
    Started {
        dialogues: usize,
        held_out_dialogues: usize,
        /// Dialogues left out: assistant's voice, or not two speakers.
        left_out_assistant: usize,
        /// Conversations practised (a dialogue counts once per side).
        practice: usize,
        earlier_items: usize,
        classroom_total: usize,
        baseline_loss: f32,
        baseline_classroom: usize,
        steps: u64,
    },
    Batch {
        step: u64,
        steps: u64,
        tokens: u64,
        loss: f32,
        learning_rate: f32,
    },
    Checked(ConversationCheck),
    Finished {
        accepted: bool,
        best_loss: f32,
        tokens: u64,
        stopped: bool,
        early_stop: bool,
    },
}

pub struct ConversationResult {
    /// The best accepted checkpoint; the input model unchanged when none was accepted.
    pub model: Bicameral,
    pub accepted: bool,
    pub baseline_loss: f32,
    pub best_loss: f32,
    pub baseline_classroom: usize,
    pub classroom_total: usize,
    pub tokens: u64,
    pub checks: Vec<ConversationCheck>,
    pub stopped: bool,
    pub early_stop: bool,
}

/// One conversation as Nova practises it: a dialogue and the speaker Nova plays.
#[derive(Clone, Debug)]
struct Practice<'a> {
    dialogue: &'a Dialogue,
    nova: String,
}

impl Practice<'_> {
    /// The dialogue's text with Nova's speaker named `own_name`.
    fn turns(&self, cfg: &ConversationConfig) -> Vec<(bool, String)> {
        self.dialogue
            .speakers
            .iter()
            .zip(&self.dialogue.turns)
            .map(|(s, t)| (*s == self.nova, rename(t, &self.nova, &cfg.own_name)))
            .collect()
    }

    /// The training row: turn-aligned, cut to the context (a random start turn with `rng`, the
    /// first otherwise). `None` when no trained reply fits.
    fn row(
        &self,
        model: &Bicameral,
        cfg: &ConversationConfig,
        rng: Option<&mut Rng>,
    ) -> Result<Option<Row>, String> {
        let tok = &model.tokenizer;
        let mut pieces: Vec<(Vec<TokenId>, Vec<f32>, bool)> = Vec::new();
        let situation = if cfg.situations && !self.dialogue.situation.is_empty() {
            let s =
                tok.encode(&format!("{}\n", rename(&self.dialogue.situation, &self.nova, &cfg.own_name)))?;
            let w = vec![0.; s.len()];
            Some((s, w))
        } else {
            None
        };
        for (i, (mine, text)) in self.turns(cfg).into_iter().enumerate() {
            if mine {
                let p = tok.encode("N: ")?;
                let r = tok.encode(&format!("{text}\n"))?;
                let trained = self.dialogue.trainable[i];
                let mut w = vec![cfg.user_weight; p.len()];
                w.resize(w.len() + r.len(), if trained { 1. } else { 0. });
                let mut ids = p;
                ids.extend(r);
                pieces.push((ids, w, trained));
            } else {
                let ids = tok.encode(&format!("U: {text}\n"))?;
                let w = vec![cfg.user_weight; ids.len()];
                pieces.push((ids, w, false));
            }
        }
        let limit = model.config.max_tokens + 1;
        let starts: Vec<usize> = {
            // A start turn from which a trained reply begins within the window.
            let mut offsets = Vec::with_capacity(pieces.len());
            let mut at = 0;
            for p in &pieces {
                offsets.push(at);
                at += p.0.len();
            }
            (0..pieces.len())
                .filter(|&s| (s..pieces.len()).any(|j| pieces[j].2 && offsets[j] - offsets[s] + 3 < limit))
                .collect()
        };
        let Some(&first) = starts.first() else { return Ok(None) };
        let start = match rng {
            Some(r) => starts[r.below(starts.len())],
            None => first,
        };
        let (mut ids, mut w) = situation.filter(|_| start == 0).unwrap_or_default();
        for p in &pieces[start..] {
            ids.extend(&p.0);
            w.extend(&p.1);
            if ids.len() >= limit {
                break;
            }
        }
        ids.truncate(limit);
        w.truncate(limit);
        if ids.len() < 2 || w[1..].iter().all(|&x| x == 0.) {
            return Ok(None);
        }
        let targets = ids[1..].to_vec();
        let mask = w[1..].to_vec();
        ids.pop();
        Ok(Some(Row { ids, targets, mask }))
    }

    /// The conversation up to Nova's first reply, and that reply.
    fn sample_prompt(&self, cfg: &ConversationConfig) -> Option<(String, String)> {
        let mut prompt = String::new();
        for (mine, text) in self.turns(cfg) {
            if mine {
                return (!prompt.is_empty()).then(|| (format!("{prompt}N: "), text));
            }
            prompt.push_str(&format!("U: {text}\n"));
        }
        None
    }
}

/// Every conversation practised, split into training and held-out. Returns them and the number
/// of dialogues left out for an assistant's voice.
fn practices<'a>(
    sets: &'a [ConversationSet],
    cfg: &ConversationConfig,
) -> (Vec<Practice<'a>>, Vec<Practice<'a>>, usize, usize, usize) {
    let (mut train, mut held) = (Vec::new(), Vec::new());
    let (mut left_out, mut dialogues, mut held_dialogues) = (0, 0, 0);
    for set in sets {
        for d in &set.dialogues {
            if assistant_voice(d) {
                left_out += 1;
                continue;
            }
            dialogues += 1;
            let held_out = set.hold_out && is_held_out(&d.text());
            held_dialogues += held_out as usize;
            let (a, b) = d.pair();
            let sides = match &set.nova {
                Some(n) => vec![n.clone()],
                None if cfg.both_sides => vec![b, a],
                None => vec![b],
            };
            for nova in sides {
                let p = Practice { dialogue: d, nova };
                if held_out {
                    held.push(p);
                } else {
                    train.push(p);
                }
            }
        }
    }
    (train, held, left_out, dialogues, held_dialogues)
}

/// Evenly spaced picks of at most `n` from `len`.
fn spread(len: usize, n: usize) -> Vec<usize> {
    if len <= n {
        return (0..len).collect();
    }
    (0..n).map(|i| i * len / n).collect()
}

/// A checkpoint and its check.
struct Measured {
    model: Bicameral,
    step: u64,
    tokens: u64,
    train_loss: f32,
    loss: f32,
    classroom: usize,
    samples: Vec<Sample>,
}

/// Practises conversations: trains on rows of the training conversations (and earlier lessons),
/// checks the held-out conversations and Classroom questions at regular intervals and keeps the
/// best checkpoint. `on_accepted` is called with every accepted checkpoint; `stop` ends the run
/// at the next batch (the best accepted checkpoint is returned).
pub fn converse(
    model: &Bicameral,
    sets: &[ConversationSet],
    earlier: &[Item],
    cfg: &ConversationConfig,
    observer: &mut dyn FnMut(&ConversationEvent),
    on_accepted: &mut dyn FnMut(&Bicameral) -> Result<(), String>,
    stop: &AtomicBool,
) -> Result<ConversationResult, String> {
    if cfg.batch_size == 0 || !(cfg.learning_rate > 0. && cfg.learning_rate.is_finite()) || cfg.tokens == 0 {
        return Err("Invalid conversation settings".into());
    }
    if !(0. ..1.).contains(&cfg.earlier_share) || !(0. ..=1.).contains(&cfg.guard_drop) {
        return Err("The earlier-lesson share must be below 1 and the allowed drop between 0 and 1".into());
    }
    let (train, held, left_out, dialogues, held_dialogues) = practices(sets, cfg);
    if train.is_empty() {
        return Err("No conversations to practise: choose a conversation dataset".into());
    }
    if held.is_empty() {
        return Err(
            "No held-out conversations to check progress: choose a larger conversation dataset".into()
        );
    }
    let mut current = model.clone();
    let mut rng = Rng::new(cfg.seed);
    // Earlier lessons: taught units (not held out) for the mix, held-out questions for the guard.
    let earlier_train: Vec<&Item> = earlier.iter().filter(|i| !i.held_out).collect();
    let questions: Vec<&Item> = earlier
        .iter()
        .filter(|i| i.held_out && i.kind != Kind::Prose && (i.answer.is_some() || !i.replies.is_empty()))
        .collect();
    let guard: Vec<&Item> =
        spread(questions.len(), cfg.guard_questions).into_iter().map(|i| questions[i]).collect();
    let classroom = |m: &Bicameral| -> Result<(usize, usize), String> {
        if guard.is_empty() {
            return Ok((0, 0));
        }
        let s = evaluate(m, &guard, Ablation::default())?.overall();
        Ok((s.correct, s.total))
    };
    let probes: Vec<Row> = {
        let mut rows = Vec::new();
        for i in spread(held.len(), cfg.eval_rows) {
            if let Some(r) = held[i].row(&current, cfg, None)? {
                rows.push(r);
            }
        }
        rows
    };
    if probes.is_empty() {
        return Err("The held-out conversations are too short to check progress".into());
    }
    let prompts: Vec<(String, String)> = held
        .iter()
        .filter(|p| p.nova == p.dialogue.pair().1)
        .filter_map(|p| p.sample_prompt(cfg))
        .filter(|(p, _)| p.len() < 400)
        .take(3)
        .collect();
    let baseline_loss = batch_loss(&current, &probes, Ablation::default())?;
    let (baseline_classroom, classroom_total) = classroom(&current)?;
    let allowed_loss = (cfg.guard_drop * baseline_classroom as f32).floor() as usize;
    let floor = baseline_classroom.saturating_sub(allowed_loss);
    // Tokens per row, estimated from the training conversations.
    let mut estimate = Rng::new(cfg.seed ^ 0xC0FFEE);
    let mut lengths = Vec::new();
    for i in spread(train.len(), 256) {
        if let Some(r) = train[i].row(&current, cfg, Some(&mut estimate))? {
            lengths.push(r.ids.len());
        }
    }
    let per_row = (lengths.iter().sum::<usize>() / lengths.len().max(1)).max(8) as u64;
    let steps = cfg.tokens.div_ceil(per_row * cfg.batch_size as u64).max(1);
    observer(&ConversationEvent::Started {
        dialogues,
        held_out_dialogues: held_dialogues,
        left_out_assistant: left_out,
        practice: train.len(),
        earlier_items: earlier_train.len(),
        classroom_total,
        baseline_loss,
        baseline_classroom,
        steps,
    });
    let warmup = (steps / 20).clamp(1, 200);
    let eval_every = (steps / cfg.evaluations.max(1) as u64).max(1);

    let mut best: Option<Bicameral> = None;
    let mut best_loss = baseline_loss;
    let mut checks = Vec::new();
    let mut misses = 0;
    let (mut stopped, mut early_stop) = (false, false);
    let mut tokens = 0u64;
    let (mut since, mut since_n) = (0f64, 0usize);
    let mut judge = |m: Measured, observer: &mut dyn FnMut(&ConversationEvent)| -> Result<bool, String> {
        let improves = m.loss.is_finite() && m.loss < best_loss - cfg.min_loss_improvement;
        let keeps = guard.is_empty() || m.classroom >= floor;
        let accepted = improves && keeps;
        let reason = if accepted {
            None
        } else if !improves {
            Some(format!("not better than {best_loss:.4} by at least {}", cfg.min_loss_improvement))
        } else {
            Some(format!(
                "better, but it answers {} of the {classroom_total} Classroom questions; at least {floor} are needed (it started with {baseline_classroom})",
                m.classroom
            ))
        };
        if accepted {
            best_loss = m.loss;
            on_accepted(&m.model)?;
            best = Some(m.model);
            misses = 0;
        } else {
            misses += 1;
        }
        let c = ConversationCheck {
            step: m.step,
            tokens: m.tokens,
            train_loss: m.train_loss,
            held_out_loss: m.loss,
            best_loss,
            classroom_correct: m.classroom,
            classroom_total,
            accepted,
            reason,
            samples: m.samples,
        };
        observer(&ConversationEvent::Checked(c.clone()));
        checks.push(c);
        Ok(misses >= cfg.patience.max(1))
    };
    // The Classroom check and the samples (CPU); the held-out loss is given when the GPU
    // measured it.
    let measure = |model: Bicameral, step: u64, tokens: u64, train_loss: f32, loss: Option<f32>| {
        let loss = match loss {
            Some(l) => l,
            None => batch_loss(&model, &probes, Ablation::default())?,
        };
        let (classroom, _) = classroom(&model)?;
        let mut samples = Vec::new();
        for (prompt, original) in &prompts {
            samples.push(Sample {
                prompt: prompt.clone(),
                reply: reply(&model, prompt, 48, Ablation::default()).unwrap_or_default(),
                original: original.clone(),
            });
        }
        Ok::<_, String>(Measured { model, step, tokens, train_loss, loss, classroom, samples })
    };

    let mut trainer = super::Trainer::new(&current);
    std::thread::scope(|scope| -> Result<(), String> {
        let mut pending: Option<std::thread::ScopedJoinHandle<'_, Result<Measured, String>>> = None;
        let join = |h: std::thread::ScopedJoinHandle<'_, Result<Measured, String>>| {
            h.join().map_err(|_| "A conversation check failed".to_string())?
        };
        for step in 0..steps {
            if stop.load(Ordering::Acquire) {
                stopped = true;
                break;
            }
            if pending.as_ref().is_some_and(|h| h.is_finished())
                && judge(join(pending.take().unwrap())?, observer)?
            {
                early_stop = true;
                break;
            }
            let mut rows = Vec::with_capacity(cfg.batch_size);
            let mut tries = 0;
            while rows.len() < cfg.batch_size && tries < cfg.batch_size * 20 {
                tries += 1;
                let from_earlier = !earlier_train.is_empty() && rng.unit() < cfg.earlier_share as f64;
                let row = if from_earlier {
                    let item = earlier_train[rng.below(earlier_train.len())];
                    encode_item(&current.tokenizer, &item.segments, current.config.max_tokens, &mut rng)?
                } else {
                    let p = &train[rng.below(train.len())];
                    p.row(&current, cfg, Some(&mut rng))?
                };
                rows.extend(row);
            }
            if rows.is_empty() {
                return Err("No training rows could be made from these conversations".into());
            }
            let lr = lr_schedule(step, cfg.learning_rate, warmup, steps, 0.1);
            let loss = trainer.step(&mut current, &rows, Ablation::default(), lr, Focus::Joint)?;
            if !loss.is_finite() {
                return Err("Training diverged (loss is not finite); try a lower learning rate".into());
            }
            tokens += rows.iter().map(|r| r.ids.len() as u64).sum::<u64>();
            since += loss as f64;
            since_n += 1;
            observer(&ConversationEvent::Batch { step: step + 1, steps, tokens, loss, learning_rate: lr });
            if (step + 1) % eval_every == 0 || step + 1 == steps {
                if let Some(h) = pending.take() {
                    if judge(join(h)?, observer)? {
                        early_stop = true;
                        break;
                    }
                }
                trainer.sync(&mut current)?;
                let train_loss = (since / since_n.max(1) as f64) as f32;
                (since, since_n) = (0., 0);
                let snapshot = current.clone();
                if trainer.on_gpu() {
                    // The held-out loss on the GPU; the Classroom check and samples on a CPU
                    // thread while the GPU keeps training.
                    let loss = trainer
                        .gpu_loss(&current, &probes, Ablation::default())
                        .unwrap_or_else(|| batch_loss(&current, &probes, Ablation::default()))?;
                    let measure = &measure;
                    pending = Some(
                        scope.spawn(move || measure(snapshot, step + 1, tokens, train_loss, Some(loss))),
                    );
                } else if judge(measure(snapshot, step + 1, tokens, train_loss, None)?, observer)? {
                    early_stop = true;
                    break;
                }
            }
        }
        if let Some(h) = pending.take() {
            if judge(join(h)?, observer)? && !stopped {
                early_stop = true;
            }
        }
        Ok(())
    })?;
    let accepted = best.is_some();
    observer(&ConversationEvent::Finished { accepted, best_loss, tokens, stopped, early_stop });
    Ok(ConversationResult {
        model: best.unwrap_or_else(|| model.clone()),
        accepted,
        baseline_loss,
        best_loss,
        baseline_classroom,
        classroom_total,
        tokens,
        checks,
        stopped,
        early_stop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Config;

    fn small() -> Bicameral {
        let cfg = Config {
            d: 16,
            heads: 2,
            ff: 24,
            blocks: 2,
            split: 1,
            r1: 12,
            r2: 10,
            max_tokens: 64,
            ..Config::default()
        };
        Bicameral::with_config(cfg, 3)
    }

    #[test]
    fn documents_become_dialogues() {
        let d = parse_dialogue(
            "Situation: Veda went to church.\nVeda: Hi, Father.\nPriest: Hello, Veda.\nVeda: Tell me more.",
        )
        .unwrap();
        assert_eq!(d.situation, "Veda went to church.");
        assert_eq!(d.speakers, ["Veda", "Priest", "Veda"]);
        assert_eq!(d.pair(), ("Veda".to_string(), "Priest".to_string()));
        let u = parse_dialogue("Once upon a time.\nU: hello\nN: hi there").unwrap();
        assert_eq!((u.situation.as_str(), u.speakers.len()), ("Once upon a time.", 2));
        assert!(parse_dialogue("Ann: hi\nBen: hi\nCal: hi").is_none(), "three speakers");
        assert!(parse_dialogue("Ann: hi").is_none(), "one turn");
        assert!(parse_dialogue("just some prose\nwith two lines").is_none());
        let a = parse_dialogue("Ann: Can you help?\nBen: As an AI, I can help you with that.").unwrap();
        assert!(assistant_voice(&a));
    }

    #[test]
    fn names_are_replaced_as_whole_words() {
        assert_eq!(
            rename("Hello, Veda. Vedas and Veda's cat.", "Veda", "Nova"),
            "Hello, Nova. Vedas and Nova's cat."
        );
        assert_eq!(rename("Ann and Anna", "Ann", "Nova"), "Nova and Anna");
    }

    #[test]
    fn only_novas_lines_are_trained_and_both_sides_are_practised() {
        let m = small();
        let d = parse_dialogue("Ann: Hi Ben, how are you?\nBen: Good, Ann. And you?\nAnn: Fine, thanks.")
            .unwrap();
        let cfg = ConversationConfig::default();
        let as_ben = Practice { dialogue: &d, nova: "Ben".into() };
        assert_eq!(
            as_ben.turns(&cfg),
            [
                (false, "Hi Nova, how are you?".into()),
                (true, "Good, Ann. And you?".into()),
                (false, "Fine, thanks.".into())
            ]
        );
        let row = as_ben.row(&m, &cfg, None).unwrap().unwrap();
        let text = m.tokenizer.decode(&row.ids).unwrap()
            + &m.tokenizer.decode(&row.targets[row.targets.len() - 1..]).unwrap();
        assert_eq!(text, "U: Hi Nova, how are you?\nN: Good, Ann. And you?\nU: Fine, thanks.\n");
        // Weighted targets are exactly Nova's reply and its line break.
        let trained: Vec<TokenId> =
            row.targets.iter().zip(&row.mask).filter(|(_, &w)| w > 0.).map(|(&t, _)| t).collect();
        assert_eq!(m.tokenizer.decode(&trained).unwrap(), "Good, Ann. And you?\n");
        let as_ann = Practice { dialogue: &d, nova: "Ann".into() };
        assert_eq!(as_ann.sample_prompt(&cfg), None, "Ann speaks first");
        assert_eq!(
            as_ben.sample_prompt(&cfg),
            Some(("U: Hi Nova, how are you?\nN: ".into(), "Good, Ann. And you?".into()))
        );
        let sets =
            [ConversationSet { name: "x".into(), dialogues: vec![d.clone()], nova: None, hold_out: false }];
        let (train, held, ..) = practices(&sets, &cfg);
        assert_eq!((train.len(), held.len()), (2, 0));
        // A chat: only the marked replies are trained.
        let mut chat = d.clone();
        chat.trainable = vec![true, false, true];
        let p = Practice { dialogue: &chat, nova: "Ben".into() };
        assert!(p.row(&m, &cfg, None).unwrap().is_none(), "Ben's only reply is not marked");
    }

    #[test]
    fn long_conversations_are_cut_at_turns_to_the_context() {
        let m = small();
        let mut doc = String::new();
        for i in 0..30 {
            doc.push_str(&format!("Ann: question {i} goes here\nBen: answer {i} goes here\n"));
        }
        let d = parse_dialogue(&doc).unwrap();
        let p = Practice { dialogue: &d, nova: "Ben".into() };
        let cfg = ConversationConfig::default();
        let mut rng = Rng::new(4);
        for _ in 0..20 {
            let r = p.row(&m, &cfg, Some(&mut rng)).unwrap().unwrap();
            assert!(r.ids.len() <= m.config.max_tokens);
            let text = m.tokenizer.decode(&r.ids).unwrap();
            assert!(text.starts_with("U: ") || text.starts_with("N: "), "{text:?}");
            assert!(r.mask.iter().any(|&w| w > 0.));
        }
    }

    #[test]
    fn practice_runs_and_keeps_only_better_checkpoints() {
        let m = small();
        let mut dialogues = Vec::new();
        for i in 0..120 {
            let doc = format!(
                "Ann: Do you like the color number {i}?\nBen: Yes, I like it.\nAnn: Why?\nBen: Because it is bright."
            );
            dialogues.push(parse_dialogue(&doc).unwrap());
        }
        let sets = [ConversationSet { name: "x".into(), dialogues, nova: None, hold_out: true }];
        let cfg = ConversationConfig {
            tokens: 20_000,
            batch_size: 8,
            learning_rate: 3e-3,
            evaluations: 4,
            ..ConversationConfig::default()
        };
        let mut events = Vec::new();
        let mut saved = 0;
        let r = converse(
            &m,
            &sets,
            &[],
            &cfg,
            &mut |e| events.push(e.clone()),
            &mut |_| {
                saved += 1;
                Ok(())
            },
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(r.accepted, "the replies are easy to learn");
        assert!(r.best_loss < r.baseline_loss);
        assert_eq!(saved, r.checks.iter().filter(|c| c.accepted).count());
        assert!(matches!(events.first(), Some(ConversationEvent::Started { .. })));
        assert!(r.checks.iter().all(|c| c.samples.iter().all(|s| s.prompt.ends_with("N: "))));
        let stopped =
            converse(&m, &sets, &[], &cfg, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(true)).unwrap();
        assert!(stopped.stopped && !stopped.accepted);
    }
}
