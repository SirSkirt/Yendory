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
//! * **Earlier material.** Shares of the training rows come from the lessons the model was
//!   taught (`earlier_share`) and from text it read (`reading_share`), and a checkpoint may not
//!   raise the held-out loss on that text by more than `reading_rise`.
//!
//! How Nova is taught (each part can be switched off, to measure what it changes):
//!
//! * **Easy to hard** (`curriculum`): short exchanges, a few turns, whole conversations. A stage
//!   is left when it is *mastered* (its held-out loss below `competency_loss`, or
//!   `competency_gain` lower than when the stage began) *and* no longer improving. A plateau
//!   before mastery means Nova is stuck, not ready: the learning rate is halved (up to
//!   `stuck_backoffs` times), and if she is still stuck the run stops there (or moves on,
//!   marked as not mastered, with `advance_when_stuck`). Earlier stages are revisited
//!   (`review_share`).
//! * **Targeted practice** (`targeted`): conversations are drawn by Nova's own loss on them,
//!   relative to the median, between `target_floor` (what she handles is still practised) and
//!   `target_cap` (hard ones cannot take over); a conversation just practised is drawn less until
//!   the pool is measured again at the next check, when a share of the learned ones make way for
//!   new conversations.
//! * **Listening** (`context_weight`, `context_practice`, `context_tests`): replies that use
//!   something said before the turn they answer weigh more; generated listening practice is mixed
//!   in; and every check runs the context-sensitivity tests of `learn::listening` (pairs whose
//!   answers must change with an earlier fact, with words never practised) and measures how much
//!   earlier turns lower the loss of replies that use them.
//! * **Reply checks** (`reply_checks`): replies Nova writes mid-conversation are checked for
//!   emptiness, repetition, echoing, staying on topic, questions, its own name and an assistant's
//!   voice. They are measured, never trained on.
use super::eval::{evaluate, reply};
use super::lessons::{is_held_out, Item, Kind};
#[cfg(test)]
use super::listening::Skill;
use super::listening::{context_practice, context_tests, score, ContextScore, ContextTest};
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
    /// Held-out conversation windows scored at each check (shared between the stages).
    pub eval_rows: usize,
    /// A checkpoint must lower the best held-out loss by at least this much.
    pub min_loss_improvement: f32,
    /// Stop after this many checks in a row without an accepted checkpoint (in the last stage).
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
    /// Share of the Classroom answers that were right before practice that a checkpoint may get
    /// wrong (each answer counts on its own: new right answers do not make up for lost ones).
    pub guard_drop: f32,
    pub seed: u64,

    // How Nova is taught. Each can be switched off, to measure what it changes.
    /// Easy to hard: short exchanges, a few turns, whole conversations (`STAGES`).
    pub curriculum: bool,
    /// A stage is mastered when the held-out loss of its conversations is at most this...
    pub competency_loss: f32,
    /// ...or has fallen by at least this share since the stage began.
    pub competency_gain: f32,
    /// Checks in a row without a lower held-out loss in the stage: a plateau.
    pub stage_patience: usize,
    /// A plateau before mastery means Nova is stuck: the learning rate is halved, at most this
    /// many times in a stage...
    pub stuck_backoffs: u32,
    /// ...and then the run moves on to the next stage (recorded as not mastered) if true, or
    /// stops there if false.
    pub advance_when_stuck: bool,
    /// Share of conversation rows taken from earlier stages, to keep them up.
    pub review_share: f32,
    /// Practise most what Nova finds hard: conversations are drawn by Nova's own loss on them.
    pub targeted: bool,
    /// Conversations measured and drawn from at a time (refreshed at every check).
    pub pool_size: usize,
    /// A conversation is drawn at most this many times as often as an average one...
    pub target_cap: f32,
    /// ...and at least this often, so what Nova already handles is still practised.
    pub target_floor: f32,
    /// Loss weight of a reply that uses something said before the turn it answers (a name, a
    /// place, a fact from earlier); 1 gives no emphasis.
    pub context_weight: f32,
    /// Share of training rows from generated conversations that practise the listening skills
    /// (with other words than the context tests).
    pub context_practice: f32,
    /// Context-sensitivity test pairs per skill at each check (0: none).
    pub context_tests: usize,
    /// Automatic checks of Nova's replies at each check.
    pub reply_checks: bool,
    /// Share of training rows from text the model read (rehearsal of its English).
    pub reading_share: f32,
    /// The held-out loss on that text may rise at most this share above where it started.
    pub reading_rise: f32,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        ConversationConfig {
            tokens: 2_000_000,
            batch_size: 16,
            learning_rate: 3e-4,
            evaluations: 30,
            eval_rows: 384,
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
            curriculum: true,
            competency_loss: 2.0,
            competency_gain: 0.1,
            stage_patience: 3,
            stuck_backoffs: 2,
            advance_when_stuck: false,
            review_share: 0.15,
            targeted: true,
            pool_size: 4096,
            target_cap: 3.,
            target_floor: 0.25,
            context_weight: 2.,
            context_practice: 0.1,
            context_tests: 12,
            reply_checks: true,
            reading_share: 0.1,
            reading_rise: 0.02,
        }
    }
}

/// The curriculum's stages: a name and the most turns a practice window holds (`None`: as many
/// as fit the context).
pub const STAGES: [(&str, Option<usize>); 3] =
    [("short exchanges", Some(2)), ("a few turns", Some(6)), ("whole conversations", None)];

/// One reply written by the model at a check, next to the reply in the conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub prompt: String,
    pub reply: String,
    /// What the person in the held-out conversation said.
    pub original: String,
}

/// Automatic checks of replies Nova wrote in the middle of held-out conversations: how many of
/// `replies` showed each trait. They describe the replies; none is required of a checkpoint.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReplyChecks {
    pub replies: usize,
    /// Nothing written.
    pub empty: usize,
    /// The same three words twice in the reply, or Nova's previous reply said again.
    pub repeats: usize,
    /// Mostly the other person's last line said back.
    pub echoes: usize,
    /// Shares a topic word with the last two turns (short replies count as on topic).
    pub on_topic: usize,
    /// Asks something.
    pub questions: usize,
    /// Calls itself by another name ("I am Ben", "my name is Ben").
    pub wrong_name: usize,
    /// Speaks in an assistant's voice.
    pub assistant: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversationCheck {
    pub step: u64,
    pub tokens: u64,
    pub train_loss: f32,
    /// Loss of Nova's lines in the held-out conversations (the mean of the stages' losses).
    pub held_out_loss: f32,
    pub best_loss: f32,
    /// The curriculum stage being practised (0-based) and each stage's held-out loss.
    pub stage: usize,
    pub stage_losses: Vec<f32>,
    /// How much the earlier turns lower the loss of replies that use them (nats per token;
    /// higher: Nova uses the conversation more).
    pub listening: Option<f32>,
    /// Sampled held-out Classroom questions: right now, of all, right before practice and wrong
    /// now, and wrong before and right now.
    pub classroom_correct: usize,
    pub classroom_total: usize,
    pub classroom_lost: usize,
    pub classroom_gained: usize,
    /// Held-out loss on text the model read.
    pub reading_loss: Option<f32>,
    pub context: Option<ContextScore>,
    pub reply_checks: Option<ReplyChecks>,
    pub learning_rate_scale: f32,
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
        reading_documents: usize,
        classroom_total: usize,
        baseline_loss: f32,
        baseline_stage_losses: Vec<f32>,
        baseline_classroom: usize,
        baseline_listening: Option<f32>,
        baseline_reading: Option<f32>,
        baseline_context: Option<ContextScore>,
        baseline_reply_checks: Option<ReplyChecks>,
        steps: u64,
        stages: Vec<String>,
    },
    Batch {
        step: u64,
        steps: u64,
        tokens: u64,
        loss: f32,
        learning_rate: f32,
        stage: usize,
    },
    Checked(ConversationCheck),
    /// The curriculum moved to another stage.
    Stage {
        stage: usize,
        name: String,
        /// The previous stage was mastered (false: moved on while stuck).
        mastered: bool,
        reason: String,
    },
    /// Stuck in a stage: the learning rate was lowered.
    LearningRateLowered {
        scale: f32,
        reason: String,
    },
    Finished {
        accepted: bool,
        best_loss: f32,
        tokens: u64,
        stopped: bool,
        early_stop: bool,
        stage: usize,
        /// Why the run ended early when Nova was stuck.
        stuck: Option<String>,
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
    pub baseline_context: Option<ContextScore>,
    /// The check of the best accepted checkpoint.
    pub best_check: Option<ConversationCheck>,
    pub tokens: u64,
    pub checks: Vec<ConversationCheck>,
    pub stopped: bool,
    pub early_stop: bool,
    pub stage: usize,
    pub stuck: Option<String>,
}

/// Words too common to show that a reply uses something said earlier.
const COMMON: [&str; 72] = [
    "the", "and", "you", "your", "are", "was", "were", "for", "that", "this", "with", "have", "has", "had",
    "not", "but", "what", "when", "where", "who", "how", "why", "can", "could", "will", "would", "should",
    "just", "like", "yes", "yeah", "okay", "really", "very", "too", "all", "any", "some", "about", "there",
    "they", "them", "then", "than", "she", "her", "his", "him", "its", "our", "out", "get", "got", "know",
    "think", "good", "well", "one", "now", "want", "going", "from", "did", "does", "doing", "been", "being",
    "it's", "i'm", "don't", "that's", "here",
];

fn content_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
        .map(|w| w.trim_matches('\'').to_ascii_lowercase())
        .filter(|w| w.len() >= 3 && !COMMON.contains(&w.as_str()))
        .collect()
}

/// A reply uses something said earlier: a topic word of the turns before the one it answers
/// (within `earlier`) that the turn it answers does not have.
fn uses_earlier(reply: &str, answered: &str, earlier: &[&str]) -> bool {
    let near = content_words(answered);
    let before: Vec<String> = earlier.iter().flat_map(|t| content_words(t)).collect();
    content_words(reply).iter().any(|w| before.contains(w) && !near.contains(w))
}

/// Turns of a conversation: `(true, text)` is Nova's.
type Turns = Vec<(bool, String)>;

/// One conversation as Nova practises it: a dialogue and the speaker Nova plays.
#[derive(Clone, Debug)]
struct Practice<'a> {
    dialogue: &'a Dialogue,
    nova: String,
}

/// How a row is cut and weighted.
#[derive(Clone, Copy)]
struct Window {
    /// Most turns in the window (`None`: as many as fit).
    turns: Option<usize>,
    /// Weight replies that use earlier turns by `context_weight` (training only; checks weigh
    /// every reply the same).
    emphasis: bool,
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

    /// The training row: turn-aligned, at most `window.turns` turns, cut to the context (a
    /// random start turn with `rng`, the first possible otherwise). `None` when no trained reply
    /// fits.
    fn row(
        &self,
        model: &Bicameral,
        cfg: &ConversationConfig,
        rng: Option<&mut Rng>,
        window: Window,
    ) -> Result<Option<Row>, String> {
        let tok = &model.tokenizer;
        let turns = self.turns(cfg);
        let mut pieces: Vec<(Vec<TokenId>, usize, bool)> = Vec::new();
        for (i, (mine, text)) in turns.iter().enumerate() {
            if *mine {
                let p = tok.encode("N: ")?;
                let prefix = p.len();
                let mut ids = p;
                ids.extend(tok.encode(&format!("{text}\n"))?);
                pieces.push((ids, prefix, self.dialogue.trainable[i]));
            } else {
                pieces.push((tok.encode(&format!("U: {text}\n"))?, 0, false));
            }
        }
        let limit = model.config.max_tokens + 1;
        let most = window.turns.unwrap_or(usize::MAX);
        let mut offsets = Vec::with_capacity(pieces.len());
        let mut at = 0;
        for p in &pieces {
            offsets.push(at);
            at += p.0.len();
        }
        // A start turn from which a trained reply begins within the window.
        let starts: Vec<usize> = (0..pieces.len())
            .filter(|&s| {
                (s..pieces.len().min(s.saturating_add(most)))
                    .any(|j| pieces[j].2 && offsets[j] - offsets[s] + pieces[j].1 + 2 < limit)
            })
            .collect();
        let Some(&first) = starts.first() else { return Ok(None) };
        let start = match rng {
            Some(r) => starts[r.below(starts.len())],
            None => first,
        };
        let end = pieces.len().min(start.saturating_add(most));
        let (mut ids, mut w) = (Vec::new(), Vec::new());
        if cfg.situations && start == 0 && !self.dialogue.situation.is_empty() {
            let s =
                tok.encode(&format!("{}\n", rename(&self.dialogue.situation, &self.nova, &cfg.own_name)))?;
            w.resize(s.len(), 0.);
            ids = s;
        }
        for j in start..end {
            let (p, prefix, trained) = &pieces[j];
            let weight = if !trained {
                0.
            } else if window.emphasis && cfg.context_weight != 1. && j >= start + 2 {
                let earlier: Vec<&str> = turns[start..j - 1].iter().map(|t| t.1.as_str()).collect();
                if uses_earlier(&turns[j].1, &turns[j - 1].1, &earlier) {
                    cfg.context_weight
                } else {
                    1.
                }
            } else {
                1.
            };
            if *prefix > 0 || !turns[j].0 {
                // The other person's line, or Nova's "N: " prefix.
                let n = if turns[j].0 { *prefix } else { p.len() };
                w.resize(w.len() + n, cfg.user_weight);
            }
            if turns[j].0 {
                w.resize(w.len() + p.len() - prefix, weight);
            }
            ids.extend(p);
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

    /// The conversation up to Nova's `n`-th reply (counting from 0) that has something before
    /// it, and that reply.
    fn prompt_at(&self, cfg: &ConversationConfig, n: usize) -> Option<(String, String, Turns)> {
        let turns = self.turns(cfg);
        let mut prompt = String::new();
        let mut seen = 0;
        for (j, (mine, text)) in turns.iter().enumerate() {
            if *mine && j > 0 {
                if seen == n {
                    return Some((format!("{prompt}N: "), text.clone(), turns[..j].to_vec()));
                }
                seen += 1;
            }
            prompt.push_str(if *mine { "N: " } else { "U: " });
            prompt.push_str(text);
            prompt.push('\n');
        }
        None
    }

    /// Rows for the listening measure: replies that use earlier turns, scored with the whole
    /// conversation before them and with only the turn they answer.
    fn listening_rows(&self, model: &Bicameral, cfg: &ConversationConfig) -> Result<Vec<(Row, Row)>, String> {
        let tok = &model.tokenizer;
        let turns = self.turns(cfg);
        let mut out = Vec::new();
        for j in 2..turns.len() {
            if !turns[j].0 || turns[j - 1].0 {
                continue;
            }
            let earlier: Vec<&str> = turns[..j - 1].iter().map(|t| t.1.as_str()).collect();
            if !uses_earlier(&turns[j].1, &turns[j - 1].1, &earlier) {
                continue;
            }
            let reply = tok.encode(&format!("{}\n", turns[j].1))?;
            let row = |context: &[(bool, String)]| -> Result<Option<Row>, String> {
                let mut ids = Vec::new();
                for (mine, t) in context {
                    ids.extend(tok.encode(&format!("{}{t}\n", if *mine { "N: " } else { "U: " }))?);
                }
                ids.extend(tok.encode("N: ")?);
                let skip = ids.len();
                ids.extend(&reply);
                if ids.len() > model.config.max_tokens + 1 {
                    return Ok(None);
                }
                let mut mask = vec![0.; ids.len()];
                mask[skip..].fill(1.);
                let targets = ids[1..].to_vec();
                let mask = mask[1..].to_vec();
                ids.pop();
                Ok(Some(Row { ids, targets, mask }))
            };
            if let (Some(full), Some(short)) = (row(&turns[..j])?, row(&turns[j - 1..j])?) {
                out.push((full, short));
            }
        }
        Ok(out)
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

/// Greedy replies to many prompts (in parallel threads).
fn replies(model: &Bicameral, prompts: &[String], max_tokens: usize) -> Vec<String> {
    if prompts.is_empty() {
        return Vec::new();
    }
    let threads = super::cpu_threads().max(1).min(prompts.len());
    let chunk = prompts.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = prompts
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|p| reply(model, p, max_tokens, Ablation::default()).unwrap_or_default())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
    })
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .collect()
}

/// The automatic reply checks of one reply. `before` is the conversation it answers.
fn check_reply(reply: &str, before: &[(bool, String)], own_name: &str, out: &mut ReplyChecks) {
    out.replies += 1;
    let w = words(reply);
    if w.is_empty() {
        out.empty += 1;
        return;
    }
    let trigrams: Vec<&[String]> = w.windows(3).collect();
    let repeated_trigram = trigrams.iter().enumerate().any(|(i, t)| trigrams[i + 1..].contains(t));
    let previous_own = before.iter().rev().find(|t| t.0).map(|t| words(&t.1));
    if repeated_trigram || previous_own.is_some_and(|p| p.len() >= 3 && p == w) {
        out.repeats += 1;
    }
    if let Some(last) = before.last().filter(|t| !t.0) {
        let lw = words(&last.1);
        let shared = w.iter().filter(|x| lw.contains(x)).count();
        if lw.len() >= 3 && shared * 10 >= w.len() * 8 {
            out.echoes += 1;
        }
    }
    let recent: Vec<String> = before.iter().rev().take(2).flat_map(|t| content_words(&t.1)).collect();
    if w.len() <= 4 || content_words(reply).iter().any(|x| recent.contains(x)) {
        out.on_topic += 1;
    }
    if reply.contains('?') {
        out.questions += 1;
    }
    let lower = reply.to_ascii_lowercase();
    if ASSISTANT_VOICE.iter().any(|p| lower.contains(p)) {
        out.assistant += 1;
    }
    // "my name is X", "I am X", "I'm X" with a capitalised X that is not Nova's name.
    let raw: Vec<&str> =
        reply.split(|c: char| !c.is_ascii_alphanumeric() && c != '\'').filter(|x| !x.is_empty()).collect();
    let named = raw.windows(4).any(|q| {
        q[0].eq_ignore_ascii_case("my") && q[1] == "name" && q[2] == "is" && other_name(q[3], own_name)
    }) || raw.windows(3).any(|q| q[0] == "I" && q[1] == "am" && other_name(q[2], own_name))
        || raw.windows(2).any(|q| q[0] == "I'm" && other_name(q[1], own_name));
    if named {
        out.wrong_name += 1;
    }
}

fn other_name(word: &str, own: &str) -> bool {
    const NOT_NAMES: [&str; 6] = ["I", "Not", "So", "Very", "Sorry", "Fine"];
    word.starts_with(|c: char| c.is_ascii_uppercase())
        && !NOT_NAMES.contains(&word)
        && !own.split(' ').any(|o| o.eq_ignore_ascii_case(word))
}

/// What is fixed for a whole run: the held-out material and the tests.
struct Fixed<'a> {
    cfg: &'a ConversationConfig,
    /// Held-out rows per stage.
    probes: Vec<Vec<Row>>,
    listening: (Vec<Row>, Vec<Row>),
    guard: Vec<&'a Item>,
    reading_probes: Vec<Row>,
    tests: Vec<ContextTest>,
    /// Mid-conversation prompts for the reply checks, with the turns before them.
    check_prompts: Vec<(String, Vec<(bool, String)>)>,
    /// Prompts shown with Nova's reply and the person's.
    samples: Vec<(String, String)>,
}

/// A checkpoint and its measurements.
struct Measured {
    model: Bicameral,
    step: u64,
    tokens: u64,
    train_loss: f32,
    stage_losses: Vec<f32>,
    listening: Option<f32>,
    classroom: Vec<bool>,
    reading: Option<f32>,
    context: Option<ContextScore>,
    reply_checks: Option<ReplyChecks>,
    samples: Vec<Sample>,
}

/// Losses measured on the GPU before the rest of a check runs on a CPU thread.
struct GpuPart {
    stage_losses: Vec<f32>,
    listening: Option<f32>,
    reading: Option<f32>,
}

impl Fixed<'_> {
    fn losses(&self, model: &Bicameral) -> Result<GpuPart, String> {
        let stage_losses = self
            .probes
            .iter()
            .map(|rows| batch_loss(model, rows, Ablation::default()))
            .collect::<Result<_, _>>()?;
        let listening = if self.listening.0.is_empty() {
            None
        } else {
            Some(
                batch_loss(model, &self.listening.1, Ablation::default())?
                    - batch_loss(model, &self.listening.0, Ablation::default())?,
            )
        };
        let reading = if self.reading_probes.is_empty() {
            None
        } else {
            Some(batch_loss(model, &self.reading_probes, Ablation::default())?)
        };
        Ok(GpuPart { stage_losses, listening, reading })
    }

    fn classroom(&self, model: &Bicameral) -> Result<Vec<bool>, String> {
        if self.guard.is_empty() {
            return Ok(Vec::new());
        }
        Ok(evaluate(model, &self.guard, Ablation::default())?.outcomes.iter().map(|o| o.correct).collect())
    }

    fn context(&self, model: &Bicameral) -> Option<ContextScore> {
        if self.tests.is_empty() {
            return None;
        }
        let prompts: Vec<String> = self.tests.iter().flat_map(|t| [t.a.prompt(), t.b.prompt()]).collect();
        Some(score(&self.tests, &replies(model, &prompts, 24)))
    }

    fn reply_checks(&self, model: &Bicameral) -> Option<ReplyChecks> {
        if !self.cfg.reply_checks || self.check_prompts.is_empty() {
            return None;
        }
        let prompts: Vec<String> = self.check_prompts.iter().map(|p| p.0.clone()).collect();
        let mut out = ReplyChecks::default();
        for (r, (_, before)) in replies(model, &prompts, 48).iter().zip(&self.check_prompts) {
            check_reply(r, before, &self.cfg.own_name, &mut out);
        }
        Some(out)
    }

    /// Measures a checkpoint (the answers Nova writes: Classroom, context tests, reply checks,
    /// samples). `gpu` holds the losses when the GPU measured them.
    fn measure(
        &self,
        model: Bicameral,
        step: u64,
        tokens: u64,
        train_loss: f32,
        gpu: Option<GpuPart>,
    ) -> Result<Measured, String> {
        let GpuPart { stage_losses, listening, reading } = match gpu {
            Some(g) => g,
            None => self.losses(&model)?,
        };
        let classroom = self.classroom(&model)?;
        let context = self.context(&model);
        let reply_checks = self.reply_checks(&model);
        let samples = self
            .samples
            .iter()
            .map(|(prompt, original)| Sample {
                prompt: prompt.clone(),
                reply: reply(&model, prompt, 48, Ablation::default()).unwrap_or_default(),
                original: original.clone(),
            })
            .collect();
        Ok(Measured {
            model,
            step,
            tokens,
            train_loss,
            stage_losses,
            listening,
            classroom,
            reading,
            context,
            reply_checks,
            samples,
        })
    }
}

/// Nova's loss on each pool conversation (its first window of the stage): on the GPU when
/// training is there (the trainer's weights, synced to `model`), else on the CPU.
fn measure_pool(
    trainer: Option<&super::Trainer>,
    model: &Bicameral,
    cfg: &ConversationConfig,
    train: &[Practice<'_>],
    ids: &[usize],
    window: Window,
) -> Result<Vec<(usize, f32)>, String> {
    let mut rows = Vec::new();
    let mut keep = Vec::new();
    for &i in ids {
        if let Some(r) = train[i].row(model, cfg, None, Window { emphasis: false, ..window })? {
            rows.push(r);
            keep.push(i);
        }
    }
    let losses = match trainer.and_then(|t| t.gpu_row_losses(model, &rows, Ablation::default())) {
        Some(l) => l?,
        None => super::row_losses(model, &rows, Ablation::default())?,
    };
    Ok(keep.into_iter().zip(losses).collect())
}

/// The practice pool: conversations drawn by how hard Nova finds them.
struct Pool {
    /// (practice index, loss, weight).
    entries: Vec<(usize, f32, f32)>,
}

impl Pool {
    fn new(measured: Vec<(usize, f32)>, cfg: &ConversationConfig) -> Pool {
        let mut p = Pool { entries: measured.into_iter().map(|(i, l)| (i, l, 1.)).collect() };
        p.reweigh(cfg);
        p
    }
    /// Weights: the loss relative to the median, between the floor and the cap.
    fn reweigh(&mut self, cfg: &ConversationConfig) {
        let mut losses: Vec<f32> = self.entries.iter().map(|e| e.1).filter(|l| l.is_finite()).collect();
        losses.sort_by(|a, b| a.total_cmp(b));
        let median = losses.get(losses.len() / 2).copied().unwrap_or(1.).max(1e-3);
        for e in &mut self.entries {
            e.2 = if e.1.is_finite() {
                (e.1 / median).clamp(cfg.target_floor, cfg.target_cap)
            } else {
                cfg.target_cap
            };
        }
    }
    /// A conversation, by weight; it is drawn less until the pool is measured again (spaced
    /// repetition).
    fn draw(&mut self, rng: &mut Rng) -> Option<usize> {
        let total: f32 = self.entries.iter().map(|e| e.2).sum();
        if self.entries.is_empty() || total <= 0. {
            return None;
        }
        let mut r = rng.unit() as f32 * total;
        let mut k = self.entries.len() - 1;
        for (i, e) in self.entries.iter().enumerate() {
            if r < e.2 {
                k = i;
                break;
            }
            r -= e.2;
        }
        self.entries[k].2 *= 0.5;
        Some(self.entries[k].0)
    }
    /// The next pool to measure: up to a quarter of what Nova has learned (loss below half the
    /// median) makes way for new conversations; the rest of the learned ones stay.
    fn next_ids(&self, train_len: usize, rng: &mut Rng) -> Vec<usize> {
        let mut losses: Vec<f32> = self.entries.iter().map(|e| e.1).collect();
        losses.sort_by(|a, b| a.total_cmp(b));
        let median = losses.get(losses.len() / 2).copied().unwrap_or(0.);
        let mut retire = self.entries.len() / 4;
        let mut ids: Vec<usize> = self
            .entries
            .iter()
            .filter(|e| {
                let go = retire > 0 && e.1 < 0.5 * median;
                retire -= go as usize;
                !go
            })
            .map(|e| e.0)
            .collect();
        let want = self.entries.len();
        let mut tries = 0;
        while ids.len() < want && tries < want * 4 {
            tries += 1;
            let i = rng.below(train_len);
            if !ids.contains(&i) {
                ids.push(i);
            }
        }
        ids
    }
}

/// The curriculum's state.
struct Curriculum {
    stage: usize,
    /// The stage's held-out loss when it began, and its best since.
    start_loss: f32,
    best_loss: f32,
    stale: usize,
    backoffs: u32,
    lr_scale: f32,
}

/// What the curriculum decides after a check.
enum Step {
    Stay,
    Advance { mastered: bool, reason: String },
    Lower { reason: String },
    Stop(String),
}

impl Curriculum {
    fn decide(&mut self, loss: f32, cfg: &ConversationConfig) -> Step {
        if loss < self.best_loss - cfg.min_loss_improvement {
            self.best_loss = loss;
            self.stale = 0;
        } else {
            self.stale += 1;
        }
        if self.stage + 1 >= STAGES.len() {
            return Step::Stay;
        }
        let mastered = loss <= cfg.competency_loss || loss <= self.start_loss * (1. - cfg.competency_gain);
        let plateau = self.stale >= cfg.stage_patience.max(1);
        let name = STAGES[self.stage].0;
        match (mastered, plateau) {
            (true, true) => Step::Advance {
                mastered: true,
                reason: format!(
                    "{name}: mastered (held-out loss {loss:.3}, from {:.3}) and no longer improving",
                    self.start_loss
                ),
            },
            (false, true) if self.backoffs < cfg.stuck_backoffs => {
                self.backoffs += 1;
                self.stale = 0;
                self.lr_scale *= 0.5;
                Step::Lower {
                    reason: format!(
                        "{name}: stuck before mastery (held-out loss {loss:.3}; mastery is {:.3} or {:.0}% below {:.3})",
                        cfg.competency_loss,
                        100. * cfg.competency_gain,
                        self.start_loss
                    ),
                }
            }
            (false, true) if cfg.advance_when_stuck => Step::Advance {
                mastered: false,
                reason: format!("{name}: still stuck after lowering the learning rate; moving on without mastery"),
            },
            (false, true) => Step::Stop(format!(
                "Stuck in {name} without mastery after lowering the learning rate {} time(s): held-out loss {loss:.3}, mastery is {:.3} or {:.0}% below {:.3}",
                self.backoffs,
                cfg.competency_loss,
                100. * cfg.competency_gain,
                self.start_loss
            )),
            _ => Step::Stay,
        }
    }
}

/// Text the model read, as rows: (training windows, held-out windows).
fn reading_rows(
    model: &Bicameral,
    docs: &[String],
    window: usize,
) -> Result<(Vec<TokenId>, Vec<Row>), String> {
    let (mut train, mut held) = (Vec::new(), Vec::new());
    for d in docs {
        let ids = model.tokenizer.encode(&format!("{d}\n\n"))?;
        if is_held_out(d) {
            held.extend(ids);
        } else {
            train.extend(ids);
        }
    }
    let row = |s: &[TokenId], at: usize| Row {
        ids: s[at..at + window].to_vec(),
        targets: s[at + 1..at + window + 1].to_vec(),
        mask: vec![1.; window],
    };
    let mut probes = Vec::new();
    if held.len() > window + 1 {
        let room = held.len() - window - 1;
        for i in spread(room / window.max(1) + 1, 64) {
            probes.push(row(&held, (i * window).min(room)));
        }
    }
    Ok((train, probes))
}

/// Practises conversations: trains on rows of the training conversations (and earlier lessons,
/// read text and listening practice), checks the held-out conversations, the Classroom
/// questions, the read text and the context tests at regular intervals, and keeps the best
/// checkpoint. `reading` is text the model read (rehearsed and guarded; may be empty).
/// `on_accepted` is called with every accepted checkpoint; `stop` ends the run at the next batch
/// (the best accepted checkpoint is returned).
#[allow(clippy::too_many_arguments)]
pub fn converse(
    model: &Bicameral,
    sets: &[ConversationSet],
    earlier: &[Item],
    reading: &[String],
    cfg: &ConversationConfig,
    observer: &mut dyn FnMut(&ConversationEvent),
    on_accepted: &mut dyn FnMut(&Bicameral) -> Result<(), String>,
    stop: &AtomicBool,
) -> Result<ConversationResult, String> {
    if cfg.batch_size == 0 || !(cfg.learning_rate > 0. && cfg.learning_rate.is_finite()) || cfg.tokens == 0 {
        return Err("Invalid conversation settings".into());
    }
    let shares = [cfg.earlier_share, cfg.reading_share, cfg.context_practice, cfg.review_share];
    if shares.iter().any(|s| !(0. ..1.).contains(s))
        || cfg.earlier_share + cfg.reading_share + cfg.context_practice >= 0.9
        || !(0. ..=1.).contains(&cfg.guard_drop)
    {
        return Err("The shares of earlier lessons, reading and listening practice must leave most rows to conversations".into());
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
    let nova_label = if cfg.own_name.is_empty() { "Nova".to_string() } else { cfg.own_name.clone() };
    let listening_practice =
        if cfg.context_practice > 0. { context_practice(4000, cfg.seed, &nova_label) } else { Vec::new() };
    let listening_sets = [ConversationSet {
        name: "listening practice".into(),
        dialogues: listening_practice,
        nova: Some(nova_label.clone()),
        hold_out: false,
    }];
    let (listen_train, ..) = practices(&listening_sets, cfg);
    let earlier_train: Vec<&Item> = earlier.iter().filter(|i| !i.held_out).collect();
    let questions: Vec<&Item> = earlier
        .iter()
        .filter(|i| i.held_out && i.kind != Kind::Prose && (i.answer.is_some() || !i.replies.is_empty()))
        .collect();
    let window = 128.min(current.config.max_tokens);
    let (reading_train, reading_probes) = reading_rows(&current, reading, window)?;
    let windows: Vec<Window> = STAGES.iter().map(|s| Window { turns: s.1, emphasis: true }).collect();
    let per_stage = (cfg.eval_rows / STAGES.len()).max(32);
    let mut probes = Vec::new();
    for w in &windows {
        let mut rows = Vec::new();
        for i in spread(held.len(), per_stage) {
            rows.extend(held[i].row(&current, cfg, None, Window { emphasis: false, ..*w })?);
        }
        probes.push(rows);
    }
    if probes.iter().any(Vec::is_empty) {
        return Err("The held-out conversations are too short to check progress".into());
    }
    let mut listen = (Vec::new(), Vec::new());
    for p in &held {
        for (full, short) in p.listening_rows(&current, cfg)? {
            listen.0.push(full);
            listen.1.push(short);
        }
        if listen.0.len() >= 128 {
            break;
        }
    }
    // Samples show Nova following a conversation: her reply deep in a held-out conversation
    // (the latest of her first five replies whose conversation so far fits with room for the
    // reply), with at least three turns before it.
    let fits = |prompt: &str| {
        current.tokenizer.encode(prompt).is_ok_and(|t| t.len() + 48 <= current.config.max_tokens)
    };
    let samples: Vec<(String, String)> = held
        .iter()
        .filter_map(|p| {
            (1..5)
                .rev()
                .filter_map(|n| p.prompt_at(cfg, n))
                .find(|(prompt, _, before)| before.len() >= 3 && fits(prompt))
        })
        .map(|(p, o, _)| (p, o))
        .take(3)
        .collect();
    let check_prompts: Vec<(String, Vec<(bool, String)>)> = held
        .iter()
        .filter_map(|p| p.prompt_at(cfg, 1))
        .filter(|(p, ..)| {
            current.tokenizer.encode(p).is_ok_and(|t| t.len() + 48 <= current.config.max_tokens)
        })
        .map(|(p, _, before)| (p, before))
        .take(24)
        .collect();
    let fixed = Fixed {
        cfg,
        probes,
        listening: listen,
        guard: spread(questions.len(), cfg.guard_questions).into_iter().map(|i| questions[i]).collect(),
        reading_probes,
        tests: if cfg.context_tests > 0 { context_tests(cfg.context_tests) } else { Vec::new() },
        check_prompts,
        samples,
    };
    let baseline = fixed.measure(current.clone(), 0, 0, 0., None)?;
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let baseline_loss = mean(&baseline.stage_losses);
    let baseline_right: Vec<bool> = baseline.classroom.clone();
    let baseline_classroom = baseline_right.iter().filter(|&&r| r).count();
    let classroom_total = baseline_right.len();
    let allowed_lost = (cfg.guard_drop * baseline_classroom as f32).floor() as usize;
    let baseline_reading = baseline.reading;
    // Tokens per row, estimated from whole-conversation windows.
    let mut estimate = Rng::new(cfg.seed ^ 0xC0FFEE);
    let mut lengths = Vec::new();
    for i in spread(train.len(), 256) {
        if let Some(r) = train[i].row(&current, cfg, Some(&mut estimate), windows[2])? {
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
        reading_documents: reading.len(),
        classroom_total,
        baseline_loss,
        baseline_stage_losses: baseline.stage_losses.clone(),
        baseline_classroom,
        baseline_listening: baseline.listening,
        baseline_reading,
        baseline_context: baseline.context.clone(),
        baseline_reply_checks: baseline.reply_checks.clone(),
        steps,
        stages: STAGES.iter().map(|s| s.0.to_string()).collect(),
    });
    let first_stage = if cfg.curriculum { 0 } else { STAGES.len() - 1 };
    let mut curriculum = Curriculum {
        stage: first_stage,
        start_loss: baseline.stage_losses[first_stage],
        best_loss: baseline.stage_losses[first_stage],
        stale: 0,
        backoffs: 0,
        lr_scale: 1.,
    };
    // Nova's loss on a new pool of conversations for a stage: on the GPU when training is there.
    let new_pool = |trainer: &super::Trainer,
                    model: &Bicameral,
                    stage: usize,
                    rng: &mut Rng|
     -> Result<Option<Pool>, String> {
        if !cfg.targeted {
            return Ok(None);
        }
        let n = cfg.pool_size.max(16).min(train.len());
        let mut ids: Vec<usize> = (0..n).map(|_| rng.below(train.len())).collect();
        ids.sort_unstable();
        ids.dedup();
        Ok(Some(Pool::new(measure_pool(Some(trainer), model, cfg, &train, &ids, windows[stage])?, cfg)))
    };
    let mut trainer = super::Trainer::new(&current);
    let mut pool = new_pool(&trainer, &current, curriculum.stage, &mut rng)?;

    let mut best: Option<(Bicameral, ConversationCheck)> = None;
    let mut best_loss = baseline_loss;
    let mut checks = Vec::new();
    let mut misses = 0;
    let (mut stopped, mut early_stop) = (false, false);
    let mut stuck = None;
    let mut tokens = 0u64;
    let (mut since, mut since_n) = (0f64, 0usize);
    let warmup = (steps / 20).clamp(1, 200);
    let eval_every = (steps / cfg.evaluations.max(1) as u64).max(1);

    // Judges a measured checkpoint: accepted or not, then the curriculum's decision. Returns
    // true when the run should end.
    let mut judge = |m: Measured,
                     curriculum: &mut Curriculum,
                     pool: &mut Option<Pool>,
                     rebuild: &mut Option<usize>,
                     observer: &mut dyn FnMut(&ConversationEvent)|
     -> Result<bool, String> {
        let loss = mean(&m.stage_losses);
        let lost = baseline_right.iter().zip(&m.classroom).filter(|(b, n)| **b && !**n).count();
        let gained = baseline_right.iter().zip(&m.classroom).filter(|(b, n)| !**b && **n).count();
        let correct = m.classroom.iter().filter(|&&r| r).count();
        let improves = loss.is_finite() && loss < best_loss - cfg.min_loss_improvement;
        let keeps_classroom = lost <= allowed_lost;
        let reading_limit = baseline_reading.map(|b| b * (1. + cfg.reading_rise));
        let keeps_reading = match (m.reading, reading_limit) {
            (Some(r), Some(limit)) => r <= limit,
            _ => true,
        };
        let accepted = improves && keeps_classroom && keeps_reading;
        let reason = if accepted {
            None
        } else if !improves {
            Some(format!("not better than {best_loss:.4} by at least {}", cfg.min_loss_improvement))
        } else if !keeps_classroom {
            Some(format!(
                "better, but {lost} Classroom answer(s) that were right before practice are wrong now (at most {allowed_lost} may be; {gained} new right answer(s) do not make up for them)"
            ))
        } else {
            Some(format!(
                "better, but the held-out loss on text it read rose to {:.4} (at most {:.4})",
                m.reading.unwrap_or(0.),
                reading_limit.unwrap_or(0.)
            ))
        };
        let c = ConversationCheck {
            step: m.step,
            tokens: m.tokens,
            train_loss: m.train_loss,
            held_out_loss: loss,
            best_loss: if accepted { loss } else { best_loss },
            stage: curriculum.stage,
            stage_losses: m.stage_losses.clone(),
            listening: m.listening,
            classroom_correct: correct,
            classroom_total,
            classroom_lost: lost,
            classroom_gained: gained,
            reading_loss: m.reading,
            context: m.context,
            reply_checks: m.reply_checks,
            learning_rate_scale: curriculum.lr_scale,
            accepted,
            reason,
            samples: m.samples,
        };
        if accepted {
            best_loss = loss;
            on_accepted(&m.model)?;
            best = Some((m.model.clone(), c.clone()));
            misses = 0;
        } else {
            misses += 1;
        }
        observer(&ConversationEvent::Checked(c.clone()));
        checks.push(c);
        let mut end = false;
        if cfg.curriculum {
            match curriculum.decide(m.stage_losses[curriculum.stage], cfg) {
                Step::Stay => {}
                Step::Lower { reason } => {
                    observer(&ConversationEvent::LearningRateLowered { scale: curriculum.lr_scale, reason })
                }
                Step::Advance { mastered, reason } => {
                    let next = curriculum.stage + 1;
                    *curriculum = Curriculum {
                        stage: next,
                        start_loss: m.stage_losses[next],
                        best_loss: m.stage_losses[next],
                        stale: 0,
                        backoffs: 0,
                        lr_scale: 1.,
                    };
                    // The next stage's pool is measured by the training loop (on the GPU).
                    *pool = None;
                    *rebuild = Some(next);
                    observer(&ConversationEvent::Stage {
                        stage: next,
                        name: STAGES[next].0.into(),
                        mastered,
                        reason,
                    });
                }
                Step::Stop(why) => {
                    stuck = Some(why);
                    end = true;
                }
            }
        }
        // Early stopping applies in the last stage (earlier stages end by the curriculum).
        let last = curriculum.stage + 1 >= STAGES.len();
        Ok(end || (last && misses >= cfg.patience.max(1)))
    };

    std::thread::scope(|scope| -> Result<(), String> {
        let mut pending: Option<std::thread::ScopedJoinHandle<'_, Result<Measured, String>>> = None;
        let join = |h: std::thread::ScopedJoinHandle<'_, Result<Measured, String>>| {
            h.join().map_err(|_| "A conversation check failed".to_string())?
        };
        // A check fell due while the previous one was still being measured on the CPU: training
        // carries on, and the check starts as soon as that one is judged (the GPU never waits).
        let mut due = false;
        let mut rebuild: Option<usize> = None;
        for step in 0..steps {
            if stop.load(Ordering::Acquire) {
                stopped = true;
                break;
            }
            let mut check_now = false;
            if pending.as_ref().is_some_and(|h| h.is_finished()) {
                if judge(join(pending.take().unwrap())?, &mut curriculum, &mut pool, &mut rebuild, observer)?
                {
                    early_stop = true;
                    break;
                }
                check_now = std::mem::take(&mut due);
            }
            if let Some(stage) = rebuild.take() {
                trainer.sync(&mut current)?;
                pool = new_pool(&trainer, &current, stage, &mut rng)?;
            }
            let stage = curriculum.stage;
            let mut rows = Vec::with_capacity(cfg.batch_size);
            let mut tries = 0;
            while rows.len() < cfg.batch_size && tries < cfg.batch_size * 20 {
                tries += 1;
                let u = rng.unit() as f32;
                let row = if u < cfg.earlier_share && !earlier_train.is_empty() {
                    let item = earlier_train[rng.below(earlier_train.len())];
                    encode_item(&current.tokenizer, &item.segments, current.config.max_tokens, &mut rng)?
                } else if u < cfg.earlier_share + cfg.reading_share && reading_train.len() > window + 1 {
                    let at = rng.below(reading_train.len() - window - 1);
                    Some(Row {
                        ids: reading_train[at..at + window].to_vec(),
                        targets: reading_train[at + 1..at + window + 1].to_vec(),
                        mask: vec![1.; window],
                    })
                } else if u < cfg.earlier_share + cfg.reading_share + cfg.context_practice
                    && !listen_train.is_empty()
                {
                    let p = &listen_train[rng.below(listen_train.len())];
                    p.row(&current, cfg, Some(&mut rng), windows[STAGES.len() - 1])?
                } else {
                    // A conversation: from an earlier stage now and then, else from the pool
                    // (by difficulty) or at random.
                    let review = stage > 0 && (rng.unit() as f32) < cfg.review_share;
                    let w = if review { windows[rng.below(stage)] } else { windows[stage] };
                    let i = match pool.as_mut().filter(|_| !review).and_then(|p| p.draw(&mut rng)) {
                        Some(i) => i,
                        None => rng.below(train.len()),
                    };
                    train[i].row(&current, cfg, Some(&mut rng), w)?
                };
                rows.extend(row);
            }
            if rows.is_empty() {
                return Err("No training rows could be made from these conversations".into());
            }
            let lr = lr_schedule(step, cfg.learning_rate, warmup, steps, 0.1) * curriculum.lr_scale;
            let loss = trainer.step(&mut current, &rows, Ablation::default(), lr, Focus::Joint)?;
            if !loss.is_finite() {
                return Err("Training diverged (loss is not finite); try a lower learning rate".into());
            }
            tokens += rows.iter().map(|r| r.ids.len() as u64).sum::<u64>();
            since += loss as f64;
            since_n += 1;
            observer(&ConversationEvent::Batch {
                step: step + 1,
                steps,
                tokens,
                loss,
                learning_rate: lr,
                stage,
            });
            if (step + 1) % eval_every == 0 || step + 1 == steps {
                if pending.is_some() {
                    due = true;
                } else {
                    check_now = true;
                }
            }
            if !check_now {
                continue;
            }
            trainer.sync(&mut current)?;
            let train_loss = (since / since_n.max(1) as f64) as f32;
            (since, since_n) = (0., 0);
            // The practice pool is measured again now (on the GPU when training is there): a
            // share of the learned conversations make way for new ones.
            if let Some(p) = pool.as_mut() {
                let ids = p.next_ids(train.len(), &mut rng);
                *p = Pool::new(
                    measure_pool(Some(&trainer), &current, cfg, &train, &ids, windows[curriculum.stage])?,
                    cfg,
                );
            }
            let snapshot = current.clone();
            if trainer.on_gpu() {
                // The losses on the GPU; the answers Nova writes (Classroom, context tests,
                // replies) on a CPU thread while the GPU keeps training.
                let gpu = |rows: &[Row]| {
                    trainer
                        .gpu_loss(&current, rows, Ablation::default())
                        .unwrap_or_else(|| batch_loss(&current, rows, Ablation::default()))
                };
                let stage_losses = fixed.probes.iter().map(|r| gpu(r)).collect::<Result<Vec<_>, _>>()?;
                let listening = if fixed.listening.0.is_empty() {
                    None
                } else {
                    Some(gpu(&fixed.listening.1)? - gpu(&fixed.listening.0)?)
                };
                let reading =
                    if fixed.reading_probes.is_empty() { None } else { Some(gpu(&fixed.reading_probes)?) };
                let part = GpuPart { stage_losses, listening, reading };
                let fixed = &fixed;
                pending = Some(
                    scope.spawn(move || fixed.measure(snapshot, step + 1, tokens, train_loss, Some(part))),
                );
            } else if judge(
                fixed.measure(snapshot, step + 1, tokens, train_loss, None)?,
                &mut curriculum,
                &mut pool,
                &mut rebuild,
                observer,
            )? {
                early_stop = true;
                break;
            }
        }
        if let Some(h) = pending.take() {
            if judge(join(h)?, &mut curriculum, &mut pool, &mut rebuild, observer)? && !stopped {
                early_stop = true;
            }
        }
        // The last check fell due while another was being measured: it runs now.
        if due && !stopped && !early_stop {
            trainer.sync(&mut current)?;
            let train_loss = (since / since_n.max(1) as f64) as f32;
            if judge(
                fixed.measure(current.clone(), steps, tokens, train_loss, None)?,
                &mut curriculum,
                &mut pool,
                &mut rebuild,
                observer,
            )? {
                early_stop = true;
            }
        }
        Ok(())
    })?;
    let accepted = best.is_some();
    let stage = curriculum.stage;
    observer(&ConversationEvent::Finished {
        accepted,
        best_loss,
        tokens,
        stopped,
        early_stop,
        stage,
        stuck: stuck.clone(),
    });
    let (best_model, best_check) = match best {
        Some((m, c)) => (m, Some(c)),
        None => (model.clone(), None),
    };
    Ok(ConversationResult {
        model: best_model,
        accepted,
        baseline_loss,
        best_loss,
        baseline_classroom,
        classroom_total,
        baseline_context: baseline.context,
        best_check,
        tokens,
        checks,
        stopped,
        early_stop,
        stage,
        stuck,
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
            max_tokens: 96,
            ..Config::default()
        };
        Bicameral::with_config(cfg, 3)
    }
    const WHOLE: Window = Window { turns: None, emphasis: false };

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

    fn trained_text(m: &Bicameral, row: &Row) -> String {
        let t: Vec<TokenId> =
            row.targets.iter().zip(&row.mask).filter(|(_, &w)| w > 0.).map(|(&t, _)| t).collect();
        m.tokenizer.decode(&t).unwrap()
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
        let row = as_ben.row(&m, &cfg, None, WHOLE).unwrap().unwrap();
        let text = m.tokenizer.decode(&row.ids).unwrap()
            + &m.tokenizer.decode(&row.targets[row.targets.len() - 1..]).unwrap();
        assert_eq!(text, "U: Hi Nova, how are you?\nN: Good, Ann. And you?\nU: Fine, thanks.\n");
        assert_eq!(trained_text(&m, &row), "Good, Ann. And you?\n");
        assert_eq!(
            as_ben.prompt_at(&cfg, 0).map(|p| (p.0, p.1)),
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
        assert!(p.row(&m, &cfg, None, WHOLE).unwrap().is_none(), "Ben's only reply is not marked");
    }

    #[test]
    fn stages_cut_windows_and_replies_using_earlier_turns_weigh_more() {
        let m = small();
        let d = parse_dialogue(
            "Ann: I went to the bakery today.\nBen: Nice.\nAnn: It was busy.\nBen: Was the bakery out of bread?\nAnn: No.\nBen: Good.",
        )
        .unwrap();
        let cfg = ConversationConfig::default();
        let p = Practice { dialogue: &d, nova: "Ben".into() };
        let short = p.row(&m, &cfg, None, Window { turns: Some(2), emphasis: false }).unwrap().unwrap();
        assert_eq!(m.tokenizer.decode(&short.ids).unwrap(), "U: I went to the bakery today.\nN: Nice.");
        let mut rng = Rng::new(3);
        for _ in 0..20 {
            let r =
                p.row(&m, &cfg, Some(&mut rng), Window { turns: Some(2), emphasis: true }).unwrap().unwrap();
            let text = m.tokenizer.decode(&r.ids).unwrap();
            assert!(text.matches("U: ").count() + text.matches("N: ").count() <= 2, "{text:?}");
        }
        // "bakery" comes from two turns back: that reply weighs `context_weight`.
        let whole = p.row(&m, &cfg, None, Window { turns: None, emphasis: true }).unwrap().unwrap();
        let weights: Vec<f32> = whole.mask.iter().copied().filter(|&w| w > 0.).collect();
        assert!(weights.contains(&cfg.context_weight) && weights.contains(&1.));
        let plain = p.row(&m, &cfg, None, WHOLE).unwrap().unwrap();
        assert!(plain.mask.iter().all(|&w| w == 0. || w == 1.), "checks weigh every reply the same");
        let pairs = p.listening_rows(&m, &cfg).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(trained_text(&m, &pairs[0].0), "Was the bakery out of bread?\n");
        assert_eq!(trained_text(&m, &pairs[0].1), "Was the bakery out of bread?\n");
        assert!(pairs[0].0.ids.len() > pairs[0].1.ids.len());
    }

    #[test]
    fn the_curriculum_moves_on_only_when_mastered_and_lowers_the_rate_when_stuck() {
        let cfg = ConversationConfig {
            competency_loss: 1.0,
            competency_gain: 0.2,
            stage_patience: 2,
            stuck_backoffs: 1,
            ..ConversationConfig::default()
        };
        let mut c =
            Curriculum { stage: 0, start_loss: 3.0, best_loss: 3.0, stale: 0, backoffs: 0, lr_scale: 1. };
        assert!(matches!(c.decide(2.9, &cfg), Step::Stay));
        assert!(matches!(c.decide(2.9, &cfg), Step::Stay));
        // A plateau at 2.9: not mastered (needs 1.0 or 2.4), so stuck, not ready.
        assert!(matches!(c.decide(2.9, &cfg), Step::Lower { .. }));
        assert_eq!(c.lr_scale, 0.5);
        assert!(matches!(c.decide(2.3, &cfg), Step::Stay), "improving and mastered: stays while it gains");
        assert!(matches!(c.decide(2.3, &cfg), Step::Stay));
        assert!(matches!(c.decide(2.3, &cfg), Step::Advance { mastered: true, .. }));
        let mut stuck =
            Curriculum { stage: 0, start_loss: 3.0, best_loss: 3.0, stale: 0, backoffs: 1, lr_scale: 0.5 };
        stuck.decide(3.0, &cfg);
        assert!(matches!(stuck.decide(3.0, &cfg), Step::Stop(_)), "still stuck after its back-off: stops");
        let lenient = ConversationConfig { advance_when_stuck: true, ..cfg.clone() };
        let mut s2 =
            Curriculum { stage: 0, start_loss: 3.0, best_loss: 3.0, stale: 0, backoffs: 1, lr_scale: 0.5 };
        s2.decide(3.0, &lenient);
        assert!(matches!(s2.decide(3.0, &lenient), Step::Advance { mastered: false, .. }));
        let mut last =
            Curriculum { stage: 2, start_loss: 3.0, best_loss: 3.0, stale: 9, backoffs: 0, lr_scale: 1. };
        assert!(matches!(last.decide(3.0, &cfg), Step::Stay), "the last stage has nowhere to go");
    }

    #[test]
    fn the_pool_favours_hard_conversations_within_the_floor_and_cap() {
        let cfg = ConversationConfig::default();
        let mut pool = Pool::new(vec![(0, 0.1), (1, 1.0), (2, 1.0), (3, 50.0)], &cfg);
        let w: Vec<f32> = pool.entries.iter().map(|e| e.2).collect();
        assert_eq!(w, [cfg.target_floor, 1., 1., cfg.target_cap]);
        let mut rng = Rng::new(1);
        let mut counts = [0; 4];
        for _ in 0..4000 {
            counts[pool.draw(&mut rng).unwrap()] += 1;
            pool.reweigh(&cfg); // measured again: the recency penalty goes
        }
        assert!(counts[3] > counts[1] && counts[1] > counts[0] && counts[0] > 0, "{counts:?}");
        let next = pool.next_ids(100, &mut rng);
        assert_eq!(next.len(), 4);
        assert!(!next.contains(&0) && next.contains(&3), "a learned conversation makes way; hard ones stay");
    }

    #[test]
    fn replies_are_checked_for_traits_not_wording() {
        let before = vec![(false, "I went to the bakery and bought bread.".to_string())];
        let mut c = ReplyChecks::default();
        check_reply("Was the bread fresh?", &before, "Nova", &mut c);
        check_reply("I went to the bakery and bought bread.", &before, "Nova", &mut c);
        check_reply("", &before, "Nova", &mut c);
        check_reply("My name is Ben and I like it, I like it, I like it.", &before, "Nova", &mut c);
        check_reply("As an AI, I cannot eat bread.", &before, "Nova", &mut c);
        check_reply("I am Nova.", &before, "Nova", &mut c);
        assert_eq!(c.replies, 6);
        assert_eq!((c.questions, c.echoes, c.empty, c.wrong_name, c.assistant), (1, 1, 1, 1, 1));
        assert_eq!(c.repeats, 1);
        assert!(c.on_topic >= 3);
    }

    #[test]
    fn practice_runs_keeps_only_better_checkpoints_and_reports_every_measure() {
        let m = Bicameral::with_config(Config { max_tokens: 256, ..small().config }, 3);
        let mut dialogues = Vec::new();
        for i in 0..160 {
            let doc = format!(
                "Ann: Do you like the color number {i}?\nBen: Yes, I like it.\nAnn: Why?\nBen: Because it is bright.\nAnn: Is number {i} your favourite?\nBen: Yes, number {i} is."
            );
            dialogues.push(parse_dialogue(&doc).unwrap());
        }
        let sets = [ConversationSet { name: "x".into(), dialogues, nova: None, hold_out: true }];
        let cfg = ConversationConfig {
            tokens: 30_000,
            batch_size: 8,
            learning_rate: 3e-3,
            evaluations: 5,
            pool_size: 64,
            context_tests: 2,
            ..ConversationConfig::default()
        };
        let reading: Vec<String> =
            (0..60).map(|i| format!("The little dog number {i} ran to the park and played.")).collect();
        let mut events = Vec::new();
        let mut saved = 0;
        let r = converse(
            &m,
            &sets,
            &[],
            &reading,
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
        let Some(ConversationEvent::Started {
            baseline_context,
            baseline_reading,
            baseline_stage_losses,
            ..
        }) = events.first()
        else {
            panic!("no start")
        };
        assert!(baseline_context.as_ref().is_some_and(|c| c.total == 2 * Skill::ALL.len()));
        assert!(baseline_reading.is_some());
        assert_eq!(baseline_stage_losses.len(), STAGES.len());
        for c in &r.checks {
            assert_eq!(c.stage_losses.len(), 3);
            assert!(c.context.is_some() && c.reply_checks.is_some() && c.reading_loss.is_some());
            assert!(c.samples.iter().all(|s| s.prompt.ends_with("N: ")));
            assert!(
                !c.samples.is_empty() && c.samples.iter().all(|s| s.prompt.lines().count() >= 4),
                "samples show a conversation: {:?}",
                c.samples
            );
        }
        let stopped =
            converse(&m, &sets, &[], &[], &cfg, &mut |_| {}, &mut |_| Ok(()), &AtomicBool::new(true))
                .unwrap();
        assert!(stopped.stopped && !stopped.accepted);
    }
}
