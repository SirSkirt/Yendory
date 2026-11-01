//! Choosing each next token of a conversation reply. Exams, grading and teaching always take
//! the most likely token (greedy); a conversation may instead sample, so the same message can
//! get different replies. Every reply is still the model's own: sampling only decides which of
//! the tokens the model itself proposes is written next.
use super::Rng;
use crate::model::inspect::{ConversationTrace, TraceEvent, TraceReply};
use crate::model::{argmax, Ablation, Bicameral};
use crate::tokenizer::TokenId;
use serde::{Deserialize, Serialize};

/// How a reply's tokens are chosen. The default is greedy, exactly as exams answer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Sampling {
    /// 0 takes the most likely token every time. Above 0 the next token is drawn at random in
    /// proportion to the model's probabilities raised to 1/temperature: below 1 the likely
    /// tokens gain, above 1 the unlikely ones do.
    pub temperature: f32,
    /// Only the k most likely tokens may be drawn (0: no limit).
    pub top_k: usize,
    /// Only the most likely tokens whose probabilities add up to p may be drawn (1: no limit).
    pub top_p: f32,
    /// A token already written in this reply is this many times less likely to be written
    /// again (1: no penalty). Applies to greedy replies too.
    pub repetition_penalty: f32,
    /// Seed of the random draws: the same seed, settings and prompt give the same reply.
    pub seed: u64,
}
impl Default for Sampling {
    fn default() -> Self {
        Sampling { temperature: 0., top_k: 0, top_p: 1., repetition_penalty: 1., seed: 0 }
    }
}
impl Sampling {
    pub fn is_greedy(&self) -> bool {
        self.temperature.is_nan() || self.temperature <= 0.
    }
    /// Settings within their ranges: temperature 0-5, top-p in (0, 1], penalty 1-5.
    pub fn clamped(mut self) -> Self {
        self.temperature = if self.temperature.is_finite() { self.temperature.clamp(0., 5.) } else { 0. };
        self.top_p = if self.top_p.is_finite() && self.top_p > 0. { self.top_p.min(1.) } else { 1. };
        self.repetition_penalty =
            if self.repetition_penalty.is_finite() { self.repetition_penalty.clamp(1., 5.) } else { 1. };
        self
    }
    /// The next token from the model's next-token probabilities, given the tokens of this reply
    /// so far.
    pub fn pick(&self, probs: &[f32], so_far: &[TokenId], rng: &mut Rng) -> TokenId {
        let penalty = self.repetition_penalty;
        let penalize = penalty > 1.;
        if self.is_greedy() && !penalize {
            return argmax(probs);
        }
        let mut p = probs.to_vec();
        if penalize {
            let mut seen = so_far.to_vec();
            seen.sort_unstable();
            seen.dedup();
            for t in seen {
                if let Some(x) = p.get_mut(t as usize) {
                    *x /= penalty;
                }
            }
        }
        if self.is_greedy() {
            return argmax(&p);
        }
        // Weights p^(1/T), computed from logarithms so small probabilities do not underflow.
        let inv = 1. / self.temperature as f64;
        let max = p.iter().copied().fold(0f32, f32::max) as f64;
        if max <= 0. {
            return argmax(probs);
        }
        let mut w: Vec<(usize, f64)> = p
            .iter()
            .enumerate()
            .filter(|(_, &x)| x > 0.)
            .map(|(i, &x)| (i, ((x as f64).ln() - max.ln()) * inv))
            .map(|(i, l)| (i, l.exp()))
            .filter(|(_, x)| *x > 0.)
            .collect();
        let by_weight = |a: &(usize, f64), b: &(usize, f64)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        if self.top_k > 0 && self.top_k < w.len() {
            w.select_nth_unstable_by(self.top_k - 1, by_weight);
            w.truncate(self.top_k);
        }
        if self.top_p < 1. {
            w.sort_unstable_by(by_weight);
            let total: f64 = w.iter().map(|x| x.1).sum();
            let mut sum = 0.;
            let mut keep = w.len();
            for (n, x) in w.iter().enumerate() {
                sum += x.1;
                if sum >= self.top_p as f64 * total {
                    keep = n + 1;
                    break;
                }
            }
            w.truncate(keep.max(1));
        } else {
            // A fixed order, so a seed always gives the same draw.
            w.sort_unstable_by_key(|x| x.0);
        }
        let total: f64 = w.iter().map(|x| x.1).sum();
        let mut r = rng.unit() * total;
        for &(i, x) in &w {
            if r < x {
                return i as TokenId;
            }
            r -= x;
        }
        w.last().map_or_else(|| argmax(probs), |x| x.0 as TokenId)
    }
}

/// A reply as it was generated.
#[derive(Clone, Debug, PartialEq)]
pub struct Reply {
    /// The reply's text without the final newline.
    pub text: String,
    /// Tokens generated (the final newline included).
    pub tokens: usize,
    /// The reply ended with a newline (rather than at the length limit or a stop).
    pub finished: bool,
    /// `each` stopped it.
    pub stopped: bool,
}

/// A conversation reply to `prompt`, token by token: `each` receives every token's text as it
/// is written (the final newline excluded) and stops the reply by returning false.
pub fn reply_with(
    model: &Bicameral,
    prompt: &str,
    max_tokens: usize,
    sampling: &Sampling,
    mut each: impl FnMut(&str) -> bool,
) -> Result<Reply, String> {
    let ids = model.tokenizer.encode(prompt)?;
    let sampling = sampling.clamped();
    let mut rng = Rng::new(sampling.seed);
    let mut stopped = false;
    let mut failed = None;
    let out = model.generate_cached_with(
        &ids,
        max_tokens,
        Ablation::default(),
        |probs, so_far| sampling.pick(probs, so_far, &mut rng),
        |t| {
            if t == b'\n' as TokenId {
                return true;
            }
            match model.tokenizer.decode(&[t]) {
                Ok(text) => {
                    stopped = !each(&text);
                    !stopped
                }
                Err(e) => {
                    failed = Some(e);
                    false
                }
            }
        },
    )?;
    if let Some(e) = failed {
        return Err(e);
    }
    let finished = out.last() == Some(&(b'\n' as TokenId));
    let text = model.tokenizer.decode(&out)?.trim_end_matches('\n').to_string();
    Ok(Reply { text, tokens: out.len(), finished, stopped })
}

/// [`reply_with`] with a per-network trace recorded as the reply is written: the same tokens
/// (the same probabilities and draws; tracing only copies values out), the same `each`, and
/// `event` sees the prompt once read and every step (for live display).
pub fn reply_traced(
    model: &Bicameral,
    prompt: &str,
    max_tokens: usize,
    sampling: &Sampling,
    mut each: impl FnMut(&str) -> bool,
    mut event: impl FnMut(TraceEvent<'_>),
) -> Result<(Reply, ConversationTrace), String> {
    let ids = model.tokenizer.encode(prompt)?;
    let sampling = sampling.clamped();
    let mut rng = Rng::new(sampling.seed);
    let (mut stopped, mut failed) = (false, None);
    let mut choose = |probs: &[f32], so_far: &[TokenId]| sampling.pick(probs, so_far, &mut rng);
    let trace = model.trace_turn(&ids, TraceReply::Generate { max_tokens, choose: &mut choose }, |e| {
        let go = match &e {
            TraceEvent::Step(s, _) if s.chosen.id != b'\n' as TokenId => {
                match model.tokenizer.decode(&[s.chosen.id]) {
                    Ok(text) => {
                        stopped = !each(&text);
                        !stopped
                    }
                    Err(e) => {
                        failed = Some(e);
                        false
                    }
                }
            }
            _ => true,
        };
        event(e);
        go
    })?;
    if let Some(e) = failed {
        return Err(e);
    }
    let out: Vec<TokenId> = trace.steps.iter().map(|s| s.chosen.id).collect();
    let text = model.tokenizer.decode(&out)?.trim_end_matches('\n').to_string();
    Ok((Reply { text, tokens: out.len(), finished: trace.finished, stopped }, trace))
}
