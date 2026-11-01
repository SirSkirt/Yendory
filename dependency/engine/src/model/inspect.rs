//! Per-network trace of one conversation turn: what each of Nova's networks did while she read
//! the prompt and wrote (or is shown) the reply, token by token.
//!
//! Every number comes from the inference pass itself ([`KvCache`] with tracing, which computes
//! exactly the values generation computes and only copies some of them out) or from extra
//! passes of the same model with one network switched off ("influence"). Nothing here names a
//! concept, guesses an intention or writes a sentence about what Nova "thought": the trace is
//! measurements, and each one says what it measures.
//!
//! Positions are counted over everything read (prompt, then reply). "Lines" are the prompt's
//! lines as written (a line ends with a line break): the context text, Study Mode memories,
//! earlier turns (`U:` the user, `N:` Nova), the current message, and Nova's reply.
use super::cache::{KvCache, NeuroRound, StageTrace, Trace};
use super::think::{Choice, Readout};
use super::{Ablation, Bicameral};
use crate::tokenizer::TokenId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Version of the saved trace format.
pub const TRACE_VERSION: u32 = 1;
/// Recalled patterns kept per token and round in the profile.
const PROFILE_PATTERNS: usize = 16;
/// Positions kept per attention head.
const HEAD_TOP: usize = 5;
/// Next-token alternatives kept per step.
const ALTERNATIVES: usize = 5;

/// A network of the model, as built (from its design, not from a description).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Network {
    /// "transformer", "memory", "associative", "relational", "experts", "consolidation",
    /// "controller", "bus" (Nova Neuro) or "dense" (Nova Bicameral's dense pathway).
    pub id: String,
    pub name: String,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenInfo {
    pub id: TokenId,
    /// The token as text (a space shown as ␠, a line break as ↵).
    pub text: String,
    pub line: usize,
    /// Written by Nova in this turn (otherwise read from the prompt).
    pub reply: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Line {
    pub index: usize,
    /// Token positions `start..end`.
    pub start: usize,
    pub end: usize,
    /// "context", "memory", "user", "nova", "message" (the message being answered) or "reply".
    pub role: String,
    pub text: String,
}

/// Measured for every token read (from its own forward pass): small, kept for long prompts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TokenProfile {
    /// How unexpected this token was: -log2 of the probability the model gave it before
    /// reading it, in bits (`None` for the first token).
    pub surprise: Option<f32>,
    /// The Executive Controller's gates, `[round][network]`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gates: Vec<Vec<f32>>,
    /// Working memory per layer: mean update gate, share of channels overwritten, change.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub memory_update: Vec<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub memory_overwrite: Vec<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub memory_change: Vec<f32>,
    /// Associative memory per round: retrieval entropy (bits) and the strongest patterns.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub retrieval_entropy: Vec<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub patterns: Vec<Vec<(usize, f32)>>,
    /// Per round: length of what the bus wrote into the transformer's state divided by the
    /// state's length.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub injection: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HeadSummary {
    /// Entropy of the head's attention over the window, in bits (0 = one position).
    pub entropy: f32,
    /// The positions it attended to most (position, weight).
    pub top: Vec<(usize, f32)>,
}

/// One transformer block execution for the newest token.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StageSummary {
    pub block: usize,
    /// 0 = lower stage, 1.. = thinking round.
    pub round: usize,
    pub state_norm: f32,
    pub update_norm: f32,
    pub heads: Vec<HeadSummary>,
    /// Attention mass (mean over heads) on each line (lines below 0.001 left out).
    pub lines: Vec<(usize, f32)>,
    /// Mean distance back, in tokens, of the attention (mean over heads).
    pub distance: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TransformerStep {
    pub stages: Vec<StageSummary>,
    /// Attention mass on each line, mean over every block execution and head.
    pub lines: Vec<(usize, f32)>,
    /// What the model would predict from its state after the lower stage and after each round
    /// (the output layer applied there); the last is its actual output.
    pub readouts: Vec<Readout>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryLayerStep {
    /// Mean update gate (0 keep, 1 overwrite) and the share of channels above one half.
    pub update: f32,
    pub overwrite: f32,
    /// Length of the state's change with this token, and of the state.
    pub change: f32,
    pub norm: f32,
    /// Length of what each line still contributes to the state (exact split of the state).
    pub lines: Vec<(usize, f32)>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModuleStep {
    pub name: String,
    /// The Executive Controller's gate for it, and the length of its output before the gate.
    pub gate: f32,
    pub output_norm: f32,
}

/// One round of the networks around the bus for the newest token.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RoundStep {
    pub round: usize,
    /// Onto the bus: the transformer's projection and the working memory's output (lengths).
    pub read_norm: f32,
    pub memory_in_norm: f32,
    /// The bus state the networks read (length, and the values).
    pub bus_norm: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub bus: Vec<f32>,
    pub modules: Vec<ModuleStep>,
    /// The gated outputs summed (length), written back into the transformer's state through
    /// a learned gate: the write-back's length, the state's length, and the gate's mean.
    pub write_norm: f32,
    pub injection_norm: f32,
    pub state_norm: f32,
    pub write_gate: f32,
    /// The Executive Controller's hidden activations.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub controller: Vec<f32>,
    /// Associative memory: strongest patterns, retrieval entropy (bits).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub patterns: Vec<(usize, f32)>,
    pub retrieval_entropy: f32,
    /// Associative memory: the lines whose tokens recalled the most similar patterns (cosine
    /// of the recalled pattern weights, this round), most similar first.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub similar_lines: Vec<(usize, f32)>,
    /// Relational network (each hop's read over the slots), experts (router weights),
    /// consolidation (code length).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub slot_reads: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub experts: Vec<f32>,
    pub code_norm: f32,
}

/// How the step's prediction changed with one network switched off for the whole turn.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Influence {
    pub network: String,
    /// Total-variation distance between the two next-token distributions (0 same, 1 disjoint).
    pub change: f32,
    /// The probability the switched-off model gives the token actually written.
    pub chosen_probability: f32,
    /// The switched-off model's most likely token, and whether it differs from the full
    /// model's most likely token.
    pub top: Choice,
    pub flips: bool,
}

/// One generation step: the forward pass that read the token before and predicted this one.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StepTrace {
    pub index: usize,
    /// Position of the token written at this step.
    pub position: usize,
    pub chosen: Choice,
    pub alternatives: Vec<Choice>,
    /// Entropy of the next-token distribution, in bits.
    pub entropy: f32,
    pub transformer: TransformerStep,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory: Vec<MemoryLayerStep>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rounds: Vec<RoundStep>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub influence: Vec<Influence>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConversationTrace {
    pub version: u32,
    /// The model's training step when traced.
    pub model_step: u64,
    /// "neuro" or "bicameral".
    pub architecture: String,
    pub networks: Vec<Network>,
    /// The controller's networks, in gate order.
    pub modules: Vec<String>,
    pub rounds: usize,
    /// The model's context (tokens read at once).
    pub context: usize,
    pub prompt_tokens: usize,
    pub tokens: Vec<TokenInfo>,
    pub lines: Vec<Line>,
    pub profile: Vec<TokenProfile>,
    pub steps: Vec<StepTrace>,
    pub reply: String,
    /// The reply was given (replayed through the model) rather than generated now.
    pub forced: bool,
    /// The reply ended with a line break (rather than at the length limit or a stop).
    pub finished: bool,
    /// The networks whose influence was measured (each by one extra pass).
    pub influence: Vec<String>,
    pub millis: u64,
    /// What the viewer must know about how this trace was made.
    pub notes: Vec<String>,
}

/// What happens as a trace is made (for live display).
pub enum TraceEvent<'a> {
    /// The prompt has been read: tokens, lines and profile so far.
    Read(&'a ConversationTrace),
    /// A step is done (its token is the last in `tokens`).
    Step(&'a StepTrace, &'a TokenInfo),
}

/// The reply of a traced turn.
pub enum TraceReply<'a> {
    /// These tokens, read through the model as if it had written them (the same computation:
    /// each step is the pass that read the token before).
    Forced(&'a [TokenId]),
    /// Written now: `choose` picks each token from the model's probabilities (it also sees the
    /// tokens written so far), as [`Bicameral::generate_cached_with`] does.
    Generate { max_tokens: usize, choose: &'a mut dyn FnMut(&[f32], &[TokenId]) -> TokenId },
}

fn entropy_bits(p: &[f32]) -> f32 {
    -p.iter().filter(|&&q| q > 0.).map(|&q| q * q.log2()).sum::<f32>()
}

fn top_k(p: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    let k = k.min(p.len());
    if k == 0 {
        return Vec::new();
    }
    idx.select_nth_unstable_by(k - 1, |&a, &b| p[b].total_cmp(&p[a]));
    idx.truncate(k);
    idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    idx.into_iter().map(|i| (i, p[i])).collect()
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Total-variation distance between two distributions.
fn total_variation(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / 2.
}

fn sparse_cosine(a: &BTreeMap<usize, f32>, b: &[(usize, f32)]) -> f32 {
    let dot: f32 = b.iter().filter_map(|(i, w)| a.get(i).map(|v| v * w)).sum();
    let na = a.values().map(|v| v * v).sum::<f32>().sqrt();
    let nb = b.iter().map(|(_, w)| w * w).sum::<f32>().sqrt();
    if na == 0. || nb == 0. {
        0.
    } else {
        dot / (na * nb)
    }
}

impl Bicameral {
    fn trace_choice(&self, id: usize, p: f32) -> Choice {
        Choice { token: self.tokenizer.token_label(id as TokenId), id: id as TokenId, probability: p }
    }

    /// The networks this model is built from.
    pub fn trace_networks(&self) -> Vec<Network> {
        let c = &self.config;
        let upper = c.blocks - c.split;
        let mut out = vec![Network {
            id: "transformer".into(),
            name: "Causal transformer".into(),
            detail: format!(
                "{} blocks ({} lower, {} upper run {} times), {} attention heads, width {}, reads {} tokens at once",
                c.blocks, c.split, upper, c.loops, c.heads, c.d, c.max_tokens
            ),
        }];
        let Some(n) = &self.neuro else {
            out.push(Network {
                id: "dense".into(),
                name: "Dense pathway".into(),
                detail: format!(
                    "reads the last {} characters; exchanges with the transformer once before the upper blocks and once per round",
                    c.window_chars
                ),
            });
            return out;
        };
        if let Some(m) = &n.memory {
            out.push(Network {
                id: "memory".into(),
                name: "Working memory (minGRU)".into(),
                detail: format!(
                    "{} layers of {} channels; its state belongs to this conversation only",
                    m.layers, m.width
                ),
            });
        }
        if let Some(a) = &n.associative {
            out.push(Network {
                id: "associative".into(),
                name: "Associative memory (modern Hopfield)".into(),
                detail: format!(
                    "{} learned patterns (keys of width {}); learned in training, not written during a conversation",
                    a.patterns, a.key
                ),
            });
        }
        if let Some(r) = &n.relational {
            out.push(Network {
                id: "relational".into(),
                name: "Relational network".into(),
                detail: format!("{} slots of width {}, {} hops", r.slots, r.width, r.hops),
            });
        }
        if let Some(e) = &n.experts {
            out.push(Network {
                id: "experts".into(),
                name: "Experts".into(),
                detail: format!("{} experts of width {}", e.count, e.width),
            });
        }
        if let Some(k) = &n.consolidation {
            out.push(Network {
                id: "consolidation".into(),
                name: "Consolidation".into(),
                detail: format!("code of width {}", k.code),
            });
        }
        if !n.modules().is_empty() {
            out.push(Network {
                id: "controller".into(),
                name: "Executive Controller (gating)".into(),
                detail: format!(
                    "hidden width {}; one gate per network per round, read from the bus",
                    n.controller
                ),
            });
        }
        if n.has_bus() {
            out.push(Network {
                id: "bus".into(),
                name: "Communication bus".into(),
                detail: format!(
                    "width {}; carries the transformer's state to the networks and their gated outputs back, once per round (a connection, not a network that computes on its own)",
                    n.bus
                ),
            });
        }
        out
    }

    /// The switches for measuring each network's influence: (network, what is switched off).
    pub fn influence_switches(&self) -> Vec<(String, Ablation)> {
        let off = Ablation::default();
        let Some(n) = &self.neuro else {
            return vec![
                ("dense".into(), Ablation { no_afferent: true, ..off }),
                ("dense input".into(), Ablation { no_efferent: true, ..off }),
            ];
        };
        let mut out = Vec::new();
        if n.memory.is_some() {
            out.push(("memory".into(), Ablation { no_memory: true, ..off }));
        }
        if n.associative.is_some() {
            out.push(("associative".into(), Ablation { no_associative: true, ..off }));
        }
        if n.relational.is_some() {
            out.push(("relational".into(), Ablation { no_relational: true, ..off }));
        }
        if n.experts.is_some() {
            out.push(("experts".into(), Ablation { no_experts: true, ..off }));
        }
        if n.consolidation.is_some() {
            out.push(("consolidation".into(), Ablation { no_consolidation: true, ..off }));
        }
        if !n.modules().is_empty() {
            out.push(("controller".into(), Ablation { no_controller: true, ..off }));
        }
        if n.has_bus() {
            out.push(("bus write-back".into(), Ablation { no_afferent: true, ..off }));
            out.push(("bus input".into(), Ablation { no_efferent: true, ..off }));
        }
        out
    }

    /// Traces one turn: reads `prompt` with tracing on, then writes or replays the reply,
    /// recording every step. `each` sees the prompt once read and every step; returning false
    /// stops after that step. The reply's tokens are exactly those generation gives for the
    /// same choices: tracing only copies values out of the computation.
    pub fn trace_turn(
        &self,
        prompt: &[TokenId],
        reply: TraceReply<'_>,
        mut each: impl FnMut(TraceEvent<'_>) -> bool,
    ) -> Result<ConversationTrace, String> {
        if prompt.is_empty() {
            return Err("Empty prompt".into());
        }
        let started = std::time::Instant::now();
        let mut cache = KvCache::with_trace(self, Ablation::default());
        let mut t = ConversationTrace {
            version: TRACE_VERSION,
            model_step: self.step,
            architecture: if self.neuro.is_some() { "neuro" } else { "bicameral" }.into(),
            networks: self.trace_networks(),
            modules: self
                .neuro
                .as_ref()
                .map(|n| n.modules().iter().map(|m| m.name().to_string()).collect())
                .unwrap_or_default(),
            rounds: self.config.loops,
            context: self.config.max_tokens,
            prompt_tokens: prompt.len(),
            ..Default::default()
        };
        if prompt.len() >= self.config.max_tokens {
            t.notes.push(format!(
                "The prompt is {} tokens, at least the model's context of {}: only the last {} are read at once (the working memory restarts with that window).",
                prompt.len(),
                self.config.max_tokens,
                self.config.max_tokens
            ));
        }
        let mut line = 0;
        let mut probs: Vec<f32> = Vec::new();
        for &id in prompt {
            let surprise = (!probs.is_empty()).then(|| -probs[id as usize].max(1e-30).log2());
            let logits = cache.append(self, id)?;
            t.tokens.push(TokenInfo { id, text: self.tokenizer.token_label(id), line, reply: false });
            t.profile.push(profile_of(cache.trace(), surprise));
            if id == b'\n' as TokenId {
                line += 1;
            }
            probs = self.next_token_probabilities(logits);
        }
        t.lines = lines_of(self, &t.tokens, false)?;
        if !each(TraceEvent::Read(&t)) {
            t.millis = started.elapsed().as_millis() as u64;
            return Ok(t);
        }
        // Each prompt line's recalled patterns, per round (for "similar lines").
        let line_patterns = line_patterns(&t);
        let (forced, max_tokens, mut choose) = match reply {
            TraceReply::Forced(ids) => (Some(ids), ids.len(), None),
            TraceReply::Generate { max_tokens, choose } => (None, max_tokens, Some(choose)),
        };
        t.forced = forced.is_some();
        let mut written: Vec<TokenId> = Vec::new();
        for index in 0..max_tokens {
            let next = match (&forced, choose.as_mut()) {
                (Some(ids), _) => ids[index],
                (None, Some(c)) => c(&probs, &written),
                _ => unreachable!(),
            };
            if (next as usize) >= probs.len() {
                return Err("Invalid reply token".into());
            }
            let position = t.tokens.len();
            let step = self.step_trace(index, position, next, &probs, cache.trace(), &t, &line_patterns);
            t.tokens.push(TokenInfo { id: next, text: self.tokenizer.token_label(next), line, reply: true });
            t.profile.push(TokenProfile {
                surprise: Some(-probs[next as usize].max(1e-30).log2()),
                ..Default::default()
            });
            t.steps.push(step);
            written.push(next);
            let go = each(TraceEvent::Step(t.steps.last().unwrap(), t.tokens.last().unwrap()));
            if next == b'\n' as TokenId {
                t.finished = true;
                break;
            }
            if !go || index + 1 == max_tokens {
                break;
            }
            let logits = cache.append(self, next)?;
            // The written token's own pass completes its profile.
            let surprise = t.profile.last().and_then(|p| p.surprise);
            *t.profile.last_mut().unwrap() = profile_of(cache.trace(), surprise);
            probs = self.next_token_probabilities(logits);
        }
        t.lines = lines_of(self, &t.tokens, true)?;
        t.reply = self.tokenizer.decode(&written)?.trim_end_matches('\n').to_string();
        t.millis = started.elapsed().as_millis() as u64;
        Ok(t)
    }

    #[allow(clippy::too_many_arguments)]
    fn step_trace(
        &self,
        index: usize,
        position: usize,
        next: TokenId,
        probs: &[f32],
        trace: Option<&Trace>,
        t: &ConversationTrace,
        line_patterns: &[Vec<BTreeMap<usize, f32>>],
    ) -> StepTrace {
        let next = next as usize;
        let alternatives = top_k(probs, ALTERNATIVES + 1)
            .into_iter()
            .filter(|&(i, _)| i != next)
            .take(ALTERNATIVES)
            .map(|(i, p)| self.trace_choice(i, p))
            .collect();
        let mut step = StepTrace {
            index,
            position,
            chosen: self.trace_choice(next, probs[next]),
            alternatives,
            entropy: entropy_bits(probs),
            ..Default::default()
        };
        let Some(tr) = trace else { return step };
        let read = position; // tokens read so far
        let line_of = |pos: usize| t.tokens.get(pos).map_or(usize::MAX, |x| x.line);
        let mut all_lines: BTreeMap<usize, f32> = BTreeMap::new();
        for s in &tr.stages {
            let sum = stage_summary(s, read, &line_of);
            for &(l, w) in &sum.lines {
                *all_lines.entry(l).or_default() += w / tr.stages.len() as f32;
            }
            step.transformer.stages.push(sum);
        }
        let mut lines: Vec<(usize, f32)> = all_lines.into_iter().filter(|(_, w)| *w >= 0.001).collect();
        lines.sort_by_key(|(l, _)| *l);
        step.transformer.lines = lines;
        let rounds = tr.readouts.len().saturating_sub(1);
        step.transformer.readouts = tr
            .readouts
            .iter()
            .enumerate()
            .map(|(i, z)| Readout {
                stage: if i == 0 {
                    "lower stage".into()
                } else if i == rounds {
                    format!("round {i} (output)")
                } else {
                    format!("round {i}")
                },
                top: top_k(&self.next_token_probabilities(z.clone()), 3)
                    .into_iter()
                    .map(|(t, q)| self.trace_choice(t, q))
                    .collect(),
            })
            .collect();
        match &tr.neuro {
            Some(n) => {
                for l in 0..n.memory_update.len() {
                    step.memory.push(MemoryLayerStep {
                        update: n.memory_update[l],
                        overwrite: n.memory_overwrite.get(l).copied().unwrap_or_default(),
                        change: n.memory_change[l],
                        norm: n.memory_norm[l],
                        lines: n.memory_lines.get(l).cloned().unwrap_or_default(),
                    });
                }
                for (i, r) in n.rounds.iter().enumerate() {
                    let g = tr.gates.get(i);
                    step.rounds.push(round_step(i, r, &n.modules, g, line_patterns));
                }
            }
            None => {
                // Nova Bicameral: the dense pathway's exchanges (before the upper blocks, then
                // each round).
                for (i, g) in tr.gates.iter().enumerate() {
                    step.rounds.push(RoundStep {
                        round: i,
                        injection_norm: g.injection_norm,
                        state_norm: g.state_norm,
                        write_gate: g.gate.iter().sum::<f32>() / g.gate.len().max(1) as f32,
                        ..Default::default()
                    });
                }
            }
        }
        step
    }

    /// Measures each network's influence on every step of `trace`: the whole turn (prompt and
    /// reply) is read again with that network switched off, one pass per network (in
    /// parallel), plus one with nothing switched off as the reference.
    pub fn measure_influence(&self, trace: &mut ConversationTrace) -> Result<(), String> {
        let ids: Vec<TokenId> = trace.tokens.iter().map(|t| t.id).collect();
        let first = trace.prompt_tokens;
        if trace.steps.is_empty() || first == 0 {
            return Ok(());
        }
        let mut runs = vec![("full".to_string(), Ablation::default())];
        runs.extend(self.influence_switches());
        // Step i is predicted by the pass that read position first + i - 1.
        let at: Vec<usize> = trace.steps.iter().map(|s| s.position - 1).collect();
        let results: Vec<Result<Vec<Vec<f32>>, String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = runs
                .iter()
                .map(|(_, abl)| {
                    let (ids, at) = (&ids, &at);
                    scope.spawn(move || {
                        let mut cache = KvCache::new(self, *abl);
                        let mut out = Vec::with_capacity(at.len());
                        let last = at.last().copied().unwrap_or(0);
                        for (pos, &id) in ids.iter().enumerate().take(last + 1) {
                            let logits = cache.append(self, id)?;
                            if at.contains(&pos) {
                                out.push(self.next_token_probabilities(logits));
                            }
                        }
                        Ok(out)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("A pass failed".into()))).collect()
        });
        let mut results = results.into_iter().collect::<Result<Vec<_>, _>>()?;
        let full = results.remove(0);
        for step in &mut trace.steps {
            step.influence.clear();
        }
        for ((name, _), probs) in runs[1..].iter().zip(&results) {
            for (k, step) in trace.steps.iter_mut().enumerate() {
                let (p, q) = (&full[k], &probs[k]);
                let top_q = top_k(q, 1)[0];
                let top_p = top_k(p, 1)[0];
                step.influence.push(Influence {
                    network: name.clone(),
                    change: total_variation(p, q),
                    chosen_probability: q[step.chosen.id as usize],
                    top: self.trace_choice(top_q.0, top_q.1),
                    flips: top_q.0 != top_p.0,
                });
            }
        }
        trace.influence = runs[1..].iter().map(|(n, _)| n.clone()).collect();
        Ok(())
    }
}

fn stage_summary(s: &StageTrace, read: usize, line_of: &dyn Fn(usize) -> usize) -> StageSummary {
    let n = s.attention.first().map_or(0, |h| h.len());
    let start = read.saturating_sub(n);
    let heads_n = s.attention.len().max(1) as f32;
    let mut lines: BTreeMap<usize, f32> = BTreeMap::new();
    let mut distance = 0f32;
    let heads = s
        .attention
        .iter()
        .map(|h| {
            for (t, &w) in h.iter().enumerate() {
                *lines.entry(line_of(start + t)).or_default() += w / heads_n;
                distance += w * (n - 1 - t) as f32 / heads_n;
            }
            HeadSummary {
                entropy: entropy_bits(h),
                top: top_k(h, HEAD_TOP).into_iter().map(|(t, w)| (start + t, w)).collect(),
            }
        })
        .collect();
    let mut lines: Vec<(usize, f32)> = lines.into_iter().filter(|(_, w)| *w >= 0.001).collect();
    lines.sort_by_key(|(l, _)| *l);
    StageSummary {
        block: s.block,
        round: s.round,
        state_norm: s.state_norm,
        update_norm: s.update_norm,
        heads,
        lines,
        distance,
    }
}

fn round_step(
    i: usize,
    r: &NeuroRound,
    modules: &[String],
    gate: Option<&super::cache::GateTrace>,
    line_patterns: &[Vec<BTreeMap<usize, f32>>],
) -> RoundStep {
    let mut similar: Vec<(usize, f32)> = if r.patterns.is_empty() {
        Vec::new()
    } else {
        line_patterns
            .iter()
            .enumerate()
            .filter_map(|(l, per_round)| per_round.get(i).map(|m| (l, sparse_cosine(m, &r.patterns))))
            .filter(|(_, c)| *c > 0.)
            .collect()
    };
    similar.sort_by(|a, b| b.1.total_cmp(&a.1));
    similar.truncate(8);
    RoundStep {
        round: i + 1,
        read_norm: r.read_norm,
        memory_in_norm: r.memory_in_norm,
        bus_norm: r.bus_norm,
        bus: r.bus.clone(),
        modules: modules
            .iter()
            .enumerate()
            .map(|(k, name)| ModuleStep {
                name: name.clone(),
                gate: r.gates.get(k).copied().unwrap_or_default(),
                output_norm: r.outputs.get(k).copied().unwrap_or_default(),
            })
            .collect(),
        write_norm: r.write_norm,
        injection_norm: gate.map_or(0., |g| g.injection_norm),
        state_norm: gate.map_or(0., |g| g.state_norm),
        write_gate: gate.map_or(0., |g| g.gate.iter().sum::<f32>() / g.gate.len().max(1) as f32),
        controller: r.controller.clone(),
        patterns: r.patterns.clone(),
        retrieval_entropy: r.retrieval_entropy,
        similar_lines: similar,
        slot_reads: r.slot_reads.clone(),
        experts: r.experts.clone(),
        code_norm: l2(&r.code),
    }
}

/// The light per-token record of a token's own pass.
fn profile_of(trace: Option<&Trace>, surprise: Option<f32>) -> TokenProfile {
    let mut p = TokenProfile { surprise, ..Default::default() };
    let Some(tr) = trace else { return p };
    p.injection = tr.gates.iter().map(|g| g.injection_norm / g.state_norm.max(1e-9)).collect();
    if let Some(n) = &tr.neuro {
        p.gates = n.rounds.iter().map(|r| r.gates.clone()).collect();
        p.memory_update = n.memory_update.clone();
        p.memory_overwrite = n.memory_overwrite.clone();
        p.memory_change = n.memory_change.clone();
        if n.rounds.iter().any(|r| !r.patterns.is_empty()) {
            p.retrieval_entropy = n.rounds.iter().map(|r| r.retrieval_entropy).collect();
            p.patterns = n
                .rounds
                .iter()
                .map(|r| r.patterns.iter().take(PROFILE_PATTERNS).copied().collect())
                .collect();
        }
    }
    p
}

/// Each line's recalled patterns, per round: the mean of its tokens' strongest patterns.
fn line_patterns(t: &ConversationTrace) -> Vec<Vec<BTreeMap<usize, f32>>> {
    let mut out: Vec<Vec<BTreeMap<usize, f32>>> = vec![Vec::new(); t.lines.len()];
    for (tok, p) in t.tokens.iter().zip(&t.profile) {
        let per_round = &mut out[tok.line];
        if per_round.len() < p.patterns.len() {
            per_round.resize(p.patterns.len(), BTreeMap::new());
        }
        for (r, pats) in p.patterns.iter().enumerate() {
            for &(i, w) in pats {
                *per_round[r].entry(i).or_default() += w;
            }
        }
    }
    out
}

/// The lines of the tokens read, with their roles.
fn lines_of(model: &Bicameral, tokens: &[TokenInfo], with_reply: bool) -> Result<Vec<Line>, String> {
    let count = tokens.last().map_or(0, |t| t.line + 1);
    let mut lines: Vec<Line> = Vec::with_capacity(count);
    for l in 0..count {
        let start = tokens.iter().position(|t| t.line == l).unwrap_or(0);
        let end = tokens.iter().rposition(|t| t.line == l).map_or(start, |e| e + 1);
        let ids: Vec<TokenId> = tokens[start..end].iter().map(|t| t.id).collect();
        let text = model.tokenizer.decode(&ids)?;
        let role = if text.starts_with(crate::memory::MEMORY) {
            "memory"
        } else if text.starts_with("U: ") || text.starts_with("U:") {
            "user"
        } else if text.starts_with("N: ") || text.starts_with("N:") {
            "nova"
        } else {
            "context"
        };
        let mut shown: String = text.trim_end_matches('\n').chars().take(200).collect();
        if shown.starts_with(crate::memory::MEMORY) {
            shown = format!("[memory] {}", &shown[crate::memory::MEMORY.len_utf8()..]);
        }
        lines.push(Line { index: l, start, end, role: role.into(), text: shown });
    }
    // The last line is the reply (it starts with the prompt's "N: "); the user line before it is
    // the message being answered.
    if let Some(last) = lines.last_mut() {
        if last.role == "nova" || with_reply || tokens[last.start..].iter().any(|t| t.reply) {
            last.role = "reply".into();
        }
    }
    if let Some(m) = lines.iter_mut().rev().find(|l| l.role == "user") {
        m.role = "message".into();
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::neuro::presets;
    use crate::model::Focus;

    fn neuro_model() -> Bicameral {
        let p = presets().into_iter().next().unwrap();
        let mut m = Bicameral::with_design(p.config, Some(p.design), 9);
        for _ in 0..3 {
            m.train_text(
                "U: Where is the key?\nN: In the box.\nU: No, it is in the bag.\nN: In the bag.\n",
                3e-3,
                Focus::Joint,
            )
            .unwrap();
        }
        m
    }

    fn prompt(m: &Bicameral) -> Vec<TokenId> {
        m.tokenizer.encode("Talk kindly.\nU: Where is the key?\nN: In the box.\nU: No, it is in the bag. Where is it?\nN: ").unwrap()
    }

    /// Tracing changes nothing: the traced reply is the reply generation gives, and every
    /// step's probability is the one generation computed.
    #[test]
    fn tracing_does_not_change_generation() {
        for m in [neuro_model(), Bicameral::new()] {
            let ids = prompt(&m);
            let plain = m.generate_cached_ids(&ids, 12, Ablation::default()).unwrap();
            let mut choose = |p: &[f32], _: &[TokenId]| crate::model::argmax(p);
            let t = m
                .trace_turn(&ids, TraceReply::Generate { max_tokens: 12, choose: &mut choose }, |_| true)
                .unwrap();
            let traced: Vec<TokenId> = t.steps.iter().map(|s| s.chosen.id).collect();
            assert_eq!(traced, plain);
            // The same probabilities, bit for bit, as an untraced cache.
            let mut cache = KvCache::new(&m, Ablation::default());
            let mut probs = Vec::new();
            for &id in &ids {
                probs = m.predict_cached(&mut cache, id).unwrap();
            }
            for (s, &id) in t.steps.iter().zip(&plain) {
                assert_eq!(s.chosen.probability.to_bits(), probs[id as usize].to_bits());
                probs = m.predict_cached(&mut cache, id).unwrap();
            }
            // Replaying the reply gives the same trace.
            let forced = m.trace_turn(&ids, TraceReply::Forced(&plain), |_| true).unwrap();
            assert_eq!(forced.steps.len(), t.steps.len());
            for (a, b) in forced.steps.iter().zip(&t.steps) {
                assert_eq!(a.chosen, b.chosen, "chosen at {}", a.index);
                assert_eq!(a.transformer, b.transformer, "transformer at {}", a.index);
                assert_eq!(a.memory, b.memory, "memory at {}", a.index);
                assert_eq!(a.rounds, b.rounds, "rounds at {}", a.index);
            }
            assert_eq!(forced.steps, t.steps);
            assert_eq!(forced.profile, t.profile);
        }
    }

    /// A sampled chat reply written with the trace on is the reply written with it off.
    #[test]
    fn traced_chat_replies_are_the_same_replies() {
        use crate::learn::sampling::{reply_traced, reply_with, Sampling};
        let m = neuro_model();
        let prompt = "U: Where is the key?\nN: In the box.\nU: Where is it?\nN: ";
        for seed in [1, 7, 42] {
            let s = Sampling { temperature: 0.9, top_k: 20, top_p: 0.95, repetition_penalty: 1.3, seed };
            let (mut a, mut b) = (String::new(), String::new());
            let plain = reply_with(&m, prompt, 16, &s, |t| {
                a.push_str(t);
                true
            })
            .unwrap();
            let mut steps = 0;
            let (traced, trace) = reply_traced(
                &m,
                prompt,
                16,
                &s,
                |t| {
                    b.push_str(t);
                    true
                },
                |e| {
                    if let TraceEvent::Step(..) = e {
                        steps += 1;
                    }
                },
            )
            .unwrap();
            assert_eq!(plain, traced);
            assert_eq!(a, b);
            assert_eq!(steps, trace.steps.len());
            assert_eq!(trace.reply, traced.text);
        }
    }

    /// Steps line up with tokens: step i wrote the token at `position`, read after everything
    /// before it; lines and roles follow the prompt.
    #[test]
    fn steps_and_lines_line_up() {
        let m = neuro_model();
        let ids = prompt(&m);
        let reply = m.tokenizer.encode("In the bag.\n").unwrap();
        let t = m.trace_turn(&ids, TraceReply::Forced(&reply), |_| true).unwrap();
        assert_eq!(t.steps.len(), reply.len());
        assert_eq!(t.tokens.len(), ids.len() + reply.len());
        for (i, s) in t.steps.iter().enumerate() {
            assert_eq!(s.index, i);
            assert_eq!(s.position, ids.len() + i);
            assert_eq!(t.tokens[s.position].id, s.chosen.id);
            assert!(t.tokens[s.position].reply);
            // Attention covers what was read: the newest position read is position - 1.
            for st in &s.transformer.stages {
                for h in &st.heads {
                    assert!(h.top.iter().all(|&(p, _)| p < s.position));
                }
                let mass: f32 = st.lines.iter().map(|(_, w)| w).sum();
                assert!(mass > 0.95 && mass < 1.001, "attention mass {mass}");
            }
        }
        let roles: Vec<&str> = t.lines.iter().map(|l| l.role.as_str()).collect();
        assert_eq!(roles, ["context", "user", "nova", "message", "reply"]);
        assert_eq!(t.reply, "In the bag.");
        assert!(t.finished);
        // The networks are those built: transformer, working memory, associative memory,
        // controller, bus.
        let ids: Vec<&str> = t.networks.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, ["transformer", "memory", "associative", "controller", "bus"]);
        assert_eq!(t.steps[0].rounds.len(), m.config.loops);
        assert_eq!(t.steps[0].memory.len(), m.neuro.as_ref().unwrap().memory.as_ref().unwrap().layers);
    }

    /// The working memory's split by line is exact: the contributions add up to the state.
    #[test]
    fn memory_split_by_line_adds_up_to_the_state() {
        let m = neuro_model();
        let ids = prompt(&m);
        let mut cache = KvCache::with_trace(&m, Ablation::default());
        for &id in &ids {
            cache.append(&m, id).unwrap();
        }
        let tr = cache.trace().unwrap().neuro.clone().unwrap();
        for (l, state) in tr.memory_state.iter().enumerate() {
            let parts = &cache.memory_lines[l];
            assert!(parts.len() >= 4, "every line wrote into layer {l}");
            let mut sum = vec![0f32; state.len()];
            for (_, v) in parts {
                for (s, x) in sum.iter_mut().zip(v) {
                    *s += x;
                }
            }
            for (a, b) in sum.iter().zip(state) {
                assert!((a - b).abs() <= 1e-4 * (1. + b.abs()), "layer {l}: {a} vs {b}");
            }
        }
    }

    /// Influence: switching a network off for the turn changes the prediction by a measured
    /// amount, and nothing switched off changes nothing.
    #[test]
    fn influence_is_measured_per_network() {
        let m = neuro_model();
        let ids = prompt(&m);
        let reply = m.tokenizer.encode("In the bag.\n").unwrap();
        let mut t = m.trace_turn(&ids, TraceReply::Forced(&reply), |_| true).unwrap();
        m.measure_influence(&mut t).unwrap();
        assert_eq!(t.influence, ["memory", "associative", "controller", "bus write-back", "bus input"]);
        for s in &t.steps {
            assert_eq!(s.influence.len(), 5);
            for i in &s.influence {
                assert!((0. ..=1.).contains(&i.change));
            }
        }
        assert!(t.steps.iter().any(|s| s.influence.iter().any(|i| i.change > 1e-4)));
    }
}
