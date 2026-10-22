//! Growing a trained network without losing what it learned (docs/GROWTH.md).
//!
//! Function-preserving growth: the grown network computes what the original computed (exactly,
//! or within float rounding where a norm's epsilon is involved), so trained knowledge is kept;
//! the new capacity is then trained like any other. New units are appended after the existing
//! ones in every dimension, so each existing weight keeps its place. The rule for new weights:
//! a new unit's *outgoing* weights (everything that writes into a computation that already
//! exists) start at zero; its *incoming* weights start small and random, so gradients reach the
//! outgoing weights at once and training can start using the unit. RMS norms whose input gains
//! new (zero) dimensions have their gains rescaled by sqrt(old / new width), because the mean
//! square is then taken over more values.
//!
//! What cannot be grown this way is refused, never approximated: more associative-memory
//! patterns (the softmax over patterns renormalises), more working-memory layers and more
//! thinking rounds (neither is an identity).
use super::{Bicameral, Config, NeuroSpec, Slot};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// How much to add to each part of a network (zero: unchanged).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Growth {
    /// Attention heads; the transformer's width grows by one head width each.
    pub heads: usize,
    /// Feed-forward units in every transformer block.
    pub ff: usize,
    /// Transformer blocks added at the end of the lower blocks (run once).
    pub lower_blocks: usize,
    /// Transformer blocks added at the end of the upper blocks (run in every round).
    pub upper_blocks: usize,
    /// Bicameral: units of Dense's first and second layers.
    pub dense1: usize,
    pub dense2: usize,
    /// Nova Neuro: bus width, working-memory width (every layer), controller width, and the
    /// associative memory's key width.
    pub bus: usize,
    pub memory_width: usize,
    pub controller: usize,
    pub key: usize,
}

impl Growth {
    pub fn is_empty(&self) -> bool {
        *self == Growth::default()
    }
}

/// How closely a grown network reproduces the original at the moment of growth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Preservation {
    /// The same computation; only float rounding of the wider sums differs.
    Exact,
    /// A norm's epsilon is divided over more dimensions: differences of about one part in a
    /// million of the activations.
    NearExact,
}

/// A growth step Studio can offer, with its exact result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrowthOption {
    pub id: String,
    pub label: String,
    pub growth: Growth,
    pub preservation: Preservation,
    pub parameters_before: usize,
    pub parameters_after: usize,
    /// Values stored per token while training, after growth (multiply by 4 bytes per value).
    pub activation_floats_per_token: usize,
}

/// A part that cannot be grown without changing what the network computes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NotOffered {
    pub part: String,
    pub reason: String,
}

/// Outputs of two networks on the same probe text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutputComparison {
    /// The largest change of any next-token logit at the probes' last positions.
    pub max_logit_change: f32,
    /// Mean teacher-forced loss on the probes, before and after.
    pub loss_before: f32,
    pub loss_after: f32,
}

impl OutputComparison {
    /// Within what float rounding (and, for width growth, a norm's epsilon) explains.
    pub fn preserved(&self) -> bool {
        self.max_logit_change <= 2e-3
            && (self.loss_after - self.loss_before).abs() <= 1e-3 * self.loss_before.abs().max(1.)
    }
}

/// Which weights of a slot start at zero when its rows or columns grow.
#[derive(Clone, Copy, Default)]
struct Rule {
    zero_new_rows: bool,
    zero_new_cols: bool,
    /// An RMS norm's gains: rescale when the normalised width grows.
    norm: bool,
}

/// The slot's name without its block prefix ("b3.wo" -> "wo").
fn base(name: &str) -> &str {
    match name.split_once('.') {
        Some((b, rest)) if b.starts_with('b') && b[1..].chars().all(|c| c.is_ascii_digit()) => rest,
        _ => name,
    }
}

fn rule(name: &str) -> Rule {
    let b = base(name);
    let zero = |rows, cols| Rule { zero_new_rows: rows, zero_new_cols: cols, norm: false };
    if matches!(b, "ln1" | "ln2" | "ln_f" | "c.n1" | "c.n2" | "n.read_norm" | "n.bus_norm" | "n.as_norm")
        || (b.starts_with("n.wm") && b.ends_with(".norm"))
    {
        return Rule { norm: true, ..Rule::default() };
    }
    if b.starts_with("n.wm") && b.ends_with(".c") {
        // New memory units' candidates: zero, so the units hold zero until trained.
        return zero(true, false);
    }
    match b {
        // Writes into the residual stream: new dimensions and new sources start at zero.
        "embed" => zero(false, true),
        "wo" | "w_down" | "c.a1" | "c.a2" | "n.write" => zero(true, true),
        // New Dense-1 units feed Dense 2 only once trained.
        "r.w2" => zero(false, true),
        // New bus dimensions start at zero (the bus is normalised; nonzero values would change it).
        "n.read" | "n.wm_out" => zero(true, false),
        // New controller units do not move the gates until trained.
        "n.ctl2" => zero(false, true),
        // New key dimensions do not change the pattern scores until trained.
        "n.as_q" => zero(true, false),
        _ => Rule::default(),
    }
}

impl Bicameral {
    /// The configuration and design after `growth`, checked (or why it cannot be grown).
    pub fn grown_shape(&self, g: &Growth) -> Result<(Config, Option<NeuroSpec>), String> {
        if g.is_empty() {
            return Err("Choose something to grow".into());
        }
        let c = &self.config;
        let head = c.d / c.heads.max(1);
        let mut config = Config {
            d: c.d + g.heads * head,
            heads: c.heads + g.heads,
            ff: c.ff + g.ff,
            blocks: c.blocks + g.lower_blocks + g.upper_blocks,
            split: c.split + g.lower_blocks,
            ..c.clone()
        };
        let design = match &self.neuro {
            None => {
                if g.bus + g.memory_width + g.controller + g.key > 0 {
                    return Err("A Bicameral model has no bus, working memory, controller or associative memory to grow".into());
                }
                config.r1 = c.r1 + g.dense1;
                config.r2 = c.r2 + g.dense2;
                None
            }
            Some(n) => {
                if g.dense1 + g.dense2 > 0 {
                    return Err("A Nova Neuro model has no Dense hemisphere; grow its modules instead".into());
                }
                let mut n = n.clone();
                n.bus += g.bus;
                if g.memory_width > 0 {
                    let m = n.memory.as_mut().ok_or("This design has no working memory to grow")?;
                    m.width += g.memory_width;
                }
                if g.controller > 0 {
                    if n.modules().is_empty() {
                        return Err("This design has no Executive Controller to grow".into());
                    }
                    n.controller += g.controller;
                }
                if g.key > 0 {
                    let a = n.associative.as_mut().ok_or("This design has no associative memory to grow")?;
                    a.key += g.key;
                }
                Some(n)
            }
        };
        Bicameral::check_config(&config)?;
        if let Some(n) = &design {
            n.validate(&config)?;
        }
        let params: usize =
            Bicameral::tensors_for(&config, design.as_ref()).iter().map(|(_, r, c)| r * c).sum();
        if params > super::MAX_PARAMETERS {
            return Err(format!("{params} parameters: beyond what the Engine supports"));
        }
        Ok((config, design))
    }

    /// The network grown by `growth`: every trained weight and its AdamW moments kept in
    /// place, new weights initialised by the growth rules (`seed` draws the random ones), the
    /// step and vocabulary unchanged. Check the result with [`compare_outputs`].
    pub fn grow(&self, g: &Growth, seed: u64) -> Result<Bicameral, String> {
        let (config, design) = self.grown_shape(g)?;
        let mut out = Bicameral::with_design(config, design, seed);
        out.tokenizer = self.tokenizer.clone();
        out.step = self.step;
        let old: HashMap<&str, &Slot> = self.slots.iter().map(|s| (s.name.as_str(), s)).collect();
        let (split, blocks) = (self.config.split, self.config.blocks);
        // The original block a new block index corresponds to (None: a new block).
        let old_block = |j: usize| -> Option<usize> {
            if j < split {
                Some(j)
            } else if j < split + g.lower_blocks {
                None
            } else if j - g.lower_blocks < blocks {
                Some(j - g.lower_blocks)
            } else {
                None
            }
        };
        let slots = out.slots.clone();
        for ns in &slots {
            let b = base(&ns.name);
            let old_name = match ns.name.split_once('.') {
                Some((p, rest)) if p.starts_with('b') && p[1..].chars().all(|c| c.is_ascii_digit()) => {
                    let j: usize = p[1..].parse().map_err(|_| "Bad block slot name")?;
                    old_block(j).map(|i| format!("b{i}.{rest}"))
                }
                _ => Some(ns.name.clone()),
            };
            let Some(os) = old_name.as_deref().and_then(|n| old.get(n)) else {
                // A new block: an identity on the residual stream until trained.
                if matches!(b, "wo" | "w_down") {
                    out.weights[ns.range.clone()].iter_mut().for_each(|w| *w = 0.);
                }
                continue;
            };
            if os.rows > ns.rows || os.cols > ns.cols {
                return Err(format!("{} would shrink", ns.name));
            }
            let r = rule(&ns.name);
            let scale =
                if r.norm && ns.cols > os.cols { (os.cols as f32 / ns.cols as f32).sqrt() } else { 1. };
            for row in 0..ns.rows {
                for col in 0..ns.cols {
                    let i = ns.range.start + row * ns.cols + col;
                    if row < os.rows && col < os.cols {
                        let j = os.range.start + row * os.cols + col;
                        out.weights[i] = self.weights[j] * scale;
                        out.momentum[i] = self.momentum[j];
                        out.variance[i] = self.variance[j];
                    } else {
                        out.momentum[i] = 0.;
                        out.variance[i] = 0.;
                        if (row >= os.rows && r.zero_new_rows) || (col >= os.cols && r.zero_new_cols) {
                            out.weights[i] = 0.;
                        } else if r.norm {
                            out.weights[i] = scale;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Growth steps for this model, with their parameter counts, and the parts that cannot be
    /// grown without changing what the network computes.
    pub fn growth_options(&self) -> (Vec<GrowthOption>, Vec<NotOffered>) {
        let c = &self.config;
        let quarter = |x: usize, step: usize| (x / 4).div_ceil(step).max(1) * step;
        let mut list: Vec<(&str, String, Growth, Preservation)> = vec![
            (
                "ff",
                format!("Feed-forward width +25% ({} -> {} units per block)", c.ff, c.ff + quarter(c.ff, 16)),
                Growth { ff: quarter(c.ff, 16), ..Growth::default() },
                Preservation::Exact,
            ),
            (
                "upper_block",
                "One more upper transformer block (runs in every thinking round)".into(),
                Growth { upper_blocks: 1, ..Growth::default() },
                Preservation::Exact,
            ),
            (
                "lower_block",
                "One more lower transformer block".into(),
                Growth { lower_blocks: 1, ..Growth::default() },
                Preservation::Exact,
            ),
            (
                "heads",
                format!("Transformer width +1 attention head ({} -> {})", c.d, c.d + c.d / c.heads.max(1)),
                Growth { heads: 1, ..Growth::default() },
                Preservation::NearExact,
            ),
        ];
        let mut not = vec![NotOffered {
            part: "Thinking rounds".into(),
            reason: "Another round adds another pass through the upper blocks; it is not an identity, so it would change what the model computes".into(),
        }];
        match &self.neuro {
            None => {
                list.push((
                    "dense",
                    format!(
                        "Dense hemisphere +25% ({} -> {} and {} -> {} units)",
                        c.r1,
                        c.r1 + quarter(c.r1, 8),
                        c.r2,
                        c.r2 + quarter(c.r2, 8)
                    ),
                    Growth { dense1: quarter(c.r1, 8), dense2: quarter(c.r2, 8), ..Growth::default() },
                    Preservation::Exact,
                ));
                list.push((
                    "balanced",
                    "Balanced: width +25%, feed-forward +25%, one more upper block, Dense +25%".into(),
                    Growth {
                        heads: (c.heads / 4).max(1),
                        ff: quarter(c.ff, 16),
                        upper_blocks: 1,
                        dense1: quarter(c.r1, 8),
                        dense2: quarter(c.r2, 8),
                        ..Growth::default()
                    },
                    Preservation::NearExact,
                ));
            }
            Some(n) => {
                if let Some(m) = &n.memory {
                    list.push((
                        "memory",
                        format!(
                            "Working memory +25% ({} -> {} units per layer)",
                            m.width,
                            m.width + quarter(m.width, 8)
                        ),
                        Growth { memory_width: quarter(m.width, 8), ..Growth::default() },
                        Preservation::Exact,
                    ));
                }
                if !n.modules().is_empty() {
                    list.push((
                        "controller",
                        format!(
                            "Executive Controller +50% ({} -> {} units)",
                            n.controller,
                            n.controller + (n.controller / 2).max(8)
                        ),
                        Growth { controller: (n.controller / 2).max(8), ..Growth::default() },
                        Preservation::Exact,
                    ));
                }
                list.push((
                    "bus",
                    format!("Communication bus +25% ({} -> {})", n.bus, n.bus + quarter(n.bus, 8)),
                    Growth { bus: quarter(n.bus, 8), ..Growth::default() },
                    Preservation::NearExact,
                ));
                if let Some(a) = &n.associative {
                    list.push((
                        "key",
                        format!("Associative-memory keys +25% ({} -> {})", a.key, a.key + quarter(a.key, 8)),
                        Growth { key: quarter(a.key, 8), ..Growth::default() },
                        Preservation::Exact,
                    ));
                    not.push(NotOffered {
                        part: "Associative-memory patterns".into(),
                        reason: "The softmax over patterns renormalises: a new pattern takes weight from the learned ones, so recall would change".into(),
                    });
                }
                if n.memory.is_some() {
                    not.push(NotOffered {
                        part: "Working-memory layers".into(),
                        reason:
                            "A new minGRU layer is not an identity: it would change what the memory passes on"
                                .into(),
                    });
                }
                list.push((
                    "balanced",
                    "Balanced: width +25%, feed-forward +25%, one more upper block, working memory and bus +25%".into(),
                    Growth {
                        heads: (c.heads / 4).max(1),
                        ff: quarter(c.ff, 16),
                        upper_blocks: 1,
                        memory_width: n.memory.as_ref().map_or(0, |m| quarter(m.width, 8)),
                        bus: quarter(n.bus, 8),
                        ..Growth::default()
                    },
                    Preservation::NearExact,
                ));
            }
        }
        let before = self.parameter_count();
        let options = list
            .into_iter()
            .filter_map(|(id, label, growth, preservation)| {
                let (config, design) = self.grown_shape(&growth).ok()?;
                let after =
                    Bicameral::tensors_for(&config, design.as_ref()).iter().map(|(_, r, c)| r * c).sum();
                Some(GrowthOption {
                    id: id.into(),
                    label,
                    preservation,
                    parameters_before: before,
                    parameters_after: after,
                    activation_floats_per_token: Bicameral::activation_floats_for(&config, design.as_ref()),
                    growth,
                })
            })
            .collect();
        (options, not)
    }
}

/// Compares two networks' outputs on probe texts: next-token logits at each probe's last
/// position and the mean teacher-forced loss. Used before saving a grown model.
pub fn compare_outputs(
    before: &Bicameral,
    after: &Bicameral,
    probes: &[&str],
) -> Result<OutputComparison, String> {
    let abl = super::Ablation::default();
    let (mut max, mut lb, mut la, mut n) = (0f32, 0f64, 0f64, 0usize);
    for text in probes {
        let ids = before.tokenizer.encode(text)?;
        if ids.len() < 2 {
            continue;
        }
        let ids = &ids[ids.len().saturating_sub(before.config.max_tokens + 1)..];
        let (a, b) = (
            super::KvCache::new(before, abl).prefill(before, &ids[..ids.len() - 1])?,
            super::KvCache::new(after, abl).prefill(after, &ids[..ids.len() - 1])?,
        );
        for (x, y) in a.iter().zip(&b) {
            max = max.max((x - y).abs());
        }
        let (x, y) = (&ids[..ids.len() - 1], &ids[1..]);
        let mask = vec![1.; x.len()];
        lb += before.loss(x, y, &mask, abl)? as f64;
        la += after.loss(x, y, &mask, abl)? as f64;
        n += 1;
    }
    if n == 0 {
        return Err("No probe text to compare on".into());
    }
    Ok(OutputComparison {
        max_logit_change: max,
        loss_before: (lb / n as f64) as f32,
        loss_after: (la / n as f64) as f32,
    })
}

/// Probe texts for checking a growth: everyday prose and dialogue in Nova's formats.
pub const PROBES: [&str; 6] = [
    "Once upon a time, a little girl named Lily went to the park with her dog.",
    "U: Hello! How are you today?\nN: I am fine, thank you.",
    "U: Tom went to the kitchen. Then Tom moved to the garden. Where is Tom?\nN: Tom is in the garden.",
    "The sun is hot. Snow is cold. Birds can fly and fish can swim.",
    "U: What is 3 plus 4?\nN: 3 + 4 = 7.",
    "In the beginning there was a small town by the river, and the people there worked hard every day.",
];
