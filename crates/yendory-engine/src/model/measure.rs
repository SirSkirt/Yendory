//! Measurements for diagnostics and the brain visualizer, computed from a real forward pass of
//! the model. Nothing here is estimated or simulated, and nothing feeds back into training or
//! inference.
use super::{Ablation, Bicameral, Focus, KvCache, Trace};
use crate::tokenizer::TokenId;
use serde::{Deserialize, Serialize};

/// How much the next-token distribution changes when a pathway is cut, as total variation
/// distance (0 = identical, 1 = completely different).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AblationEffects {
    /// Right hemisphere's input to the left cut. (Silencing the right hemisphere and cutting the
    /// afferent path are the same operation for the output, so this is one measurement.)
    pub without_right_to_left: f32,
    /// The right hemisphere no longer hears the left (efferent path cut).
    pub without_left_to_right: f32,
}

/// Nova Neuro: how much the next-token distribution changes when one of its systems is switched
/// off (total variation distance; `None` when the model does not have that network).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NeuroEffects {
    pub without_memory: Option<f32>,
    pub without_associative: Option<f32>,
    /// Every controller gate fully open (the controller's choices removed).
    pub without_controller: Option<f32>,
    /// One computation round instead of the model's rounds (`None` with one round).
    pub one_round: Option<f32>,
    #[serde(default)]
    pub without_relational: Option<f32>,
    #[serde(default)]
    pub without_experts: Option<f32>,
    #[serde(default)]
    pub without_consolidation: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Measurements {
    pub step: u64,
    /// Labels of the tokens in the context window (oldest first).
    pub tokens: Vec<String>,
    /// Internal activity while reading the last prompt token.
    pub trace: Trace,
    /// Mean gate opening of each right->left injection (exchange 1, then each thinking round).
    pub gate_means: Vec<f32>,
    /// Mean absolute activation of the right hemisphere's bottom layer and of its upper layer
    /// in each round.
    pub right_trunk_activity: f32,
    pub right_upper_activity: Vec<f32>,
    /// The five most likely next tokens of the full model.
    pub next_tokens: Vec<(String, f32)>,
    pub ablation: AblationEffects,
    /// Parameters per pathway: transformer (left), dense (right), interconnect.
    pub parameters: [usize; 3],
    /// Nova Neuro: the effect of switching off each of its systems.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neuro_effects: Option<NeuroEffects>,
}

pub(super) fn probabilities(model: &Bicameral, logits: &[f32]) -> Vec<f64> {
    let valid = (128 + model.tokenizer.words().len()).min(model.config.vocab);
    let z = &logits[..valid];
    let max = z.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64));
    let e: Vec<f64> = z.iter().map(|&x| (x as f64 - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|x| x / sum).collect()
}

pub(super) fn total_variation(a: &[f64], b: &[f64]) -> f32 {
    (0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()) as f32
}

fn mean_abs(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).sum::<f32>() / v.len().max(1) as f32
}

impl Bicameral {
    /// Reads `prompt` and measures the forward pass of its last token.
    pub fn measure(&self, prompt: &str) -> Result<Measurements, String> {
        self.measure_ids(&self.tokenizer.encode(prompt)?)
    }
    /// Measures the forward pass of the last of `ids` (e.g. a prompt plus generated tokens).
    pub fn measure_ids(&self, ids: &[TokenId]) -> Result<Measurements, String> {
        let ids = ids.to_vec();
        let mut cache = KvCache::with_trace(self, Ablation::default());
        let logits = cache.prefill(self, &ids)?;
        let trace = cache.trace().cloned().unwrap_or_default();
        let full = probabilities(self, &logits);
        let ablated = |abl: Ablation| -> Result<Vec<f64>, String> {
            Ok(probabilities(self, &KvCache::new(self, abl).prefill(self, &ids)?))
        };
        let without_right = ablated(Ablation { no_afferent: true, ..Ablation::default() })?;
        let without_efferent = ablated(Ablation { no_efferent: true, ..Ablation::default() })?;
        let neuro_effects = match &self.neuro {
            None => None,
            Some(n) => {
                let effect = |on: bool, abl: Ablation| -> Result<Option<f32>, String> {
                    Ok(if on { Some(total_variation(&full, &ablated(abl)?)) } else { None })
                };
                let has_modules = !n.modules().is_empty();
                Some(NeuroEffects {
                    without_memory: effect(
                        n.memory.is_some(),
                        Ablation { no_memory: true, ..Ablation::default() },
                    )?,
                    without_associative: effect(
                        n.associative.is_some(),
                        Ablation { no_associative: true, ..Ablation::default() },
                    )?,
                    without_controller: effect(
                        has_modules,
                        Ablation { no_controller: true, ..Ablation::default() },
                    )?,
                    one_round: effect(
                        self.config.loops > 1,
                        Ablation { one_round: true, ..Ablation::default() },
                    )?,
                    without_relational: effect(
                        n.relational.is_some(),
                        Ablation { no_relational: true, ..Ablation::default() },
                    )?,
                    without_experts: effect(
                        n.experts.is_some(),
                        Ablation { no_experts: true, ..Ablation::default() },
                    )?,
                    without_consolidation: effect(
                        n.consolidation.is_some(),
                        Ablation { no_consolidation: true, ..Ablation::default() },
                    )?,
                })
            }
        };
        let mut top: Vec<(usize, f64)> = full.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let window = &ids[ids.len().saturating_sub(self.config.max_tokens)..];
        Ok(Measurements {
            step: self.step,
            tokens: window.iter().map(|&t| self.tokenizer.token_label(t)).collect(),
            gate_means: trace
                .gates
                .iter()
                .map(|g| g.gate.iter().sum::<f32>() / g.gate.len().max(1) as f32)
                .collect(),
            right_trunk_activity: mean_abs(&trace.right_trunk),
            right_upper_activity: trace.right_upper.iter().map(|z| mean_abs(z)).collect(),
            next_tokens: top
                .iter()
                .take(5)
                .map(|&(t, p)| (self.tokenizer.token_label(t as TokenId), p as f32))
                .collect(),
            ablation: AblationEffects {
                without_right_to_left: total_variation(&full, &without_right),
                without_left_to_right: total_variation(&full, &without_efferent),
            },
            parameters: [
                self.group_parameters(Focus::Left),
                self.group_parameters(Focus::Right),
                self.group_parameters(Focus::Interconnect),
            ],
            neuro_effects,
            trace,
        })
    }
}
