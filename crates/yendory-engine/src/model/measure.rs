//! Measurements for diagnostics and the brain visualizer, computed from a real forward pass of
//! the model. Nothing here is estimated or simulated, and nothing feeds back into training or
//! inference.
use super::{Ablation, Bicameral, Focus, KvCache, Trace};
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
}

pub(super) fn probabilities(model: &Bicameral, logits: &[f32]) -> Vec<f64> {
    let valid = (128 + model.tokenizer.words.len()).min(model.config.vocab);
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
    pub fn measure_ids(&self, ids: &[u16]) -> Result<Measurements, String> {
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
                .map(|&(t, p)| (self.tokenizer.token_label(t as u16), p as f32))
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
            trace,
        })
    }
}
