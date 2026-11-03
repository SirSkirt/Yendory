//! The core sensing something other than words: a model (Nova Bicameral or Nova Neuro) reading
//! a sequence of vectors, one per position, in place of its words' embeddings, and giving its
//! final state at every position. Yendory Tools uses it to let the model's own core take part in
//! playing a game (one position per game tick; docs/GAME_TRAINING.md on yendory-tools).
//!
//! Everything after the first layer is the model as it is: the same blocks, hemispheres or
//! networks, the same weights, attention over earlier positions only. A Bicameral model's right
//! hemisphere reads a second vector per position (its input width, in place of the characters'
//! features); a Neuro model has none.
use super::{tape, Ablation, Bicameral, First};

impl Bicameral {
    /// The widths of what the core reads at each position when it senses: vectors of the model's
    /// width (where words' embeddings go), and the right hemisphere's input (0 for Neuro).
    pub fn sense_widths(&self) -> (usize, usize) {
        (self.config.d, if self.neuro.is_some() { 0 } else { self.config.window_chars * 16 })
    }

    fn sense_positions(&self, left: &[f32], right: &[f32]) -> Result<usize, String> {
        let (d, r) = self.sense_widths();
        let n = left.len() / d;
        if n == 0 || left.len() != n * d || right.len() != n * r {
            return Err("The core's input does not have its width at every position".into());
        }
        if n > self.config.max_tokens {
            return Err(format!("The core reads at most {} positions at once", self.config.max_tokens));
        }
        if left.iter().chain(right).any(|v| !v.is_finite()) {
            return Err("The core's input holds values that are not numbers".into());
        }
        Ok(n)
    }

    /// The core's final (normalised) state at each position, the model's width each, for the
    /// vectors `left` (positions × width) and, for Bicameral, `right` (positions × its width).
    pub fn sense(&self, left: &[f32], right: &[f32], abl: Ablation) -> Result<Vec<f32>, String> {
        let n = self.sense_positions(left, right)?;
        let mut t = tape::Tape::new(&self.weights, None);
        let node = t.input(left.to_vec());
        let feat = self.neuro.is_none().then(|| t.input(right.to_vec()));
        let h = self.hidden_from(&mut t, First::Node { node, positions: n }, feat, n, abl);
        Ok(t.values(h).to_vec())
    }

    /// [`Bicameral::sense`] and its backward pass in one: `back` gets the states and gives the
    /// gradient at each; the weights' gradient is added to `grads`; returns the gradients at
    /// `left` and `right`. (One forward pass, where `sense` then `sense_gradients` make two.)
    pub fn sense_train(
        &self,
        left: &[f32],
        right: &[f32],
        grads: &mut [f32],
        abl: Ablation,
        back: impl FnOnce(&[f32]) -> Result<Vec<f32>, String>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n = self.sense_positions(left, right)?;
        if grads.len() != self.weights.len() {
            return Err("The gradients do not fit the core".into());
        }
        let mut t = tape::Tape::new(&self.weights, Some(grads));
        let node = t.input(left.to_vec());
        let feat = self.neuro.is_none().then(|| t.input(right.to_vec()));
        let h = self.hidden_from(&mut t, First::Node { node, positions: n }, feat, n, abl);
        let dstates = back(t.values(h))?;
        if dstates.len() != n * self.config.d {
            return Err("The gradients do not fit the core's states".into());
        }
        t.backward_from(h, dstates);
        let dleft = t.grad(node);
        let dright = feat.map(|f| t.grad(f)).unwrap_or_default();
        Ok((dleft, dright))
    }

    /// The backward pass of [`Bicameral::sense`]: with `dstates`, the gradient at every state,
    /// adds the weights' gradient to `grads` and returns the gradients at `left` and `right`.
    pub fn sense_gradients(
        &self,
        left: &[f32],
        right: &[f32],
        dstates: &[f32],
        grads: &mut [f32],
        abl: Ablation,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n = self.sense_positions(left, right)?;
        if dstates.len() != n * self.config.d || grads.len() != self.weights.len() {
            return Err("The gradients do not fit the core".into());
        }
        let mut t = tape::Tape::new(&self.weights, Some(grads));
        let node = t.input(left.to_vec());
        let feat = self.neuro.is_none().then(|| t.input(right.to_vec()));
        let h = self.hidden_from(&mut t, First::Node { node, positions: n }, feat, n, abl);
        t.backward_from(h, dstates.to_vec());
        let dleft = t.grad(node);
        let dright = feat.map(|f| t.grad(f)).unwrap_or_default();
        Ok((dleft, dright))
    }
}

#[cfg(test)]
mod tests {
    use super::super::brain::{design_for, BrainChoice};
    use super::super::Config;
    use super::*;

    fn noise(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2. * scale
            })
            .collect()
    }

    fn bicameral() -> Bicameral {
        let c = Config {
            vocab: 140,
            d: 16,
            heads: 2,
            ff: 24,
            blocks: 2,
            split: 1,
            loops: 2,
            window_chars: 2,
            r1: 8,
            r2: 8,
            max_tokens: 16,
            ..Config::default()
        };
        let mut m = Bicameral::with_config(c, 5);
        // Open the gates a little so every path carries gradient.
        let jitter = noise(3, m.weights.len(), 0.1);
        for (w, j) in m.weights.iter_mut().zip(jitter) {
            *w += j;
        }
        m
    }

    fn neuro() -> Bicameral {
        let choice = BrainChoice {
            width: 64,
            transformer: Some(60_000),
            memory: Some(8_000),
            associative: Some(6_000),
            relational: Some(6_000),
            experts: Some(8_000),
            consolidation: Some(6_000),
            controller: Some(2_000),
        };
        let (mut c, n) = design_for(&choice, 140).unwrap();
        c.max_tokens = 16;
        let mut m = Bicameral::with_design(c, Some(n), 7);
        let jitter = noise(4, m.weights.len(), 0.1);
        for (w, j) in m.weights.iter_mut().zip(jitter) {
            *w += j;
        }
        m
    }

    /// The gradients of a weighted sum of the states against finite differences, at weights in
    /// every tensor and at inputs.
    fn check(m: &Bicameral) {
        let (d, r) = m.sense_widths();
        let n = 5;
        let left = noise(11, n * d, 1.0);
        let right = noise(12, n * r, 1.0);
        let proj = noise(13, n * d, 1.0);
        let abl = Ablation::default();
        let value = |m: &Bicameral, l: &[f32], rt: &[f32]| -> f64 {
            let s = m.sense(l, rt, abl).unwrap();
            s.iter().zip(&proj).map(|(a, b)| (*a as f64) * (*b as f64)).sum()
        };
        let mut g = vec![0f32; m.weights.len()];
        let (dl, dr) = m.sense_gradients(&left, &right, &proj, &mut g, abl).unwrap();
        let biggest = g.iter().chain(&dl).chain(&dr).fold(0f32, |a, x| a.max(x.abs()));
        let close = |analytic: f32, numeric: f32, what: &str| {
            assert!(
                (numeric - analytic).abs() <= 3e-3 * biggest.max(1e-3) + 0.05 * analytic.abs(),
                "{what}: analytic {analytic} numeric {numeric} (largest {biggest})"
            );
        };
        let eps = 1e-2f32;
        let mut checked = 0;
        for slot in &m.slots {
            for k in [0, slot.range.len() / 2, slot.range.len() - 1] {
                let i = slot.range.start + k;
                let mut up = m.clone();
                up.weights[i] += eps;
                let mut down = m.clone();
                down.weights[i] -= eps;
                let numeric =
                    ((value(&up, &left, &right) - value(&down, &left, &right)) / (2. * eps as f64)) as f32;
                close(g[i], numeric, &format!("{} [{k}]", slot.name));
                checked += 1;
            }
        }
        for (i, &a) in dl.iter().enumerate().step_by(7) {
            let (mut up, mut down) = (left.clone(), left.clone());
            up[i] += eps;
            down[i] -= eps;
            let numeric = ((value(m, &up, &right) - value(m, &down, &right)) / (2. * eps as f64)) as f32;
            close(a, numeric, &format!("left [{i}]"));
        }
        for (i, &a) in dr.iter().enumerate().step_by(5) {
            let (mut up, mut down) = (right.clone(), right.clone());
            up[i] += eps;
            down[i] -= eps;
            let numeric = ((value(m, &left, &up) - value(m, &left, &down)) / (2. * eps as f64)) as f32;
            close(a, numeric, &format!("right [{i}]"));
        }
        assert!(checked > 30, "{checked}");
        assert!(dl.iter().any(|v| *v != 0.));
    }

    #[test]
    fn a_bicameral_core_senses_and_its_gradients_are_right() {
        let m = bicameral();
        assert_eq!(m.sense_widths(), (16, 32));
        check(&m);
    }

    #[test]
    fn a_neuro_core_with_every_network_senses_and_its_gradients_are_right() {
        let m = neuro();
        assert_eq!(m.sense_widths().1, 0);
        check(&m);
    }

    #[test]
    fn a_position_sees_only_itself_and_earlier_ones() {
        for m in [bicameral(), neuro()] {
            let (d, r) = m.sense_widths();
            let n = 6;
            let left = noise(21, n * d, 1.0);
            let right = noise(22, n * r, 1.0);
            let s = m.sense(&left, &right, Ablation::default()).unwrap();
            let mut later = left.clone();
            for v in &mut later[4 * d..] {
                *v += 1.0;
            }
            let s2 = m.sense(&later, &right, Ablation::default()).unwrap();
            assert_eq!(s[..4 * d], s2[..4 * d], "an earlier state changed with a later input");
            assert_ne!(s[4 * d..], s2[4 * d..]);
            // The first positions alone give the same states as within the longer sequence.
            let short = m.sense(&left[..3 * d], &right[..3 * r], Ablation::default()).unwrap();
            for (a, b) in short.iter().zip(&s[..3 * d]) {
                assert!((a - b).abs() < 1e-4, "{a} {b}");
            }
        }
    }

    #[test]
    fn one_tick_at_a_time_gives_the_same_states_as_the_whole_sequence() {
        for m in [bicameral(), neuro()] {
            let (d, r) = m.sense_widths();
            let n = m.config.max_tokens;
            let left = noise(31, n * d, 1.0);
            let right = noise(32, n * r, 1.0);
            let all = m.sense(&left, &right, Ablation::default()).unwrap();
            let mut cache = super::super::KvCache::new(&m, Ablation::default());
            for t in 0..n {
                let s =
                    cache.sense_append(&m, &left[t * d..(t + 1) * d], &right[t * r..(t + 1) * r]).unwrap();
                for (a, b) in s.iter().zip(&all[t * d..(t + 1) * d]) {
                    assert!((a - b).abs() < 2e-4, "position {t}: {a} {b}");
                }
            }
            assert_eq!(cache.positions(), n);
            // Full: refused, and the caller starts again.
            assert!(cache.sense_append(&m, &left[..d], &right[..r]).is_err());
            cache.reset(&m);
            let s = cache.sense_append(&m, &left[..d], &right[..r]).unwrap();
            for (a, b) in s.iter().zip(&all[..d]) {
                assert!((a - b).abs() < 2e-4);
            }
        }
    }

    #[test]
    fn wrong_widths_are_refused() {
        let m = bicameral();
        assert!(m.sense(&[0.; 15], &[], Ablation::default()).is_err());
        assert!(m.sense(&[0.; 16], &[0.; 31], Ablation::default()).is_err());
        assert!(
            m.sense(&[0.; 16 * 17], &[0.; 32 * 17], Ablation::default()).is_err(),
            "longer than its context"
        );
        assert!(m.sense(&[f32::NAN; 16], &[0.; 32], Ablation::default()).is_err());
    }
}
