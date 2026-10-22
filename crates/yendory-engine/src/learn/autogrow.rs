//! When automatic growth grows a network (docs/GROWTH.md, Stage 5): from the held-out checks of
//! a reading run, never from a token count. Growth helps a network that is short of capacity:
//! its held-out loss has stopped improving *and* its training loss is about as high (it is not
//! memorising its training text). When the training loss is well below the held-out loss, more
//! capacity would mostly memorise more, and the network is left as it is.
use serde::{Deserialize, Serialize};

/// The limits and thresholds of automatic growth (kept per model).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AutoGrowthRules {
    /// Never grow past this many parameters (`None`: only the memory limits apply).
    pub max_parameters: Option<usize>,
    /// Training steps between two growths, so the new capacity is trained before judging again.
    pub min_steps_between: u64,
    /// Held-out checks that make a plateau: the best held-out loss improved by less than
    /// `min_gain` (relative) over the last this many checks.
    pub plateau_checks: usize,
    pub min_gain: f32,
    /// At most this relative gap between held-out and training loss (above it the network is
    /// memorising rather than short of capacity).
    pub max_gap: f32,
}

impl Default for AutoGrowthRules {
    fn default() -> Self {
        AutoGrowthRules {
            max_parameters: None,
            min_steps_between: 3000,
            plateau_checks: 4,
            min_gain: 0.005,
            max_gap: 0.1,
        }
    }
}

/// One held-out check of a training run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Check {
    /// Mean training loss since the previous check.
    pub train_loss: f32,
    pub held_out_loss: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "verdict", content = "reason")]
pub enum Verdict {
    /// Grow now, and why.
    Grow(String),
    /// Not now, and why.
    Wait(String),
}

/// Judges a finished run: its held-out checks in order, whether it stopped early because the
/// held-out loss stopped improving, and the training steps since the last growth.
pub fn judge(
    checks: &[Check],
    early_stop: bool,
    steps_since_growth: u64,
    rules: &AutoGrowthRules,
) -> Verdict {
    if steps_since_growth < rules.min_steps_between {
        return Verdict::Wait(format!(
            "{steps_since_growth} training steps since the last growth; at least {} are needed to train the new capacity first",
            rules.min_steps_between
        ));
    }
    let k = rules.plateau_checks.max(1);
    let valid: Vec<Check> =
        checks.iter().copied().filter(|c| c.train_loss.is_finite() && c.held_out_loss.is_finite()).collect();
    if valid.len() <= k && !(early_stop && valid.len() >= 2) {
        return Verdict::Wait(format!(
            "{} held-out check(s) in this run; more than {k} are needed to see whether the held-out loss has stopped improving",
            valid.len()
        ));
    }
    let k = k.min(valid.len() - 1);
    let (before, recent) = valid.split_at(valid.len() - k);
    let best = |c: &[Check]| c.iter().map(|c| c.held_out_loss).fold(f32::INFINITY, f32::min);
    let (best_before, best_now) = (best(before), best(&valid));
    let gain = (best_before - best_now) / best_before.max(1e-6);
    if gain >= rules.min_gain && !early_stop {
        return Verdict::Wait(format!(
            "the held-out loss is still improving ({best_before:.4} -> {best_now:.4}, {:.1}% over the last {k} checks)",
            100. * gain
        ));
    }
    let n = recent.len() as f32;
    let train = recent.iter().map(|c| c.train_loss).sum::<f32>() / n;
    let held = recent.iter().map(|c| c.held_out_loss).sum::<f32>() / n;
    let gap = (held - train) / held.max(1e-6);
    if gap > rules.max_gap {
        return Verdict::Wait(format!(
            "the held-out loss has levelled off, but the training loss ({train:.3}) is {:.0}% below it ({held:.3}): the network is memorising its text, so more varied text would help more than more capacity",
            100. * gap
        ));
    }
    Verdict::Grow(format!(
        "the held-out loss has levelled off ({:.1}% better over the last {k} checks{}) while the training loss ({train:.3}) is as high as the held-out loss ({held:.3}): the network looks short of capacity",
        100. * gain.max(0.),
        if early_stop { "; reading stopped early" } else { "" }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(held: &[f32], gap: f32) -> Vec<Check> {
        held.iter().map(|&h| Check { train_loss: h * (1. - gap), held_out_loss: h }).collect()
    }

    #[test]
    fn grows_only_on_a_plateau_without_memorising() {
        let r = AutoGrowthRules::default();
        let improving = run(&[3.0, 2.6, 2.3, 2.1, 1.95, 1.85], 0.02);
        assert!(
            matches!(judge(&improving, false, 10_000, &r), Verdict::Wait(w) if w.contains("still improving"))
        );
        let flat = run(&[3.0, 2.2, 2.0, 1.999, 1.998, 1.998, 1.997], 0.02);
        assert!(matches!(judge(&flat, false, 10_000, &r), Verdict::Grow(_)));
        let memorising = run(&[3.0, 2.2, 2.0, 1.999, 1.998, 1.998, 1.997], 0.3);
        assert!(
            matches!(judge(&memorising, false, 10_000, &r), Verdict::Wait(w) if w.contains("memorising"))
        );
        assert!(
            matches!(judge(&flat, false, 100, &r), Verdict::Wait(w) if w.contains("since the last growth"))
        );
        assert!(matches!(judge(&flat[..3], false, 10_000, &r), Verdict::Wait(w) if w.contains("check(s)")));
        // A run that stopped early has already seen the plateau.
        assert!(matches!(judge(&run(&[2.0, 1.99, 1.995], 0.01), true, 10_000, &r), Verdict::Grow(_)));
        // Broken numbers are never evidence.
        let nan = vec![Check { train_loss: f32::NAN, held_out_loss: 2. }; 8];
        assert!(matches!(judge(&nan, true, 10_000, &r), Verdict::Wait(_)));
    }
}
