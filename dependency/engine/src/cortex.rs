//! A game skill playing with its model's own core: the skill is the fast "reflex" (eyes,
//! working memory, the controls, every tick), the core (Nova Bicameral or Nova Neuro, the same
//! weights that hold conversation) is the "cortex". Every tick the core reads the skill's
//! goal-free encoding of the picture (its sight) through the link, as one position of a sequence
//! (`Bicameral::sense`); its state at that tick comes back through the link as the skill's goal
//! input. Nothing in it knows any game.
//!
//! Training ([`sequence_loss`]) is end to end through the goal input: the controls' loss reaches
//! the link and every weight of the core. The sight is held fixed for the core: the skill's eyes
//! learn from the skill's own path only (gradient through the core into the eyes would need a
//! pass of the core for every tick). With the way back at zero (a new link) the skill plays
//! exactly as without a core.
use crate::model::{Ablation, Bicameral};
use crate::skill::{Loss, Policy, Target};

/// Why the skill's link does not fit this core, if it doesn't.
pub fn fits(policy: &Policy, core: &Bicameral) -> Result<(), String> {
    let link = policy.arch.cortex.as_ref().ok_or("This skill has no link to a core")?;
    let (d, r) = core.sense_widths();
    if link.width != d || link.right != r {
        return Err(format!(
            "The skill's link was made for a core of width {} (right input {}); this model's core is {d} ({r})",
            link.width, link.right
        ));
    }
    Ok(())
}

impl Policy {
    fn weights_of(&self, name: &str) -> &[f32] {
        self.slot(name).map(|s| &self.w[s.range()]).unwrap_or(&[])
    }

    /// The core's input for these sights, one per tick: (ticks × the core's width, ticks × its
    /// right input).
    pub fn link_in(&self, sights: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
        let Some(link) = &self.arch.cortex else { return (vec![], vec![]) };
        let h = self.arch.hidden;
        let project = |wn: &str, bn: &str, rows: usize| -> Vec<f32> {
            let (w, b) = (self.weights_of(wn), self.weights_of(bn));
            let mut out = Vec::with_capacity(sights.len() * rows);
            for s in sights {
                for r in 0..rows {
                    out.push(b[r] + s.iter().zip(&w[r * h..(r + 1) * h]).map(|(a, b)| a * b).sum::<f32>());
                }
            }
            out
        };
        let left = project("link.in.w", "link.in.b", link.width);
        let right = if link.right > 0 { project("link.right.w", "link.right.b", link.right) } else { vec![] };
        (left, right)
    }

    /// The goal input for one state of the core.
    pub fn link_out(&self, state: &[f32]) -> Vec<f32> {
        let Some(link) = &self.arch.cortex else { return vec![0.; self.arch.goal] };
        let (w, b) = (self.weights_of("link.out.w"), self.weights_of("link.out.b"));
        (0..self.arch.goal)
            .map(|g| {
                b[g] + state
                    .iter()
                    .zip(&w[g * link.width..(g + 1) * link.width])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            })
            .collect()
    }
}

/// Runs a sequence of ticks with the core, measures the loss against `targets`, and adds the
/// gradients (averaged over the ticks, times `scale`): the skill's to `pgrad`, the core's to
/// `cgrad`.
#[allow(clippy::too_many_arguments)]
pub fn sequence_loss(
    policy: &Policy,
    core: &Bicameral,
    inputs: &[Vec<f32>],
    targets: &[Target],
    pgrad: &mut [f32],
    cgrad: &mut [f32],
    scale: f32,
    abl: Ablation,
) -> Result<Loss, String> {
    fits(policy, core)?;
    let link = policy.arch.cortex.clone().expect("checked");
    let (d, h, goal) = (link.width, policy.arch.hidden, policy.arch.goal);
    let seen: Vec<_> = inputs.iter().map(|x| policy.see(x)).collect();
    let sights: Vec<Vec<f32>> = seen.iter().map(|s| policy.sight(s)).collect();
    let (left, right) = policy.link_in(&sights);
    let slot = |n: &str| policy.slot(n).map(|s| s.range()).expect("a linked skill has its link's weights");
    let (ow, ob) = (slot("link.out.w"), slot("link.out.b"));
    let mut loss = Loss::default();
    let (dleft, dright) = core.sense_train(&left, &right, cgrad, abl, |states| {
        let goals: Vec<Vec<f32>> = states.chunks(d).map(|s| policy.link_out(s)).collect();
        let (l, dgoals) = policy.sequence_loss_seen(seen, Some(&goals), targets, pgrad, scale);
        loss = l;
        // Back through the way out: its weights, and the gradient at each state.
        let mut dstates = vec![0f32; states.len()];
        for (t, (dg, s)) in dgoals.iter().zip(states.chunks(d)).enumerate() {
            for (g, &v) in dg.iter().enumerate().take(goal) {
                if v == 0.0 {
                    continue;
                }
                pgrad[ob.start + g] += v;
                let row = ow.start + g * d;
                for k in 0..d {
                    pgrad[row + k] += v * s[k];
                    dstates[t * d + k] += v * policy.w[row + k];
                }
            }
        }
        Ok(dstates)
    })?;
    // Back through the way in (the sights held fixed).
    let mut back_in = |wn: &str, bn: &str, rows: usize, dx: &[f32]| {
        let (w, b) = (slot(wn), slot(bn));
        for (t, s) in sights.iter().enumerate() {
            for r in 0..rows {
                let v = dx[t * rows + r];
                if v == 0.0 {
                    continue;
                }
                pgrad[b.start + r] += v;
                for k in 0..h {
                    pgrad[w.start + r * h + k] += v * s[k];
                }
            }
        }
    };
    back_in("link.in.w", "link.in.b", d, &dleft);
    if link.right > 0 {
        back_in("link.right.w", "link.right.b", link.right, &dright);
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::brain::{design_for, BrainChoice};
    use crate::model::Config;
    use crate::skill::{check_arch, random_sequence, CortexLink};

    fn jitter(w: &mut [f32], seed: u64, scale: f32) {
        let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        for v in w {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v += ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2. * scale;
        }
    }

    fn bicameral() -> Bicameral {
        let c = Config {
            vocab: 140,
            d: 16,
            heads: 2,
            ff: 24,
            blocks: 2,
            split: 1,
            loops: 1,
            window_chars: 2,
            r1: 8,
            r2: 8,
            max_tokens: 16,
            ..Config::default()
        };
        let mut m = Bicameral::with_config(c, 5);
        jitter(&mut m.weights, 3, 0.1);
        m
    }

    fn neuro() -> Bicameral {
        let choice = BrainChoice {
            width: 64,
            transformer: Some(60_000),
            memory: Some(8_000),
            experts: Some(8_000),
            controller: Some(2_000),
            ..BrainChoice::default()
        };
        let (mut c, n) = design_for(&choice, 140).unwrap();
        c.max_tokens = 16;
        let mut m = Bicameral::with_design(c, Some(n), 7);
        jitter(&mut m.weights, 4, 0.1);
        m
    }

    fn linked(core: &Bicameral, open: bool) -> Policy {
        let (width, right) = core.sense_widths();
        let arch = crate::skill::Arch { cortex: Some(CortexLink { width, right }), ..check_arch() };
        let mut p = Policy::new(arch, 9).unwrap();
        if open {
            let r = p.slot("link.out.w").unwrap().range();
            jitter(&mut p.w[r], 6, 0.3);
        }
        p
    }

    #[test]
    fn a_new_link_plays_exactly_as_the_skill_alone() {
        for core in [bicameral(), neuro()] {
            let p = linked(&core, false);
            let (inputs, targets) = random_sequence(&p.arch, 3, 6);
            let mut g1 = vec![0f32; p.parameters()];
            let alone = p.sequence_loss(&inputs, &targets, &mut g1, 1.0);
            let mut g2 = vec![0f32; p.parameters()];
            let mut cg = vec![0f32; core.weights.len()];
            let with =
                sequence_loss(&p, &core, &inputs, &targets, &mut g2, &mut cg, 1.0, Ablation::default())
                    .unwrap();
            assert!((alone.total - with.total).abs() < 1e-6, "{} {}", alone.total, with.total);
            // Nothing comes back from the core yet, so it learns nothing yet; the way back does.
            assert!(cg.iter().all(|v| *v == 0.0));
            let out = p.slot("link.out.w").unwrap().range();
            assert!(g2[out].iter().any(|v| *v != 0.0), "the way back must learn to open");
        }
    }

    /// Every gradient of the skill, its link and the core against finite differences.
    fn check(core: &Bicameral) {
        let p = linked(core, true);
        let (inputs, targets) = random_sequence(&p.arch, 4, 5);
        let abl = Ablation::default();
        let mut pg = vec![0f32; p.parameters()];
        let mut cg = vec![0f32; core.weights.len()];
        sequence_loss(&p, core, &inputs, &targets, &mut pg, &mut cg, 1.0, abl).unwrap();
        let value = |p: &Policy, c: &Bicameral| -> f64 {
            let mut a = vec![0f32; p.parameters()];
            let mut b = vec![0f32; c.weights.len()];
            sequence_loss(p, c, &inputs, &targets, &mut a, &mut b, 1.0, abl).unwrap().total
        };
        let biggest = pg.iter().chain(&cg).fold(0f32, |a, x| a.max(x.abs()));
        let close = |analytic: f32, numeric: f32, what: &str| {
            assert!(
                (numeric - analytic).abs() <= 3e-3 * biggest.max(1e-3) + 0.05 * analytic.abs(),
                "{what}: analytic {analytic} numeric {numeric} (largest {biggest})"
            );
        };
        let eps = 1e-2f32;
        // The link and the heads (the eyes' gradient through the core is held back on purpose:
        // check those slots only for the skill's own path, as `Policy::sequence_loss` does).
        for s in p.slots.iter().filter(|s| {
            s.name.starts_with("link.")
                || s.name.starts_with("goal")
                || s.name.starts_with("mem")
                || s.name.starts_with("key")
        }) {
            let r = s.range();
            for i in [r.start, r.start + r.len() / 2, r.end - 1] {
                let (mut up, mut down) = (p.clone(), p.clone());
                up.w[i] += eps;
                down.w[i] -= eps;
                let numeric = ((value(&up, core) - value(&down, core)) / (2. * eps as f64)) as f32;
                close(pg[i], numeric, &format!("skill {} [{i}]", s.name));
            }
        }
        let mut checked = 0;
        for s in &core.slots {
            for k in [0, s.range.len() / 2, s.range.len() - 1] {
                let i = s.range.start + k;
                let (mut up, mut down) = (core.clone(), core.clone());
                up.weights[i] += eps;
                down.weights[i] -= eps;
                let numeric = ((value(&p, &up) - value(&p, &down)) / (2. * eps as f64)) as f32;
                close(cg[i], numeric, &format!("core {} [{k}]", s.name));
                checked += 1;
            }
        }
        assert!(checked > 20);
        assert!(cg.iter().any(|v| *v != 0.0), "the core learns through the goal input");
    }

    #[test]
    fn gradients_through_a_bicameral_core_are_right() {
        check(&bicameral());
    }

    #[test]
    fn gradients_through_a_neuro_core_are_right() {
        check(&neuro());
    }

    #[test]
    fn a_link_that_does_not_fit_the_core_is_refused() {
        let p = linked(&bicameral(), false);
        assert!(fits(&p, &neuro()).is_err());
        let plain = Policy::new(check_arch(), 1).unwrap();
        assert!(fits(&plain, &bicameral()).is_err());
    }

    #[test]
    fn a_skill_keeps_its_weights_when_given_a_link_and_back() {
        let spec = crate::skill::SkillSpec {
            game: "g".into(),
            name: "G".into(),
            size: crate::skill::Size::Small,
            seed: 2,
            arch: check_arch(),
            keys: vec!["w".into(), "a".into()],
            buttons: vec!["left".into()],
            lineage: crate::skill::Lineage { generation: 0, parent: None, method: "new".into() },
            steps: 0,
            accepted_step: 0,
            created: 0,
            notes: serde_json::Value::Null,
        };
        let mut k = crate::skill::Skill::new(spec).unwrap();
        k.latest[3] = 7.0;
        let core = bicameral();
        let (width, right) = core.sense_widths();
        let l = k.with_cortex(Some(CortexLink { width, right })).unwrap();
        assert_eq!(l.latest[..k.latest.len()], k.latest[..]);
        assert!(l.latest.len() > k.latest.len());
        let out = l.policy().unwrap();
        let r = out.slot("link.out.w").unwrap().range();
        assert!(l.accepted[r].iter().all(|v| *v == 0.0));
        let back = l.with_cortex(None).unwrap();
        assert_eq!(back.latest, k.latest);
        assert_eq!(back.adam.m.len(), k.adam.m.len());
    }
}
