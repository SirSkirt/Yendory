//! Building a Nova Neuro brain by choosing its networks and their sizes: the simple way to
//! design a modular model (the Hub's create screen). The person picks the networks they want and
//! how many weights each should have; the Engine finds each network's internal widths that come
//! closest, and reports the exact count of every network. Studio's brain builder sets every
//! width by hand; both make an ordinary [`NeuroSpec`] and [`Config`].
use super::neuro::{Associative, Consolidation, Experts, Relational, WorkingMemory, VERSION};
use super::{spec, Config, NeuroSpec};
use serde::{Deserialize, Serialize};

/// The networks a Neuro brain's weights belong to, in the order they are shown.
pub const NETWORKS: [&str; 9] = [
    "words",
    "transformer",
    "bus",
    "controller",
    "memory",
    "associative",
    "relational",
    "experts",
    "consolidation",
];

/// Which network a weight slot belongs to.
pub fn network_of(slot: &str) -> &'static str {
    if slot.starts_with('b') && slot[1..].chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return "transformer";
    }
    match slot {
        "embed" | "out_bias" | "ln_f" => return "words",
        "n.read" | "n.read_norm" | "n.bus_norm" | "n.gate_w" | "n.gate_b" | "n.write" => return "bus",
        _ => {}
    }
    for (prefix, name) in [
        ("n.ctl", "controller"),
        ("n.wm", "memory"),
        ("n.as", "associative"),
        ("n.rel", "relational"),
        ("n.ex", "experts"),
        ("n.co", "consolidation"),
    ] {
        if slot.starts_with(prefix) {
            return name;
        }
    }
    "other"
}

/// The weights of each network in a design (only those it has), in [`NETWORKS`] order.
pub fn breakdown(c: &Config, n: &NeuroSpec) -> Vec<(&'static str, usize)> {
    let mut out: Vec<(&'static str, usize)> = NETWORKS.iter().map(|&k| (k, 0)).collect();
    for s in spec(c, Some(n)) {
        let k = network_of(&s.name);
        match out.iter_mut().find(|x| x.0 == k) {
            Some(x) => x.1 += s.range.len(),
            None => out.push((k, s.range.len())),
        }
    }
    out.retain(|x| x.1 > 0);
    out
}

fn weights_of(c: &Config, n: &NeuroSpec, network: &str) -> usize {
    breakdown(c, n).into_iter().find(|x| x.0 == network).map(|x| x.1).unwrap_or(0)
}

/// A brain chosen network by network: the width they share, and for each network wanted, about
/// how many weights it should have (`None`: left out).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrainChoice {
    /// The width every network reads and writes: the words' vectors, the transformer and the
    /// communication bus (64 to 512, a multiple of 32).
    pub width: usize,
    pub transformer: Option<usize>,
    pub memory: Option<usize>,
    pub associative: Option<usize>,
    pub relational: Option<usize>,
    pub experts: Option<usize>,
    pub consolidation: Option<usize>,
    /// The Executive Controller, which weighs the other networks (needed when any of memory,
    /// associative, relational, experts or consolidation is chosen; a small default otherwise).
    pub controller: Option<usize>,
}

/// The smallest `x` in `lo..=hi` whose count reaches `target` (or the closer of it and the one
/// before), for a count that grows with `x`.
fn fit(lo: usize, hi: usize, target: usize, count: impl Fn(usize) -> usize) -> usize {
    let (mut a, mut b) = (lo, hi);
    if count(b) <= target {
        return b;
    }
    while a < b {
        let m = (a + b) / 2;
        if count(m) < target {
            a = m + 1;
        } else {
            b = m;
        }
    }
    if a > lo && target.abs_diff(count(a - 1)) < target.abs_diff(count(a)) {
        a - 1
    } else {
        a
    }
}

/// The design a choice describes, each network sized as close to its target as its widths
/// allow. `vocab`: the word slots' count (the Network sets it).
pub fn design_for(choice: &BrainChoice, vocab: usize) -> Result<(Config, NeuroSpec), String> {
    let d = choice.width;
    if !(64..=512).contains(&d) || d % 32 != 0 {
        return Err("The brain's width must be 64 to 512, in steps of 32".into());
    }
    let wanted = [
        choice.transformer,
        choice.memory,
        choice.associative,
        choice.relational,
        choice.experts,
        choice.consolidation,
    ];
    if wanted.iter().all(Option::is_none) {
        return Err("Choose at least one network".into());
    }
    if wanted.iter().flatten().chain(choice.controller.iter()).any(|&t| t == 0 || t > 400_000_000) {
        return Err("Each network's size must be 1 to 400 million weights".into());
    }
    let mut c = Config {
        vocab,
        d,
        heads: d / 32,
        ff: d * 9 / 4 / 16 * 16,
        blocks: 0,
        split: 0,
        loops: 2,
        window_chars: 1,
        r1: 1,
        r2: 1,
        ..Config::default()
    };
    let mut n = NeuroSpec {
        version: VERSION,
        bus: d,
        memory: choice.memory.map(|_| WorkingMemory { layers: 2, width: 64 }),
        associative: choice.associative.map(|_| Associative { patterns: 64, key: d.min(64) }),
        controller: 64,
        relational: choice.relational.map(|_| Relational { slots: 16, width: 32, hops: 2 }),
        experts: choice.experts.map(|_| Experts { count: 4, width: 64 }),
        consolidation: choice
            .consolidation
            .map(|_| Consolidation { code: (d / 4).clamp(16, 128), hidden: 64 }),
    };

    // The transformer: blocks of width d; the feed-forward width chosen so the blocks come
    // closest to the target, keeping it between the width and four times it when possible.
    if let Some(t) = choice.transformer {
        let block = |ff: usize| 4 * d * d + 3 * d * ff + 2 * d;
        let mut best: Option<(usize, usize, f64)> = None;
        for blocks in 1..=48usize {
            let per = t as f64 / blocks as f64;
            let ff =
                ((per - (4 * d * d + 2 * d) as f64) / (3 * d) as f64 / 16.0).round().max(1.0) as usize * 16;
            let ff = ff.clamp(d, 4 * d);
            let miss = (blocks * block(ff)).abs_diff(t) as f64 / t as f64;
            // Prefer a feed-forward width near 2.5 times the width when the sizes are close.
            let shape = ((ff as f64 / d as f64) - 2.5).abs() * 0.02;
            if best.map_or(true, |b| miss + shape < b.2) {
                best = Some((blocks, ff, miss + shape));
            }
        }
        let (blocks, ff, _) = best.unwrap_or((1, d, 0.0));
        // Smaller than one block allows: one block with a narrower feed-forward network.
        let ff = if blocks == 1 && block(d) > t { fit(1, d / 16, t, |f| block(f * 16)) * 16 } else { ff };
        c.blocks = blocks;
        c.ff = ff;
        c.split = blocks / 2;
    }

    // Each module by its own width (the others held), then the controller (it has one gate per
    // module). Consolidation before associative memory (it may read the consolidation code).
    if let Some(t) = choice.memory {
        let w = fit(1, 4096, t, |w| {
            weights_of(
                &c,
                &NeuroSpec { memory: Some(WorkingMemory { layers: 2, width: w }), ..n.clone() },
                "memory",
            )
        });
        n.memory = Some(WorkingMemory { layers: 2, width: w });
    }
    if let Some(t) = choice.relational {
        let count = |slots: usize, w: usize| {
            weights_of(
                &c,
                &NeuroSpec { relational: Some(Relational { slots, width: w, hops: 2 }), ..n.clone() },
                "relational",
            )
        };
        let w = fit(1, 512, t, |w| count(16, w));
        let slots = if w == 512 { fit(16, 64, t, |s| count(s, 512)) } else { 16 };
        n.relational = Some(Relational { slots, width: w, hops: 2 });
    }
    if let Some(t) = choice.experts {
        let count = |k: usize, w: usize| {
            weights_of(
                &c,
                &NeuroSpec { experts: Some(Experts { count: k, width: w }), ..n.clone() },
                "experts",
            )
        };
        let w = fit(1, 4096, t, |w| count(4, w));
        let k = if w == 4096 { fit(4, 16, t, |k| count(k, 4096)) } else { 4 };
        n.experts = Some(Experts { count: k, width: w });
    }
    if let Some(t) = choice.consolidation {
        let code = (d / 4).clamp(16, 128);
        let h = fit(1, 4096, t, |h| {
            weights_of(
                &c,
                &NeuroSpec { consolidation: Some(Consolidation { code, hidden: h }), ..n.clone() },
                "consolidation",
            )
        });
        n.consolidation = Some(Consolidation { code, hidden: h });
    }
    if let Some(t) = choice.associative {
        let key = d.min(64);
        let p = fit(1, 16384, t, |p| {
            weights_of(
                &c,
                &NeuroSpec { associative: Some(Associative { patterns: p, key }), ..n.clone() },
                "associative",
            )
        });
        n.associative = Some(Associative { patterns: p, key });
    }
    if !n.modules().is_empty() {
        let t = choice.controller.unwrap_or(4 * d * n.modules().len() + 64 * d);
        let h = fit(1, 4096, t, |h| weights_of(&c, &NeuroSpec { controller: h, ..n.clone() }, "controller"));
        n.controller = h;
    }
    n.validate(&c)?;
    Ok((c, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(c: &Config, n: &NeuroSpec) -> usize {
        breakdown(c, n).iter().map(|x| x.1).sum()
    }

    #[test]
    fn every_weight_belongs_to_a_known_network() {
        for p in super::super::neuro::presets() {
            let parts = breakdown(&p.config, &p.design);
            assert!(parts.iter().all(|x| x.0 != "other"), "{}: {parts:?}", p.id);
            let n = spec(&p.config, Some(&p.design)).last().unwrap().range.end;
            assert_eq!(total(&p.config, &p.design), n, "{}", p.id);
        }
    }

    #[test]
    fn networks_come_close_to_their_sizes() {
        // The owner's example: a 6M transformer and a 1M executive controller, with working
        // memory; and one network alone; and every network at very different sizes.
        let choices = [
            BrainChoice {
                width: 256,
                transformer: Some(6_000_000),
                memory: Some(500_000),
                controller: Some(1_000_000),
                ..BrainChoice::default()
            },
            BrainChoice { width: 128, memory: Some(2_000_000), ..BrainChoice::default() },
            BrainChoice {
                width: 192,
                transformer: Some(3_000_000),
                memory: Some(1_000_000),
                associative: Some(250_000),
                relational: Some(400_000),
                experts: Some(2_000_000),
                consolidation: Some(300_000),
                controller: Some(50_000),
            },
        ];
        for ch in choices {
            let (c, n) = design_for(&ch, 384 + 128).unwrap();
            let parts = breakdown(&c, &n);
            let got = |k: &str| parts.iter().find(|x| x.0 == k).map(|x| x.1).unwrap_or(0);
            for (k, want) in [
                ("transformer", ch.transformer),
                ("memory", ch.memory),
                ("associative", ch.associative),
                ("relational", ch.relational),
                ("experts", ch.experts),
                ("consolidation", ch.consolidation),
                ("controller", ch.controller),
            ] {
                if let Some(w) = want {
                    let g = got(k);
                    assert!(g.abs_diff(w) as f64 / (w as f64) < 0.08, "{k}: {g} for {w} in {ch:?}");
                }
            }
            // A network not chosen has no weights.
            if ch.transformer.is_none() {
                assert_eq!(c.blocks, 0);
                assert_eq!(got("transformer"), 0);
            }
            // The brain builds and runs.
            let m = super::super::Bicameral::with_design(c.clone(), Some(n.clone()), 1);
            assert_eq!(m.parameter_count(), total(&c, &n));
        }
    }

    #[test]
    fn impossible_choices_are_refused() {
        assert!(design_for(&BrainChoice { width: 256, ..BrainChoice::default() }, 384).is_err());
        assert!(design_for(&BrainChoice { width: 100, memory: Some(1000), ..BrainChoice::default() }, 384)
            .is_err());
        assert!(
            design_for(&BrainChoice { width: 128, memory: Some(0), ..BrainChoice::default() }, 384).is_err()
        );
    }
}
