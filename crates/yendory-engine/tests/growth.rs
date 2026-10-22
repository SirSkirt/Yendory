//! Growing trained networks (docs/GROWTH.md): every offered growth keeps what the network
//! computes, keeps every trained weight and its optimizer state, gives new units gradient, and
//! the grown model saves, loads and trains on. Existing checkpoints can be grown.
use yendory_engine::learn::{batch_loss_and_gradients, Row};
use yendory_engine::model::grow::PROBES;
use yendory_engine::model::neuro::presets;
use yendory_engine::{compare_outputs, Ablation, Bicameral, Config, Focus, Growth};

fn rows(m: &Bicameral) -> Vec<Row> {
    PROBES
        .iter()
        .map(|t| {
            let ids = m.tokenizer.encode(t).unwrap();
            let n = ids.len() - 1;
            Row { ids: ids[..n].to_vec(), targets: ids[1..].to_vec(), mask: vec![1.; n] }
        })
        .collect()
}

/// A model with trained (not initial) weights and optimizer state.
fn trained(mut m: Bicameral) -> Bicameral {
    let r = rows(&m);
    for _ in 0..6 {
        let (_, g) = batch_loss_and_gradients(&m, &r, Ablation::default()).unwrap();
        m.adam_step(&g, 3e-3, Focus::Joint).unwrap();
    }
    m
}

fn models() -> Vec<(&'static str, Bicameral)> {
    let p = presets().into_iter().next().unwrap();
    vec![
        ("bicameral", trained(Bicameral::with_config(Config::default(), 3))),
        ("neuro", trained(Bicameral::with_design(p.config, Some(p.design), 3))),
    ]
}

#[test]
fn every_offered_growth_preserves_the_network() {
    for (arch, m) in models() {
        let (options, not) = m.growth_options();
        assert!(options.len() >= 5, "{arch}: {} options", options.len());
        for o in &options {
            let g = m.grow(&o.growth, 11).unwrap();
            assert_eq!(g.parameter_count(), o.parameters_after, "{arch} {}", o.id);
            assert!(o.parameters_after > o.parameters_before);
            let c = compare_outputs(&m, &g, &PROBES).unwrap();
            println!(
                "{arch} {:12} {:>9} -> {:>9} parameters: largest logit change {:.2e}, loss {:.6} -> {:.6}",
                o.id,
                o.parameters_before,
                o.parameters_after,
                c.max_logit_change,
                c.loss_before,
                c.loss_after
            );
            assert!(c.preserved(), "{arch} {}: {c:?}", o.id);
            assert_eq!(g.step, m.step);
            assert_eq!(g.tokenizer.words(), m.tokenizer.words());
            assert_eq!(g.neuro.is_some(), m.neuro.is_some(), "the architecture is kept");
        }
        // Everything at once, then again on the grown model (growth composes).
        let all = Growth {
            heads: 2,
            ff: 64,
            lower_blocks: 1,
            upper_blocks: 1,
            ..options.iter().find(|o| o.id == "balanced").unwrap().growth.clone()
        };
        let g1 = m.grow(&all, 5).unwrap();
        let g2 = g1.grow(&Growth { ff: 32, upper_blocks: 1, ..Growth::default() }, 6).unwrap();
        let c = compare_outputs(&m, &g2, &PROBES).unwrap();
        println!("{arch} twice: {} -> {} parameters, {c:?}", m.parameter_count(), g2.parameter_count());
        assert!(c.preserved(), "{arch}: {c:?}");
        if arch == "neuro" {
            assert!(not.iter().any(|n| n.part.contains("patterns")));
            assert!(not.iter().any(|n| n.part.contains("layers")));
        }
    }
}

#[test]
fn optimizer_state_kept_new_units_learn_and_checkpoints_round_trip() {
    for (arch, m) in models() {
        let g = m.grow(&Growth { ff: 32, heads: 1, upper_blocks: 1, ..Growth::default() }, 9).unwrap();
        // Existing weights and moments kept in place (block 0's first query row).
        let n0 = m.config.d;
        let (a, b) = (m.slot_range("b0.wq"), g.slot_range("b0.wq"));
        assert_eq!(&g.weights[b.start..b.start + n0], &m.weights[a.start..a.start + n0], "{arch}");
        assert_eq!(&g.momentum[b.start..b.start + n0], &m.momentum[a.start..a.start + n0]);
        assert_eq!(&g.variance[b.start..b.start + n0], &m.variance[a.start..a.start + n0]);
        // New feed-forward units: their outgoing weights start at zero and get gradient at once.
        let down = g.slot_range("b0.w_down");
        let (d, ff_old, ff_new) = (g.config.d, m.config.ff, g.config.ff);
        let new_cols: Vec<usize> =
            (0..d).flat_map(|r| (ff_old..ff_new).map(move |c| r * ff_new + c)).collect();
        assert!(new_cols.iter().all(|&i| g.weights[down.start + i] == 0.));
        let (_, grads) = batch_loss_and_gradients(&g, &rows(&g), Ablation::default()).unwrap();
        let reach = new_cols.iter().filter(|&&i| grads[down.start + i] != 0.).count();
        assert!(reach > new_cols.len() / 2, "{arch}: {reach} of {} new weights get gradient", new_cols.len());
        // The grown model saves and loads with its new dimensions and trains on.
        let back = Bicameral::from_bytes(&g.to_bytes().unwrap()).unwrap();
        assert_eq!(back.config, g.config);
        assert_eq!(back.neuro, g.neuro);
        assert_eq!(back.weights, g.weights);
        assert_eq!(back.momentum, g.momentum);
        let mut t = back.clone();
        let r = rows(&t);
        let first = batch_loss_and_gradients(&t, &r, Ablation::default()).unwrap().0;
        for _ in 0..8 {
            let (_, gr) = batch_loss_and_gradients(&t, &r, Ablation::default()).unwrap();
            t.adam_step(&gr, 2e-3, Focus::Joint).unwrap();
        }
        let last = batch_loss_and_gradients(&t, &r, Ablation::default()).unwrap().0;
        println!("{arch}: training after growth {first:.4} -> {last:.4}");
        assert!(last < first);
        // The trained, grown model scores text and generates from its KV cache.
        let ids = t.tokenizer.encode("U: hello\nN: ").unwrap();
        let full =
            t.loss(&ids[..ids.len() - 1], &ids[1..], &vec![1.; ids.len() - 1], Ablation::default()).unwrap();
        let mut cache = yendory_engine::KvCache::new(&t, Ablation::default());
        let logits = cache.prefill(&t, &ids).unwrap();
        assert!(full.is_finite() && logits.iter().all(|x| x.is_finite()));
        assert!(!t.generate_cached_ids(&ids, 8, Ablation::default()).unwrap().is_empty());
    }
}

#[test]
fn growth_refuses_what_it_cannot_preserve() {
    let bica = Bicameral::with_config(Config::default(), 1);
    let p = presets().into_iter().next().unwrap();
    let neuro = Bicameral::with_design(p.config, Some(p.design), 1);
    assert!(bica.grow(&Growth::default(), 1).is_err(), "nothing to grow");
    assert!(bica.grow(&Growth { bus: 8, ..Growth::default() }, 1).is_err());
    assert!(bica.grow(&Growth { memory_width: 8, ..Growth::default() }, 1).is_err());
    assert!(neuro.grow(&Growth { dense1: 8, ..Growth::default() }, 1).is_err());
}

/// The real step-15,000 checkpoint (an existing, trained model) grows without losing anything.
#[test]
fn an_existing_checkpoint_grows() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/compat/fixtures/nova2-step15000.nnova2");
    let m = Bicameral::load(path).unwrap();
    let (options, _) = m.growth_options();
    let balanced = options.iter().find(|o| o.id == "balanced").unwrap();
    let g = m.grow(&balanced.growth, 1).unwrap();
    let c = compare_outputs(&m, &g, &PROBES).unwrap();
    println!("step 15000: {} -> {} parameters, {c:?}", m.parameter_count(), g.parameter_count());
    assert!(c.preserved(), "{c:?}");
    let ids = m.tokenizer.encode("U: What is your name?\nN: ").unwrap();
    assert_eq!(
        g.generate_cached_ids(&ids, 12, Ablation::default()).unwrap(),
        m.generate_cached_ids(&ids, 12, Ablation::default()).unwrap(),
        "the same replies"
    );
}
