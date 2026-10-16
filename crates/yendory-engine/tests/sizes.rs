//! Size presets keep Nova-2's division of parameters and work end to end.
use yendory_engine::model::MAX_PARAMETERS;
use yendory_engine::{size_presets, Ablation, Bicameral, Config, Focus};

#[test]
fn presets_keep_the_nova2_division() {
    let share = |m: &Bicameral| {
        let t = m.parameter_count() as f64;
        [Focus::Left, Focus::Right, Focus::Interconnect].map(|f| m.group_parameters(f) as f64 / t)
    };
    let nova = share(&Bicameral::new());
    let mut last = 0;
    for p in size_presets() {
        assert_eq!(
            Bicameral::parameters_for(&p.config),
            Bicameral::with_config(p.config.clone(), 1).parameter_count()
        );
        let m = Bicameral::with_config(p.config.clone(), 1);
        assert!(m.parameter_count() > last && m.parameter_count() <= MAX_PARAMETERS, "{}", p.id);
        last = m.parameter_count();
        for (a, b) in share(&m).iter().zip(nova) {
            assert!((a - b).abs() < 0.006, "{}: {a:.4} vs {b:.4}", p.id);
        }
        // Same tokenizer, context, thinking rounds and character window as Nova-2.
        let d = Config::default();
        assert_eq!(
            (p.config.vocab, p.config.max_tokens, p.config.loops, p.config.window_chars),
            (d.vocab, d.max_tokens, d.loops, d.window_chars)
        );
    }
    assert!(last > 14_000_000);
}

#[test]
fn a_larger_model_trains_generates_and_round_trips() {
    let p = size_presets().into_iter().find(|p| p.id == "4m").unwrap();
    let mut m = Bicameral::with_config(p.config, 3);
    let ids = m.tokenizer.encode("U: hello\nN: Hi!\n").unwrap();
    let before = m
        .loss_and_gradients(&ids[..ids.len() - 1], &ids[1..], &vec![1.; ids.len() - 1], Ablation::default())
        .unwrap()
        .0;
    for _ in 0..5 {
        m.train_text("U: hello\nN: Hi!\n", 3e-3, Focus::Joint).unwrap();
    }
    let after = m
        .loss_and_gradients(&ids[..ids.len() - 1], &ids[1..], &vec![1.; ids.len() - 1], Ablation::default())
        .unwrap()
        .0;
    assert!(after < before, "{before} -> {after}");
    // Cached generation and the full forward pass agree for this shape too.
    let prompt = m.tokenizer.encode("U: hello\nN: ").unwrap();
    let full = m.logits(&prompt, Ablation::default()).unwrap();
    let cached = yendory_engine::KvCache::new(&m, Ablation::default()).prefill(&m, &prompt).unwrap();
    let last = &full[full.len() - m.config.vocab..];
    for (a, b) in last.iter().zip(&cached) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
    let bytes = m.to_bytes().unwrap();
    let back = Bicameral::from_bytes(&bytes).unwrap();
    assert_eq!(back.to_bytes().unwrap(), bytes);
}

#[test]
fn checkpoints_beyond_the_cap_are_refused_before_allocating() {
    let mut bytes = Bicameral::new().to_bytes().unwrap();
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let json = String::from_utf8(bytes[12..12 + len].to_vec()).unwrap();
    let mut cfg: serde_json::Value = serde_json::from_str(&json).unwrap();
    cfg["d"] = 4096.into();
    cfg["ff"] = 16384.into();
    let big = serde_json::to_vec(&cfg).unwrap();
    bytes.splice(8..12 + len, (big.len() as u32).to_le_bytes().into_iter().chain(big));
    let err = Bicameral::from_bytes(&bytes).err().unwrap();
    assert!(err.contains("supports up to"), "{err}");
}
