//! Golden vectors from the float64 numpy reference (tests/compat/golden/*.txt, written by
//! reference/make_golden.py): loss, 12 logits and 62 sampled gradients per file.
use yendory_engine::TokenId;
use yendory_engine::{Ablation, Bicameral, Config};

fn field<'a>(text: &'a str, key: &str) -> &'a str {
    text.lines()
        .find_map(|l| l.strip_prefix(key).and_then(|r| r.strip_prefix(' ')))
        .unwrap_or_else(|| panic!("golden file has no `{key}` line"))
}
fn nums<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    s.split_whitespace().map(|x| x.parse().unwrap()).collect()
}
fn kv(s: &str, key: &str) -> usize {
    s.split_whitespace().find_map(|p| p.strip_prefix(&format!("{key}="))).unwrap().parse().unwrap()
}

fn check(name: &str) {
    let path = format!("{}/../../tests/compat/golden/{name}.txt", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap();
    let c = field(&text, "config");
    let config = Config {
        vocab: kv(c, "vocab"),
        d: kv(c, "d"),
        heads: kv(c, "heads"),
        ff: kv(c, "ff"),
        blocks: kv(c, "blocks"),
        split: kv(c, "split"),
        loops: kv(c, "loops"),
        window_chars: kv(c, "window_chars"),
        r1: kv(c, "r1"),
        r2: kv(c, "r2"),
        max_tokens: kv(c, "max_tokens"),
        ..Config::default()
    };
    let a = field(&text, "ablation");
    let abl = Ablation {
        no_afferent: kv(a, "no_afferent") == 1,
        no_efferent: kv(a, "no_efferent") == 1,
        no_right: kv(a, "no_right") == 1,
    };
    let ids: Vec<TokenId> = nums(field(&text, "ids"));
    let targets: Vec<TokenId> = nums(field(&text, "targets"));
    let mask: Vec<f32> = nums(field(&text, "weights"));
    let want_logits: Vec<f64> = nums(field(&text, "last_logits"));
    let want_loss: f64 = field(&text, "loss").parse().unwrap();

    let mut m = Bicameral::with_config(config.clone(), 1);
    assert_eq!(m.parameter_count(), field(&text, "param_count").parse::<usize>().unwrap());
    for (i, x) in m.weights.iter_mut().enumerate() {
        *x = (0.2_f64 * (0.37_f64 * i as f64 + 1.).sin()) as f32;
    }
    let feat = config.window_chars * 16;
    let feats: Vec<f32> = (0..ids.len())
        .flat_map(|t| (0..feat).map(move |j| (0.11 * (t * feat + j) as f64).sin() as f32))
        .collect();

    let logits = m.logits_with_features(&ids, &feats, abl).unwrap();
    let last = &logits[logits.len() - config.vocab..];
    for (k, (&got, &want)) in last.iter().zip(&want_logits).enumerate() {
        assert!((got as f64 - want).abs() <= 1e-5 + 1e-4 * want.abs(), "{name}: logit {k}: {got} vs {want}");
    }
    let (loss, grads) = m.loss_and_gradients_with_features(&ids, &feats, &targets, &mask, abl).unwrap();
    assert!((loss as f64 - want_loss).abs() <= 1e-4 * want_loss.abs(), "{name}: loss {loss} vs {want_loss}");
    let samples: Vec<(usize, f64)> = text
        .lines()
        .filter_map(|l| l.strip_prefix("grad "))
        .map(|r| {
            let v: Vec<&str> = r.split(' ').collect();
            (v[0].parse().unwrap(), v[1].parse().unwrap())
        })
        .collect();
    assert_eq!(samples.len(), 62);
    let scale = samples.iter().map(|s| s.1.abs()).fold(0., f64::max);
    let mut worst = 0f64;
    for &(i, want) in &samples {
        let err = (grads[i] as f64 - want).abs();
        worst = worst.max(err / scale);
        assert!(err <= 1e-4 * scale + 1e-3 * want.abs(), "{name}: grad[{i}] {} vs {want}", grads[i]);
    }
    eprintln!("{name}: worst gradient error {worst:.2e} of the largest sampled gradient");
}

#[test]
fn golden_full_model() {
    check("small_full");
}
#[test]
fn golden_no_efferent() {
    check("small_no_efferent");
}
#[test]
fn golden_one_loop() {
    check("small_loops1");
}
