//! Rust Engine vs the numpy reference on REAL trained checkpoints (tests/compat/README.md).
//! Expected values come from reference/make_compat.py (float64 evaluation of the stored weights).
use serde_json::Value;
use std::path::PathBuf;
use yendory_engine::TokenId;
use yendory_engine::{Ablation, Bicameral, Focus, KvCache};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/compat")
}

fn ablation(mode: &str) -> Ablation {
    match mode {
        "full" => Ablation::default(),
        "no_afferent" => Ablation { no_afferent: true, ..Ablation::default() },
        "no_efferent" => Ablation { no_efferent: true, ..Ablation::default() },
        "no_right" => Ablation { no_right: true, ..Ablation::default() },
        other => panic!("unknown mode {other}"),
    }
}

fn ids(v: &Value) -> Vec<TokenId> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as TokenId).collect()
}

/// Largest |got - want| relative to max(1, |want|) across a logit vector.
fn logit_error(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).map(|(&g, &w)| (g as f64 - w).abs() / w.abs().max(1.)).fold(0., f64::max)
}

/// Last-position logits without the KV cache, windowed like the reference's
/// `next_token_probs`: features over the whole history, then the last `max_tokens` positions.
fn window_logits(m: &Bicameral, seq: &[TokenId], abl: Ablation) -> Vec<f32> {
    let start = seq.len().saturating_sub(m.config.max_tokens);
    let row = m.config.window_chars * 16;
    let feats = m.features(seq).unwrap();
    let z = m.logits_with_features(&seq[start..], &feats[start * row..], abl).unwrap();
    z[z.len() - m.config.vocab..].to_vec()
}

/// Greedy decoding without the KV cache: the full window is recomputed for every token,
/// exactly like the reference's `generate`.
fn greedy_uncached(m: &Bicameral, prompt: &[TokenId], max_new: usize, abl: Ablation) -> Vec<TokenId> {
    let valid = (128 + m.tokenizer.words().len()).min(m.config.vocab);
    let mut seq = prompt.to_vec();
    let mut out = Vec::new();
    for _ in 0..max_new {
        let z = window_logits(m, &seq, abl);
        let last = &z[..valid];
        let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as TokenId;
        out.push(next);
        seq.push(next);
        if next == b'\n' as TokenId {
            break;
        }
    }
    out
}

const LOGIT_TOLERANCE: f64 = 1e-4;

fn check(ckpt: PathBuf, expected: PathBuf) {
    let bytes = std::fs::read(&ckpt).unwrap();
    let want: Value = serde_json::from_slice(&std::fs::read(&expected).unwrap()).unwrap();
    let m = Bicameral::from_bytes(&bytes).unwrap();
    assert_eq!(m.step, want["step"].as_u64().unwrap());
    assert_eq!(m.parameter_count() as u64, want["params"].as_u64().unwrap());
    assert_eq!(m.tokenizer.words().len() as u64, want["words"].as_u64().unwrap());

    // 1.4: load -> save reproduces the file byte for byte.
    assert!(m.to_bytes().unwrap() == bytes, "checkpoint round trip is not byte-identical");

    let max_new = want["max_new_tokens"].as_u64().unwrap() as usize;
    let (mut worst_uncached, mut worst_cached) = (0f64, 0f64);
    let mut longest = 0;
    for case in want["cases"].as_array().unwrap() {
        let prompt = case["prompt"].as_str().unwrap();
        let mode = case["mode"].as_str().unwrap();
        let abl = ablation(mode);
        let prompt_ids = m.tokenizer.encode(prompt).unwrap();
        longest = longest.max(prompt_ids.len());
        assert_eq!(prompt_ids, ids(&case["ids"]), "tokenizer differs on {prompt:?}");
        let expect: Vec<f64> =
            case["last_logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();

        let err = logit_error(&window_logits(&m, &prompt_ids, abl), &expect);
        assert!(err < LOGIT_TOLERANCE, "{mode} {prompt:?}: uncached logits off by {err:.2e}");
        worst_uncached = worst_uncached.max(err);

        let mut cache = KvCache::new(&m, abl);
        let cached = cache.prefill(&m, &prompt_ids).unwrap();
        let err = logit_error(&cached, &expect);
        assert!(err < LOGIT_TOLERANCE, "{mode} {prompt:?}: KV-cached logits off by {err:.2e}");
        worst_cached = worst_cached.max(err);

        let expect_ids = ids(&case["greedy_ids"]);
        let text = case["greedy_text"].as_str().unwrap();
        // The uncached path re-runs the whole window per token; on long prompts it is checked in
        // the full mode only (the KV-cached path, which apps use, is checked in every mode).
        if prompt_ids.len() < m.config.max_tokens / 2 || mode == "full" {
            assert_eq!(
                greedy_uncached(&m, &prompt_ids, max_new, abl),
                expect_ids,
                "{mode} {prompt:?}: uncached greedy differs from reference {text:?}"
            );
        }
        assert_eq!(
            m.generate_cached_ids(&prompt_ids, max_new, abl).unwrap(),
            expect_ids,
            "{mode} {prompt:?}: KV-cached greedy differs from reference {text:?}"
        );
    }

    let train = &want["train"];
    let (inp, tgt) = (ids(&train["ids"]), ids(&train["targets"]));
    let (loss, grads) = m.loss_and_gradients(&inp, &tgt, &vec![1.; inp.len()], Ablation::default()).unwrap();
    let want_loss = train["loss"].as_f64().unwrap();
    assert!((loss as f64 - want_loss).abs() <= 1e-4 * want_loss.abs().max(1.), "loss {loss} vs {want_loss}");
    let norm = train["grad_norm"].as_f64().unwrap();
    let got_norm = grads.iter().map(|&g| (g as f64) * (g as f64)).sum::<f64>().sqrt();
    assert!((got_norm - norm).abs() <= 1e-3 * norm, "gradient norm {got_norm} vs {norm}");
    let mut worst_grad = 0f64;
    for pair in train["grads"].as_array().unwrap() {
        let (i, w) = (pair[0].as_u64().unwrap() as usize, pair[1].as_f64().unwrap());
        let err = (grads[i] as f64 - w).abs() / norm;
        assert!(err < 1e-4, "grad[{i}] {} vs {w}", grads[i]);
        worst_grad = worst_grad.max(err);
    }
    // One AdamW step from the stored optimizer state, compared as parameter deltas.
    let mut worst_adam = 0f64;
    for (focus_name, focus) in [("joint", Focus::Joint), ("right", Focus::Right)] {
        let step = &want["adam"][focus_name];
        let lr = step["lr"].as_f64().unwrap();
        let mut c = m.clone();
        c.adam_step(&grads, lr as f32, focus).unwrap();
        assert_eq!(c.step, step["step_after"].as_u64().unwrap());
        let unchanged = c.weights.iter().zip(&m.weights).filter(|(a, b)| a == b).count();
        // Outside the focus nothing may move. Inside it, an update smaller than half a float32
        // ulp of its weight rounds away (the float64 reference moves it), so allow a few.
        let expected = step["unchanged_params"].as_u64().unwrap() as usize;
        assert!(
            unchanged >= expected && unchanged - expected <= m.parameter_count() / 10_000,
            "{focus_name}: {unchanged} unchanged parameters, reference {expected}"
        );
        for pair in step["delta"].as_array().unwrap() {
            let (i, w) = (pair[0].as_u64().unwrap() as usize, pair[1].as_f64().unwrap());
            let got = c.weights[i] as f64 - m.weights[i] as f64;
            let err = (got - w).abs() / lr;
            assert!(err < 1e-3, "{focus_name}: delta[{i}] {got} vs {w}");
            worst_adam = worst_adam.max(err);
        }
    }
    eprintln!("AdamW step: worst parameter-delta error {worst_adam:.1e} of the learning rate");

    assert!(longest > m.config.max_tokens, "no case exercises the full context window");
    eprintln!(
        "{}: {} cases; worst logit error uncached {worst_uncached:.1e}, cached {worst_cached:.1e}; \
               loss {loss}; worst sampled gradient error {worst_grad:.1e} of the gradient norm",
        ckpt.file_name().unwrap().to_string_lossy(),
        want["cases"].as_array().unwrap().len()
    );
}

#[test]
fn step15000_matches_reference() {
    let dir = root().join("fixtures");
    check(dir.join("nova2-step15000.nnova2"), dir.join("step15000.expected.json"));
}

/// The step-19,000 checkpoint is not published (owner's choice). It runs when the files are in
/// tests/compat/private/; set YENDORY_REQUIRE_PRIVATE=1 to make their absence a failure.
#[test]
fn step19000_matches_reference_when_present() {
    let dir = root().join("private");
    let (ckpt, exp) = (dir.join("nova2-lessons-step19000.nnova2"), dir.join("step19000.expected.json"));
    if !ckpt.exists() || !exp.exists() {
        assert!(std::env::var_os("YENDORY_REQUIRE_PRIVATE").is_none(), "private step-19000 fixtures missing");
        eprintln!("step-19000 fixtures not present; not checked (see tests/compat/README.md)");
        return;
    }
    check(ckpt, exp);
}
