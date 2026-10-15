use super::{Ablation, Bicameral, Config};
#[test]
fn reproduces_claude_numpy_golden_cases() {
    let cases = [
        (
            2,
            1,
            false,
            3.162531302466491f32,
            -0.17660530665063534f32,
            0.0006197456564347281f32,
            0.0005480848717520259f32,
        ),
        (
            2,
            1,
            true,
            3.1609651961372625f32,
            -0.1797787118679941f32,
            0.0016373169459711404f32,
            -0.00012817856126001324f32,
        ),
        (
            1,
            2,
            false,
            3.162443854807051f32,
            -0.17620015781408105f32,
            -0.002106335374108788f32,
            -0.00015319383290789568f32,
        ),
    ];
    for &(loops, split, no_efferent, want_loss, want_first, want_g0, want_last) in &cases {
        let c = Config {
            vocab: 24,
            d: 16,
            heads: 2,
            ff: 24,
            blocks: 3,
            split,
            loops,
            window_chars: 4,
            r1: 12,
            r2: 10,
            max_tokens: 16,
            ..Config::default()
        };
        let mut m = Bicameral::with_config(c, 1);
        assert_eq!(m.parameter_count(), 9046);
        for (i, x) in m.weights.iter_mut().enumerate() {
            *x = (0.2_f64 * (0.37_f64 * i as f64 + 1.).sin()) as f32;
        }
        let ids = [3, 10, 17, 0, 7, 14, 21];
        let targets = [1, 6, 11, 16, 21, 2, 7];
        let mask = [1., 0.5, 1., 0., 1., 0.5, 1.];
        let a = Ablation { no_efferent, ..Ablation::default() };
        let mut feats = Vec::new();
        for t in 0..ids.len() {
            for j in 0..(4 * 16) {
                feats.push((0.11_f64 * ((t * 64 + j) as f64)).sin() as f32);
            }
        }
        let logits = m.logits_with_features(&ids, &feats, a).unwrap();
        let last = &logits[logits.len() - 24..];
        assert!((last[0] - want_first).abs() < 1e-3, "logits parity: {} vs {}", last[0], want_first);
        let (loss, g) = m.loss_and_gradients_with_features(&ids, &feats, &targets, &mask, a).unwrap();
        assert!((loss - want_loss).abs() < 1e-3, "loss parity: {loss} vs {want_loss}");
        assert!((g[0] - want_g0).abs() < 2e-3, "gradient[0] parity: {} vs {}", g[0], want_g0);
        assert!((g[9045] - want_last).abs() < 2e-3, "gradient[9045] parity: {} vs {}", g[9045], want_last);
    }
}
