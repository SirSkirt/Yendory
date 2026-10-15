//! The Engine's learning pipeline vs reference/lessons.py + train_lessons.py
//! (expected values from reference/make_learn.py).
use serde_json::Value;
use std::path::{Path, PathBuf};
use yendory_engine::learn::eval::{evaluate, Report};
use yendory_engine::learn::{
    batch_loss_and_gradients, encode_item, grow_vocabulary, parse_lesson, Item, Rng,
};
use yendory_engine::{Ablation, Bicameral};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/compat")
}

fn parse_dir(dir: &Path) -> Vec<Item> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    let mut items = Vec::new();
    for f in files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        items.extend(parse_lesson(&std::fs::read_to_string(&f).unwrap(), &name).unwrap());
    }
    items
}

fn opt(v: &Value) -> Option<String> {
    v.as_str().map(String::from)
}
fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
}
fn u16s(v: &Value) -> Vec<u16> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u16).collect()
}

fn check_items(items: &[Item], want: &Value) {
    let want = want.as_array().unwrap();
    assert_eq!(items.len(), want.len(), "item count");
    for (i, (it, w)) in items.iter().zip(want).enumerate() {
        assert_eq!(it.kind.name(), w["kind"].as_str().unwrap(), "item {i} kind");
        assert_eq!(it.source, w["source"].as_str().unwrap(), "item {i} source");
        assert_eq!(it.held_out, w["held_out"].as_bool().unwrap(), "item {i} held-out split");
        let segs: Vec<(String, f32)> = w["segments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| (s[0].as_str().unwrap().to_string(), s[1].as_f64().unwrap() as f32))
            .collect();
        let got: Vec<(String, f32)> = it.segments.iter().map(|s| (s.text.clone(), s.weight)).collect();
        assert_eq!(got, segs, "item {i} segments");
        assert_eq!(it.prompt, opt(&w["prompt"]), "item {i} prompt");
        assert_eq!(it.answer, opt(&w["answer"]), "item {i} answer");
        assert_eq!(it.contradicted, opt(&w["contradicted"]), "item {i} contradicted");
        assert_eq!(it.users, strings(&w["users"]), "item {i} users");
        assert_eq!(it.replies, strings(&w["replies"]), "item {i} replies");
    }
}

/// Parsing, vocabulary growth, row encoding and one batch gradient against the reference.
fn check_pipeline(lessons: &Path, expected: &Path) -> (Bicameral, Vec<Item>, Value) {
    let want: Value = serde_json::from_slice(&std::fs::read(expected).unwrap()).unwrap();
    let items = parse_dir(lessons);
    check_items(&items, &want["items"]);

    let model = Bicameral::load(root().join("fixtures/nova2-step15000.nnova2")).unwrap();
    assert_eq!(model.tokenizer.words.len() as u64, want["words_before"].as_u64().unwrap());
    let min_count = want["min_count"].as_u64().unwrap() as usize;
    let words = grow_vocabulary(&model.tokenizer.words, &items, 256, min_count);
    assert_eq!(words, strings(&want["words"]), "grown vocabulary");

    let mut grown = model.clone();
    grown.tokenizer.words = words;
    let max = grown.config.max_tokens;
    let train: Vec<&Item> = items.iter().filter(|it| !it.held_out).collect();
    let mut rng = Rng::new(1);
    let mut rows = Vec::new();
    for it in train.iter().step_by(7) {
        let len: usize = it.segments.iter().map(|s| grown.tokenizer.encode(&s.text).unwrap().len()).sum();
        if (2..=max + 1).contains(&len) {
            rows.push(encode_item(&grown.tokenizer, &it.segments, max, &mut rng).unwrap().unwrap());
        }
    }
    rows.truncate(8);
    let want_rows = want["rows"].as_array().unwrap();
    assert_eq!(rows.len(), want_rows.len());
    for (r, w) in rows.iter().zip(want_rows) {
        assert_eq!(r.ids, u16s(&w["ids"]));
        assert_eq!(r.targets, u16s(&w["targets"]));
        let mask: Vec<f32> =
            w["mask"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        assert_eq!(r.mask, mask);
    }

    let (loss, grads) = batch_loss_and_gradients(&grown, &rows, Ablation::default()).unwrap();
    let b = &want["batch"];
    let want_loss = b["loss"].as_f64().unwrap();
    assert!((loss as f64 - want_loss).abs() <= 1e-4 * want_loss.abs(), "batch loss {loss} vs {want_loss}");
    let norm = b["grad_norm"].as_f64().unwrap();
    for pair in b["grads"].as_array().unwrap() {
        let (i, w) = (pair[0].as_u64().unwrap() as usize, pair[1].as_f64().unwrap());
        assert!((grads[i] as f64 - w).abs() < 1e-4 * norm, "batch grad[{i}] {} vs {w}", grads[i]);
    }
    (model, items, want)
}

fn held_out(items: &[Item]) -> Vec<&Item> {
    items.iter().filter(|it| it.held_out).collect()
}

#[test]
fn sample_course_matches_reference() {
    let (model, items, want) =
        check_pipeline(&root().join("fixtures/lessons"), &root().join("fixtures/learn.expected.json"));
    let report: Report = evaluate(&model, &held_out(&items), Ablation::default()).unwrap();
    let units = want["eval"].as_array().unwrap();
    assert_eq!(report.outcomes.len(), units.len());
    for (o, u) in report.outcomes.iter().zip(units) {
        assert_eq!(o.prompt, u["prompt"].as_str().unwrap());
        assert_eq!(o.expected, u["expected"].as_str().unwrap());
        assert_eq!(o.reply, u["reply"].as_str().unwrap(), "greedy reply to {:?}", o.prompt);
        assert_eq!(o.prefers_supported, u["prefers_supported"].as_bool(), "preference for {:?}", o.prompt);
    }
}

/// The owner's 27 lessons and the step-19,000 checkpoint (not published; see tests/compat/README.md).
#[test]
fn all_lessons_match_reference_when_present() {
    let dir = root().join("private");
    let lessons = dir.join("lessons");
    let expected = dir.join("learn-all-lessons.expected.json");
    let ckpt = dir.join("nova2-lessons-step19000.nnova2");
    let report_path = dir.join("step19000.eval_report.json");
    if ![&lessons, &expected, &ckpt, &report_path].iter().all(|p| p.exists()) {
        assert!(std::env::var_os("YENDORY_REQUIRE_PRIVATE").is_none(), "private lesson fixtures missing");
        eprintln!("private lessons not present; not checked (see tests/compat/README.md)");
        return;
    }
    let (_, items, want) = check_pipeline(&lessons, &expected);
    let step19000 = Bicameral::load(&ckpt).unwrap();
    assert_eq!(
        step19000.tokenizer.words,
        strings(&want["words"]),
        "vocabulary growth reproduces step 19,000"
    );

    // The published held-out results of the step-19,000 checkpoint (all 666 units).
    let started = std::time::Instant::now();
    let report = evaluate(&step19000, &held_out(&items), Ablation::default()).unwrap();
    let published: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    let after = &published["held_out_after_training"];
    let round = |x: f64| (x * 1000.).round() / 1000.;
    for (key, score) in &report.exact {
        eprintln!("{key}: {}/{}", score.correct, score.total);
        assert_eq!(round(score.rate().unwrap()), after[key].as_f64().unwrap(), "{key}");
    }
    assert_eq!(
        round(report.prefers_supported.rate().unwrap()),
        after["reasoning_prefers_supported"].as_f64().unwrap()
    );
    assert_eq!(round(report.overall().rate().unwrap()), after["overall_exact"].as_f64().unwrap());
    eprintln!("evaluated {} held-out units in {:?}", held_out(&items).len(), started.elapsed());
}
