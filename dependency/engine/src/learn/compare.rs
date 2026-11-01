//! Comparing teaching settings on one model (an experiment; the model is never changed).
//!
//! The same lessons are taught from the same starting model once per variant of the settings,
//! the way the Classroom teaches lessons together (saved earlier lessons trained alongside them
//! when `train_earlier` is on). Every lesson's held-out questions are split in two by position:
//! the judged half is what the sessions see and choose checkpoints by; the TEST half is never
//! trained on and never used to choose anything, so it measures what each variant generalizes.
//! After every epoch the accepted model and the model training continues from are both asked
//! the test half; at the end the context tests of `learn::listening` are asked too.
use super::course::CourseFile;
use super::eval::evaluate;
use super::lessons::{parse_lesson, Item, Kind};
use super::listening::{context_tests, score as context_score, ContextScore};
use super::session::{teach_rewarded, EpochHook, Event, GrowthCheck, Probation, SessionConfig};
use super::{batch_loss, encode_item, Rng};
use crate::model::{Ablation, Bicameral};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// One set of settings to compare.
#[derive(Clone, Debug)]
pub struct Variant {
    pub name: String,
    pub cfg: SessionConfig,
}

/// Answers on the test half (never trained on, never used to choose a checkpoint).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TestScore {
    /// The lessons being taught.
    pub new_right: usize,
    pub new_total: usize,
    /// The saved earlier lessons.
    pub earlier_right: usize,
    pub earlier_total: usize,
    /// Teacher-forced loss on each half.
    pub new_loss: f32,
    pub earlier_loss: f32,
    /// Replies stopped by their length limit before they ended.
    pub cut_off: usize,
    /// Which test questions were right (new, then earlier), for gained and lost counts.
    #[serde(skip)]
    pub right: Vec<bool>,
}

/// After one epoch of a variant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpochTest {
    pub epoch: u32,
    pub accepted: TestScore,
    pub working: TestScore,
    pub accepted_step: u64,
    pub working_step: u64,
    /// Context-test pairs passed and asked, of the accepted model and of the working model
    /// (when the comparison asks context tests).
    #[serde(default)]
    pub accepted_context: Option<(usize, usize)>,
    #[serde(default)]
    pub working_context: Option<(usize, usize)>,
}

/// What one variant did.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariantResult {
    pub name: String,
    pub end: TestScore,
    /// Test answers right at the end that were wrong at the start, and the reverse.
    pub gained: usize,
    pub lost: usize,
    pub epochs: u32,
    pub accepted_epochs: u32,
    /// Probation branches promoted and dropped.
    pub promoted: u32,
    pub dropped: u32,
    /// Why candidates were rejected: reason and how many.
    pub rejections: std::collections::BTreeMap<String, usize>,
    pub growth: Option<GrowthCheck>,
    pub context: Option<ContextScore>,
    pub trajectory: Vec<EpochTest>,
    pub stopped: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CompareEvent {
    Started {
        variants: Vec<String>,
        /// Test questions of the lessons taught and of the earlier lessons.
        new_test: usize,
        earlier_test: usize,
        start: TestScore,
        start_context: Option<ContextScore>,
    },
    /// A variant began.
    VariantStarted {
        index: usize,
        name: String,
    },
    /// An event of the variant's session.
    Session {
        index: usize,
        event: Event,
    },
    /// The test half after an epoch.
    Tested {
        index: usize,
        test: EpochTest,
    },
    VariantFinished {
        index: usize,
        result: VariantResult,
    },
    Finished {
        stopped: bool,
    },
}

/// Splits each lesson's held-out questions by position: odd ones are judged, even ones tested.
pub fn split_test(items: Vec<Item>) -> (Vec<Item>, Vec<Item>) {
    let mut seen: std::collections::HashMap<String, usize> = Default::default();
    let (mut keep, mut test) = (Vec::new(), Vec::new());
    for it in items {
        if it.held_out && it.kind != Kind::Prose {
            let k = seen.entry(it.source.clone()).or_default();
            *k += 1;
            if *k % 2 == 0 {
                test.push(it);
                continue;
            }
        }
        keep.push(it);
    }
    (keep, test)
}

/// Asks the test halves (`new`, `earlier`).
pub fn test(model: &Bicameral, new: &[Item], earlier: &[Item]) -> Result<TestScore, String> {
    let ask = |items: &[Item]| -> Result<(Vec<bool>, f32, usize), String> {
        let refs: Vec<&Item> = items.iter().collect();
        let r = evaluate(model, &refs, Ablation::default())?;
        let mut rng = Rng::new(1);
        let mut rows = Vec::new();
        for it in items {
            if let Some(row) = encode_item(&model.tokenizer, &it.segments, model.config.max_tokens, &mut rng)?
            {
                rows.push(row);
            }
        }
        Ok((
            r.outcomes.iter().map(|o| o.correct).collect(),
            batch_loss(model, &rows, Ablation::default())?,
            r.cut_off,
        ))
    };
    let (n, nl, nc) = ask(new)?;
    let (e, el, ec) = ask(earlier)?;
    Ok(TestScore {
        new_right: n.iter().filter(|x| **x).count(),
        new_total: n.len(),
        earlier_right: e.iter().filter(|x| **x).count(),
        earlier_total: e.len(),
        new_loss: nl,
        earlier_loss: el,
        cut_off: nc + ec,
        right: n.into_iter().chain(e).collect(),
    })
}

/// The context tests of `learn::listening` (a few per skill), asked of `model`.
pub fn context(model: &Bicameral, per_skill: usize) -> Option<ContextScore> {
    if per_skill == 0 {
        return None;
    }
    let tests = context_tests(per_skill);
    let prompts: Vec<String> = tests.iter().flat_map(|t| [t.a.prompt(), t.b.prompt()]).collect();
    Some(context_score(&tests, &super::conversation::replies(model, &prompts, 24)))
}

/// Teaches `files` (together, one session per variant) from `model` with each variant's
/// settings, and measures every variant on the test halves. `archive`: the model's saved
/// earlier lessons. The model is not changed. `context_per_skill`: context tests per skill
/// asked at the start and of each variant's result (0: none).
pub fn compare(
    model: &Bicameral,
    files: &[CourseFile],
    archive: &[Item],
    variants: &[Variant],
    context_per_skill: usize,
    observer: &mut dyn FnMut(&CompareEvent),
    stop: &AtomicBool,
) -> Result<Vec<VariantResult>, String> {
    if variants.is_empty() {
        return Err("Choose at least one set of settings to compare".into());
    }
    let mut lesson = Vec::new();
    for f in files {
        lesson.extend(parse_lesson(&f.text, &f.name)?);
    }
    let sources: std::collections::HashSet<String> = lesson.iter().map(|i| i.source.clone()).collect();
    let earlier_items: Vec<Item> = archive.iter().filter(|i| !sources.contains(&i.source)).cloned().collect();
    let (lesson, new_test) = split_test(lesson);
    let (earlier, earlier_test) = split_test(earlier_items);
    if new_test.is_empty() {
        return Err("The lessons have too few held-out questions to keep half of them for testing".into());
    }
    let start = test(model, &new_test, &earlier_test)?;
    let start_context = context(model, context_per_skill);
    observer(&CompareEvent::Started {
        variants: variants.iter().map(|v| v.name.clone()).collect(),
        new_test: new_test.len(),
        earlier_test: earlier_test.len(),
        start: start.clone(),
        start_context,
    });
    let mut results = Vec::new();
    let mut stopped = false;
    for (index, v) in variants.iter().enumerate() {
        if stop.load(Ordering::Acquire) {
            stopped = true;
            break;
        }
        observer(&CompareEvent::VariantStarted { index, name: v.name.clone() });
        // The test halves after every epoch, collected by the session's hook and reported
        // after its epoch event.
        let tested: Arc<Mutex<Vec<Result<EpochTest, String>>>> = Default::default();
        let hook = {
            let (tested, nt, et) = (tested.clone(), new_test.clone(), earlier_test.clone());
            EpochHook(Arc::new(move |epoch: u32, accepted: &Bicameral, working: &Bicameral| {
                let same = working.step == accepted.step;
                let passed = |m: &Bicameral| context(m, context_per_skill).map(|c| (c.passed, c.total));
                let r = test(accepted, &nt, &et).and_then(|a| {
                    let w = if same { a.clone() } else { test(working, &nt, &et)? };
                    let accepted_context = passed(accepted);
                    Ok(EpochTest {
                        epoch,
                        accepted: a,
                        working: w,
                        accepted_step: accepted.step,
                        working_step: working.step,
                        working_context: if same { accepted_context } else { passed(working) },
                        accepted_context,
                    })
                });
                tested.lock().unwrap().push(r);
            }))
        };
        let cfg = SessionConfig { epoch_hook: Some(hook), ..v.cfg.clone() };
        // As the Classroom teaches lessons together: the earlier lessons trained in the mix
        // (`train_earlier`) and guarded either way.
        let all: Vec<Item> =
            if cfg.train_earlier { lesson.iter().chain(&earlier).cloned().collect() } else { lesson.clone() };
        let cfg = SessionConfig { eval_limit: None, ..cfg };
        let mut trajectory = Vec::new();
        let mut growth = None;
        let mut rejections = std::collections::BTreeMap::new();
        let (mut promoted, mut dropped, mut accepted_epochs, mut epochs) = (0, 0, 0, 0);
        let mut failed = None;
        let mut drain = |trajectory: &mut Vec<EpochTest>, observer: &mut dyn FnMut(&CompareEvent)| {
            for r in tested.lock().unwrap().drain(..) {
                match r {
                    Ok(t) => {
                        observer(&CompareEvent::Tested { index, test: t.clone() });
                        trajectory.push(t);
                    }
                    Err(e) => failed = Some(e),
                }
            }
        };
        let r = {
            let mut forward = |e: &Event| {
                drain(&mut trajectory, observer);
                match e {
                    Event::Started { growth: g, .. } => growth = g.clone(),
                    Event::Epoch(rec) => {
                        epochs += 1;
                        accepted_epochs += u32::from(rec.selected.is_some());
                        match rec.probation {
                            Some(Probation::Promoted { .. }) => promoted += 1,
                            Some(Probation::Discarded { .. }) => dropped += 1,
                            _ => {}
                        }
                        for c in &rec.candidates {
                            for r in &c.rejections {
                                *rejections.entry(r.name().to_string()).or_insert(0usize) += 1;
                            }
                        }
                    }
                    _ => {}
                }
                observer(&CompareEvent::Session { index, event: e.clone() });
            };
            teach_rewarded(model, &all, None, &earlier, &cfg, None, &mut forward, stop)
        };
        drain(&mut trajectory, observer);
        if let Some(e) = failed {
            return Err(e);
        }
        let r = r?;
        let end = test(&r.model, &new_test, &earlier_test)?;
        let gained = start.right.iter().zip(&end.right).filter(|(a, b)| !**a && **b).count();
        let lost = start.right.iter().zip(&end.right).filter(|(a, b)| **a && !**b).count();
        let result = VariantResult {
            name: v.name.clone(),
            end,
            gained,
            lost,
            epochs,
            accepted_epochs,
            promoted,
            dropped,
            rejections,
            growth,
            context: context(&r.model, context_per_skill),
            trajectory,
            stopped: r.stopped,
        };
        observer(&CompareEvent::VariantFinished { index, result: result.clone() });
        results.push(result);
        if r.stopped {
            stopped = true;
            break;
        }
    }
    observer(&CompareEvent::Finished { stopped });
    Ok(results)
}
