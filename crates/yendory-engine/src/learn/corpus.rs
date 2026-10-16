//! Corpus pre-training ("reading"): plain next-token training on a text corpus before the
//! teacher lessons. The model reads ordinary text (for example short stories) so it starts the
//! lessons already knowing how English words and sentences go together.
//!
//! The same rules as lessons apply: about 10% of documents are held out by a stable hash of their
//! text and never trained on, word units are learned from the training documents only, and the
//! trainer decides what is kept: a checkpoint is accepted only when it lowers the held-out loss,
//! and the model returned is the best accepted checkpoint (the input model unchanged when no
//! checkpoint ever improved on it).
use super::lessons::{grow_vocabulary_from_texts, is_held_out};
use super::{batch_loss_and_gradients, lr_schedule, Rng, Row};
use crate::model::{Ablation, Bicameral, Focus};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

/// Separator between stories in the TinyStories files.
pub const END_OF_TEXT: &str = "<|endoftext|>";
/// Documents shorter than this (in characters, after normalisation) are skipped.
pub const MIN_DOCUMENT_CHARS: usize = 16;

/// Converts text to the ASCII the tokenizer reads: typographic quotes, dashes, ellipses and
/// spaces become their ASCII forms, common accented Latin letters lose their accents, and
/// anything else is dropped. Line endings become `\n`. Returns the text and the number of
/// characters dropped.
pub fn normalize_ascii(text: &str) -> (String, usize) {
    let mut out = String::with_capacity(text.len());
    let mut dropped = 0;
    for c in text.chars() {
        match c {
            '\n' => out.push('\n'),
            '\r' => {}
            '\t' => out.push(' '),
            ' '..='~' => out.push(c),
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' | '\u{00B4}' => out.push('\''),
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{2033}' | '\u{00AB}' | '\u{00BB}' => {
                out.push('"')
            }
            '\u{2010}'..='\u{2015}' | '\u{2212}' => out.push('-'),
            '\u{2026}' => out.push_str("..."),
            '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => out.push(' '),
            '\u{200B}'..='\u{200D}' | '\u{FEFF}' => {}
            _ => match fold_latin(c) {
                Some(s) => out.push_str(s),
                None => dropped += 1,
            },
        }
    }
    (out, dropped)
}

fn fold_latin(c: char) -> Option<&'static str> {
    Some(match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => "a",
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => "A",
        'è' | 'é' | 'ê' | 'ë' => "e",
        'È' | 'É' | 'Ê' | 'Ë' => "E",
        'ì' | 'í' | 'î' | 'ï' => "i",
        'Ì' | 'Í' | 'Î' | 'Ï' => "I",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' => "o",
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' => "O",
        'ù' | 'ú' | 'û' | 'ü' => "u",
        'Ù' | 'Ú' | 'Û' | 'Ü' => "U",
        'ý' | 'ÿ' => "y",
        'Ý' => "Y",
        'ñ' => "n",
        'Ñ' => "N",
        'ç' => "c",
        'Ç' => "C",
        'ß' => "ss",
        'æ' => "ae",
        'Æ' => "AE",
        'œ' => "oe",
        'Œ' => "OE",
        _ => return None,
    })
}

/// Splits normalised text into documents: at `<|endoftext|>` when the corpus uses it
/// (TinyStories), otherwise at blank lines (paragraphs). Lines are trimmed, runs of blank lines
/// inside a document are collapsed, and very short documents are skipped.
pub fn split_documents(text: &str) -> Vec<String> {
    let parts: Vec<&str> = if text.contains(END_OF_TEXT) {
        text.split(END_OF_TEXT).collect()
    } else {
        text.split("\n\n").collect()
    };
    parts
        .into_iter()
        .filter_map(|p| {
            let lines: Vec<&str> = p.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
            let doc = lines.join("\n");
            (doc.len() >= MIN_DOCUMENT_CHARS).then_some(doc)
        })
        .collect()
}

/// A corpus ready for training: documents split into training and held-out sets.
#[derive(Clone, Debug)]
pub struct Corpus {
    pub train: Vec<String>,
    pub held_out: Vec<String>,
    /// Characters that could not be represented in ASCII and were dropped.
    pub dropped_chars: usize,
}

impl Corpus {
    /// Normalises `text`, splits it into documents and holds out about 10% of them by a stable
    /// hash of their text (the same rule as lesson units).
    pub fn prepare(text: &str) -> Corpus {
        let (ascii, dropped_chars) = normalize_ascii(text);
        let (mut train, mut held_out) = (Vec::new(), Vec::new());
        for doc in split_documents(&ascii) {
            if is_held_out(&doc) {
                held_out.push(doc);
            } else {
                train.push(doc);
            }
        }
        Corpus { train, held_out, dropped_chars }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PretrainConfig {
    /// How many tokens to read (target tokens trained on); the run may stop earlier.
    pub tokens: u64,
    pub batch_size: usize,
    /// Tokens per training window (at most the model's context length).
    pub window: usize,
    pub learning_rate: f32,
    /// Evaluate the held-out loss this many times over the run (at least).
    pub evaluations: usize,
    /// Held-out windows scored at each evaluation.
    pub eval_windows: usize,
    /// A checkpoint is accepted only if it lowers the best held-out loss by at least this much.
    pub min_loss_improvement: f32,
    /// Stop after this many evaluations in a row without an accepted checkpoint.
    pub patience: usize,
    /// Learn word units from the corpus, up to `max_words` in total (leave room for lessons).
    pub grow_vocabulary: bool,
    pub max_words: usize,
    pub min_word_count: usize,
    pub seed: u64,
}

impl Default for PretrainConfig {
    fn default() -> Self {
        Self {
            tokens: 2_000_000,
            batch_size: 16,
            window: 128,
            learning_rate: 1e-3,
            evaluations: 20,
            eval_windows: 48,
            min_loss_improvement: 0.002,
            patience: 4,
            grow_vocabulary: true,
            max_words: 192,
            min_word_count: 12,
            seed: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PretrainEvent {
    Started {
        train_documents: usize,
        held_out_documents: usize,
        train_tokens: usize,
        held_out_tokens: usize,
        new_words: usize,
        steps: u64,
        baseline_loss: f32,
    },
    Batch {
        step: u64,
        steps: u64,
        tokens: u64,
        loss: f32,
        learning_rate: f32,
    },
    Evaluated(Evaluation),
    Finished {
        accepted: bool,
        best_loss: f32,
        tokens: u64,
        stopped: bool,
        early_stop: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub step: u64,
    pub tokens: u64,
    /// Mean training loss since the previous evaluation.
    pub train_loss: f32,
    pub held_out_loss: f32,
    pub best_loss: f32,
    pub accepted: bool,
    /// Greedy continuation of the start of a held-out document, written by the model.
    pub sample_prompt: String,
    pub sample: String,
}

pub struct PretrainResult {
    /// The best accepted checkpoint; the input model unchanged when none was accepted.
    pub model: Bicameral,
    pub accepted: bool,
    pub baseline_loss: f32,
    pub best_loss: f32,
    /// Target tokens trained on (including any after the best checkpoint).
    pub tokens: u64,
    pub new_words: usize,
    pub evaluations: Vec<Evaluation>,
    pub stopped: bool,
    pub early_stop: bool,
}

/// Encodes documents into one token stream, each document followed by a blank line.
fn encode_stream(model: &Bicameral, docs: &[String]) -> Result<Vec<u16>, String> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(1);
    let chunk = docs.len().div_ceil(threads).max(1);
    let parts: Vec<Result<Vec<u16>, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = docs
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for d in part {
                        out.extend(model.tokenizer.encode(d)?);
                        out.extend([b'\n' as u16, b'\n' as u16]);
                    }
                    Ok(out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("Encoding thread failed".into())))
            .collect()
    });
    let mut stream = Vec::new();
    for p in parts {
        stream.extend(p?);
    }
    Ok(stream)
}

fn window_row(stream: &[u16], start: usize, len: usize) -> Row {
    let w = &stream[start..start + len + 1];
    Row { ids: w[..len].to_vec(), targets: w[1..].to_vec(), mask: vec![1.; len] }
}

/// Fixed, evenly spaced held-out windows (the same ones at every evaluation).
fn eval_rows(stream: &[u16], window: usize, count: usize) -> Vec<Row> {
    if stream.len() < 2 {
        return Vec::new();
    }
    let len = window.min(stream.len() - 1);
    let room = stream.len() - len - 1;
    let n = count.min(room / len.max(1) + 1).max(1);
    (0..n).map(|i| window_row(stream, if n == 1 { 0 } else { i * room / (n - 1) }, len)).collect()
}

fn mean_loss(model: &Bicameral, rows: &[Row]) -> Result<f32, String> {
    let mut total = 0f64;
    let mut weight = 0f64;
    for chunk in rows.chunks(16) {
        let (l, _) = batch_loss_and_gradients(model, chunk, Ablation::default())?;
        let w: f64 = chunk.iter().map(|r| r.mask.len() as f64).sum();
        total += l as f64 * w;
        weight += w;
    }
    Ok((total / weight.max(1.)) as f32)
}

/// The first words of a held-out document, used as the sample prompt.
fn sample_prompt(corpus: &Corpus) -> String {
    let doc = corpus.held_out.first().map(String::as_str).unwrap_or("Once upon a time");
    let line = doc.lines().next().unwrap_or(doc);
    let words: Vec<&str> = line.split(' ').take(6).collect();
    words.join(" ")
}

fn sample(model: &Bicameral, prompt: &str) -> String {
    super::eval::reply(model, prompt, 40, Ablation::default()).unwrap_or_default()
}

/// Reads the corpus: trains the whole model (Joint) on random windows of the training
/// documents, evaluates the held-out loss at regular intervals and keeps the best checkpoint.
/// `on_accepted` is called with every newly accepted checkpoint (to save progress); `stop` ends
/// the run at the next batch and the best accepted checkpoint is returned.
pub fn pretrain(
    model: &Bicameral,
    corpus: &Corpus,
    cfg: &PretrainConfig,
    observer: &mut dyn FnMut(&PretrainEvent),
    on_accepted: &mut dyn FnMut(&Bicameral) -> Result<(), String>,
    stop: &AtomicBool,
) -> Result<PretrainResult, String> {
    if corpus.train.is_empty() || corpus.held_out.is_empty() {
        return Err("The corpus is too small: it needs training documents and held-out documents \
                    (about one document in ten is held out)"
            .into());
    }
    if cfg.batch_size == 0
        || cfg.window < 2
        || !cfg.learning_rate.is_finite()
        || cfg.learning_rate <= 0.
        || cfg.tokens == 0
    {
        return Err("Invalid pre-training settings".into());
    }
    let mut current = model.clone();
    let mut new_words = 0;
    if cfg.grow_vocabulary {
        let cap = cfg.max_words.min(crate::tokenizer::MAX_WORDS).max(current.tokenizer.words.len());
        let words = grow_vocabulary_from_texts(
            &current.tokenizer.words,
            corpus.train.iter().map(String::as_str),
            cap,
            cfg.min_word_count,
        );
        new_words = words.len() - current.tokenizer.words.len();
        current.tokenizer.words = words;
    }
    let window = cfg.window.min(current.config.max_tokens);
    let train = encode_stream(&current, &corpus.train)?;
    let held = encode_stream(&current, &corpus.held_out)?;
    if train.len() < window + 1 {
        return Err("The corpus is too small for one training window".into());
    }
    let probes = eval_rows(&held, window, cfg.eval_windows);
    let prompt = sample_prompt(corpus);
    let tokens_per_step = (cfg.batch_size * window) as u64;
    let steps = cfg.tokens.div_ceil(tokens_per_step).max(1);
    let warmup = (steps / 20).clamp(1, 200);
    let eval_every = (steps / cfg.evaluations.max(1) as u64).max(1);

    let baseline_loss = mean_loss(&current, &probes)?;
    observer(&PretrainEvent::Started {
        train_documents: corpus.train.len(),
        held_out_documents: corpus.held_out.len(),
        train_tokens: train.len(),
        held_out_tokens: held.len(),
        new_words,
        steps,
        baseline_loss,
    });

    let mut rng = Rng::new(cfg.seed);
    let mut best: Option<Bicameral> = None;
    let mut best_loss = baseline_loss;
    let mut evaluations = Vec::new();
    let (mut since_eval, mut since_n) = (0f64, 0usize);
    let mut misses = 0;
    let (mut stopped, mut early_stop) = (false, false);
    let mut tokens = 0u64;
    for step in 0..steps {
        if stop.load(Ordering::Acquire) {
            stopped = true;
            break;
        }
        let rows: Vec<Row> = (0..cfg.batch_size)
            .map(|_| window_row(&train, rng.below(train.len() - window), window))
            .collect();
        let (loss, grads) = batch_loss_and_gradients(&current, &rows, Ablation::default())?;
        if !loss.is_finite() {
            return Err("Training diverged (loss is not finite); try a lower learning rate".into());
        }
        let lr = lr_schedule(step, cfg.learning_rate, warmup, steps, 0.1);
        current.adam_step(&grads, lr, Focus::Joint)?;
        tokens += tokens_per_step;
        since_eval += loss as f64;
        since_n += 1;
        observer(&PretrainEvent::Batch { step: step + 1, steps, tokens, loss, learning_rate: lr });

        if (step + 1) % eval_every == 0 || step + 1 == steps {
            let held_out_loss = mean_loss(&current, &probes)?;
            let accepted = held_out_loss.is_finite() && held_out_loss < best_loss - cfg.min_loss_improvement;
            if accepted {
                best_loss = held_out_loss;
                best = Some(current.clone());
                misses = 0;
                on_accepted(&current)?;
            } else {
                misses += 1;
            }
            let e = Evaluation {
                step: step + 1,
                tokens,
                train_loss: (since_eval / since_n.max(1) as f64) as f32,
                held_out_loss,
                best_loss,
                accepted,
                sample_prompt: prompt.clone(),
                sample: sample(&current, &prompt),
            };
            observer(&PretrainEvent::Evaluated(e.clone()));
            evaluations.push(e);
            since_eval = 0.;
            since_n = 0;
            if misses >= cfg.patience.max(1) {
                early_stop = true;
                break;
            }
        }
    }
    let accepted = best.is_some();
    observer(&PretrainEvent::Finished { accepted, best_loss, tokens, stopped, early_stop });
    Ok(PretrainResult {
        model: best.unwrap_or_else(|| model.clone()),
        accepted,
        baseline_loss,
        best_loss,
        tokens,
        new_words: if accepted { new_words } else { 0 },
        evaluations,
        stopped,
        early_stop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalises_to_ascii() {
        let (t, dropped) =
            normalize_ascii("\u{201C}Caf\u{00E9}\u{201D} \u{2014} it\u{2019}s\u{2026}\r\n\u{4E2D}ok");
        assert_eq!(t, "\"Cafe\" - it's...\nok");
        assert_eq!(dropped, 1);
        assert!(t.is_ascii());
    }

    #[test]
    fn splits_stories_and_paragraphs() {
        let stories = "One day a cat sat down.\n<|endoftext|>\nTom saw a big red ball.\n\nHe was happy.\n<|endoftext|>\nx";
        assert_eq!(
            split_documents(stories),
            ["One day a cat sat down.", "Tom saw a big red ball.\nHe was happy."]
        );
        let book = "  First paragraph here.  \n\n\n\nSecond paragraph, longer.\n\nshort";
        assert_eq!(split_documents(book), ["First paragraph here.", "Second paragraph, longer."]);
    }

    #[test]
    fn held_out_split_is_stable() {
        let text: String = (0..300).map(|i| format!("Story number {i} is about a small dog.\n\n")).collect();
        let a = Corpus::prepare(&text);
        let b = Corpus::prepare(&text);
        assert_eq!(a.held_out, b.held_out);
        assert!(!a.held_out.is_empty() && a.held_out.len() < 60, "{}", a.held_out.len());
        assert!(a.train.iter().all(|d| !a.held_out.contains(d)));
    }
}
