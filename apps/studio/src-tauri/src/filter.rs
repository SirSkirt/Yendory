//! Quality filter for downloaded conversation datasets: what Nova practises with is fewer, better
//! conversations, not as many as possible.
//!
//! Since 0.8.3 a download keeps every usable conversation (the dataset's own flagged ones in a
//! file apart) and the filter is applied when practice starts, with the reasons chosen on the
//! Conversation page (`FilterChoice`, all on by default) and as much of each dataset as asked
//! for. The download still reports what the full filter would keep. Two things are always left
//! out: conversations that are not usable (malformed, broken turn order) and conversations that
//! hold one of Yendory's tests (so the tests stay unseen). Datasets you add yourself are never
//! filtered.
//!
//! Left out, in this order:
//! * flagged by the dataset itself (LMSYS-Chat-1M: not English, flagged by its moderation, or
//!   names removed for privacy; see `sources::screen_of`);
//! * malformed: not a conversation between two people with at least two turns, or a turn with
//!   no words;
//! * broken turn order: the same speaker twice in a row (roles mixed up or turns missing);
//! * too long: a turn over 600 characters, or code;
//! * not English;
//! * unsafe or inappropriate: explicit, hateful, violent or self-harm words;
//! * low quality: links, e-mail addresses or placeholders, mostly symbols, shouting, a turn said
//!   twice in a row, or too few words to be a conversation;
//! * assistant style: an assistant's phrases ("Certainly! Here are", "I hope this helps", "as an
//!   AI", "Is there anything else"), lists and headings, or one speaker answering at length every
//!   time, so Nova does not learn to sound like a customer-service helper;
//! * Yendory's tests: a conversation that holds a context or understanding test (its fact and
//!   its question), so the tests stay unseen;
//! * duplicates: the same conversation again (ignoring case, spacing and punctuation).
use crate::sources::{english, Convert, Screen};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use yendory_engine::learn::conversation::parse_dialogue;

/// Why a conversation was left out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    Flagged,
    Malformed,
    TurnOrder,
    TooLong,
    NotEnglish,
    Unsafe,
    LowQuality,
    AssistantStyle,
    OverlapsTests,
    Duplicate,
}

/// Which reasons leave a conversation out (the page's switches; all on by default). Malformed
/// conversations, broken turn order and Yendory's tests are always left out.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FilterChoice {
    pub assistant_style: bool,
    pub too_long: bool,
    pub not_english: bool,
    /// Flagged by the dataset itself (LMSYS-Chat-1M: its language tag, moderation, names removed).
    pub flagged: bool,
    pub unsafe_content: bool,
    pub low_quality: bool,
    pub duplicates: bool,
}

impl Default for FilterChoice {
    fn default() -> Self {
        FilterChoice {
            assistant_style: true,
            too_long: true,
            not_english: true,
            flagged: true,
            unsafe_content: true,
            low_quality: true,
            duplicates: true,
        }
    }
}

impl FilterChoice {
    pub fn applies(&self, r: Reason) -> bool {
        match r {
            Reason::Malformed | Reason::TurnOrder | Reason::OverlapsTests => true,
            Reason::Flagged => self.flagged,
            Reason::TooLong => self.too_long,
            Reason::NotEnglish => self.not_english,
            Reason::Unsafe => self.unsafe_content,
            Reason::LowQuality => self.low_quality,
            Reason::AssistantStyle => self.assistant_style,
            Reason::Duplicate => self.duplicates,
        }
    }
}

/// What the filter did with a download.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FilterReport {
    /// Conversations read, and kept.
    pub read: usize,
    pub kept: usize,
    pub flagged: usize,
    pub malformed: usize,
    pub turn_order: usize,
    pub too_long: usize,
    pub not_english: usize,
    pub unsafe_content: usize,
    pub low_quality: usize,
    pub assistant_style: usize,
    pub overlaps_tests: usize,
    pub duplicates: usize,
}

impl FilterReport {
    fn count(&mut self, r: Reason) {
        *match r {
            Reason::Flagged => &mut self.flagged,
            Reason::Malformed => &mut self.malformed,
            Reason::TurnOrder => &mut self.turn_order,
            Reason::TooLong => &mut self.too_long,
            Reason::NotEnglish => &mut self.not_english,
            Reason::Unsafe => &mut self.unsafe_content,
            Reason::LowQuality => &mut self.low_quality,
            Reason::AssistantStyle => &mut self.assistant_style,
            Reason::OverlapsTests => &mut self.overlaps_tests,
            Reason::Duplicate => &mut self.duplicates,
        } += 1;
    }

    /// "kept 812 of 3,000 (left out: 1,500 assistant-style, ...)".
    pub fn summary(&self) -> String {
        let parts: Vec<String> = [
            (self.assistant_style, "assistant-style"),
            (self.not_english, "not English"),
            (self.flagged, "flagged by the dataset"),
            (self.too_long, "too long or code"),
            (self.low_quality, "low quality"),
            (self.unsafe_content, "unsafe or inappropriate"),
            (self.turn_order, "broken turn order"),
            (self.malformed, "malformed"),
            (self.duplicates, "duplicates"),
            (self.overlaps_tests, "overlapping Yendory's tests"),
        ]
        .iter()
        .filter(|(n, _)| *n > 0)
        .map(|(n, what)| format!("{} {what}", thousands(*n)))
        .collect();
        let mut s = format!("kept {} of {} conversations", thousands(self.kept), thousands(self.read));
        if !parts.is_empty() {
            s.push_str(&format!(" (left out: {})", parts.join(", ")));
        }
        s
    }
}

impl FilterReport {
    /// For a download that keeps everything: what the full filter would keep.
    pub fn estimate(&self) -> String {
        let s = self.summary();
        format!(
            "with every filter on, {}",
            s.replacen("kept", "practice would use", 1).replace("(left out:", "(leaving out")
        )
    }
}

fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Longest turn kept.
const LONGEST: usize = 600;

/// Phrases of an assistant's voice (lower case).
const ASSISTANT: [&str; 36] = [
    "as an ai",
    "language model",
    "i'm sorry, but i",
    "i am sorry, but i",
    "i apologize",
    "certainly!",
    "certainly,",
    "certainly.",
    "sure! here",
    "sure, here",
    "here are some",
    "here are a few",
    "here is a",
    "here's a",
    "i hope this helps",
    "hope this helps",
    "let me know if",
    "feel free to",
    "great question",
    "i'd be happy to",
    "i would be happy to",
    "happy to help",
    "in conclusion",
    "in summary",
    "additionally,",
    "furthermore,",
    "it's important to",
    "it is important to",
    "how can i help",
    "how can i assist",
    "is there anything else",
    "as an assistant",
    "virtual assistant",
    "step 1",
    "the following",
    "please note",
];

/// Words that make a conversation unsuitable (whole words, lower case).
const UNSAFE: [&str; 40] = [
    "fuck",
    "fucking",
    "fucked",
    "motherfucker",
    "shit",
    "bitch",
    "bastard",
    "cunt",
    "dick",
    "cock",
    "pussy",
    "porn",
    "porno",
    "pornography",
    "xxx",
    "nude",
    "nudes",
    "naked",
    "sex",
    "sexy",
    "sexual",
    "orgasm",
    "erotic",
    "horny",
    "rape",
    "raped",
    "rapist",
    "molest",
    "suicide",
    "suicidal",
    "nigger",
    "nigga",
    "faggot",
    "retard",
    "kike",
    "spic",
    "chink",
    "whore",
    "slut",
    "dildo",
];

/// Phrases that make a conversation unsuitable (lower case).
const UNSAFE_PHRASES: [&str; 6] =
    ["kill yourself", "kill myself", "cut myself", "hurt myself", "how to make a bomb", "child abuse"];

/// A test conversation's fact and question (lower case, single spaces): a downloaded
/// conversation holding both holds the test.
struct TestSignature {
    fact: String,
    question: String,
}

fn normal(text: &str) -> String {
    text.to_ascii_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The context and understanding tests' signatures (the first thing the user says and the
/// question asked).
fn test_signatures() -> Vec<TestSignature> {
    use yendory_engine::learn::listening::{context_tests, understanding_tests};
    let mut out = Vec::new();
    for t in context_tests(50).into_iter().chain(understanding_tests(50, "Nova")) {
        for p in [t.a, t.b] {
            if let Some((_, fact)) = p.turns.iter().find(|(mine, _)| !mine) {
                out.push(TestSignature { fact: normal(fact), question: normal(&p.question) });
            }
        }
    }
    out
}

/// The filter's state through one download (or one preparation).
pub struct ConversationFilter {
    pub report: FilterReport,
    pub choice: FilterChoice,
    seen: HashSet<u64>,
    tests: Vec<TestSignature>,
}

/// A conversation a download keeps, and whether the dataset itself flagged it (`screen`).
#[derive(Debug, PartialEq)]
pub struct Taken {
    pub doc: String,
    pub screened: bool,
}

impl Default for ConversationFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationFilter {
    pub fn new() -> Self {
        Self::with(FilterChoice::default())
    }

    pub fn with(choice: FilterChoice) -> Self {
        ConversationFilter {
            report: FilterReport::default(),
            choice,
            seen: HashSet::new(),
            tests: test_signatures(),
        }
    }

    /// One dataset row at download: every usable conversation is returned (flagged by the
    /// dataset or not); the report counts what the full filter would keep.
    pub fn take_all(&mut self, row: &Value, screen: Option<Screen>, convert: Convert) -> Vec<Taken> {
        let screened = screen.and_then(|s| s(row));
        let docs = convert(row);
        if docs.is_empty() {
            self.report.read += 1;
            self.report.count(screened.unwrap_or(Reason::Malformed));
            return Vec::new();
        }
        let mut out = Vec::new();
        for doc in docs {
            self.report.read += 1;
            let verdict = match screened {
                Some(r) => {
                    // Counted as the dataset's flag; usable unless it cannot be read.
                    self.report.count(r);
                    reasons(&doc)
                        .first()
                        .copied()
                        .filter(|r| matches!(r, Reason::Malformed | Reason::TurnOrder))
                }
                None => {
                    let v = self.verdict(&doc);
                    match v {
                        Some(r) => self.report.count(r),
                        None => self.report.kept += 1,
                    }
                    v
                }
            };
            // Never usable: unreadable conversations and Yendory's tests. Everything else is kept
            // and filtered when practice starts.
            if !matches!(verdict, Some(Reason::Malformed | Reason::TurnOrder | Reason::OverlapsTests)) {
                out.push(Taken { doc, screened: screened.is_some() });
            }
        }
        out
    }

    /// Checks one converted conversation; true when it is kept (and counts it either way).
    pub fn keep(&mut self, doc: &str) -> bool {
        self.report.read += 1;
        match self.verdict(doc) {
            Some(why) => {
                self.report.count(why);
                false
            }
            None => {
                self.report.kept += 1;
                true
            }
        }
    }

    /// The first reason (of the ones chosen) the conversation is left out for.
    fn verdict(&mut self, doc: &str) -> Option<Reason> {
        if let Some(why) = reasons(doc).into_iter().find(|r| self.choice.applies(*r)) {
            return Some(why);
        }
        let text = normal(doc);
        if self.tests.iter().any(|t| text.contains(&t.question) && text.contains(&t.fact)) {
            return Some(Reason::OverlapsTests);
        }
        // Duplicates: the same words in the same order.
        let key: String = text.chars().filter(|c| c.is_ascii_alphanumeric() || *c == ' ').collect();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&key, &mut h);
        if !self.seen.insert(std::hash::Hasher::finish(&h)) && self.choice.duplicates {
            return Some(Reason::Duplicate);
        }
        None
    }
}

/// The conversations of a downloaded dataset that pass the chosen filters, up to `cap` bytes:
/// `docs` are the dataset's conversations, `flagged` the ones the dataset itself flagged (used
/// only when that filter is off). Returns them with the filter's counts.
pub fn select<'a>(
    docs: impl IntoIterator<Item = &'a str>,
    flagged: impl IntoIterator<Item = &'a str>,
    choice: &FilterChoice,
    cap: Option<u64>,
) -> (Vec<String>, FilterReport) {
    let mut f = ConversationFilter::with(choice.clone());
    let mut out = Vec::new();
    let mut bytes = 0u64;
    let flagged_too = !choice.flagged;
    let all = docs.into_iter().map(|d| (d, false)).chain(flagged.into_iter().map(|d| (d, true)));
    for (doc, screened) in all {
        if cap.is_some_and(|c| bytes >= c) {
            break;
        }
        if screened && !flagged_too {
            f.report.read += 1;
            f.report.count(Reason::Flagged);
            continue;
        }
        if f.keep(doc) {
            bytes += doc.len() as u64 + 16;
            out.push(doc.to_string());
        }
    }
    (out, f.report)
}

/// Every reason a conversation would be left out for, in the filter's order (apart from the
/// tests and duplicates; empty: it is kept). Malformed and broken turn order end the list (the
/// rest cannot be judged).
pub fn reasons(doc: &str) -> Vec<Reason> {
    let Some(d) = parse_dialogue(doc) else { return vec![Reason::Malformed] };
    if d.turns.iter().any(|t| !t.chars().any(|c| c.is_ascii_alphabetic())) {
        return vec![Reason::Malformed];
    }
    if d.speakers.windows(2).any(|w| w[0] == w[1]) {
        return vec![Reason::TurnOrder];
    }
    let mut out = Vec::new();
    if d.turns.iter().any(|t| t.len() > LONGEST || t.contains("```")) {
        out.push(Reason::TooLong);
    }
    let all = d.turns.join("\n");
    if !english(&all) {
        out.push(Reason::NotEnglish);
    }
    let lower = all.to_ascii_lowercase();
    let words: Vec<&str> =
        lower.split(|c: char| !c.is_ascii_alphanumeric() && c != '\'').filter(|w| !w.is_empty()).collect();
    if words.iter().any(|w| UNSAFE.contains(w)) || UNSAFE_PHRASES.iter().any(|p| lower.contains(p)) {
        out.push(Reason::Unsafe);
    }
    if low_quality(&d.turns, &lower, words.len()) {
        out.push(Reason::LowQuality);
    }
    if assistant_style(&d.speakers, &d.turns, &lower) {
        out.push(Reason::AssistantStyle);
    }
    out
}

fn low_quality(turns: &[String], lower: &str, words: usize) -> bool {
    let links = ["http://", "https://", "www.", ".com", "@"];
    let placeholder = lower.contains("name_") || lower.contains("[name]") || lower.contains("<name>");
    let symbols = |t: &str| {
        let odd = t.chars().filter(|c| !c.is_ascii_alphanumeric() && !" .,!?'\"-:;()".contains(*c)).count();
        odd * 10 > t.len().max(1) * 3
    };
    let shouting = |t: &str| {
        let letters: Vec<char> = t.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        letters.len() >= 20 && letters.iter().all(|c| c.is_ascii_uppercase())
    };
    links.iter().any(|l| lower.contains(l))
        || placeholder
        || words < 6
        || turns.iter().any(|t| symbols(t) || shouting(t))
        // A longer turn said back word for word (a short "Hi!" / "Hi!" is how people talk).
        || turns.windows(2).any(|w| w[0].split_whitespace().count() >= 4 && w[0].trim().eq_ignore_ascii_case(w[1].trim()))
}

fn assistant_style(speakers: &[String], turns: &[String], lower: &str) -> bool {
    let phrases = ASSISTANT.iter().filter(|p| lower.contains(*p)).count();
    // Lists and headings, written inline once the turn's lines were joined.
    let lists = turns
        .iter()
        .filter(|t| t.contains(" 2. ") || t.contains("**") || t.contains(" - ") || t.starts_with("1. "))
        .count();
    // One speaker answering at length every time.
    let mut long_answers = false;
    for who in speakers.iter().collect::<HashSet<_>>() {
        let mine: Vec<&String> =
            speakers.iter().zip(turns).filter(|(s, _)| *s == who).map(|(_, t)| t).collect();
        let mean = mine.iter().map(|t| t.len()).sum::<usize>() / mine.len().max(1);
        if mean > 300 {
            long_answers = true;
        }
    }
    phrases >= 2 || (phrases >= 1 && (lists > 0 || long_answers)) || lists >= 2 || long_answers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::whole_dialogue;
    use serde_json::json;

    fn verdict(doc: &str) -> Option<Reason> {
        ConversationFilter::new().verdict(doc)
    }

    #[test]
    fn natural_conversations_are_kept() {
        let doc = "Mia: Hi! How was your weekend?\nLeo: Great, I went hiking with my cousin.\nMia: Where did you go?\nLeo: Up the hill behind town. The view was lovely.";
        assert_eq!(verdict(doc), None);
    }

    #[test]
    fn each_problem_is_counted_under_its_reason() {
        let cases = [
            ("Mia: Hi there.", Reason::Malformed),
            ("Mia: Hi there, how are you?\nMia: Hello?\nLeo: Sorry, I'm here.", Reason::TurnOrder),
            (&*format!("Mia: Tell me about it.\nLeo: {}", "word ".repeat(150)), Reason::TooLong),
            ("Mia: Wie geht es dir heute?\nLeo: Mir geht es gut, danke. Und dir?", Reason::NotEnglish),
            ("Mia: That movie was so sexy.\nLeo: You think so? I found it boring.", Reason::Unsafe),
            ("Mia: Look at this www.example.com page.\nLeo: Nice, I will look later.", Reason::LowQuality),
            ("Mia: Hello NAME_1, how are you today?\nLeo: I am fine, thanks for asking.", Reason::LowQuality),
            ("Mia: How are you doing today, my friend?\nLeo: How are you doing today, my friend?", Reason::LowQuality),
            (
                "Mia: How do I bake bread?\nLeo: Certainly! Here are some steps: 1. Mix flour and water 2. Knead it. I hope this helps!",
                Reason::AssistantStyle,
            ),
            (
                "Mia: What is rain?\nLeo: As an AI language model, I can explain that rain is water falling from clouds.",
                Reason::AssistantStyle,
            ),
        ];
        for (doc, want) in cases {
            assert_eq!(verdict(doc), Some(want), "{doc}");
        }
    }

    #[test]
    fn duplicates_and_test_conversations_are_left_out() {
        let mut f = ConversationFilter::new();
        let doc = "Mia: Do you want to go to the park later?\nLeo: Sure, what time works for you?";
        assert!(f.keep(doc));
        assert!(!f.keep("Mia: do you want to go to the park later??\nLeo:  Sure, what time works for you?"));
        assert_eq!(f.report.duplicates, 1);
        // A conversation holding one of Yendory's context tests.
        let t = &yendory_engine::learn::listening::context_tests(1)[0];
        let fact = &t.a.turns[0].1;
        let leaked =
            format!("Mia: {fact}\nLeo: Okay, good to know.\nMia: {}\nLeo: I think so.", t.a.question);
        assert!(!f.keep(&leaked));
        assert_eq!(f.report.overlaps_tests, 1);
        assert_eq!((f.report.read, f.report.kept), (3, 1));
        assert!(
            f.report.summary().starts_with("kept 1 of 3 conversations (left out:"),
            "{}",
            f.report.summary()
        );
    }

    /// LMSYS-Chat-1M rows: its own flags are honoured before anything else.
    #[test]
    fn lmsys_rows_are_screened_by_their_flags() {
        let screen = crate::sources::screen_of("lmsys-chat").expect("LMSYS is screened");
        let row = |language: &str, flagged: bool, redacted: bool| {
            json!({
                "conversation_id": "x", "model": "vicuna-13b", "turn": 2, "language": language, "redacted": redacted,
                "conversation": [
                    {"role": "user", "content": "Hi! How are you today?"},
                    {"role": "assistant", "content": "I'm good, thanks. How about you?"},
                    {"role": "user", "content": "Pretty tired. I worked late."},
                    {"role": "assistant", "content": "Oh no. Get some rest tonight."}
                ],
                "openai_moderation": [
                    {"flagged": false}, {"flagged": flagged}, {"flagged": false}, {"flagged": false}
                ]
            })
        };
        let mut f = ConversationFilter::new();
        let taken = |f: &mut ConversationFilter, r| f.take_all(&r, Some(screen), whole_dialogue);
        assert_eq!(taken(&mut f, row("English", false, false)).iter().filter(|t| !t.screened).count(), 1);
        // Flagged rows are kept apart (for when that filter is switched off), not dropped.
        for r in [row("Portuguese", false, false), row("English", true, false), row("English", false, true)] {
            let t = taken(&mut f, r);
            assert!(t.len() == 1 && t[0].screened);
        }
        assert_eq!((f.report.read, f.report.kept, f.report.not_english, f.report.flagged), (4, 1, 1, 2));
    }

    /// At practice: the chosen reasons only, the dataset's flagged conversations only when that
    /// filter is off, duplicates counted whether or not they are left out, the tests always left
    /// out, and no more than the size asked for.
    #[test]
    fn practice_uses_the_chosen_filters_and_size() {
        let natural = "Mia: Hi! How was your weekend?\nLeo: Great, I went hiking with my cousin.\nMia: Where did you go?\nLeo: Up the hill behind town.";
        let assistant = "Mia: How do I bake bread?\nLeo: Certainly! Here are some steps: 1. Mix flour and water 2. Knead it. I hope this helps!";
        let long = format!(
            "Mia: Tell me about your trip.\nLeo: {}",
            "We walked far and saw many birds. ".repeat(30)
        );
        let t = &yendory_engine::learn::listening::context_tests(1)[0];
        let leaked = format!(
            "Mia: {}\nLeo: Okay, good to know.\nMia: {}\nLeo: I think so.",
            t.a.turns[0].1, t.a.question
        );
        let flagged = "Ana: Oi, tudo bem com voce hoje?\nBea: Tudo otimo, obrigada por perguntar.";
        let docs = [natural, assistant, &long, natural, &leaked];
        let all_on = FilterChoice::default();
        let (kept, r) = select(docs, [flagged], &all_on, None);
        assert_eq!(kept, [natural]);
        assert_eq!(
            (r.assistant_style, r.too_long, r.duplicates, r.overlaps_tests, r.flagged),
            (1, 1, 1, 1, 1)
        );
        let loose = FilterChoice {
            assistant_style: false,
            too_long: false,
            duplicates: false,
            flagged: false,
            not_english: false,
            ..all_on.clone()
        };
        let (kept, _) = select(docs, [flagged], &loose, None);
        assert_eq!(
            kept,
            [natural, assistant, &long, natural, flagged],
            "the test conversation is always left out"
        );
        let (kept, _) = select(docs, [flagged], &loose, Some(10));
        assert_eq!(kept, [natural], "stops at the size asked for");
    }
}
