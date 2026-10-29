//! Quality filter for downloaded conversation datasets: what Nova practises with is fewer, better
//! conversations, not as many as possible. Every conversation read is either kept or counted
//! under the first reason it was left out, and the counts are shown during the download and kept
//! with the dataset.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// The filter's state through one download.
pub struct ConversationFilter {
    pub report: FilterReport,
    seen: HashSet<u64>,
    tests: Vec<TestSignature>,
}

impl Default for ConversationFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationFilter {
    pub fn new() -> Self {
        ConversationFilter { report: FilterReport::default(), seen: HashSet::new(), tests: test_signatures() }
    }

    /// One dataset row: screened by the dataset's own flags (`screen`), converted, then checked.
    /// Returns the conversations kept.
    pub fn take(&mut self, row: &Value, screen: Option<Screen>, convert: Convert) -> Vec<String> {
        if let Some(why) = screen.and_then(|s| s(row)) {
            self.report.read += 1;
            self.report.count(why);
            return Vec::new();
        }
        let docs = convert(row);
        if docs.is_empty() {
            self.report.read += 1;
            self.report.count(Reason::Malformed);
            return Vec::new();
        }
        docs.into_iter().filter(|d| self.keep(d)).collect()
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

    fn verdict(&mut self, doc: &str) -> Option<Reason> {
        let why = reason(doc);
        if why.is_some() {
            return why;
        }
        let text = normal(doc);
        if self.tests.iter().any(|t| text.contains(&t.question) && text.contains(&t.fact)) {
            return Some(Reason::OverlapsTests);
        }
        // Duplicates: the same words in the same order.
        let key: String = text.chars().filter(|c| c.is_ascii_alphanumeric() || *c == ' ').collect();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&key, &mut h);
        if !self.seen.insert(std::hash::Hasher::finish(&h)) {
            return Some(Reason::Duplicate);
        }
        None
    }
}

/// Why a conversation is left out, apart from the tests and duplicates (`None`: it is kept).
pub fn reason(doc: &str) -> Option<Reason> {
    let Some(d) = parse_dialogue(doc) else { return Some(Reason::Malformed) };
    if d.turns.iter().any(|t| !t.chars().any(|c| c.is_ascii_alphabetic())) {
        return Some(Reason::Malformed);
    }
    if d.speakers.windows(2).any(|w| w[0] == w[1]) {
        return Some(Reason::TurnOrder);
    }
    if d.turns.iter().any(|t| t.len() > LONGEST || t.contains("```")) {
        return Some(Reason::TooLong);
    }
    let all = d.turns.join("\n");
    if !english(&all) {
        return Some(Reason::NotEnglish);
    }
    let lower = all.to_ascii_lowercase();
    let words: Vec<&str> =
        lower.split(|c: char| !c.is_ascii_alphanumeric() && c != '\'').filter(|w| !w.is_empty()).collect();
    if words.iter().any(|w| UNSAFE.contains(w)) || UNSAFE_PHRASES.iter().any(|p| lower.contains(p)) {
        return Some(Reason::Unsafe);
    }
    if low_quality(&d.turns, &lower, words.len()) {
        return Some(Reason::LowQuality);
    }
    if assistant_style(&d.speakers, &d.turns, &lower) {
        return Some(Reason::AssistantStyle);
    }
    None
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
        assert_eq!(f.take(&row("English", false, false), Some(screen), whole_dialogue).len(), 1);
        assert!(f.take(&row("Portuguese", false, false), Some(screen), whole_dialogue).is_empty());
        assert!(f.take(&row("English", true, false), Some(screen), whole_dialogue).is_empty());
        assert!(f.take(&row("English", false, true), Some(screen), whole_dialogue).is_empty());
        assert_eq!((f.report.read, f.report.kept, f.report.not_english, f.report.flagged), (4, 1, 1, 2));
    }
}
