//! Built-in example conversations for the Conversation stage (`conversation_examples.txt`).
//!
//! For each kind of conversation (greetings and greeting again, remembering what someone said,
//! who said what, the same question in other words, answering the latest message, changing the
//! topic, misunderstandings, ...) there are three different conversations at each level of the
//! curriculum: short exchanges, a few turns and whole conversations. They are written by hand,
//! so the three differ in what happens, not only in words. Nova is one of the two speakers and
//! every one of her turns is trained (with the conversation before it, as in
//! `conversation::converse`).
//!
//! They are practice only: none of them is a test, and they use none of the names, words or
//! wordings of the context and understanding tests (`learn::listening`), so those tests still
//! measure what Nova does with conversations she has never seen.
use super::conversation::{parse_dialogue, rename, Dialogue};

/// The text of the examples: `# kind`, `## level` (short, a few turns, whole) and conversations
/// of `Name: text` lines separated by blank lines; Nova's lines are `Nova: ...`.
const TEXT: &str = include_str!("conversation_examples.txt");

/// The levels, in the order of the curriculum's stages.
pub const LEVELS: [&str; 3] = ["short", "a few turns", "whole"];

/// One example conversation.
#[derive(Clone, Debug)]
pub struct Example {
    /// The kind of conversation (e.g. "Who said what").
    pub kind: String,
    /// Index into `LEVELS` (and the curriculum's stages).
    pub level: usize,
    pub dialogue: Dialogue,
}

/// The example conversations, with Nova's speaker (and her name in the text) called `nova`.
pub fn examples(nova: &str) -> Vec<Example> {
    let mut out = Vec::new();
    let (mut kind, mut level) = (String::new(), 0);
    for block in TEXT.split("\n\n") {
        let mut lines = Vec::new();
        for line in block.lines() {
            if let Some(k) = line.strip_prefix("# ") {
                kind = k.trim().to_string();
            } else if let Some(l) = line.strip_prefix("## ") {
                level = LEVELS.iter().position(|x| *x == l.trim()).unwrap_or(0);
            } else if !line.trim().is_empty() {
                lines.push(rename(line, "Nova", nova));
            }
        }
        if let Some(dialogue) = parse_dialogue(&lines.join("\n")) {
            out.push(Example { kind: kind.clone(), level, dialogue });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    fn words(text: &str) -> BTreeSet<String> {
        text.split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .map(|w| w.to_ascii_lowercase())
            .collect()
    }

    /// Three different conversations for every kind at every level, of the right length, with
    /// Nova as one speaker and the speakers taking turns.
    #[test]
    fn three_different_conversations_per_kind_and_level() {
        let all = examples("Nova");
        let mut per: BTreeMap<(String, usize), Vec<&Example>> = BTreeMap::new();
        for e in &all {
            per.entry((e.kind.clone(), e.level)).or_default().push(e);
        }
        let kinds: BTreeSet<&String> = all.iter().map(|e| &e.kind).collect();
        assert!(kinds.len() >= 12, "{kinds:?}");
        assert_eq!(per.len(), kinds.len() * LEVELS.len(), "every kind has every level");
        for ((kind, level), list) in &per {
            assert_eq!(list.len(), 3, "{kind} / {}", LEVELS[*level]);
            for e in list {
                let d = &e.dialogue;
                let turns = d.turns.len();
                let fits = match level {
                    0 => (2..=4).contains(&turns),
                    1 => (5..=8).contains(&turns),
                    _ => (10..=18).contains(&turns),
                };
                assert!(fits, "{kind} / {}: {turns} turns: {:?}", LEVELS[*level], d.turns);
                assert!(d.speakers.contains(&"Nova".to_string()), "{:?}", d.turns);
                assert!(d.speakers.windows(2).all(|w| w[0] != w[1]), "turns alternate: {:?}", d.turns);
            }
            // Different conversations, not the same one with a few words changed: no two share
            // more than a third of their words.
            for i in 0..list.len() {
                for j in i + 1..list.len() {
                    let (a, b) =
                        (words(&list[i].dialogue.turns.join(" ")), words(&list[j].dialogue.turns.join(" ")));
                    let shared = a.intersection(&b).count() as f32 / a.union(&b).count() as f32;
                    assert!(
                        shared < 0.34,
                        "{kind}: {:?} / {:?} share {shared:.2}",
                        list[i].dialogue.turns[0],
                        list[j].dialogue.turns[0]
                    );
                }
            }
        }
        // The name given to Nova replaces hers.
        let other = examples("Ivo");
        assert!(other.iter().all(|e| !e.dialogue.speakers.contains(&"Nova".to_string())));
        assert!(other.iter().any(|e| e.dialogue.turns.iter().any(|t| t.contains("I'm Ivo"))));
    }

    /// No example uses a word or a wording of the tests, so the tests stay unseen.
    #[test]
    fn examples_never_use_the_tests_words() {
        let text: String = examples("Nova").iter().map(|e| e.dialogue.turns.join("\n") + "\n").collect();
        let used = words(&text);
        let lower = text.to_ascii_lowercase();
        for t in super::super::listening::context_tests(12)
            .into_iter()
            .chain(super::super::listening::understanding_tests(12, "Nova"))
        {
            for p in [&t.a, &t.b] {
                for w in p.expected.iter().chain(&p.wrong) {
                    // Words that only say "I do not know" or greet are allowed: they are how
                    // anyone talks, not the tests' facts.
                    let w = w.to_ascii_lowercase();
                    let common = super::super::listening::UNSURE.contains(&w.as_str())
                        || [
                            "where",
                            "hello",
                            "hi",
                            "hey",
                            "morning",
                            "good",
                            "fine",
                            "well",
                            "great",
                            "okay",
                            "alright",
                            "happy",
                            "nova",
                            "welcome",
                            "good morning",
                            "not bad",
                            "good to see you",
                            "nice to see you",
                            "evening",
                        ]
                        .contains(&w.as_str());
                    if !common {
                        assert!(!used.contains(&w), "{w} is in the examples");
                    }
                }
                let question = p.question.to_ascii_lowercase();
                assert!(!lower.contains(&question), "{question}");
            }
        }
    }
}
