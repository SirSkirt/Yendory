//! Answer keys for course lessons and exams: deterministic, lightweight grading rules that are
//! more flexible than exact matching without guessing at meaning.
//!
//! An answer in a course lesson or exam may list several accepted forms and rules, separated by
//! `|`:
//!
//! ```text
//! What is a cat?  animal | creature | !plant
//! Is a robin an animal?  Yes, a robin is an animal. | A robin is an animal. | must: robin animal | never: not
//! ```
//!
//! The first accepted form is the training target; every form is accepted when grading.
//!
//! * **Keyword** grading (Dense lessons and the Dense exam section): the reply is correct when it
//!   contains an accepted keyword (or keyword phrase) as whole words, not negated, and no
//!   conflicting keyword that is not negated. Conflicts are the `!word` entries plus, by default,
//!   the accepted keywords of the other questions in the same lesson ("plant" when another
//!   question's answer is "plant"). "Cat animal" and "animal, not plant" pass; "animal plant" and
//!   "not an animal" fail.
//! * **Sentence** grading (Transformer and Co-op): the reply must equal an accepted form after
//!   normalising case, punctuation, spacing and contractions ("isn't" = "is not"), or satisfy the
//!   question's `must:` words while containing none of its `never:` words. Word overlap alone
//!   never gives credit, so "The door is not open" never matches "The door is open".
//!
//! Items without a key (ordinary Classroom lessons) are graded by exact match, as in the
//! reference evaluation.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradeMode {
    /// Words, meanings and associations: an accepted keyword, no conflicting one.
    Keyword,
    /// Sentence construction: an accepted form (normalised), or the question's word rules.
    Sentence,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnswerKey {
    pub mode: GradeMode,
    /// Accepted answers as written (the first is the training target).
    pub accepted: Vec<String>,
    /// Keyword mode: keywords that make a reply wrong unless negated.
    #[serde(default)]
    pub conflicts: Vec<String>,
    /// Sentence mode: a reply containing all of these words (and none of `never`) is correct.
    #[serde(default)]
    pub must: Vec<String>,
    #[serde(default)]
    pub never: Vec<String>,
}

/// Words that negate the keyword after them (within two words).
const NEGATIONS: [&str; 5] = ["not", "no", "never", "nor", "without"];

/// Lowercase words with contractions expanded and punctuation removed.
pub fn words(text: &str) -> Vec<String> {
    let mut s = text.to_ascii_lowercase();
    for (from, to) in [
        ("can't", "can not"),
        ("won't", "will not"),
        ("n't", " not"),
        ("'re", " are"),
        ("'m", " am"),
        ("'ll", " will"),
        ("'ve", " have"),
    ] {
        s = s.replace(from, to);
    }
    s = s.replace('\'', "");
    s.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_string).collect()
}

/// The normalised form compared in sentence grading.
pub fn normalize(text: &str) -> String {
    words(text).join(" ")
}

/// Positions where `phrase` occurs in `w` as whole words.
fn occurrences(w: &[String], phrase: &[String]) -> Vec<usize> {
    if phrase.is_empty() || phrase.len() > w.len() {
        return Vec::new();
    }
    (0..=w.len() - phrase.len()).filter(|&i| w[i..i + phrase.len()] == *phrase).collect()
}

fn negated(w: &[String], at: usize) -> bool {
    w[at.saturating_sub(2)..at].iter().any(|x| NEGATIONS.contains(&x.as_str()))
}

impl AnswerKey {
    /// Parses an answer written with `|`-separated forms and rules. Returns None when no
    /// accepted form is given.
    pub fn parse(spec: &str, mode: GradeMode) -> Option<AnswerKey> {
        let mut key = AnswerKey {
            mode,
            accepted: Vec::new(),
            conflicts: Vec::new(),
            must: Vec::new(),
            never: Vec::new(),
        };
        for part in spec.split('|').map(str::trim).filter(|p| !p.is_empty()) {
            if let Some(c) = part.strip_prefix('!') {
                key.conflicts.push(c.trim().to_string());
            } else if let Some(m) = part.strip_prefix("must:") {
                key.must.extend(words(m));
            } else if let Some(n) = part.strip_prefix("never:") {
                key.never.extend(words(n));
            } else {
                key.accepted.push(part.to_string());
            }
        }
        (!key.accepted.is_empty()).then_some(key)
    }

    /// The training target: the first accepted form.
    pub fn target(&self) -> &str {
        &self.accepted[0]
    }

    pub fn is_correct(&self, reply: &str) -> bool {
        let w = words(reply);
        match self.mode {
            GradeMode::Sentence => {
                let norm = w.join(" ");
                if self.accepted.iter().any(|a| normalize(a) == norm && !norm.is_empty()) {
                    return true;
                }
                !self.must.is_empty()
                    && self.must.iter().all(|m| w.contains(m))
                    && !self.never.iter().any(|n| w.contains(n))
            }
            GradeMode::Keyword => {
                let found = self.accepted.iter().any(|a| {
                    let p = words(a);
                    occurrences(&w, &p).into_iter().any(|i| !negated(&w, i))
                });
                found
                    && !self.conflicts.iter().any(|c| {
                        let p = words(c);
                        occurrences(&w, &p).into_iter().any(|i| !negated(&w, i))
                    })
            }
        }
    }
}

/// Keyword grading: adds, as conflicts of each key, the accepted keywords of the other keys
/// (short answers of up to three words), leaving out keywords that this key accepts, that
/// contain one of its accepted keywords, or whose words all appear in its question.
pub fn derive_conflicts(keys: &mut [(&str, &mut AnswerKey)]) {
    let pool: Vec<Vec<String>> = keys
        .iter()
        .filter(|(_, k)| k.mode == GradeMode::Keyword)
        .flat_map(|(_, k)| k.accepted.iter().map(|a| words(a)))
        .filter(|p| (1..=3).contains(&p.len()))
        .collect();
    for (question, key) in keys.iter_mut() {
        if key.mode != GradeMode::Keyword {
            continue;
        }
        let q = words(question);
        let own: Vec<Vec<String>> = key.accepted.iter().map(|a| words(a)).collect();
        let mut add = Vec::new();
        for p in &pool {
            let contains_own = own.iter().any(|o| !occurrences(p, o).is_empty());
            let in_question = p.iter().all(|x| q.contains(x));
            let joined = p.join(" ");
            if !contains_own && !in_question && !add.contains(&joined) && !key.conflicts.contains(&joined) {
                add.push(joined);
            }
        }
        key.conflicts.extend(add);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyword_grading() {
        let mut k = AnswerKey::parse("animal | creature", GradeMode::Keyword).unwrap();
        k.conflicts.push("plant".into());
        for (reply, ok) in [
            ("Animal", true),
            ("Cat Animal", true),
            ("a creature.", true),
            ("Animal Plant", false),
            ("Animal, not plant", true),
            ("not an animal", false),
            ("plant", false),
            ("animals", false),
        ] {
            assert_eq!(k.is_correct(reply), ok, "{reply}");
        }
    }

    #[test]
    fn sentence_grading() {
        let k = AnswerKey::parse(
            "Yes, a robin is an animal. | A robin is an animal. | Yes, because a robin is a bird and birds are animals.",
            GradeMode::Sentence,
        )
        .unwrap();
        assert_eq!(k.target(), "Yes, a robin is an animal.");
        assert!(k.is_correct("yes a robin is an animal"));
        assert!(k.is_correct("A robin is an animal!"));
        assert!(!k.is_correct("A robin is not an animal."));
        assert!(!k.is_correct("A robin isn't an animal."));
        let door = AnswerKey::parse("The door is open.", GradeMode::Sentence).unwrap();
        assert!(!door.is_correct("The door is not open."));
        assert!(door.is_correct("the door is open"));
        let rules =
            AnswerKey::parse("A robin is an animal. | must: robin animal | never: not", GradeMode::Sentence)
                .unwrap();
        assert!(rules.is_correct("Of course, the robin is an animal."));
        assert!(!rules.is_correct("The robin is not an animal."));
        assert!(!rules.is_correct("It isn't an animal, the robin."));
        assert!(AnswerKey::parse("!plant", GradeMode::Keyword).is_none());
    }

    #[test]
    fn conflicts_come_from_other_answers() {
        let mut cat = AnswerKey::parse("animal", GradeMode::Keyword).unwrap();
        let mut rose = AnswerKey::parse("plant | flower", GradeMode::Keyword).unwrap();
        let mut meow = AnswerKey::parse("cat", GradeMode::Keyword).unwrap();
        derive_conflicts(&mut [
            ("What is a cat?", &mut cat),
            ("What is a rose?", &mut rose),
            ("Which animal says meow?", &mut meow),
        ]);
        // "cat" is in the question, so "cat animal" stays correct.
        assert_eq!(cat.conflicts, ["plant", "flower"]);
        assert!(cat.is_correct("cat animal"));
        assert!(!cat.is_correct("animal plant"));
        assert!(rose.conflicts.contains(&"animal".to_string()));
        // "animal" is in the question "Which animal says meow?".
        assert_eq!(meow.conflicts, ["plant", "flower"]);
    }
}
