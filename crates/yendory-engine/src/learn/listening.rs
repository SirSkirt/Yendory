//! Listening: whether Nova uses what was said earlier in a conversation.
//!
//! **Context-sensitivity tests.** Each test is a pair of conversations that differ in one earlier
//! fact (where someone went, a name, a correction, ...) and end with the same question. Nova
//! passes a pair only when its answer follows the fact in *both*: "kitchen" after "Daniel went to
//! the kitchen", "hallway" after "Daniel went to the hallway". What matters is that the answer
//! changes when the earlier information changes, not one wording: a reply is right when it
//! contains the expected word and not the other version's.
//!
//! Seven kinds: names, locations, changing facts, corrections, references (she / he / it),
//! follow-up questions, and uncertainty (asked about something never said: any honest "I do not
//! know" is right, a guess is not). Filler exchanges sit between the fact and the question, so the
//! fact is several turns back.
//!
//! **Practice of the same skills** is generated from *different* word lists (names, places,
//! things), so the tests measure whether the skill generalises, not whether Nova remembers the
//! test conversations. Test conversations are never trained on.
use super::conversation::Dialogue;
use serde::{Deserialize, Serialize};

/// Words a set of conversations is built from.
struct Words {
    people: (&'static [&'static str], &'static [&'static str]),
    places: &'static [&'static str],
    things: &'static [&'static str],
    colours: &'static [&'static str],
    pets: &'static [&'static str],
}

/// The tests' words (never in the practice).
const TEST: Words = Words {
    people: (
        &["Daniel", "Marcus", "Oliver", "Victor", "Hugo", "Felix", "Simon", "Arthur"],
        &["Quinn", "Nadia", "Rosa", "Clara", "Hanna", "Iris", "Julia", "Lena"],
    ),
    places: &[
        "kitchen", "hallway", "garden", "office", "garage", "bedroom", "library", "bakery", "station",
        "harbor",
    ],
    things: &["umbrella", "violin", "lantern", "basket", "compass", "notebook", "helmet", "teapot"],
    colours: &["red", "blue", "green", "yellow", "purple", "orange", "white", "black"],
    pets: &["parrot", "hamster", "turtle", "rabbit", "goldfish", "kitten"],
};

/// The practice's words (never in the tests).
const PRACTICE: Words = Words {
    people: (
        &["Peter", "George", "Henry", "Leo", "Samuel", "Thomas", "Walter", "Edgar", "Louis", "Martin"],
        &["Emma", "Sophie", "Grace", "Alice", "Maya", "Nora", "Ruth", "Ella", "Diana", "Paula"],
    ),
    places: &[
        "school", "market", "beach", "park", "museum", "church", "farm", "cinema", "lake", "forest", "shop",
        "zoo",
    ],
    things: &["ball", "kite", "scarf", "camera", "bucket", "ladder", "radio", "pillow", "candle", "wallet"],
    colours: &["pink", "brown", "gray", "gold", "silver", "navy", "beige", "lilac"],
    pets: &["dog", "cat", "pony", "duck", "lizard", "mouse"],
};

/// Exchanges with no information, put between a fact and the question.
const FILLER: [(&str, &str); 8] = [
    ("It is a nice day today.", "Yes, it is."),
    ("I had tea this morning.", "That sounds nice."),
    ("The bus was late again.", "Oh no."),
    ("I like this song.", "It is a good song."),
    ("My shoes are new.", "They look good."),
    ("It might rain later.", "Maybe it will."),
    ("I am a bit tired.", "You should rest."),
    ("The shop was busy.", "It often is."),
];

/// The turns Nova says between facts (they carry no information).
const ACKS: [&str; 4] = ["Okay.", "I see.", "Got it.", "Alright."];

/// Accepted ways of saying "I do not know".
pub const UNSURE: [&str; 9] = [
    "don't know",
    "do not know",
    "not sure",
    "didn't say",
    "did not say",
    "didn't tell",
    "did not tell",
    "no idea",
    "can't tell",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Skill {
    Names,
    Locations,
    ChangingFacts,
    Corrections,
    References,
    FollowUps,
    Uncertainty,
}

impl Skill {
    pub const ALL: [Skill; 7] = [
        Skill::Names,
        Skill::Locations,
        Skill::ChangingFacts,
        Skill::Corrections,
        Skill::References,
        Skill::FollowUps,
        Skill::Uncertainty,
    ];
}

/// One conversation of a test: the turns before the question (`(true, text)` is Nova's), the
/// question, what a right answer contains and what the other version's answer contains.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Probe {
    pub turns: Vec<(bool, String)>,
    pub question: String,
    /// Any of these (whole words, any case) makes the answer right...
    pub expected: Vec<String>,
    /// ...unless it also says one of these (the other version's answer).
    pub wrong: Vec<String>,
    /// The answer a practice conversation teaches (not used to grade).
    pub answer: String,
}

impl Probe {
    /// The prompt as Nova reads it, ending `N: `.
    pub fn prompt(&self) -> String {
        let mut s = String::new();
        for (mine, t) in &self.turns {
            s.push_str(if *mine { "N: " } else { "U: " });
            s.push_str(t);
            s.push('\n');
        }
        s.push_str(&format!("U: {}\nN: ", self.question));
        s
    }
    /// Whether a reply is right: it says an expected answer and not the other version's.
    pub fn right(&self, reply: &str) -> bool {
        let r = reply.to_ascii_lowercase();
        let says = |w: &String| contains_words(&r, &w.to_ascii_lowercase());
        self.expected.iter().any(says) && !self.wrong.iter().any(says)
    }
}

/// `phrase` occurs in `text` as whole words.
fn contains_words(text: &str, phrase: &str) -> bool {
    let mut from = 0;
    while let Some(i) = text[from..].find(phrase) {
        let at = from + i;
        let before = text[..at].chars().next_back();
        let after = text[at + phrase.len()..].chars().next();
        let word = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric());
        if !word(before) && !word(after) {
            return true;
        }
        from = at + phrase.len().max(1);
    }
    false
}

/// A pair of conversations differing in one earlier fact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextTest {
    pub skill: Skill,
    pub a: Probe,
    pub b: Probe,
}

struct Gen(u64);
impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn pick(&mut self, xs: &[&'static str]) -> &'static str {
        xs[self.below(xs.len())]
    }
    fn two(&mut self, xs: &[&'static str]) -> (&'static str, &'static str) {
        let a = self.below(xs.len());
        let b = (a + 1 + self.below(xs.len() - 1)) % xs.len();
        (xs[a], xs[b])
    }
}

fn person(g: &mut Gen, w: &Words) -> (&'static str, &'static str) {
    if g.below(2) == 0 {
        (g.pick(w.people.0), "he")
    } else {
        (g.pick(w.people.1), "she")
    }
}

/// The two versions of one conversation: `fact(x)` is told with `x` and `y` swapped.
fn pair(
    g: &mut Gen,
    skill: Skill,
    build: impl Fn(&str, &str, &mut Vec<(bool, String)>) -> (String, String),
    x: &'static str,
    y: &'static str,
    fillers: usize,
) -> ContextTest {
    let ack = ACKS[g.below(ACKS.len())];
    let fill: Vec<(&str, &str)> = (0..fillers).map(|_| FILLER[g.below(FILLER.len())]).collect();
    let one = |x: &str, y: &str| {
        let mut turns = Vec::new();
        let (question, answer) = build(x, y, &mut turns);
        turns.push((true, ack.to_string()));
        for (u, n) in &fill {
            turns.push((false, u.to_string()));
            turns.push((true, n.to_string()));
        }
        Probe { turns, question, expected: vec![x.to_string()], wrong: vec![y.to_string()], answer }
    };
    ContextTest { skill, a: one(x, y), b: one(y, x) }
}

fn make(g: &mut Gen, skill: Skill, w: &Words) -> ContextTest {
    let fillers = g.below(3);
    match skill {
        Skill::Locations => {
            let (p, _) = person(g, w);
            let (x, y) = g.two(w.places);
            pair(
                g,
                skill,
                |x, _, t| {
                    t.push((false, format!("{p} went to the {x}.")));
                    (format!("Where is {p}?"), format!("In the {x}."))
                },
                x,
                y,
                fillers,
            )
        }
        Skill::Names => {
            let (x, y) = g.two(w.people.1);
            let pet = g.pick(w.pets);
            pair(
                g,
                skill,
                |x, _, t| {
                    t.push((false, format!("My {pet} is called {x}.")));
                    (format!("What is my {pet} called?"), format!("Your {pet} is called {x}."))
                },
                x,
                y,
                fillers,
            )
        }
        Skill::ChangingFacts => {
            let thing = g.pick(w.things);
            let (x, y) = g.two(w.colours);
            let mut test = pair(
                g,
                skill,
                |x, y, t| {
                    t.push((false, format!("My {thing} was {y}.")));
                    t.push((true, ACKS[0].to_string()));
                    t.push((false, format!("I painted it, and now it is {x}.")));
                    (format!("What color is my {thing} now?"), format!("It is {x} now."))
                },
                x,
                y,
                fillers,
            );
            // Both colours are in both conversations; only the order says which is current.
            test.a.wrong = vec![y.to_string()];
            test.b.wrong = vec![x.to_string()];
            test
        }
        Skill::Corrections => {
            let (p, _) = person(g, w);
            let (x, y) = g.two(w.places);
            let mut test = pair(
                g,
                skill,
                |x, y, t| {
                    t.push((false, format!("{p} is at the {y}. Sorry, I mean the {x}.")));
                    (format!("Where is {p}?"), format!("At the {x}."))
                },
                x,
                y,
                fillers,
            );
            test.a.wrong = vec![y.to_string()];
            test.b.wrong = vec![x.to_string()];
            test
        }
        Skill::References => {
            let she = g.pick(w.people.1);
            let he = g.pick(w.people.0);
            let (x, y) = g.two(w.things);
            let ask_her = g.below(2) == 0;
            pair(
                g,
                skill,
                |x, y, t| {
                    // The asked-about person's thing is x.
                    let (hers, his) = if ask_her { (x, y) } else { (y, x) };
                    t.push((false, format!("{she} has a {hers}, and {he} has a {his}.")));
                    let (pron, verb) = if ask_her { ("she", "does") } else { ("he", "does") };
                    (format!("What {verb} {pron} have?"), format!("A {x}."))
                },
                x,
                y,
                fillers,
            )
        }
        Skill::FollowUps => {
            let pet = g.pick(w.pets);
            let (x, y) = g.two(w.people.0);
            pair(
                g,
                skill,
                |x, _, t| {
                    t.push((false, format!("I got a {pet} last week.")));
                    t.push((true, "What is its name?".to_string()));
                    t.push((false, format!("Its name is {x}.")));
                    (format!("Do you remember my {pet}'s name?"), format!("Yes, it is {x}."))
                },
                x,
                y,
                fillers,
            )
        }
        Skill::Uncertainty => {
            let (p, _) = person(g, w);
            let (q, _) = person(g, w);
            let q = if q == p {
                if p == w.people.0[0] {
                    w.people.0[1]
                } else {
                    w.people.0[0]
                }
            } else {
                q
            };
            let (x, y) = g.two(w.places);
            let mut test = pair(
                g,
                skill,
                |x, _, t| {
                    t.push((false, format!("{p} went to the {x}.")));
                    (format!("Where is {q}?"), format!("I do not know. You did not tell me where {q} is."))
                },
                x,
                y,
                fillers,
            );
            // Right: any "I do not know"; wrong: naming either place (a guess).
            for (probe, (x, y)) in [(&mut test.a, (x, y)), (&mut test.b, (y, x))] {
                probe.expected = UNSURE.iter().map(|s| s.to_string()).collect();
                probe.wrong = vec![x.to_string(), y.to_string()];
            }
            test
        }
    }
}

/// The context-sensitivity tests: `per_skill` pairs of each kind, the same every time.
pub fn context_tests(per_skill: usize) -> Vec<ContextTest> {
    let mut g = Gen(0x7E57_5EED);
    let mut out = Vec::new();
    for skill in Skill::ALL {
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..per_skill * 20 {
            if seen.len() == per_skill {
                break;
            }
            let t = make(&mut g, skill, &TEST);
            if seen.insert(t.a.prompt()) {
                out.push(t);
            }
        }
    }
    out
}

/// Practice conversations of the same skills from different words: `n` dialogues (each test
/// version becomes one dialogue whose last turn is Nova's answer), Nova's speaker called `nova`.
pub fn context_practice(n: usize, seed: u64, nova: &str) -> Vec<Dialogue> {
    let mut g = Gen(seed ^ 0xC0_7E_47);
    let mut out = Vec::new();
    while out.len() < n {
        let skill = Skill::ALL[g.below(Skill::ALL.len())];
        let t = make(&mut g, skill, &PRACTICE);
        for p in [t.a, t.b] {
            let mut speakers = Vec::new();
            let mut turns = Vec::new();
            for (mine, text) in p.turns {
                speakers.push(if mine { nova.to_string() } else { "Friend".to_string() });
                turns.push(text);
            }
            speakers.push("Friend".into());
            turns.push(p.question);
            speakers.push(nova.to_string());
            turns.push(p.answer);
            // Only the answer is trained: the turns before it are the teacher's script.
            let mut trainable = vec![false; turns.len()];
            *trainable.last_mut().unwrap() = true;
            out.push(Dialogue { situation: String::new(), speakers, turns, trainable });
        }
    }
    out.truncate(n);
    out
}

/// Results of the context tests on one model.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextScore {
    /// Pairs passed (both versions right), and pairs.
    pub passed: usize,
    pub total: usize,
    /// Pairs whose two answers differ in the way the facts do (each says its own version's
    /// word), right or not elsewhere.
    pub changed: usize,
    /// Single conversations answered right.
    pub right: usize,
    /// Per skill: (passed, total).
    pub skills: Vec<(Skill, usize, usize)>,
    /// A few answers, for reading.
    pub examples: Vec<(String, String, bool)>,
}

/// Scores replies to the tests (`replies[i]` answers `tests[i / 2]`, version a then b).
pub fn score(tests: &[ContextTest], replies: &[String]) -> ContextScore {
    let mut s = ContextScore { total: tests.len(), ..ContextScore::default() };
    let mut per: std::collections::BTreeMap<Skill, (usize, usize)> = Default::default();
    for (i, t) in tests.iter().enumerate() {
        let (ra, rb) = (&replies[2 * i], &replies[2 * i + 1]);
        let (a, b) = (t.a.right(ra), t.b.right(rb));
        s.right += a as usize + b as usize;
        let mentions = |r: &str, p: &Probe| {
            p.expected.iter().any(|w| contains_words(&r.to_ascii_lowercase(), &w.to_ascii_lowercase()))
        };
        if mentions(ra, &t.a) && mentions(rb, &t.b) {
            s.changed += 1;
        }
        let e = per.entry(t.skill).or_default();
        e.1 += 1;
        if a && b {
            s.passed += 1;
            e.0 += 1;
        }
        if s.examples.len() < 4 && i % (tests.len() / 4).max(1) == 0 {
            s.examples.push((t.a.prompt(), ra.clone(), a));
        }
    }
    s.skills = per.into_iter().map(|(k, (p, t))| (k, p, t)).collect();
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_differ_in_one_fact_and_are_graded_by_meaning() {
        let tests = context_tests(6);
        assert_eq!(tests.len(), 6 * Skill::ALL.len());
        assert_eq!(tests, context_tests(6), "the same every time");
        let loc = tests.iter().find(|t| t.skill == Skill::Locations).unwrap();
        assert_ne!(loc.a.prompt(), loc.b.prompt());
        assert_eq!(loc.a.question, loc.b.question);
        let (x, y) = (&loc.a.expected[0], &loc.b.expected[0]);
        assert!(loc.a.right(&format!("In the {x}.")) && loc.a.right(&format!("I think {x}")));
        assert!(!loc.a.right(&format!("In the {y}.")) && !loc.a.right(&format!("The {x} or the {y}.")));
        let unsure = tests.iter().find(|t| t.skill == Skill::Uncertainty).unwrap();
        assert!(unsure.a.right("I don't know.") && unsure.a.right("I am not sure, you did not say."));
        assert!(!unsure.a.right(&format!("In the {}.", unsure.a.wrong[0])));
        // Corrections: the corrected place is right, the first one wrong.
        let c = tests.iter().find(|t| t.skill == Skill::Corrections).unwrap();
        assert!(c.a.prompt().contains("Sorry, I mean the"));
        assert!(c.a.right(&format!("At the {}.", c.a.expected[0])));
        assert!(!c.a.right(&format!("At the {}.", c.a.wrong[0])));
        // Scoring: a pair passes only when both versions are right.
        let mut replies = Vec::new();
        for t in &tests {
            replies.push(t.a.answer.clone());
            replies.push(t.a.answer.clone()); // the same answer to both: never passes (except "unsure")
        }
        let s = score(&tests, &replies);
        let unsure_pairs = tests.iter().filter(|t| t.skill == Skill::Uncertainty).count();
        assert_eq!(s.passed, unsure_pairs, "a fixed answer passes only where the right answer is the same");
        let right: Vec<String> =
            tests.iter().flat_map(|t| [t.a.answer.clone(), t.b.answer.clone()]).collect();
        let s = score(&tests, &right);
        assert_eq!(
            (s.passed, s.total),
            (tests.len(), tests.len()),
            "{:?}",
            tests.iter().zip(right.chunks(2)).find(|(t, r)| !(t.a.right(&r[0]) && t.b.right(&r[1])))
        );
    }

    #[test]
    fn practice_uses_other_words_and_trains_only_the_answer() {
        let tests = context_tests(12);
        let test_text: String = tests.iter().map(|t| t.a.prompt() + &t.b.prompt()).collect();
        let practice = context_practice(400, 9, "Nova");
        assert_eq!(practice.len(), 400);
        for d in &practice {
            assert_eq!(d.trainable.iter().filter(|&&t| t).count(), 1);
            assert_eq!(d.speakers.last().unwrap(), "Nova");
            // No practice conversation is a test conversation.
            let last_question = &d.turns[d.turns.len() - 2];
            assert!(
                !test_text.contains(&d.turns[0]) || FILLER.iter().any(|f| f.0 == d.turns[0]),
                "{last_question}"
            );
        }
        for w in TEST
            .people
            .0
            .iter()
            .chain(TEST.people.1)
            .chain(TEST.places)
            .chain(TEST.things)
            .chain(TEST.pets)
            .chain(TEST.colours)
        {
            assert!(
                !PRACTICE.people.0.contains(w)
                    && !PRACTICE.people.1.contains(w)
                    && !PRACTICE.places.contains(w)
                    && !PRACTICE.things.contains(w)
                    && !PRACTICE.pets.contains(w)
                    && !PRACTICE.colours.contains(w),
                "{w} is in both"
            );
        }
    }
}
