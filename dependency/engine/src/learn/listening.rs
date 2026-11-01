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
//!
//! **Understanding tests** (`understanding_tests`, `Skill::UNDERSTANDING`, since 0.7.8): following
//! the conversation itself. Greeted again (greet back, not "I do not know"), the same question in
//! other words, who said what, the latest message (the same conversation, a different last
//! question), topic changes, misunderstandings put right, and a question asked again. Their
//! practice uses other words *and other wordings* ("What's your name?" is practised, "May I know
//! your name?" only tested). The context tests above are unchanged, so their scores compare with
//! earlier runs.
use super::conversation::Dialogue;
use serde::{Deserialize, Serialize};

/// Words a set of conversations is built from.
struct Words {
    people: (&'static [&'static str], &'static [&'static str]),
    places: &'static [&'static str],
    things: &'static [&'static str],
    colours: &'static [&'static str],
    pets: &'static [&'static str],
    foods: &'static [&'static str],
    relatives: &'static [&'static str],
}

/// Ways of saying things a set of conversations is built from (the tests' wordings are never
/// in the practice), for the understanding skills.
struct Phrases {
    greetings: &'static [&'static str],
    name_questions: &'static [&'static str],
    wellbeing_questions: &'static [&'static str],
    /// Where a relative lives (`{r}` is the relative).
    where_lives: &'static [&'static str],
    /// Asking again for a name just given (`{p}` is the pet).
    again: &'static [&'static str],
    topic_changes: &'static [&'static str],
}

const TEST_PHRASES: Phrases = Phrases {
    greetings: &["Hiya!", "Good evening!", "Hey there!", "Howdy!", "Greetings!", "Good afternoon!"],
    name_questions: &[
        "Can you tell me your name?",
        "What are you called?",
        "May I know your name?",
        "What do people call you?",
        "What do I call you?",
    ],
    wellbeing_questions: &[
        "How have you been?",
        "How are things?",
        "How do you feel today?",
        "How's your day going?",
    ],
    where_lives: &[
        "Where's my {r}'s home?",
        "Can you recall where my {r} lives?",
        "Which place does my {r} live by?",
    ],
    again: &[
        "Remind me, what is my {p}'s name?",
        "Can you say its name once more?",
        "Wait, what was it called?",
    ],
    topic_changes: &[
        "On another note,",
        "Totally unrelated, but",
        "Before I forget,",
        "Speaking of something else,",
    ],
};

const PRACTICE_PHRASES: Phrases = Phrases {
    greetings: &["Hello!", "Hi!", "Hey!", "Good morning!", "Hello there!", "Hi, you!"],
    name_questions: &[
        "What is your name?",
        "What's your name?",
        "Who are you?",
        "What should I call you?",
        "Tell me your name.",
    ],
    wellbeing_questions: &["How are you?", "How are you doing?", "How's it going?", "Are you okay?"],
    where_lives: &[
        "Where does my {r} live?",
        "Do you remember where my {r} lives?",
        "Where did I say my {r} lives?",
    ],
    again: &[
        "Sorry, what was its name again?",
        "What did you say it was called?",
        "What is it called again?",
    ],
    topic_changes: &["Anyway,", "By the way,", "Oh, different topic:", "Changing the subject,"],
};

/// Words that answer a greeting.
const GREETING_WORDS: [&str; 14] = [
    "hello",
    "hi",
    "hey",
    "good morning",
    "good evening",
    "good afternoon",
    "morning",
    "evening",
    "nice to see you",
    "good to see you",
    "welcome",
    "hiya",
    "howdy",
    "greetings",
];

/// Words that answer "how are you".
const WELL_WORDS: [&str; 8] = ["good", "fine", "well", "great", "okay", "alright", "not bad", "happy"];

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
    foods: &["pancakes", "soup", "mangoes", "noodles", "cherries", "popcorn", "oatmeal", "tacos"],
    relatives: &["sister", "uncle", "grandmother", "nephew"],
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
    foods: &["apples", "bread", "rice", "pizza", "salad", "cookies", "eggs", "pears"],
    relatives: &["brother", "aunt", "cousin", "niece"],
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
    // Understanding (`understanding_tests`): following the conversation itself.
    /// Greeted again: greet back (not "I do not know").
    RepeatedGreetings,
    /// The same question in other words (Nova's name, how she is, a fact told earlier).
    Paraphrases,
    /// What each speaker said about themselves.
    WhoSaidWhat,
    /// The same conversation, a different last question: answer the latest one.
    LatestMessage,
    /// What was said before a change of topic, and after it.
    TopicChanges,
    /// Nova misheard and was corrected.
    Misunderstandings,
    /// The same question asked again.
    RepeatedQuestions,
}

impl Skill {
    /// The context-sensitivity tests (`context_tests`; unchanged since 0.5.9, so scores compare).
    pub const ALL: [Skill; 7] = [
        Skill::Names,
        Skill::Locations,
        Skill::ChangingFacts,
        Skill::Corrections,
        Skill::References,
        Skill::FollowUps,
        Skill::Uncertainty,
    ];
    /// The understanding tests (`understanding_tests`).
    pub const UNDERSTANDING: [Skill; 7] = [
        Skill::RepeatedGreetings,
        Skill::Paraphrases,
        Skill::WhoSaidWhat,
        Skill::LatestMessage,
        Skill::TopicChanges,
        Skill::Misunderstandings,
        Skill::RepeatedQuestions,
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
        let says = |w: &String| affirms(&r, &w.to_ascii_lowercase());
        self.expected.iter().any(says) && !self.wrong.iter().any(says)
    }
}

/// Words that deny what follows them ("not called Lena", "isn't in the kitchen", "no, Lena").
const DENY: [&str; 8] = ["not", "no", "never", "nor", "neither", "cannot", "without", "nobody"];
/// The same written without an apostrophe.
const NT: [&str; 9] = ["isnt", "wasnt", "dont", "doesnt", "didnt", "cant", "wont", "arent", "werent"];

/// `phrase` occurs in `text` as whole words and is not denied: none of the three words before
/// it is a negation ("not", "no", "isn't", ...). "Not in the hallway, in the kitchen" says
/// kitchen and does not say hallway.
fn affirms(text: &str, phrase: &str) -> bool {
    let word = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '\'');
    let mut from = 0;
    while let Some(i) = text[from..].find(phrase) {
        let at = from + i;
        let before = text[..at].chars().next_back();
        let after = text[at + phrase.len()..].chars().next();
        if !word(before) && !word(after) {
            let denied = text[..at]
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
                .filter(|w| !w.is_empty())
                .rev()
                .take(3)
                .any(|w| DENY.contains(&w) || w.ends_with("n't") || NT.contains(&w));
            if !denied {
                return true;
            }
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

fn make(g: &mut Gen, skill: Skill, w: &Words, ph: &Phrases, own: &str) -> ContextTest {
    let fillers = g.below(3);
    if Skill::UNDERSTANDING.contains(&skill) {
        return understanding(g, skill, w, ph, own);
    }
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
        // Made by `understanding` (returned above).
        _ => unreachable!("an understanding skill"),
    }
}

/// Fills `{r}` (a relative) and `{p}` (a pet) in a wording.
fn fill(text: &str, r: &str, p: &str) -> String {
    text.replace("{r}", r).replace("{p}", p)
}

/// A probe of the understanding skills: the conversation, the question, what a right answer says
/// and must not say, and a natural answer for practice.
fn probe(
    turns: Vec<(bool, String)>,
    question: String,
    expected: &[&str],
    wrong: &[&str],
    answer: String,
) -> Probe {
    Probe {
        turns,
        question,
        expected: expected.iter().map(|x| x.to_string()).collect(),
        wrong: wrong.iter().map(|x| x.to_string()).collect(),
        answer,
    }
}

/// The two conversations of an understanding test. Its words come from `w` and its wordings
/// from `ph` (the tests' or the practice's); `own` is Nova's name.
fn understanding(g: &mut Gen, skill: Skill, w: &Words, ph: &Phrases, own: &str) -> ContextTest {
    let ack = |g: &mut Gen| ACKS[g.below(ACKS.len())].to_string();
    let filler = |g: &mut Gen, turns: &mut Vec<(bool, String)>| {
        for _ in 0..g.below(2) {
            let (u, n) = FILLER[g.below(FILLER.len())];
            turns.push((false, u.to_string()));
            turns.push((true, n.to_string()));
        }
    };
    let mut unsure: Vec<&str> = UNSURE.to_vec();
    unsure.push("where");
    match skill {
        Skill::RepeatedGreetings => {
            let (g1, g2) = g.two(ph.greetings);
            let first = ["Hi!", "Hello!", "Hey! How are you?", "Hello! Nice to see you."];
            let again = ["Hi again!", "Hello again!", "Hey, hello again!", "Hi! Good to see you."];
            let one = |g: &mut Gen, a: &str, b: &str| {
                let turns = vec![(false, a.to_string()), (true, first[g.below(first.len())].to_string())];
                probe(turns, b.to_string(), &GREETING_WORDS, &unsure, again[g.below(again.len())].to_string())
            };
            ContextTest { skill, a: one(g, g1, g2), b: one(g, g2, g1) }
        }
        Skill::Paraphrases => match g.below(3) {
            0 => {
                let (q1, q2) = g.two(ph.name_questions);
                let answers =
                    [format!("My name is {own}."), format!("I'm {own}."), format!("You can call me {own}.")];
                let one = |g: &mut Gen, q: &str| {
                    let turns = vec![(false, "Hi!".to_string()), (true, "Hello!".to_string())];
                    probe(turns, q.to_string(), &[own], &[], answers[g.below(answers.len())].clone())
                };
                ContextTest { skill, a: one(g, q1), b: one(g, q2) }
            }
            1 => {
                let (q1, q2) = g.two(ph.wellbeing_questions);
                let answers =
                    ["I'm good, thanks. And you?", "Pretty well, thank you!", "Fine, thanks for asking."];
                let one = |g: &mut Gen, q: &str| {
                    probe(
                        Vec::new(),
                        q.to_string(),
                        &WELL_WORDS,
                        &UNSURE,
                        answers[g.below(answers.len())].to_string(),
                    )
                };
                ContextTest { skill, a: one(g, q1), b: one(g, q2) }
            }
            _ => {
                let r = g.pick(w.relatives);
                let (x, y) = g.two(w.places);
                let (q1, q2) = g.two(ph.where_lives);
                let one = |g: &mut Gen, x: &str, y: &str, q: &str| {
                    let mut turns = vec![(false, format!("My {r} lives near the {x}.")), (true, ack(g))];
                    filler(g, &mut turns);
                    probe(turns, fill(q, r, ""), &[x], &[y], format!("Near the {x}."))
                };
                ContextTest { skill, a: one(g, x, y, q1), b: one(g, y, x, q2) }
            }
        },
        Skill::WhoSaidWhat => {
            let (x, y) = g.two(w.foods);
            let ask_me = g.below(2) == 0;
            let one = |g: &mut Gen, x: &str, y: &str| {
                let mut turns =
                    vec![(false, format!("I really like {x}.")), (true, format!("I like {y} more."))];
                filler(g, &mut turns);
                if ask_me {
                    probe(turns, "What food do I like?".into(), &[x], &[y], format!("You like {x}."))
                } else {
                    probe(turns, "And what food do you like?".into(), &[y], &[x], format!("I like {y}."))
                }
            };
            ContextTest { skill, a: one(g, x, y), b: one(g, y, x) }
        }
        Skill::LatestMessage => {
            let pet = g.pick(w.pets);
            let name = g.pick(w.people.1);
            let place = g.pick(w.places);
            let mut turns = vec![
                (false, format!("My {pet} is called {name}.")),
                (true, ack(g)),
                (false, format!("I went to the {place} today.")),
                (true, "That sounds fun.".to_string()),
            ];
            filler(g, &mut turns);
            let a = probe(
                turns.clone(),
                "Where did I go today?".into(),
                &[place],
                &[name],
                format!("You went to the {place}."),
            );
            let b = probe(
                turns,
                format!("What is my {pet} called?"),
                &[name],
                &[place],
                format!("Your {pet} is called {name}."),
            );
            ContextTest { skill, a, b }
        }
        Skill::TopicChanges => {
            let r = g.pick(w.relatives);
            let change = g.pick(ph.topic_changes);
            let ask_old = g.below(2) == 0;
            let (x, y) = g.two(w.places);
            let (t1, t2) = g.two(w.things);
            let one = |g: &mut Gen, x: &str, t: &str, other_place: &str, other_thing: &str| {
                let mut turns = vec![
                    (false, format!("I met my {r} at the {x} this morning.")),
                    (true, "That's nice.".to_string()),
                    (false, format!("{change} I bought a new {t}.")),
                    (true, "Oh, nice!".to_string()),
                ];
                filler(g, &mut turns);
                if ask_old {
                    probe(
                        turns,
                        format!("Where did I meet my {r}?"),
                        &[x],
                        &[other_place],
                        format!("At the {x}."),
                    )
                } else {
                    probe(turns, "What did I buy?".into(), &[t], &[other_thing], format!("A new {t}."))
                }
            };
            ContextTest { skill, a: one(g, x, t1, y, t2), b: one(g, y, t2, x, t1) }
        }
        Skill::Misunderstandings => {
            let (x, y) = g.two(w.things);
            let one = |g: &mut Gen, x: &str, y: &str| {
                let mut turns = vec![
                    (false, format!("I lost my {x} yesterday.")),
                    (true, format!("Oh no, your {y}?")),
                    (false, format!("No, my {x}.")),
                    (true, format!("Sorry, your {x}. I hope you find it.")),
                ];
                filler(g, &mut turns);
                probe(turns, "What did I lose?".into(), &[x], &[y], format!("Your {x}."))
            };
            ContextTest { skill, a: one(g, x, y), b: one(g, y, x) }
        }
        Skill::RepeatedQuestions => {
            let pet = g.pick(w.pets);
            let (x, y) = g.two(w.people.0);
            let q = g.pick(ph.again);
            let one = |g: &mut Gen, x: &str, y: &str| {
                let mut turns = vec![
                    (false, format!("My {pet} is called {x}.")),
                    (true, "What a nice name.".to_string()),
                ];
                filler(g, &mut turns);
                turns.push((false, format!("What is my {pet} called?")));
                turns.push((true, format!("Your {pet} is called {x}.")));
                probe(turns, fill(q, "", pet), &[x], &[y], format!("It is called {x}."))
            };
            ContextTest { skill, a: one(g, x, y), b: one(g, y, x) }
        }
        _ => unreachable!("not an understanding skill"),
    }
}

/// The context-sensitivity tests: `per_skill` pairs of each kind, the same every time.
pub fn context_tests(per_skill: usize) -> Vec<ContextTest> {
    tests_of(&Skill::ALL, per_skill, "Nova", 0x7E57_5EED)
}

/// The understanding tests: `per_skill` pairs of each `Skill::UNDERSTANDING` kind, the same every
/// time, from the tests' words and wordings (never practised). `own` is Nova's name (what she
/// answers when asked hers).
pub fn understanding_tests(per_skill: usize, own: &str) -> Vec<ContextTest> {
    tests_of(&Skill::UNDERSTANDING, per_skill, own, 0x0DE5_7A4D)
}

fn tests_of(skills: &[Skill], per_skill: usize, own: &str, seed: u64) -> Vec<ContextTest> {
    let mut g = Gen(seed);
    let mut out = Vec::new();
    for &skill in skills {
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..per_skill * 20 {
            if seen.len() == per_skill {
                break;
            }
            let t = make(&mut g, skill, &TEST, &TEST_PHRASES, own);
            // Keyed by the first conversation, as since 0.5.9 (the tests stay the same).
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
    practice_of(&Skill::ALL, n, seed, nova)
}

/// Practice conversations of `skills` (as `context_practice`), from the practice's words and
/// wordings.
pub fn practice_of(skills: &[Skill], n: usize, seed: u64, nova: &str) -> Vec<Dialogue> {
    let mut g = Gen(seed ^ 0xC0_7E_47);
    let mut out = Vec::new();
    while out.len() < n && !skills.is_empty() {
        let skill = skills[g.below(skills.len())];
        let t = make(&mut g, skill, &PRACTICE, &PRACTICE_PHRASES, nova);
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
    /// One pair of each skill in full: both conversations and both answers.
    #[serde(default)]
    pub pairs: Vec<PairExample>,
}

/// A context test as Nova answered it: the two conversations, her answers and whether each was
/// right.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairExample {
    pub skill: Skill,
    pub a: String,
    pub a_reply: String,
    pub a_right: bool,
    pub b: String,
    pub b_reply: String,
    pub b_right: bool,
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
            p.expected.iter().any(|w| affirms(&r.to_ascii_lowercase(), &w.to_ascii_lowercase()))
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
        if !s.pairs.iter().any(|p| p.skill == t.skill) {
            s.pairs.push(PairExample {
                skill: t.skill,
                a: t.a.prompt(),
                a_reply: ra.clone(),
                a_right: a,
                b: t.b.prompt(),
                b_reply: rb.clone(),
                b_right: b,
            });
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
        // A denied word is not an answer: "not called Lena" does not say the name is Lena.
        assert!(!loc.a.right(&format!("A does that is not called {x}.")));
        assert!(!loc.a.right(&format!("He isn't in the {x}.")) && !loc.a.right(&format!("No, {x}.")));
        assert!(
            loc.a.right(&format!("Not in the {y}, in the {x}.")),
            "a denied other answer is not a wrong one"
        );
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
        assert_eq!(s.pairs.len(), Skill::ALL.len(), "one example pair of each skill");
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

    /// The context tests and their practice are the same as before the understanding skills
    /// were added (0.7.7), so context scores compare with earlier runs.
    #[test]
    fn the_context_tests_are_unchanged() {
        let t = context_tests(12);
        let p: String = t.iter().map(|t| t.a.prompt() + &t.b.prompt() + &t.a.answer).collect();
        let q: String = context_practice(500, 7, "Nova").iter().map(|d| d.turns.join("|")).collect();
        let hash =
            |s: &str, h: u64, m: u64| s.bytes().fold(h, |h, b| h.wrapping_mul(m).wrapping_add(b as u64));
        assert_eq!((t.len(), p.len()), (84, 25213));
        assert_eq!(hash(&p, 0, 31) ^ hash(&q, 7, 37), 854991177357982254);
    }

    /// Understanding tests: two conversations per test, graded by meaning; held out from the
    /// practice by their words and their wordings.
    #[test]
    fn understanding_tests_are_graded_by_meaning_and_held_out_from_practice() {
        let tests = understanding_tests(6, "Nova");
        assert_eq!(tests.len(), 6 * Skill::UNDERSTANDING.len());
        assert_eq!(tests, understanding_tests(6, "Nova"), "the same every time");
        let of = |k: Skill| tests.iter().find(|t| t.skill == k).unwrap();
        // Greeted again: a greeting is right, "I do not know" is not.
        let g = of(Skill::RepeatedGreetings);
        assert!(g.a.right("Hi again!") && g.b.right("Hello! Good to see you."));
        assert!(!g.a.right("I do not know. You did not tell me where Leo is."));
        // Who said what: the user's food or Nova's, never the other one.
        let w = of(Skill::WhoSaidWhat);
        assert!(w.a.right(&w.a.answer) && w.b.right(&w.b.answer));
        assert!(!w.a.right(&format!("{} and {}", w.a.expected[0], w.a.wrong[0])));
        // The latest message: the same conversation, different last questions and answers.
        let l = of(Skill::LatestMessage);
        assert_eq!(l.a.turns, l.b.turns);
        assert_ne!(l.a.question, l.b.question);
        assert!(l.a.right(&l.a.answer) && !l.a.right(&l.b.answer));
        // Paraphrases of Nova's name expect her own name.
        let named = understanding_tests(40, "Nova")
            .into_iter()
            .find(|t| t.skill == Skill::Paraphrases && t.a.expected == ["Nova"]);
        let named = named.expect("a name paraphrase");
        assert_ne!(named.a.question, named.b.question);
        assert!(named.a.right("I'm Nova.") && !named.a.right("I'm Cara."));
        // Every test's own answer is right, and the other version's is wrong where they differ.
        for t in &tests {
            assert!(t.a.right(&t.a.answer) && t.b.right(&t.b.answer), "{t:?}");
        }
        // Practice: other words and other wordings than the tests.
        let practice = practice_of(&Skill::UNDERSTANDING, 600, 3, "Nova");
        assert_eq!(practice.len(), 600);
        let text: String = practice.iter().map(|d| d.turns.join("\n")).collect::<Vec<_>>().join("\n");
        let words = |t: &str| -> std::collections::BTreeSet<String> {
            t.split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| !w.is_empty())
                .map(str::to_string)
                .collect()
        };
        let practised = words(&text);
        for wd in TEST.foods.iter().chain(TEST.relatives).chain(TEST.people.0).chain(TEST.people.1) {
            assert!(!practised.contains(*wd), "{wd} is practised");
        }
        for ph in TEST_PHRASES
            .greetings
            .iter()
            .chain(TEST_PHRASES.name_questions)
            .chain(TEST_PHRASES.wellbeing_questions)
            .chain(TEST_PHRASES.topic_changes)
        {
            assert!(!text.contains(ph), "{ph} is practised");
        }
        for ph in TEST_PHRASES.where_lives.iter().chain(TEST_PHRASES.again) {
            let start = ph.split('{').next().unwrap();
            assert!(!text.contains(start), "{ph} is practised");
        }
        // Practice trains only Nova's answer at the end, as the context practice does.
        for d in &practice {
            assert_eq!(d.trainable.iter().filter(|&&t| t).count(), 1);
            assert_eq!(d.speakers.last().unwrap(), "Nova");
        }
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
            .chain(TEST.foods)
            .chain(TEST.relatives)
        {
            assert!(
                !PRACTICE.people.0.contains(w)
                    && !PRACTICE.people.1.contains(w)
                    && !PRACTICE.places.contains(w)
                    && !PRACTICE.things.contains(w)
                    && !PRACTICE.pets.contains(w)
                    && !PRACTICE.colours.contains(w)
                    && !PRACTICE.foods.contains(w)
                    && !PRACTICE.relatives.contains(w),
                "{w} is in both"
            );
        }
    }
}
