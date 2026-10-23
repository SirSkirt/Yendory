//! Built-in courses: Basic English, Basic Reasoning and Basic Math, in the ordinary lesson format,
//! for checking that training works independently of any teacher's own lessons.
//!
//! The lessons are generated from small word lists by fixed templates with a fixed seed, so every
//! copy of Yendory has exactly the same files. Each lesson teaches one pattern with many
//! examples; its held-out questions (the usual 10% by hash) are new combinations of the same
//! pattern (new names, colours, numbers), so a model passes only by learning the pattern, not
//! by remembering answers. Nothing here is used at inference.
use super::course::CourseFile;

/// A course shipped with Yendory.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltinCourse {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    /// Lesson files in order ("01 Greetings.txt", ...).
    pub lessons: Vec<CourseFile>,
}

/// Examples generated per lesson (fewer when the pattern has fewer combinations).
const EXAMPLES: usize = 200;

/// A small deterministic generator (SplitMix64).
struct Rng(u64);
impl Rng {
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
    fn pick<'a>(&mut self, list: &[&'a str]) -> &'a str {
        list[self.below(list.len())]
    }
    /// Two different entries.
    fn two<'a>(&mut self, list: &[&'a str]) -> (&'a str, &'a str) {
        let a = self.below(list.len());
        let b = (a + 1 + self.below(list.len() - 1)) % list.len();
        (list[a], list[b])
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

const NAMES: [&str; 24] = [
    "Ada", "Ben", "Cal", "Dan", "Eva", "Fay", "Gus", "Hal", "Ivy", "Jay", "Kai", "Lea", "Max", "Mia", "Ned",
    "Ona", "Pam", "Rex", "Sam", "Tia", "Uma", "Vic", "Wes", "Zoe",
];
const COLOURS: [&str; 10] =
    ["red", "blue", "green", "yellow", "black", "white", "pink", "brown", "orange", "purple"];
const OBJECTS: [&str; 12] =
    ["ball", "book", "cup", "hat", "key", "kite", "coat", "bag", "pen", "car", "sock", "lamp"];

fn cap(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_ascii_uppercase().to_string() + c.as_str(),
        None => String::new(),
    }
}

/// Up to `n` distinct examples from `make` (a question and its answer), in a shuffled order. A
/// question generated twice is kept once, so no question has two answers.
fn examples(
    seed: u64,
    n: usize,
    mut make: impl FnMut(&mut Rng) -> (String, String),
) -> Vec<(String, String)> {
    let mut rng = Rng(seed);
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for _ in 0..n * 20 {
        if out.len() == n {
            break;
        }
        let (q, a) = make(&mut rng);
        if seen.insert(q.to_ascii_lowercase()) {
            out.push((q, a));
        }
    }
    out
}

/// All combinations of a pattern, shuffled, at most `n`.
fn all_of(seed: u64, n: usize, mut items: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut rng = Rng(seed);
    rng.shuffle(&mut items);
    let mut seen = std::collections::BTreeSet::new();
    items.retain(|(q, _)| seen.insert(q.to_ascii_lowercase()));
    items.truncate(n);
    items
}

/// A `[Dialogue Lesson]` / `[Understanding Lesson]` block.
fn qa_block(kind: &str, explanation: &str, items: &[(String, String)]) -> String {
    let mut s = format!("[{kind}]\n{explanation}\n[examples]\n");
    for (q, a) in items {
        s += &format!("{q}  {a}\n");
    }
    s + &format!("[/examples]\n[/{kind}]\n")
}

/// A `[Reasoning Lesson]` block: (premise, question, supported, contradicted).
fn reasoning_block(explanation: &str, items: &[(String, String, String, String)]) -> String {
    let mut s = format!("[Reasoning Lesson]\n{explanation}\n[examples]\n");
    for (p, q, yes, no) in items {
        s += &format!("Premise: {p} Question: {q} Supported: {yes} Contradicted: {no}\n");
    }
    s + "[/examples]\n[/Reasoning Lesson]\n"
}

/// Up to `n` distinct reasoning examples (distinct premise and question).
fn reasoning(
    seed: u64,
    n: usize,
    mut make: impl FnMut(&mut Rng) -> (String, String, String, String),
) -> Vec<(String, String, String, String)> {
    let mut rng = Rng(seed);
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for _ in 0..n * 20 {
        if out.len() == n {
            break;
        }
        let e = make(&mut rng);
        if seen.insert(format!("{} {}", e.0, e.1).to_ascii_lowercase()) {
            out.push(e);
        }
    }
    out
}

fn file(name: &str, text: String) -> CourseFile {
    CourseFile { name: format!("{name}.txt"), text }
}

fn english() -> BuiltinCourse {
    let greetings: [(&str, &str); 15] = [
        ("hello", "Hello!"),
        ("hi", "Hi!"),
        ("good morning", "Good morning!"),
        ("good afternoon", "Good afternoon!"),
        ("good evening", "Good evening!"),
        ("good night", "Good night!"),
        ("thank you", "You are welcome."),
        ("thanks", "You are welcome."),
        ("how are you", "I am well, thank you."),
        ("bye", "Goodbye!"),
        ("goodbye", "Goodbye!"),
        ("see you later", "See you later!"),
        ("nice to meet you", "Nice to meet you too."),
        ("sorry", "That is okay."),
        ("excuse me", "Yes, how can I help?"),
    ];
    let forms = [
        "{g}",
        "{G}",
        "{g}!",
        "{G}?",
        "{g} there",
        "{G}, Nova",
        "{g} friend",
        "oh, {g}",
        "well {g}",
        "{G}!!",
    ];
    let mut greet = Vec::new();
    for (g, a) in greetings {
        for f in forms {
            greet.push((f.replace("{g}", g).replace("{G}", &cap(g)), a.to_string()));
        }
    }
    let greet = all_of(11, EXAMPLES, greet);

    let colours = examples(12, EXAMPLES, |r| {
        let n = r.pick(&NAMES);
        if r.below(2) == 0 {
            let (c, o) = (r.pick(&COLOURS), r.pick(&OBJECTS));
            (format!("{n} has a {c} {o}. What color is the {o}?"), format!("{}.", cap(c)))
        } else {
            let (c1, c2) = r.two(&COLOURS);
            let (o1, o2) = r.two(&OBJECTS);
            let (o, c) = if r.below(2) == 0 { (o1, c1) } else { (o2, c2) };
            (format!("{n} has a {c1} {o1} and a {c2} {o2}. What color is the {o}?"), format!("{}.", cap(c)))
        }
    });

    let spots: [(&str, &[&str]); 3] = [
        ("in", &["kitchen", "garden", "box", "bag", "car", "room", "yard", "shop", "park", "house"]),
        ("on", &["table", "chair", "bed", "shelf", "desk", "floor", "roof", "step"]),
        ("under", &["bed", "table", "tree", "chair", "bench", "bridge"]),
    ];
    let places = examples(13, EXAMPLES, |r| {
        let o = r.pick(&OBJECTS);
        let (prep, list) = spots[r.below(spots.len())];
        let place = r.pick(list);
        (format!("The {o} is {prep} the {place}. Where is the {o}?"), format!("{} the {place}.", cap(prep)))
    });

    let adjectives = [
        "happy", "sad", "tall", "short", "hungry", "tired", "cold", "warm", "busy", "quiet", "kind", "brave",
    ];
    let yes_no = examples(14, EXAMPLES, |r| {
        let n = r.pick(&NAMES);
        let (a, b) = r.two(&adjectives);
        if r.below(2) == 0 {
            (format!("{n} is {a}. Is {n} {a}?"), "Yes.".into())
        } else {
            (format!("{n} is {a}. Is {n} {b}?"), "No.".into())
        }
    });

    let who = examples(15, EXAMPLES, |r| {
        let (a, b) = r.two(&NAMES);
        let o = r.pick(&OBJECTS);
        if r.below(2) == 0 {
            (format!("{a} gave the {o} to {b}. Who has the {o} now?"), format!("{b}."))
        } else {
            (format!("{a} gave the {o} to {b}. Who gave the {o}?"), format!("{a}."))
        }
    });

    let nouns = [
        "cat", "dog", "cup", "hat", "book", "bird", "tree", "car", "pen", "ball", "key", "apple", "egg",
        "frog", "duck", "bag", "kite", "coat", "sock", "ship", "star", "drum", "bell", "lamp", "door",
        "shoe", "boat", "cake", "rose", "fork", "ring", "nut", "bug", "cow", "pig", "hen", "bee", "ant",
        "owl", "girl",
    ];
    let numbers = ["two", "three", "four", "five"];
    let mut plural = Vec::new();
    for n in nouns {
        for k in numbers {
            plural.push((format!("One {n}, {k} what?"), format!("{} {n}s.", cap(k))));
        }
    }
    let plural = all_of(16, EXAMPLES, plural);

    BuiltinCourse {
        id: "basic-english",
        name: "Basic English (built-in)",
        description: "Greetings, colours, where things are, yes or no, who did what, and plurals. Every answer can be read off the question or follows one pattern, so a model that learns from lessons should pass these.",
        lessons: vec![
            file("01 Greetings", qa_block("Dialogue Lesson", "A greeting is answered with a greeting. Thanks are answered politely.", &greet)),
            file("02 Colours", qa_block("Understanding Lesson", "The color of a thing is said in the sentence. Answer with that color.", &colours)),
            file("03 Where things are", qa_block("Understanding Lesson", "In, on and under tell where a thing is. Answer with the same words.", &places)),
            file("04 Yes or no", qa_block("Understanding Lesson", "Answer Yes when the question says what the sentence says, and No when it does not.", &yes_no)),
            file("05 Who did it", qa_block("Understanding Lesson", "The one who gave has given the thing away. The one it was given to has it now.", &who)),
            file("06 Plurals", qa_block("Understanding Lesson", "More than one thing: add s to the word.", &plural)),
        ],
    }
}

fn reasoning_course() -> BuiltinCourse {
    let has = reasoning(21, EXAMPLES, |r| {
        let (a, b) = r.two(&NAMES);
        let (o1, o2) = r.two(&OBJECTS);
        let (o, yes, no) = if r.below(2) == 0 { (o1, a, b) } else { (o2, b, a) };
        (
            format!("{a} has a {o1}. {b} has a {o2}."),
            format!("Who has the {o}?"),
            format!("{yes}."),
            format!("{no}."),
        )
    });

    let true_false = reasoning(22, EXAMPLES, |r| {
        let o = r.pick(&OBJECTS);
        let (c, d) = r.two(&COLOURS);
        if r.below(2) == 0 {
            (format!("The {o} is {c}."), format!("Is the {o} {c}?"), "Yes.".into(), "No.".into())
        } else {
            (format!("The {o} is {c}."), format!("Is the {o} {d}?"), "No.".into(), "Yes.".into())
        }
    });

    let relations = [
        ("taller", "tallest", "shortest"),
        ("older", "oldest", "youngest"),
        ("faster", "fastest", "slowest"),
    ];
    let compare = reasoning(23, EXAMPLES, |r| {
        let (rel, most, least) = relations[r.below(relations.len())];
        let a = r.pick(&NAMES);
        let (b, c) = loop {
            let (b, c) = r.two(&NAMES);
            if b != a && c != a {
                break (b, c);
            }
        };
        let (s1, s2) = (format!("{a} is {rel} than {b}."), format!("{b} is {rel} than {c}."));
        let premise = if r.below(2) == 0 { format!("{s1} {s2}") } else { format!("{s2} {s1}") };
        if r.below(2) == 0 {
            (premise, format!("Who is the {most}?"), format!("{a}."), format!("{c}."))
        } else {
            (premise, format!("Who is the {least}?"), format!("{c}."), format!("{a}."))
        }
    });

    let kinds = [
        ("bird", "birds"),
        ("fish", "fish"),
        ("dog", "dogs"),
        ("cat", "cats"),
        ("robot", "robots"),
        ("frog", "frogs"),
    ];
    let abilities = ["sing", "swim", "jump", "climb", "dance", "read", "fly", "dig", "count", "paint"];
    let categories = reasoning(24, EXAMPLES, |r| {
        let (kind, kinds_plural) = kinds[r.below(kinds.len())];
        let n = r.pick(&NAMES);
        let x = r.pick(&abilities);
        if r.below(2) == 0 {
            (
                format!("All {kinds_plural} can {x}. {n} is a {kind}."),
                format!("Can {n} {x}?"),
                "Yes.".into(),
                "No.".into(),
            )
        } else {
            (
                format!("No {kinds_plural} can {x}. {n} is a {kind}."),
                format!("Can {n} {x}?"),
                "No.".into(),
                "Yes.".into(),
            )
        }
    });

    let verbs = [
        "ate", "slept", "ran", "sang", "read", "swam", "jumped", "danced", "cooked", "played", "walked",
        "painted",
    ];
    let order = reasoning(25, EXAMPLES, |r| {
        let n = r.pick(&NAMES);
        let (v1, v2) = r.two(&verbs);
        let premise = format!("First {n} {v1}, then {n} {v2}.");
        if r.below(2) == 0 {
            (premise, format!("What did {n} do first?"), format!("{}.", cap(v1)), format!("{}.", cap(v2)))
        } else {
            (premise, format!("What did {n} do last?"), format!("{}.", cap(v2)), format!("{}.", cap(v1)))
        }
    });

    BuiltinCourse {
        id: "basic-reasoning",
        name: "Basic Reasoning (built-in)",
        description: "Who has what, true or false, comparisons across two facts, rules about groups, and the order of events. Every answer follows from the premise; the wrong answer is used to check the model prefers the right one.",
        lessons: vec![
            file("01 Who has it", reasoning_block("Find the person the premise says has the thing.", &has)),
            file("02 True or false", reasoning_block("Check the question against the premise: Yes when it agrees, No when it does not.", &true_false)),
            file("03 Comparisons", reasoning_block("If A is taller than B and B is taller than C, A is the tallest and C the shortest.", &compare)),
            file("04 Rules about groups", reasoning_block("What is true of all members of a group is true of each one; what no member can do, none can.", &categories)),
            file("05 Order of events", reasoning_block("First comes before then. Answer with what happened first or last.", &order)),
        ],
    }
}

fn math() -> BuiltinCourse {
    let mut counting = Vec::new();
    for n in 0..100 {
        counting.push((format!("What number comes after {n}?"), format!("{}.", n + 1)));
        counting.push((format!("What number comes before {}?", n + 1), format!("{n}.")));
    }
    let counting = all_of(31, EXAMPLES, counting);

    let mut addition = Vec::new();
    for a in 0..=10 {
        for b in 0..=10 {
            addition.push((format!("What is {a} plus {b}?"), format!("{}.", a + b)));
            addition.push((format!("{a} + {b} = ?"), format!("{}.", a + b)));
        }
    }
    let addition = all_of(32, EXAMPLES, addition);

    let mut subtraction = Vec::new();
    for a in 0..=12 {
        for b in 0..=a {
            subtraction.push((format!("What is {a} minus {b}?"), format!("{}.", a - b)));
            subtraction.push((format!("{a} - {b} = ?"), format!("{}.", a - b)));
        }
    }
    let subtraction = all_of(33, EXAMPLES, subtraction);

    let compare = examples(34, EXAMPLES, |r| {
        let a = 1 + r.below(50);
        let b = loop {
            let b = 1 + r.below(50);
            if b != a {
                break b;
            }
        };
        if r.below(2) == 0 {
            (format!("Which is bigger, {a} or {b}?"), format!("{}.", a.max(b)))
        } else {
            (format!("Which is smaller, {a} or {b}?"), format!("{}.", a.min(b)))
        }
    });

    let things = ["apples", "books", "cups", "balls", "pens", "cars", "kites", "hats"];
    let stories = examples(35, EXAMPLES, |r| {
        let n = r.pick(&NAMES);
        let t = r.pick(&things);
        let a = 1 + r.below(10);
        let b = 1 + r.below(10);
        if r.below(2) == 0 {
            (
                format!("{n} has {a} {t}. {n} gets {b} more. How many {t} does {n} have?"),
                format!("{}.", a + b),
            )
        } else {
            let (hi, lo) = (a.max(b), a.min(b));
            (
                format!("{n} has {hi} {t}. {n} gives away {lo}. How many {t} does {n} have now?"),
                format!("{}.", hi - lo),
            )
        }
    });

    BuiltinCourse {
        id: "basic-math",
        name: "Basic Math (built-in)",
        description: "Counting, adding and subtracting small numbers, comparing numbers, and short word problems. The held-out questions are sums and comparisons not taught, so passing needs real number patterns, not remembered answers.",
        lessons: vec![
            file("01 Counting", qa_block("Understanding Lesson", "Numbers go up by one: after 4 comes 5, and before 5 comes 4.", &counting)),
            file("02 Addition", qa_block("Understanding Lesson", "Plus puts two numbers together: 2 plus 3 is 5.", &addition)),
            file("03 Subtraction", qa_block("Understanding Lesson", "Minus takes a number away: 5 minus 2 is 3.", &subtraction)),
            file("04 Bigger or smaller", qa_block("Understanding Lesson", "The bigger number is the one further along when counting.", &compare)),
            file("05 Word problems", qa_block("Understanding Lesson", "Getting more is adding; giving away is taking away.", &stories)),
        ],
    }
}

/// The built-in courses, generated (the same text every time).
pub fn builtin_courses() -> Vec<BuiltinCourse> {
    vec![english(), reasoning_course(), math()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::curriculum::{prepare, CourseDef, FEW_QUESTIONS};
    use crate::learn::lessons::parse_lesson;

    #[test]
    fn courses_are_valid_lessons_with_enough_held_out_questions() {
        let courses = builtin_courses();
        assert_eq!(courses, builtin_courses(), "generated identically every time");
        for c in &courses {
            assert!(c.lessons.len() >= 5, "{}", c.id);
            for f in &c.lessons {
                assert!(f.text.is_ascii(), "{}", f.name);
                let items = parse_lesson(&f.text, &f.name).unwrap();
                let taught = items.iter().filter(|i| !i.held_out && i.answer.is_some()).count();
                let held: Vec<_> = items.iter().filter(|i| i.held_out).collect();
                println!("{} / {}: {taught} taught, {} held out", c.id, f.name, held.len());
                assert!(taught >= 100, "{}: {taught} taught", f.name);
                assert!(held.len() >= FEW_QUESTIONS, "{}: {} held out", f.name, held.len());
                // No question is both taught and held out, and none has two answers.
                let mut seen = std::collections::BTreeMap::new();
                for i in items.iter().filter(|i| i.answer.is_some()) {
                    let q = i.prompt.clone().unwrap().to_ascii_lowercase();
                    let prev = seen.insert(q.clone(), i.answer.clone());
                    assert!(prev.is_none(), "{}: {q} appears twice", f.name);
                }
            }
            // As a course: prepared without problems (exam assembled, no leaks).
            let mut def = CourseDef::new(c.id, c.name);
            def.add_files(c.lessons.clone());
            let p = prepare(&def);
            assert!(p.check.problems.is_empty(), "{}: {:?}", c.id, p.check.problems);
            assert_eq!(def.lessons.len(), c.lessons.len(), "every file is a lesson");
        }
    }
}
