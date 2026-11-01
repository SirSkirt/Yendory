//! Practice text generated on this device: arithmetic, word problems, comparisons, logic,
//! small "who is where" worlds and sequences, written as `U:` questions and `N:` answers with
//! the reasoning spelled out.
//!
//! Every answer is computed from the example itself (the sums are added, the world is
//! simulated, the logic follows the rule stated), so the text is correct by construction.
//! It is training text like any corpus: about 10% of it is held out by the usual hash and never
//! trained on, and nothing here is used when the model replies.
use super::corpus::END_OF_TEXT;
use super::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Arithmetic,
    WordProblems,
    Comparisons,
    Logic,
    World,
    Sequences,
}
impl Family {
    pub const ALL: [Family; 6] = [
        Family::Arithmetic,
        Family::WordProblems,
        Family::Comparisons,
        Family::Logic,
        Family::World,
        Family::Sequences,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Family::Arithmetic => "arithmetic",
            Family::WordProblems => "word problems",
            Family::Comparisons => "comparisons and ordering",
            Family::Logic => "if-then and all/some logic",
            Family::World => "who is where, who has what",
            Family::Sequences => "sequences and patterns",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerateConfig {
    pub families: Vec<Family>,
    /// 1 (easy) to 3 (harder): number ranges, chain lengths, steps.
    pub difficulty: u8,
    /// Approximate size of the text to write.
    pub bytes: usize,
    pub seed: u64,
}

const NAMES: [&str; 24] = [
    "Ann", "Ben", "Cara", "Dan", "Eva", "Finn", "Gia", "Hugo", "Ivy", "Jack", "Kim", "Leo", "Mia", "Noah",
    "Owen", "Pia", "Ray", "Sara", "Tom", "Uma", "Vic", "Wes", "Yara", "Zoe",
];
const PLACES: [&str; 10] =
    ["kitchen", "garden", "office", "bedroom", "hallway", "bathroom", "park", "school", "library", "shop"];
const OBJECTS: [&str; 10] = ["ball", "book", "apple", "cup", "key", "hat", "pen", "box", "coin", "toy"];
const ITEMS: [&str; 8] = ["apples", "marbles", "stickers", "books", "cookies", "pencils", "coins", "shells"];
const MOVES: [&str; 4] = ["went to", "walked to", "moved to", "ran to"];

fn pick<'a, T: ?Sized>(rng: &mut Rng, xs: &'a [&'a T]) -> &'a T {
    xs[rng.below(xs.len())]
}
fn range(rng: &mut Rng, lo: i64, hi: i64) -> i64 {
    lo + rng.below((hi - lo + 1) as usize) as i64
}
/// `n` different names.
fn names(rng: &mut Rng, n: usize) -> Vec<&'static str> {
    let mut all = NAMES.to_vec();
    for i in 0..n {
        let j = i + rng.below(all.len() - i);
        all.swap(i, j);
    }
    all.truncate(n);
    all
}
fn qa(q: &str, a: &str) -> String {
    format!("U: {q}\nN: {a}")
}

fn arithmetic(rng: &mut Rng, d: u8) -> String {
    let max = [20, 100, 1000][(d.clamp(1, 3) - 1) as usize];
    match rng.below(if d >= 2 { 4 } else { 2 }) {
        0 => {
            let (a, b) = (range(rng, 0, max), range(rng, 0, max));
            qa(&format!("What is {a} plus {b}?"), &format!("{a} + {b} = {}.", a + b))
        }
        1 => {
            let (a, b) = (range(rng, 0, max), range(rng, 0, max));
            let (a, b) = (a.max(b), a.min(b));
            qa(&format!("What is {a} minus {b}?"), &format!("{a} - {b} = {}.", a - b))
        }
        2 => {
            let lim = if d >= 3 { 20 } else { 12 };
            let (a, b) = (range(rng, 2, lim), range(rng, 2, lim));
            if rng.below(2) == 0 {
                qa(&format!("What is {a} times {b}?"), &format!("{a} x {b} = {}.", a * b))
            } else {
                qa(&format!("What is {} divided by {b}?", a * b), &format!("{} / {b} = {a}.", a * b))
            }
        }
        _ => {
            let (a, b, c) = (range(rng, 1, 30), range(rng, 1, 30), range(rng, 2, 9));
            qa(
                &format!("What is ({a} + {b}) times {c}?"),
                &format!("First {a} + {b} = {}. Then {} x {c} = {}.", a + b, a + b, (a + b) * c),
            )
        }
    }
}

fn word_problem(rng: &mut Rng, d: u8) -> String {
    let who = pick(rng, &NAMES);
    let item = pick(rng, &ITEMS);
    let max = [10, 50, 200][(d.clamp(1, 3) - 1) as usize];
    let kinds = if d >= 3 {
        5
    } else if d >= 2 {
        4
    } else {
        2
    };
    match rng.below(kinds) {
        0 => {
            let (a, b) = (range(rng, 1, max), range(rng, 1, max));
            qa(
                &format!("{who} has {a} {item}. {who} gets {b} more. How many {item} does {who} have now?"),
                &format!(
                    "{who} had {a} and got {b} more. {a} + {b} = {}. {who} has {} {item}.",
                    a + b,
                    a + b
                ),
            )
        }
        1 => {
            let a = range(rng, 2, max);
            let b = range(rng, 1, a);
            qa(
                &format!("{who} has {a} {item} and gives {b} away. How many {item} are left?"),
                &format!("{a} - {b} = {}. {} {item} are left.", a - b, a - b),
            )
        }
        2 => {
            let (c, e) = (range(rng, 2, 9), range(rng, 2, 12));
            qa(
                &format!("There are {c} bags with {e} {item} in each bag. How many {item} are there in all?"),
                &format!("{c} bags of {e} is {c} x {e} = {}. There are {} {item}.", c * e, c * e),
            )
        }
        3 => {
            let (c, e) = (range(rng, 2, 8), range(rng, 2, 10));
            qa(
                &format!(
                    "{who} shares {} {item} equally among {c} friends. How many does each friend get?",
                    c * e
                ),
                &format!("{} / {c} = {e}. Each friend gets {e} {item}.", c * e),
            )
        }
        _ => {
            let (a, b) = (range(rng, 5, max), range(rng, 1, 20));
            let c = range(rng, 1, a + b);
            qa(
                &format!(
                    "{who} has {a} {item}, buys {b} more and then gives {c} to a friend. How many {item} does {who} have?"
                ),
                &format!(
                    "First {a} + {b} = {}. Then {} - {c} = {}. {who} has {} {item}.",
                    a + b,
                    a + b,
                    a + b - c,
                    a + b - c
                ),
            )
        }
    }
}

fn comparison(rng: &mut Rng, d: u8) -> String {
    // (comparative, superlative, opposite superlative)
    const SCALES: [(&str, &str, &str); 4] = [
        ("taller", "tallest", "shortest"),
        ("older", "oldest", "youngest"),
        ("faster", "fastest", "slowest"),
        ("heavier", "heaviest", "lightest"),
    ];
    let (more, most, least) = SCALES[rng.below(SCALES.len())];
    let n = (2 + d as usize).min(5);
    let order = names(rng, n); // order[0] is the most
    let mut facts: Vec<String> =
        (0..n - 1).map(|i| format!("{} is {more} than {}.", order[i], order[i + 1])).collect();
    if d >= 2 {
        for i in (1..facts.len()).rev() {
            facts.swap(i, rng.below(i + 1));
        }
    }
    let story = facts.join(" ");
    match rng.below(3) {
        0 => qa(
            &format!("{story} Who is the {most}?"),
            &format!("{} is {more} than everyone else, so {} is the {most}.", order[0], order[0]),
        ),
        1 => qa(
            &format!("{story} Who is the {least}?"),
            &format!("Everyone else is {more} than {}, so {} is the {least}.", order[n - 1], order[n - 1]),
        ),
        _ => {
            let i = rng.below(n);
            let mut j = rng.below(n);
            if j == i {
                j = (i + 1) % n;
            }
            let (a, b) = (order[i], order[j]);
            if i < j {
                qa(&format!("{story} Is {a} {more} than {b}?"), &format!("Yes. {a} is {more} than {b}."))
            } else {
                qa(&format!("{story} Is {a} {more} than {b}?"), &format!("No. {b} is {more} than {a}."))
            }
        }
    }
}

fn logic(rng: &mut Rng, _d: u8) -> String {
    // (if-part, its fact, its denial, its question; then-part, fact, denial, question)
    const RULES: [[&str; 8]; 6] = [
        [
            "it rains",
            "It is raining.",
            "It is not raining.",
            "Is it raining?",
            "the grass gets wet",
            "The grass is wet.",
            "The grass is not wet.",
            "Is the grass wet?",
        ],
        [
            "the bell rings",
            "The bell is ringing.",
            "The bell is not ringing.",
            "Is the bell ringing?",
            "the class ends",
            "The class has ended.",
            "The class has not ended.",
            "Has the class ended?",
        ],
        [
            "the lamp is on",
            "The lamp is on.",
            "The lamp is off.",
            "Is the lamp on?",
            "the room is bright",
            "The room is bright.",
            "The room is dark.",
            "Is the room bright?",
        ],
        [
            "the dog is hungry",
            "The dog is hungry.",
            "The dog is not hungry.",
            "Is the dog hungry?",
            "the dog barks",
            "The dog is barking.",
            "The dog is not barking.",
            "Is the dog barking?",
        ],
        [
            "the shop is open",
            "The shop is open.",
            "The shop is closed.",
            "Is the shop open?",
            "the lights are on",
            "The lights are on.",
            "The lights are off.",
            "Are the lights on?",
        ],
        [
            "you press the button",
            "You press the button.",
            "You do not press the button.",
            "Did you press the button?",
            "the door opens",
            "The door opens.",
            "The door stays shut.",
            "Does the door open?",
        ],
    ];
    // (made-up group, property, members)
    const GROUPS: [(&str, &str, &str); 4] = [
        ("blickets", "blue", "Dax"),
        ("wugs", "round", "Fep"),
        ("zorps", "loud", "Kip"),
        ("grobs", "soft", "Tam"),
    ];
    let r = RULES[rng.below(RULES.len())];
    let rule = format!("If {}, then {}.", r[0], r[4]);
    match rng.below(7) {
        0 => qa(&format!("{rule} {} {}", r[1], r[7]), &format!("Yes. The rule says that if {}, then {}.", r[0], r[4])),
        1 => qa(
            &format!("{rule} {} {}", r[6], r[3]),
            &format!("No. If {} then {}, and that did not happen.", r[0], r[4]),
        ),
        2 => qa(
            &format!("{rule} {} {}", r[5], r[3]),
            "We cannot tell. The rule only says what follows from the first part; something else could have caused it.",
        ),
        3 => qa(
            &format!("{rule} {} {}", r[2], r[7]),
            "We cannot tell. The rule does not say what happens when the first part is false.",
        ),
        k => {
            let (group, prop, member) = GROUPS[rng.below(GROUPS.len())];
            let one = group.trim_end_matches('s');
            match k {
                4 => qa(
                    &format!("All {group} are {prop}. {member} is a {one}. Is {member} {prop}?"),
                    &format!("Yes. {member} is a {one}, and all {group} are {prop}."),
                ),
                5 => qa(
                    &format!("All {group} are {prop}. {member} is not {prop}. Is {member} a {one}?"),
                    &format!("No. Every {one} is {prop}, and {member} is not {prop}."),
                ),
                _ => qa(
                    &format!("Some {group} are {prop}. {member} is a {one}. Is {member} {prop}?"),
                    &format!("We cannot tell. Only some {group} are {prop}."),
                ),
            }
        }
    }
}

fn world(rng: &mut Rng, d: u8) -> String {
    let people = names(rng, 2 + d as usize);
    let mut place: Vec<Option<&str>> = vec![None; people.len()];
    let mut holder: Vec<(&str, Option<usize>, Option<&str>)> = Vec::new(); // object, held by, lies in
    let objects = d >= 2;
    let steps = 3 + 2 * d as usize + rng.below(3);
    let mut lines = Vec::new();
    for _ in 0..steps {
        let p = rng.below(people.len());
        let action = if objects { rng.below(4) } else { 0 };
        match action {
            1 if place[p].is_some() && holder.len() < 3 => {
                let free: Vec<&str> =
                    OBJECTS.iter().copied().filter(|o| !holder.iter().any(|h| h.0 == *o)).collect();
                let o = free[rng.below(free.len())];
                holder.push((o, Some(p), None));
                lines.push(format!("{} picked up the {o}.", people[p]));
            }
            2 => {
                if let Some(h) = holder.iter_mut().find(|h| h.1 == Some(p)) {
                    h.1 = None;
                    h.2 = place[p];
                    lines.push(format!("{} put down the {}.", people[p], h.0));
                } else {
                    let to = pick(rng, &PLACES);
                    place[p] = Some(to);
                    lines.push(format!("{} {} the {to}.", people[p], pick(rng, &MOVES)));
                }
            }
            _ => {
                let to = pick(rng, &PLACES);
                place[p] = Some(to);
                lines.push(format!("{} {} the {to}.", people[p], pick(rng, &MOVES)));
            }
        }
    }
    let story = lines.join(" ");
    let known: Vec<usize> = (0..people.len()).filter(|&i| place[i].is_some()).collect();
    let Some(&p) = known.get(rng.below(known.len().max(1))) else {
        return qa(&format!("{story} Where is {}?", people[0]), "We cannot tell from the story.");
    };
    let here = place[p].unwrap_or("");
    let held: Vec<&str> = holder.iter().filter(|h| h.1 == Some(p)).map(|h| h.0).collect();
    match rng.below(if objects { 4 } else { 2 }) {
        0 => qa(&format!("{story} Where is {}?", people[p]), &format!("{} is in the {here}.", people[p])),
        1 => {
            let other = PLACES.iter().copied().find(|x| *x != here).unwrap_or("park");
            let ask = if rng.below(2) == 0 { here } else { other };
            if ask == here {
                qa(
                    &format!("{story} Is {} in the {ask}?", people[p]),
                    &format!("Yes. {} is in the {here}.", people[p]),
                )
            } else {
                qa(
                    &format!("{story} Is {} in the {ask}?", people[p]),
                    &format!("No. {} is in the {here}.", people[p]),
                )
            }
        }
        2 => {
            let n = held.len();
            let words = ["nothing", "one thing", "two things", "three things"];
            let what = if n == 0 { String::new() } else { format!(": the {}", held.join(" and the ")) };
            qa(
                &format!("{story} How many things is {} carrying?", people[p]),
                &format!("{} is carrying {}{what}.", people[p], words[n.min(3)]),
            )
        }
        _ => match holder.first() {
            Some((o, Some(h), _)) => {
                let at = place[*h]
                    .map(|x| format!("in the {x}"))
                    .unwrap_or_else(|| "somewhere we were not told".into());
                qa(
                    &format!("{story} Where is the {o}?"),
                    &format!("{} has the {o}, and {} is {at}. The {o} is {at}.", people[*h], people[*h]),
                )
            }
            Some((o, None, Some(at))) => qa(
                &format!("{story} Where is the {o}?"),
                &format!("The {o} was put down in the {at}, so it is in the {at}."),
            ),
            _ => qa(&format!("{story} Where is {}?", people[p]), &format!("{} is in the {here}.", people[p])),
        },
    }
}

fn sequence(rng: &mut Rng, d: u8) -> String {
    let kinds = if d >= 2 { 4 } else { 2 };
    match rng.below(kinds) {
        0 => {
            let (start, step) = (range(rng, 0, 20), range(rng, 1, [3, 9, 25][(d.clamp(1, 3) - 1) as usize]));
            let xs: Vec<i64> = (0..4).map(|i| start + i * step).collect();
            qa(
                &format!("What comes next: {}, {}, {}, {}?", xs[0], xs[1], xs[2], xs[3]),
                &format!("Each number is {step} more than the one before, so the next is {}.", xs[3] + step),
            )
        }
        1 => {
            let start = range(rng, 0, 15) as u8;
            let step = range(rng, 1, 2) as u8;
            let xs: Vec<char> = (0..4).map(|i| (b'A' + start + i * step) as char).collect();
            let next = (b'A' + start + 4 * step) as char;
            qa(
                &format!("What comes next: {}, {}, {}, {}?", xs[0], xs[1], xs[2], xs[3]),
                &format!("The letters go forward by {step} each time, so the next is {next}."),
            )
        }
        2 => {
            let start = range(rng, 1, 5);
            let xs: Vec<i64> = (0..4).map(|i| start << i).collect();
            qa(
                &format!("What comes next: {}, {}, {}, {}?", xs[0], xs[1], xs[2], xs[3]),
                &format!("Each number is double the one before, so the next is {}.", xs[3] * 2),
            )
        }
        _ => {
            let (start, step) = (range(rng, 60, 100), range(rng, 2, 9));
            let xs: Vec<i64> = (0..4).map(|i| start - i * step).collect();
            qa(
                &format!("What comes next: {}, {}, {}, {}?", xs[0], xs[1], xs[2], xs[3]),
                &format!("Each number is {step} less than the one before, so the next is {}.", xs[3] - step),
            )
        }
    }
}

/// One example of a family.
pub fn example(family: Family, difficulty: u8, rng: &mut Rng) -> String {
    let d = difficulty.clamp(1, 3);
    match family {
        Family::Arithmetic => arithmetic(rng, d),
        Family::WordProblems => word_problem(rng, d),
        Family::Comparisons => comparison(rng, d),
        Family::Logic => logic(rng, d),
        Family::World => world(rng, d),
        Family::Sequences => sequence(rng, d),
    }
}

/// Generates about `cfg.bytes` of examples, one per document, families taken in turn.
pub fn generate(cfg: &GenerateConfig) -> Result<String, String> {
    if cfg.families.is_empty() {
        return Err("Choose at least one kind of practice".into());
    }
    let mut rng = Rng::new(cfg.seed);
    let mut out = String::with_capacity(cfg.bytes + 1024);
    let mut k = 0;
    while out.len() < cfg.bytes {
        let family = cfg.families[k % cfg.families.len()];
        out.push_str(&example(family, cfg.difficulty, &mut rng));
        out.push('\n');
        out.push_str(END_OF_TEXT);
        out.push('\n');
        k += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::corpus::Corpus;

    #[test]
    fn generated_text_is_deterministic_ascii_chat_and_splits() {
        let cfg = GenerateConfig { families: Family::ALL.to_vec(), difficulty: 3, bytes: 200_000, seed: 7 };
        let a = generate(&cfg).unwrap();
        assert_eq!(a, generate(&cfg).unwrap());
        assert!(a.is_ascii() && a.len() >= 200_000);
        let c = Corpus::prepare(&a);
        assert!(c.held_out.len() > 50 && c.train.len() > 10 * c.held_out.len() / 2);
        assert!(c.train.iter().all(|d| d.starts_with("U: ") || d.contains("\nU: ") || d.contains("U: ")));
        assert!(c.train.iter().all(|d| d.contains("\nN: ")));
    }

    #[test]
    fn answers_are_computed() {
        let mut rng = Rng::new(3);
        for _ in 0..500 {
            let t = arithmetic(&mut rng, 1);
            // "U: What is a plus b?\nN: a + b = c."
            let n = t.split("\nN: ").nth(1).unwrap().trim_end_matches('.');
            let (lhs, rhs) = n.split_once(" = ").unwrap();
            let v: Vec<i64> = lhs.split(' ').filter_map(|x| x.parse().ok()).collect();
            let r: i64 = rhs.parse().unwrap();
            assert!(if lhs.contains('+') { v[0] + v[1] == r } else { v[0] - v[1] == r }, "{t}");
        }
        for _ in 0..200 {
            let t = comparison(&mut rng, 1);
            // With 3 people in order, the tallest is the first name of the first fact.
            if t.contains("Who is the") && !t.contains("Everyone else is") {
                let first = t.trim_start_matches("U: ").split(' ').next().unwrap();
                assert!(t.contains(&format!("N: {first} is")), "{t}");
            }
        }
    }

    #[test]
    fn world_answers_follow_the_story() {
        let mut rng = Rng::new(11);
        for _ in 0..300 {
            let t = world(&mut rng, 1);
            // Without objects: the answer place is where that person last went.
            let (q, a) = t.split_once("\nN: ").unwrap();
            let q = q.trim_start_matches("U: ");
            let Some(who) = q.rsplit("Where is ").next().and_then(|x| x.strip_suffix('?')) else { continue };
            if !q.contains("Where is ") || a.starts_with("We cannot") {
                continue;
            }
            let last = q
                .split(". ")
                .filter(|s| s.starts_with(who) && s.contains(" the "))
                .last()
                .and_then(|s| s.rsplit(" the ").next())
                .map(|s| s.trim_end_matches(['.', '?']).split(' ').next().unwrap().to_string())
                .unwrap();
            assert!(a.contains(&format!("in the {last}")), "{t}");
        }
    }
}
