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

/// Making up your mind: choosing with a reason, saying no when there is a reason to, keeping to
/// what you said before, changing your mind for a reason and keeping your view without one.
/// Every choice follows from what the question says (a need, a fact, what Nova said earlier),
/// never from a preference written into the lesson: across the examples each option is chosen
/// as often as the other, so the lessons teach how to decide, not what to like.
fn deciding() -> BuiltinCourse {
    // (situation, the option that fits, the reason it fits)
    let needs: [(&str, &str, &str); 30] = [
        ("It is raining and you want to stay dry", "the umbrella", "it keeps me dry"),
        ("It is cold outside", "the warm coat", "it keeps me warm"),
        ("You are very hungry", "the sandwich", "it will fill me up"),
        ("You are thirsty", "the water", "it will quench my thirst"),
        ("It is dark in the room", "the lamp", "it gives light"),
        ("You want to write a note", "the pen", "I can write with it"),
        ("You need to open the door", "the key", "it opens the door"),
        ("You want to read a story", "the book", "it has a story in it"),
        ("The sun is very bright", "the sun hat", "it shades my eyes"),
        ("You are going to the beach", "the swimsuit", "I can swim in it"),
        ("You want to cut the paper", "the scissors", "they cut paper"),
        ("You have to carry many apples", "the big bag", "it holds many apples"),
        ("You are tired after a long day", "the soft bed", "I can rest in it"),
        ("You want to listen to music", "the radio", "it plays music"),
        ("The floor is dirty", "the broom", "it sweeps the floor"),
        ("You want to call a friend", "the phone", "I can call with it"),
        ("Your hands are wet", "the towel", "it dries my hands"),
        ("You want to plant a flower", "the shovel", "it digs the soil"),
        ("The road is long and you are in a hurry", "the bike", "it is faster than walking"),
        ("You want to see far away", "the telescope", "it shows things far away"),
        ("It is snowing", "the boots", "they keep my feet warm"),
        ("You want to paint a picture", "the brush", "I can paint with it"),
        ("You want to eat soup", "the spoon", "I can eat soup with it"),
        ("You need to fix a loose nail", "the hammer", "it drives nails in"),
        ("You want to keep the milk cold", "the fridge", "it keeps food cold"),
        ("You want to know the time", "the clock", "it shows the time"),
        ("The plants are dry", "the watering can", "it gives them water"),
        ("You want to find your way in a new town", "the map", "it shows the way"),
        ("You need to lock your bike", "the chain", "it keeps the bike safe"),
        ("It is windy and you want to fly something", "the kite", "the wind lifts it"),
    ];
    let others = [
        "the spoon",
        "the rock",
        "the sock",
        "the kite",
        "the cup",
        "the ball",
        "the hat",
        "the leaf",
        "the shoe",
        "the coin",
    ];
    let forms: [(&str, &str); 4] = [
        ("Will you take {a} or {b}?", "I will take"),
        ("Which do you pick, {a} or {b}?", "I pick"),
        ("Would you rather have {a} or {b}?", "I would rather have"),
        ("Do you want {a} or {b}?", "I want"),
    ];
    let mut choose = Vec::new();
    for (k, (need, good, why)) in needs.iter().enumerate() {
        // A distractor that does not fit the need (never the good option itself).
        let other = others.iter().cycle().skip(k).find(|o| *o != good).unwrap();
        for (q, a) in forms {
            for (x, y) in [(good, other), (other, good)] {
                choose.push((
                    format!("{need}. {}", q.replace("{a}", x).replace("{b}", y)),
                    format!("{a} {good}, because {why}."),
                ));
            }
        }
    }
    let choose = all_of(41, EXAMPLES, choose);

    // (activity, fact against it, reason no, fact for it, reason yes)
    let plans: [(&str, &str, &str, &str, &str); 25] = [
        (
            "go swimming",
            "The water is very cold today",
            "the water is too cold",
            "The water is warm today",
            "the water is warm",
        ),
        ("have a picnic", "It is raining hard", "it is raining", "The sun is shining", "the sun is shining"),
        (
            "play football",
            "The field is flooded",
            "the field is flooded",
            "The field is dry and green",
            "the field is dry",
        ),
        (
            "go for a walk",
            "There is a big storm",
            "there is a storm",
            "The weather is calm",
            "the weather is calm",
        ),
        (
            "eat the cake",
            "The cake has gone bad",
            "the cake has gone bad",
            "The cake is fresh",
            "the cake is fresh",
        ),
        (
            "go to the shop",
            "The shop is closed today",
            "the shop is closed",
            "The shop is open",
            "the shop is open",
        ),
        (
            "ride our bikes",
            "Both bikes have flat tires",
            "the tires are flat",
            "The bikes are fixed",
            "the bikes are fixed",
        ),
        (
            "go skating",
            "The ice is thin and cracking",
            "the ice is not safe",
            "The ice is thick and safe",
            "the ice is safe",
        ),
        (
            "climb the tree",
            "The branches are broken",
            "the branches are broken",
            "The branches are strong",
            "the branches are strong",
        ),
        (
            "watch the film",
            "The film is very scary for us",
            "it is too scary",
            "The film is funny",
            "the film is funny",
        ),
        (
            "stay up late",
            "We have to wake up very early",
            "we have to wake up early",
            "Tomorrow is a holiday",
            "tomorrow is a holiday",
        ),
        (
            "go to the beach",
            "There are sharks in the water",
            "there are sharks",
            "The beach is calm and sunny",
            "it is calm and sunny",
        ),
        (
            "drink the milk",
            "The milk smells sour",
            "the milk is sour",
            "The milk is fresh",
            "the milk is fresh",
        ),
        (
            "visit the zoo",
            "The zoo is closed for repairs",
            "the zoo is closed",
            "The zoo has new animals",
            "there are new animals",
        ),
        (
            "sit on the bench",
            "The paint on the bench is wet",
            "the paint is wet",
            "The bench is clean and dry",
            "the bench is dry",
        ),
        (
            "go fishing",
            "The lake is frozen",
            "the lake is frozen",
            "The fish are biting today",
            "the fish are biting",
        ),
        (
            "play in the garden",
            "The garden is full of bees",
            "there are too many bees",
            "The flowers are out",
            "the garden is lovely",
        ),
        (
            "bake cookies",
            "We have no flour",
            "we have no flour",
            "We have everything we need",
            "we have everything",
        ),
        (
            "cross the river",
            "The bridge is broken",
            "the bridge is broken",
            "The bridge is new and strong",
            "the bridge is strong",
        ),
        (
            "go camping",
            "A big storm is coming tonight",
            "a storm is coming",
            "The night will be clear",
            "the night will be clear",
        ),
        (
            "take the boat out",
            "The waves are very high",
            "the waves are too high",
            "The sea is calm",
            "the sea is calm",
        ),
        (
            "paint the fence",
            "It is about to rain",
            "the rain will wash it off",
            "It will be dry all day",
            "it will stay dry",
        ),
        (
            "go to the park",
            "The park is closed today",
            "the park is closed",
            "The park has a fair today",
            "there is a fair",
        ),
        ("eat outside", "It is freezing cold", "it is freezing", "It is a warm evening", "it is warm"),
        (
            "play tennis",
            "The court is covered in snow",
            "the court has snow on it",
            "The court is free now",
            "the court is free",
        ),
    ];
    let invites = ["Let's {x}.", "We could {x}.", "How about we {x}?", "Shall we {x}?"];
    let mut decline = Vec::new();
    for (act, bad, no_why, good, yes_why) in plans {
        for inv in invites {
            let ask = inv.replace("{x}", act);
            decline.push((format!("{ask} {bad}."), format!("No, I do not want to {act}, because {no_why}.")));
            decline.push((format!("{ask} {good}."), format!("Yes, let's {act}, because {yes_why}.")));
        }
    }
    let decline = all_of(42, EXAMPLES, decline);

    let kinds: [(&str, &[&str]); 4] = [
        (
            "food",
            &["apples", "pears", "rice", "bread", "soup", "cheese", "grapes", "carrots", "eggs", "plums"],
        ),
        ("color", &["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"]),
        ("animal", &["cats", "dogs", "birds", "horses", "rabbits", "fish", "ducks", "goats"]),
        (
            "place",
            &[
                "the sea",
                "the hills",
                "the city",
                "the woods",
                "the farm",
                "the lake",
                "the desert",
                "the town",
            ],
        ),
    ];
    let said = examples(43, EXAMPLES, |r| {
        let (_, list) = kinds[r.below(kinds.len())];
        let (liked, other) = r.two(list);
        let (x, y) = if r.below(2) == 0 { (liked, other) } else { (other, liked) };
        if r.below(2) == 0 {
            (
                format!(
                    "Earlier you said: I like {liked} more than {other}. Which do you choose, {x} or {y}?"
                ),
                format!("I choose {liked}, because I said I like {liked} more."),
            )
        } else {
            (
                format!("Earlier you said: I do not like {other} much. Which do you choose, {x} or {y}?"),
                format!("I choose {liked}, because I said I do not like {other} much."),
            )
        }
    });

    // (plan, news against it, what to do instead, news that changes nothing)
    let changes: [(&str, &str, &str, &str); 16] = [
        ("walk to school", "it is raining hard", "take the bus", "a bird is singing"),
        ("eat lunch in the park", "the park is closed", "eat lunch at home", "the clouds are white"),
        ("read the red book", "the red book is lost", "read the blue book", "the kettle is boiling"),
        ("swim in the lake", "the lake is frozen", "swim in the pool", "a dog is barking"),
        ("ride the bike", "the bike is broken", "walk", "the clock struck noon"),
        ("bake bread", "there is no flour", "make soup", "the sun came out"),
        ("visit Grandma on Monday", "Grandma is away on Monday", "visit her on Tuesday", "it is a sunny day"),
        ("paint the door blue", "the blue paint is gone", "paint it green", "the radio is on"),
        ("go to the beach", "a storm is coming", "stay at home", "the cat is asleep"),
        ("play chess with Sam", "Sam is ill today", "play chess with Mia", "the tea is ready"),
        ("take the train", "the trains are not running", "take the car", "the shop is busy"),
        ("plant the seeds today", "the ground is frozen", "plant them next week", "the bell rang"),
        ("wear the white shirt", "the white shirt is dirty", "wear the gray shirt", "the window is open"),
        ("climb the hill", "the path is blocked", "walk by the river", "a leaf fell down"),
        ("have tea", "the tea is all gone", "have juice", "the door is shut"),
        ("cross the bridge", "the bridge is closed", "take the ferry", "the grass is green"),
    ];
    let mut mind = Vec::new();
    for (plan, against, instead, nothing) in changes {
        for lead in [
            "You said you would",
            "Earlier you planned to",
            "You told me you would",
            "This morning you decided to",
            "Yesterday you said you would",
        ] {
            mind.push((
                format!("{lead} {plan}. Now {against}. Will you still {plan}?"),
                format!("No, I will {instead}, because {against}."),
            ));
            mind.push((
                format!("{lead} {plan}. Now {nothing}. Will you still {plan}?"),
                format!("Yes, I will still {plan}."),
            ));
        }
    }
    let mind = all_of(44, EXAMPLES, mind);

    let view = examples(45, EXAMPLES, |r| {
        let n = r.pick(&NAMES);
        let o = r.pick(&OBJECTS);
        let (mine, theirs) = r.two(&COLOURS);
        if r.below(2) == 0 {
            (
                format!("You saw the {o}, and it is {mine}. {n} did not see it and says it is {theirs}. What color is the {o}?"),
                format!("It is {mine}. I saw it, and {n} did not."),
            )
        } else {
            (
                format!(
                    "You did not see the {o}. {n} saw it and says it is {theirs}. What color is the {o}?"
                ),
                format!("It is {theirs}, because {n} saw it."),
            )
        }
    });

    BuiltinCourse {
        id: "deciding",
        name: "Making up your mind (built-in)",
        description: "Choosing with a reason, saying no when there is a reason to, keeping to what you said before, changing your mind for a reason, and keeping your view when someone else has no reason. Each choice follows from what the question says, never from a preference written into the lessons.",
        lessons: vec![
            file("01 Choosing for a reason", qa_block("Dialogue Lesson", "Choose the thing that fits what you need, and say why.", &choose)),
            file("02 Saying no for a reason", qa_block("Dialogue Lesson", "Say no when there is a good reason not to, and yes when there is not. Give the reason.", &decline)),
            file("03 What you said before", qa_block("Dialogue Lesson", "Choose in keeping with what you said earlier, and say so.", &said)),
            file("04 Changing your mind", qa_block("Dialogue Lesson", "Change a plan when there is a reason to, and keep it when nothing matters to it.", &mind)),
            file("05 Keeping your view", qa_block("Dialogue Lesson", "Keep what you saw when the other person has no reason. Change your view when they do.", &view)),
        ],
    }
}

/// The built-in courses, generated (the same text every time).
pub fn builtin_courses() -> Vec<BuiltinCourse> {
    vec![english(), reasoning_course(), math(), deciding()]
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
