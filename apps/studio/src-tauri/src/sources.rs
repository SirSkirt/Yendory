//! Reading sources: where each corpus comes from, its licence, and how its records become
//! documents in Nova's own format (dialogue as `U:` / `N:` turns, questions with their worked
//! answers). Every source was checked from a GitHub runner (format, licence, reachability);
//! the downloads themselves are in `reading.rs`.
use serde::Serialize;
use serde_json::Value;
use yendory_engine::learn::corpus::Category;

/// Converts one record (a JSON row) into documents.
pub type Convert = fn(&Value) -> Vec<String>;

#[derive(Clone, Copy)]
pub enum Fetch {
    /// A plain text file (stories separated by `<|endoftext|>`).
    Text(&'static str),
    /// Rows of a Hugging Face dataset through its dataset-viewer API (JSON pages of 100 rows).
    HfRows { dataset: &'static str, configs: &'static [&'static str], split: &'static str, convert: Convert },
    /// A JSON-lines file, one record per line.
    JsonLines(&'static str, Convert),
    /// Project Gutenberg books by number (plain text; the Gutenberg header and licence are removed).
    Gutenberg(&'static [(u32, &'static str)]),
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    pub id: &'static str,
    pub name: &'static str,
    pub category: Category,
    pub description: &'static str,
    pub page: &'static str,
    pub license: &'static str,
    /// How to credit the source (CC BY and CC BY-SA require attribution).
    pub attribution: &'static str,
    /// Size of the whole source in MB once converted (approximate).
    pub size_mb: u64,
    pub default_mb: u64,
    #[serde(skip)]
    pub fetch: Fetch,
}

/// Only task 1 of the 20 is available as data from the Hugging Face dataset viewer (the
/// others answer "not found"; checked from a GitHub runner).
const BABI: &[&str] = &["en-10k-qa1"];

/// Children's books checked on gutenberg.org (number, title).
const BOOKS: &[(u32, &str)] = &[
    (19994, "The Aesop for Children"),
    (11, "Alice's Adventures in Wonderland"),
    (12, "Through the Looking-Glass"),
    (16, "Peter Pan"),
    (55, "The Wonderful Wizard of Oz"),
    (2781, "Just So Stories"),
    (236, "The Jungle Book"),
    (271, "Black Beauty"),
    (289, "The Wind in the Willows"),
    (113, "The Secret Garden"),
    (146, "A Little Princess"),
    (500, "The Adventures of Pinocchio"),
    (902, "The Happy Prince, and Other Tales"),
    (1448, "Heidi"),
    (1597, "Andersen's Fairy Tales"),
    (2591, "Grimms' Fairy Tales"),
    (7439, "English Fairy Tales"),
    (17208, "The Tales of Mother Goose"),
    (11757, "The Velveteen Rabbit"),
    (120, "Treasure Island"),
    (45, "Anne of Green Gables"),
    (514, "Little Women"),
];

pub fn sources() -> Vec<Source> {
    const TINY: &str = "https://huggingface.co/datasets/roneneldan/TinyStories";
    const TINY_LICENSE: &str = "CDLA-Sharing-1.0";
    vec![
        Source {
            id: "tinystories-valid",
            name: "TinyStories (validation file)",
            category: Category::General,
            description: "About 22,000 short stories in simple English, written for training very small language models.",
            page: TINY,
            license: TINY_LICENSE,
            attribution: "TinyStories, Eldan and Li (2023)",
            size_mb: 19,
            default_mb: 19,
            fetch: Fetch::Text(
                "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
            ),
        },
        Source {
            id: "tinystories-train",
            name: "TinyStories (training file)",
            category: Category::General,
            description: "The same kind of stories, about 2.1 million of them. Download only the first part you need.",
            page: TINY,
            license: TINY_LICENSE,
            attribution: "TinyStories, Eldan and Li (2023)",
            size_mb: 1900,
            default_mb: 50,
            fetch: Fetch::Text(
                "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
            ),
        },
        Source {
            id: "gutenberg-children",
            name: "Children's classics (Project Gutenberg)",
            category: Category::General,
            description: "22 children's books: Aesop, Alice, Peter Pan, Oz, Just So Stories, fairy tales and more.",
            page: "https://www.gutenberg.org",
            license: "Public domain in the USA (Project Gutenberg; its licence text is removed)",
            attribution: "Project Gutenberg, www.gutenberg.org",
            size_mb: 8,
            default_mb: 8,
            fetch: Fetch::Gutenberg(BOOKS),
        },
        Source {
            id: "simple-wikipedia",
            name: "Simple English Wikipedia",
            category: Category::Knowledge,
            description: "Encyclopedia articles written in simple English (November 2023). Download only the first part you need.",
            page: "https://huggingface.co/datasets/wikimedia/wikipedia",
            license: "CC BY-SA 3.0 and GFDL",
            attribution: "Wikipedia contributors, Simple English Wikipedia (simple.wikipedia.org)",
            size_mb: 280,
            default_mb: 60,
            fetch: Fetch::HfRows {
                dataset: "wikimedia/wikipedia",
                configs: &["20231101.simple"],
                split: "train",
                convert: wikipedia,
            },
        },
        Source {
            id: "soda",
            name: "SODA everyday conversations",
            category: Category::Conversation,
            description: "1.2 million short social conversations, each with a one-paragraph situation. Download only the first part you need.",
            page: "https://huggingface.co/datasets/allenai/soda",
            license: "CC BY 4.0",
            attribution: "SODA, Kim et al. (2023), Allen Institute for AI",
            size_mb: 1300,
            default_mb: 60,
            fetch: Fetch::HfRows { dataset: "allenai/soda", configs: &["default"], split: "train", convert: soda },
        },
        Source {
            id: "persona-chat",
            name: "Synthetic-Persona-Chat",
            category: Category::Conversation,
            description: "Longer chats between two people getting to know each other (the training split, about 16 MB).",
            page: "https://huggingface.co/datasets/google/Synthetic-Persona-Chat",
            license: "CC BY 4.0",
            attribution: "Synthetic-Persona-Chat, Jandaghi et al. (2023), Google",
            size_mb: 15,
            default_mb: 15,
            fetch: Fetch::HfRows {
                dataset: "google/Synthetic-Persona-Chat",
                configs: &["default"],
                split: "train",
                convert: persona_chat,
            },
        },
        Source {
            id: "babi",
            name: "bAbI task 1: where is someone",
            category: Category::Reasoning,
            description: "Short stories of people moving between rooms, then \"Where is ...?\" (10,000 questions). Only task 1 of the 20 bAbI tasks can be downloaded as data; the practice generator below covers similar reasoning (where things are, who has what, counting, yes/no, comparisons, deduction).",
            page: "https://huggingface.co/datasets/facebook/babi_qa",
            license: "CC BY 3.0",
            attribution: "bAbI tasks, Weston et al. (2015), Facebook AI Research",
            size_mb: 3,
            default_mb: 3,
            fetch: Fetch::HfRows { dataset: "facebook/babi_qa", configs: BABI, split: "train", convert: babi },
        },
        Source {
            id: "gsm8k",
            name: "GSM8K school maths",
            category: Category::Reasoning,
            description: "7,473 grade-school word problems with worked step-by-step answers.",
            page: "https://github.com/openai/grade-school-math",
            license: "MIT",
            attribution: "GSM8K, Cobbe et al. (2021), OpenAI",
            size_mb: 4,
            default_mb: 4,
            fetch: Fetch::JsonLines(
                "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
                gsm8k,
            ),
        },
        Source {
            id: "proofwriter",
            name: "ProofWriter rule reasoning",
            category: Category::Reasoning,
            description: "Facts and if-then rules about people and things, and whether a statement follows (true, false or unknown).",
            page: "https://huggingface.co/datasets/tasksource/proofwriter",
            license: "CC BY 4.0 per AI2's original release (reported in the ALERT paper); this mirror states no licence",
            attribution: "ProofWriter, Tafjord et al. (2021), Allen Institute for AI",
            size_mb: 600,
            default_mb: 30,
            fetch: Fetch::HfRows {
                dataset: "tasksource/proofwriter",
                configs: &["default"],
                split: "train",
                convert: proofwriter,
            },
        },
    ]
}

/// Conversation datasets for the Conversation stage: the same conversations as the Reading
/// sources, kept with their speakers' names (`Name: text` lines), so Nova's speaker can be given
/// Nova's own name. Both are conversations between two people, not an assistant and a user.
pub fn conversation_sources() -> Vec<Source> {
    vec![
        Source {
            id: "soda-dialogues",
            name: "SODA conversations (with speakers' names)",
            category: Category::Conversation,
            description: "Everyday social conversations between two people: greetings, plans, worries, jokes, disagreements. 1.2 million in all; download only the first part you need.",
            page: "https://huggingface.co/datasets/allenai/soda",
            license: "CC BY 4.0",
            attribution: "SODA, Kim et al. (2023), Allen Institute for AI",
            size_mb: 1100,
            default_mb: 40,
            fetch: Fetch::HfRows { dataset: "allenai/soda", configs: &["default"], split: "train", convert: soda_speakers },
        },
        Source {
            id: "persona-chat-dialogues",
            name: "Synthetic-Persona-Chat (with speakers' names)",
            category: Category::Conversation,
            description: "Two people getting to know each other: their likes, families, work and plans (the training split, about 15 MB).",
            page: "https://huggingface.co/datasets/google/Synthetic-Persona-Chat",
            license: "CC BY 4.0",
            attribution: "Synthetic-Persona-Chat, Jandaghi et al. (2023), Google",
            size_mb: 15,
            default_mb: 15,
            fetch: Fetch::HfRows {
                dataset: "google/Synthetic-Persona-Chat",
                configs: &["default"],
                split: "train",
                convert: persona_speakers,
            },
        },
    ]
}

fn s<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("").trim()
}

/// Simple Wikipedia article: title, then its paragraphs up to the reference sections. Headings
/// and list fragments (short lines) are left out.
pub fn wikipedia(row: &Value) -> Vec<String> {
    const STOP: [&str; 6] =
        ["References", "Related pages", "Other websites", "Sources", "Notes", "Further reading"];
    let title = s(row, "title");
    let mut paras = Vec::new();
    for line in s(row, "text").lines().map(str::trim) {
        if STOP.contains(&line) {
            break;
        }
        if line.split_whitespace().count() >= 6 {
            paras.push(line);
        }
    }
    if title.is_empty() || paras.is_empty() {
        return Vec::new();
    }
    vec![format!("{title}\n{}", paras.join("\n"))]
}

/// SODA: the situation, then the conversation as alternating `U:` / `N:` turns.
pub fn soda(row: &Value) -> Vec<String> {
    let turns: Vec<&str> = row
        .get("dialogue")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    if turns.len() < 2 {
        return Vec::new();
    }
    let mut doc = s(row, "narrative").to_string();
    for (i, t) in turns.iter().enumerate() {
        doc.push_str(if i % 2 == 0 { "\nU: " } else { "\nN: " });
        doc.push_str(t.trim().trim_end_matches('"'));
    }
    vec![doc.trim_start().to_string()]
}

/// SODA with its speakers: the situation, then `Name: text` turns. Conversations with other than
/// two speakers, or a speaker name that is not a plain name, are left out.
pub fn soda_speakers(row: &Value) -> Vec<String> {
    let list = |k: &str| -> Vec<String> {
        row.get(k)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|t| t.trim().to_string())
            .collect()
    };
    let (turns, speakers) = (list("dialogue"), list("speakers"));
    let plain = |n: &str| {
        !n.is_empty()
            && n.len() <= 32
            && n.split(' ').count() <= 3
            && n.starts_with(|c: char| c.is_ascii_uppercase())
            && n.chars().all(|c| c.is_ascii_alphabetic() || " .'-".contains(c))
    };
    let mut names = speakers.clone();
    names.sort();
    names.dedup();
    if turns.len() < 2 || turns.len() != speakers.len() || names.len() != 2 || !names.iter().all(|n| plain(n))
    {
        return Vec::new();
    }
    let mut doc = String::new();
    let situation = s(row, "narrative");
    if !situation.is_empty() {
        doc.push_str(&format!("Situation: {}\n", situation.replace('\n', " ")));
    }
    for (who, said) in speakers.iter().zip(&turns) {
        doc.push_str(&format!("{who}: {}\n", said.trim_end_matches('"').replace('\n', " ")));
    }
    vec![doc.trim_end().to_string()]
}

/// Synthetic-Persona-Chat with its speakers: the name placeholders become names and each line
/// is `Name: text`.
pub fn persona_speakers(row: &Value) -> Vec<String> {
    const NAMES: [(&str, &str); 6] =
        [("Ann", "Ben"), ("Cara", "Dan"), ("Eva", "Finn"), ("Mia", "Leo"), ("Sara", "Tom"), ("Zoe", "Ray")];
    let text = s(row, "Best Generated Conversation");
    let (a, b) = NAMES[text.len() % NAMES.len()];
    let mut lines = Vec::new();
    for line in text.lines() {
        let line = line.replace("[user 1's name]", a).replace("[user 2's name]", b);
        if let Some(t) = line.strip_prefix("User 1:") {
            lines.push(format!("{a}: {}", t.trim()));
        } else if let Some(t) = line.strip_prefix("User 2:") {
            lines.push(format!("{b}: {}", t.trim()));
        }
    }
    if lines.len() < 2 {
        return Vec::new();
    }
    vec![lines.join("\n")]
}

/// Synthetic-Persona-Chat: "User 1:" becomes `U:` and "User 2:" `N:`; the name placeholders
/// become names.
pub fn persona_chat(row: &Value) -> Vec<String> {
    const NAMES: [(&str, &str); 6] =
        [("Ann", "Ben"), ("Cara", "Dan"), ("Eva", "Finn"), ("Mia", "Leo"), ("Sara", "Tom"), ("Zoe", "Ray")];
    let text = s(row, "Best Generated Conversation");
    let (a, b) = NAMES[text.len() % NAMES.len()];
    let mut lines = Vec::new();
    for line in text.lines() {
        let line = line.replace("[user 1's name]", a).replace("[user 2's name]", b);
        if let Some(t) = line.strip_prefix("User 1:") {
            lines.push(format!("U: {}", t.trim()));
        } else if let Some(t) = line.strip_prefix("User 2:") {
            lines.push(format!("N: {}", t.trim()));
        }
    }
    if lines.len() < 2 {
        return Vec::new();
    }
    vec![lines.join("\n")]
}

/// bAbI: for each question, the story so far, then `U:` question and `N:` answer.
pub fn babi(row: &Value) -> Vec<String> {
    let story = &row["story"];
    let list = |k: &str| story.get(k).and_then(Value::as_array).cloned().unwrap_or_default();
    let (kinds, texts, answers) = (list("type"), list("text"), list("answer"));
    let mut context: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for ((k, t), a) in kinds.iter().zip(&texts).zip(&answers) {
        let text = t.as_str().unwrap_or("").trim();
        if k.as_i64() == Some(1) {
            let answer = babi_answer(a.as_str().unwrap_or(""));
            if !answer.is_empty() {
                out.push(format!("{}\nU: {text}\nN: {answer}", context.join(" ")));
            }
        } else {
            context.push(text.to_string());
        }
    }
    out
}

fn babi_answer(a: &str) -> String {
    let parts: Vec<&str> = a.split(',').map(str::trim).filter(|p| !p.is_empty()).collect();
    let dirs = |p: &str| match p {
        "n" => Some("north"),
        "s" => Some("south"),
        "e" => Some("east"),
        "w" => Some("west"),
        _ => None,
    };
    let words: Vec<String> = if parts.iter().all(|p| dirs(p).is_some()) {
        parts.iter().filter_map(|p| dirs(p)).map(String::from).collect()
    } else {
        parts.iter().map(|p| p.to_string()).collect()
    };
    let mut answer = match words.len() {
        0 => return String::new(),
        1 => words[0].clone(),
        _ => format!("{} and {}", words[..words.len() - 1].join(", "), words[words.len() - 1]),
    };
    if let Some(c) = answer.get(..1) {
        answer = c.to_ascii_uppercase() + &answer[1..];
    }
    answer + "."
}

/// GSM8K: the question, then the worked answer without the calculator marks, ending with the
/// final number.
pub fn gsm8k(row: &Value) -> Vec<String> {
    let q = s(row, "question");
    let a = s(row, "answer");
    if q.is_empty() || a.is_empty() {
        return Vec::new();
    }
    let mut clean = String::new();
    let mut rest = a;
    while let Some(i) = rest.find("<<") {
        clean.push_str(&rest[..i]);
        rest = rest[i..].find(">>").map_or("", |j| &rest[i + j + 2..]);
    }
    clean.push_str(rest);
    let (steps, answer) = match clean.rsplit_once("####") {
        Some((steps, n)) => (steps.trim().to_string(), format!("The answer is {}.", n.trim())),
        None => (clean.trim().to_string(), String::new()),
    };
    let steps = steps.lines().map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" ");
    vec![format!("U: {q}\nN: {steps} {answer}").trim_end().to_string()]
}

/// ProofWriter: the facts and rules, the statement to check, and whether it follows.
pub fn proofwriter(row: &Value) -> Vec<String> {
    let (theory, question) = (s(row, "theory"), s(row, "question"));
    let answer = match s(row, "answer") {
        "True" => "Yes, that is true.",
        "False" => "No, that is false.",
        "Unknown" => "It cannot be known from these facts and rules.",
        _ => return Vec::new(),
    };
    if theory.is_empty() || question.is_empty() {
        return Vec::new();
    }
    vec![format!("U: {theory} Is this true: {question}\nN: {answer}")]
}

/// A Gutenberg book's text without the Project Gutenberg header and footer, as documents of a
/// few paragraphs each.
pub fn gutenberg_book(text: &str) -> Vec<String> {
    let start = text.find("*** START OF").and_then(|i| text[i..].find('\n').map(|j| i + j + 1)).unwrap_or(0);
    let end = text.find("*** END OF").unwrap_or(text.len());
    let body = &text[start..end.max(start)];
    let mut docs = Vec::new();
    let mut cur = String::new();
    for para in body.replace("\r\n", "\n").split("\n\n") {
        let p = para.split_whitespace().collect::<Vec<_>>().join(" ");
        // Illustration captions are not part of the text.
        if p.starts_with("[Illustration") {
            continue;
        }
        if p.split(' ').count() < 4 {
            continue;
        }
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(&p);
        if cur.len() > 1500 {
            docs.push(std::mem::take(&mut cur));
        }
    }
    if cur.len() > 200 {
        docs.push(cur);
    }
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Rows as the dataset viewer returned them (checked from a GitHub runner).
    #[test]
    fn records_become_documents_in_nova_format() {
        let d = soda(
            &json!({"narrative": "Veda went to church.", "dialogue": ["Hi, Father.", "Hello, Veda.", "Tell me more.\""], "speakers": ["Veda", "Priest", "Veda"]}),
        );
        assert_eq!(d, ["Veda went to church.\nU: Hi, Father.\nN: Hello, Veda.\nU: Tell me more."]);

        let d = persona_chat(
            &json!({"Best Generated Conversation": "User 1: Hi! I'm [user 1's name].\nUser 2: Hi [user 1's name], I'm [user 2's name].\nUser 1: What do you do for fun?"}),
        );
        assert_eq!(d.len(), 1);
        assert!(d[0].starts_with("U: Hi! I'm ") && d[0].contains("\nN: Hi ") && !d[0].contains('['));

        let d = babi(&json!({"story": {"id": ["1","2","3","4","5"], "type": [0,0,1,0,1],
            "text": ["Mary moved to the bathroom.", "John went to the hallway.", "Where is Mary?", "Mary went to the garden.", "How do you go from the kitchen to the garden?"],
            "supporting_ids": [[],[],["1"],[],["4"]], "answer": ["","","bathroom","","n,w"]}}));
        assert_eq!(
            d,
            [
                "Mary moved to the bathroom. John went to the hallway.\nU: Where is Mary?\nN: Bathroom.",
                "Mary moved to the bathroom. John went to the hallway. Mary went to the garden.\nU: How do you go from the kitchen to the garden?\nN: North and west."
            ]
        );

        let d = gsm8k(
            &json!({"question": "Natalia sold clips to 48 friends in April and half as many in May. How many altogether?",
            "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72"}),
        );
        assert_eq!(
            d[0],
            "U: Natalia sold clips to 48 friends in April and half as many in May. How many altogether?\nN: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether. The answer is 72."
        );

        let d = proofwriter(
            &json!({"theory": "Gary is furry. If someone is furry then they are kind.", "question": "Gary is kind.", "answer": "True"}),
        );
        assert_eq!(d, ["U: Gary is furry. If someone is furry then they are kind. Is this true: Gary is kind.\nN: Yes, that is true."]);

        let d = wikipedia(
            &json!({"title": "April", "text": "April is the fourth month of the year in the Gregorian calendar.\n\nThe Month \n\nIt has thirty days and comes between March and May.\n\nReferences\n\nSomething cited here and more words."}),
        );
        assert_eq!(d, ["April\nApril is the fourth month of the year in the Gregorian calendar.\nIt has thirty days and comes between March and May."]);
    }

    #[test]
    fn conversations_keep_their_speakers() {
        let d = soda_speakers(
            &json!({"narrative": "Veda went to church.", "dialogue": ["Hi, Father.", "Hello, Veda.", "Tell me more.\""], "speakers": ["Veda", "Priest", "Veda"]}),
        );
        assert_eq!(
            d,
            ["Situation: Veda went to church.\nVeda: Hi, Father.\nPriest: Hello, Veda.\nVeda: Tell me more."]
        );
        let parsed = yendory_engine::learn::conversation::parse_dialogue(&d[0]).unwrap();
        assert_eq!(parsed.pair(), ("Veda".to_string(), "Priest".to_string()));
        assert!(soda_speakers(
            &json!({"narrative": "", "dialogue": ["a", "b", "c"], "speakers": ["A", "B", "C"]})
        )
        .is_empty());
        let d = persona_speakers(
            &json!({"Best Generated Conversation": "User 1: Hi! I'm [user 1's name].\nUser 2: Hi [user 1's name], I'm [user 2's name].\nUser 1: What do you do for fun?"}),
        );
        let parsed = yendory_engine::learn::conversation::parse_dialogue(&d[0]).unwrap();
        assert_eq!(parsed.turns.len(), 3);
        let (a, b) = parsed.pair();
        assert!(parsed.turns[1].contains(&a) && parsed.turns[1].contains(&b), "{:?}", parsed.turns);
    }

    #[test]
    fn gutenberg_header_and_footer_are_removed() {
        let mut text = String::from("The Project Gutenberg eBook of X\nLicence words here\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n");
        for i in 0..30 {
            text.push_str(&format!(
                "Once there was a fox number {i} who lived in the wood and liked to sing songs.\n\n"
            ));
        }
        text.push_str("*** END OF THE PROJECT GUTENBERG EBOOK X ***\nMore licence text that must go.");
        let docs = gutenberg_book(&text);
        assert!(!docs.is_empty());
        assert!(docs.iter().all(|d| !d.contains("Gutenberg") && !d.contains("licence")));
        assert!(docs[0].starts_with("Once there was a fox number 0"));
    }

    #[test]
    fn every_source_has_a_licence_and_attribution() {
        for s in sources().into_iter().chain(conversation_sources()) {
            assert!(!s.license.is_empty() && !s.attribution.is_empty(), "{}", s.id);
        }
    }
}
