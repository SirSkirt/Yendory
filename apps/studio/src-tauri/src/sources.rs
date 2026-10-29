//! Reading sources: where each corpus comes from, its licence, and how its records become
//! documents in Nova's own format (dialogue as `U:` / `N:` turns, questions with their worked
//! answers). Every source was checked from a GitHub runner (format, licence, reachability);
//! the downloads themselves are in `reading.rs`.
use serde::Serialize;
use serde_json::Value;
use yendory_engine::learn::corpus::Category;

/// Converts one record (a JSON row) into documents.
pub type Convert = fn(&Value) -> Vec<String>;

/// A dataset's own reason to leave a record out (its language tag, its moderation flags, ...),
/// before it is converted (see `filter`).
pub type Screen = fn(&Value) -> Option<crate::filter::Reason>;

/// The screening a conversation source's records get, when the dataset flags them itself.
pub fn screen_of(source: &str) -> Option<Screen> {
    match source {
        "lmsys-chat" => Some(lmsys_screen),
        _ => None,
    }
}

/// LMSYS-Chat-1M: English only (its `language` tag), nothing its OpenAI moderation flagged in any
/// turn, and no conversation whose names were removed for privacy (`redacted`: they read
/// "NAME_1").
fn lmsys_screen(row: &Value) -> Option<crate::filter::Reason> {
    use crate::filter::Reason;
    if row.get("language").and_then(Value::as_str) != Some("English") {
        return Some(Reason::NotEnglish);
    }
    let flagged = row
        .get("openai_moderation")
        .and_then(Value::as_array)
        .is_some_and(|m| m.iter().any(|t| t.get("flagged").and_then(Value::as_bool) == Some(true)));
    if flagged || row.get("redacted").and_then(Value::as_bool) == Some(true) {
        return Some(Reason::Flagged);
    }
    None
}

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
    /// Rows of the first of these Hugging Face datasets (dataset, config, split) that answers
    /// with data: the same data is published under several names, and a name can stop working.
    HfFirst { datasets: &'static [(&'static str, &'static str, &'static str)], convert: Convert },
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

/// Conversation datasets for the Conversation stage, as `Name: text` lines, so Nova's speaker can
/// be given Nova's own name. SODA, Persona-Chat, DailyDialog and EmpatheticDialogues are
/// conversations between two people; UltraChat and OpenAssistant are a person and a helper, kept
/// as a conversation between two named people (system prompts are left out, so nothing tells
/// either speaker what it is).
pub fn conversation_sources() -> Vec<Source> {
    vec![
        Source {
            id: "daily-dialog",
            name: "DailyDialog everyday conversations",
            category: Category::Conversation,
            description: "About 11,000 conversations about daily life between two people: shopping, work, plans, feelings (written by English learners' teachers, simple language; about 6 MB).",
            page: "https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
            license: "CC BY-NC-SA 4.0 (non-commercial)",
            attribution: "DailyDialog, Li et al. (2017)",
            size_mb: 6,
            default_mb: 6,
            fetch: Fetch::HfFirst {
                // Checked 2026-10-29: the original (li2017dailydialog/daily_dialog) answers 501, as
                // the dataset viewer does not run its loading script; these copies answer with
                // rows. (frankdarkluo's has context/response pairs, ConvLab's fails to build,
                // Aeala's is gated and "daily_dialog" was renamed.)
                datasets: &[("OpenRL/daily_dialog", "default", "train"), ("roskoN/dailydialog", "full", "train")],
                convert: whole_dialogue,
            },
        },
        Source {
            id: "empathetic-dialogues",
            name: "EmpatheticDialogues",
            category: Category::Conversation,
            description: "About 20,000 conversations where one person tells of something that happened to them and how they felt, and the other listens and answers, each with its situation (about 10 MB).",
            page: "https://huggingface.co/datasets/facebook/empathetic_dialogues",
            license: "CC BY-NC 4.0 (non-commercial)",
            attribution: "EmpatheticDialogues, Rashkin et al. (2019), Facebook AI Research",
            size_mb: 10,
            default_mb: 10,
            fetch: Fetch::HfFirst {
                datasets: &[
                    ("Estwld/empathetic_dialogues_llm", "default", "train"),
                    ("facebook/empathetic_dialogues", "default", "train"),
                ],
                convert: whole_dialogue,
            },
        },
        Source {
            id: "ultrachat",
            name: "UltraChat 200k (assistant-style, filtered)",
            category: Category::Conversation,
            description: "Long conversations where one person asks about all kinds of topics and the other explains. Most are in an assistant's style (long answers, lists, \"Certainly!\"), as a conversation between two named people. The size you choose is downloaded whole; with every quality filter on (§ 4) practice uses only the plainer, more natural exchanges. Measured: of all 207,865 training conversations the full filter passes about 1,700 (about 2 MB), mostly because turns are over 600 characters; switch filters off to use more.",
            page: "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
            license: "MIT",
            attribution: "UltraChat, Ding et al. (2023); ultrachat_200k by Hugging Face H4",
            size_mb: 1400,
            default_mb: 40,
            fetch: Fetch::HfFirst {
                datasets: &[
                    ("HuggingFaceH4/ultrachat_200k", "default", "train_sft"),
                    ("stingning/ultrachat", "default", "train"),
                ],
                convert: whole_dialogue,
            },
        },
        Source {
            id: "openassistant",
            name: "OpenAssistant conversations (assistant-style)",
            category: Category::Conversation,
            description: "Conversations people wrote with volunteers playing a helper: questions, advice and explanations (assistant-style). Only the English ones are kept (the dataset has many languages), as a conversation between two named people (about 20 MB).",
            page: "https://huggingface.co/datasets/OpenAssistant/oasst1",
            license: "Apache 2.0",
            attribution: "OpenAssistant Conversations, Kopf et al. (2023), LAION",
            size_mb: 20,
            default_mb: 20,
            fetch: Fetch::HfFirst {
                datasets: &[
                    ("OpenAssistant/oasst_top1_2023-08-25", "default", "train"),
                    ("timdettmers/openassistant-guanaco", "default", "train"),
                ],
                convert: whole_dialogue,
            },
        },
        Source {
            id: "lmsys-chat",
            name: "LMSYS-Chat-1M (gated: needs your Hugging Face token)",
            category: Category::Conversation,
            description: "One million real conversations people had with chatbots in 2023. Most are tasks and questions for an assistant. The size you choose is downloaded whole (the conversations the dataset itself flagged, not English or moderated or with names removed, are kept apart); with every quality filter on (§ 4) practice uses few of them, so switch filters off to use more. The Parquet files (about 2.5 GB) are read and then deleted. Access: while signed in to Hugging Face, accept the licence on the dataset's page, then give Studio an access token (below). The conversations stay on this device.",
            page: "https://huggingface.co/datasets/lmsys/lmsys-chat-1m",
            license: "LMSYS-Chat-1M Dataset License Agreement (accept it on Hugging Face; no redistribution)",
            attribution: "LMSYS-Chat-1M, Zheng et al. (2023), LMSYS",
            size_mb: 2500,
            default_mb: 10,
            fetch: Fetch::HfRows {
                dataset: "lmsys/lmsys-chat-1m",
                configs: &["default"],
                split: "train",
                convert: whole_dialogue,
            },
        },
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

/// Names given to the two speakers of a conversation that has none (chosen by its text).
const PAIRS: [(&str, &str); 12] = [
    ("Ann", "Ben"),
    ("Cara", "Dan"),
    ("Eva", "Finn"),
    ("Mia", "Leo"),
    ("Sara", "Tom"),
    ("Zoe", "Ray"),
    ("Lily", "Max"),
    ("Nora", "Sam"),
    ("Ivy", "Jack"),
    ("Rose", "Owen"),
    ("Kate", "Noah"),
    ("Lucy", "Adam"),
];

/// Role labels that are not names: speakers labelled like this get names.
const ROLES: [&str; 14] = [
    "user",
    "assistant",
    "human",
    "gpt",
    "bot",
    "ai",
    "model",
    "system",
    "speaker",
    "prompter",
    "you",
    "me",
    "a",
    "b",
];

/// Longest turn kept: a conversation is cut before a longer turn (and before code).
const LONGEST_TURN: usize = 1200;

/// A speaker label that can stand as a name.
fn plain_name(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 32
        && n.split(' ').count() <= 3
        && n.starts_with(|c: char| c.is_ascii_uppercase())
        && n.chars().all(|c| c.is_ascii_alphabetic() || " .'-".contains(c))
        && !ROLES.contains(&n.to_ascii_lowercase().as_str())
}

/// Turns of the `<|im_start|>role ... <|im_end|>` or `### Human: ... ### Assistant:` layouts.
fn chat_text_turns(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if text.contains("<|im_start|>") {
        for part in text.split("<|im_start|>").skip(1) {
            let part = part.split("<|im_end|>").next().unwrap_or("");
            let (role, said) = part.split_once('\n').unwrap_or((part, ""));
            out.push((role.trim().to_string(), said.trim().to_string()));
        }
    } else if text.contains("### Human:") {
        for part in text.split("### ").filter(|p| !p.trim().is_empty()) {
            if let Some((role, said)) = part.split_once(':') {
                out.push((role.trim().to_string(), said.trim().to_string()));
            }
        }
    }
    out
}

/// Text written with spaces before punctuation (DailyDialog: "Say , Jim , how about ... ?")
/// written the usual way ("Say, Jim, how about ...?"), so that is not what Nova learns.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == ' ' && chars.peek().is_some_and(|n| ",.?!;:".contains(*n)) && !out.is_empty() {
            continue;
        }
        out.push(c);
    }
    out.replace(" n't", "n't").replace(" ' s ", "'s ")
}

/// One conversation in any common layout, as a `Situation:` line (when there is one) and
/// `Name: text` lines. Layouts: a list of utterances (`dialog`, `dialogue`, `utterances`,
/// `turns`, `conversation`, `conversations`, `messages`) as strings (the speakers alternate) or
/// as objects with a role (`role`, `from`, `speaker`) and a text (`content`, `value`, `text`,
/// `utterance`); or one `text` with `<|im_start|>` / `### Human:` turns. System turns are left
/// out. Speakers whose labels are not names (user, assistant, ...) get names. A conversation is
/// cut before code or a turn longer than 1,200 characters; fewer than two turns, more than two
/// speakers or nothing recognised: none.
pub fn any_dialogue(row: &Value) -> Vec<String> {
    dialogue_of(row, true)
}

/// [`any_dialogue`] without cutting: a conversation with code or a long turn is converted whole
/// (the download filter leaves it out and counts it, see `filter`).
pub fn whole_dialogue(row: &Value) -> Vec<String> {
    dialogue_of(row, false)
}

fn dialogue_of(row: &Value, cut: bool) -> Vec<String> {
    const LISTS: [&str; 7] =
        ["dialog", "dialogue", "utterances", "turns", "conversation", "conversations", "messages"];
    let field = |v: &Value, keys: &[&str]| -> Option<String> {
        keys.iter().find_map(|k| v.get(*k).and_then(Value::as_str)).map(|x| x.trim().to_string())
    };
    let mut turns: Vec<(Option<String>, String)> = Vec::new();
    if let Some(list) = LISTS.iter().find_map(|k| row.get(*k).and_then(Value::as_array)).or(row.as_array()) {
        for v in list {
            match v {
                Value::String(t) => turns.push((None, t.trim().to_string())),
                Value::Object(_) => {
                    let Some(text) = field(v, &["content", "value", "text", "utterance"]) else { continue };
                    turns.push((field(v, &["role", "from", "speaker"]), text));
                }
                _ => {}
            }
        }
    } else if let Some(text) = field(row, &["text", "chat", "conversation"]) {
        turns = chat_text_turns(&text).into_iter().map(|(r, t)| (Some(r), t)).collect();
    }
    turns.retain(|(r, t)| !t.is_empty() && !r.as_deref().is_some_and(|r| r.eq_ignore_ascii_case("system")));
    // Cut before code or an over-long turn.
    if let Some(i) =
        turns.iter().position(|(_, t)| t.contains("```") || t.len() > LONGEST_TURN).filter(|_| cut)
    {
        turns.truncate(i);
    }
    if turns.len() < 2 {
        return Vec::new();
    }
    // Speakers: the labels in order of appearance (alternating when there are none).
    let mut labels: Vec<String> = Vec::new();
    for (i, (r, _)) in turns.iter().enumerate() {
        let r = r.clone().unwrap_or_else(|| ["1", "2"][i % 2].to_string());
        if !labels.contains(&r) {
            labels.push(r);
        }
    }
    if labels.len() != 2 {
        return Vec::new();
    }
    let all_text: usize = turns.iter().map(|(_, t)| t.len()).sum();
    let (a, b) = PAIRS[(all_text + turns.len()) % PAIRS.len()];
    let names: Vec<String> = if labels.iter().all(|l| plain_name(l)) {
        labels.clone()
    } else {
        vec![a.to_string(), b.to_string()]
    };
    let mut doc = String::new();
    if let Some(sit) = field(row, &["situation", "narrative", "context"]).filter(|x| !x.is_empty()) {
        doc.push_str(&format!("Situation: {}\n", sit.replace('\n', " ")));
    }
    for (i, (r, t)) in turns.iter().enumerate() {
        let label = r.clone().unwrap_or_else(|| ["1", "2"][i % 2].to_string());
        let who = &names[labels.iter().position(|l| *l == label).unwrap_or(0)];
        let said = tidy(&t.split_whitespace().collect::<Vec<_>>().join(" "));
        doc.push_str(&format!("{who}: {said}\n"));
    }
    vec![doc.trim_end().to_string()]
}

/// Whether text reads as English (a downloaded dataset may mix languages): almost all letters
/// ASCII and enough of the commonest English words.
pub fn english(text: &str) -> bool {
    const COMMON: [&str; 40] = [
        "the", "a", "an", "to", "and", "i", "you", "is", "it", "of", "in", "that", "what", "for", "my", "do",
        "are", "be", "have", "this", "with", "can", "on", "not", "me", "your", "was", "so", "but", "we",
        "how", "he", "she", "they", "at", "if", "or", "about", "there", "will",
    ];
    let letters = text.chars().filter(|c| c.is_alphabetic()).count().max(1);
    let foreign = text.chars().filter(|c| c.is_alphabetic() && !c.is_ascii()).count();
    // The speaker names at the start of each line are left out of the word count.
    let words: Vec<String> = text
        .lines()
        .map(|l| l.split_once(": ").map_or(l, |(_, t)| t))
        .flat_map(|l| l.split(|c: char| !c.is_ascii_alphabetic() && c != '\''))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .collect();
    let common = words.iter().filter(|w| COMMON.contains(&w.as_str())).count();
    foreign * 50 <= letters && !words.is_empty() && common * 100 >= words.len() * 15
}

/// Conversations from a file of your own: JSON (a list of conversations, or an object with a
/// list under `conversations`, `dialogues` or `data`, or one conversation), JSON lines (one
/// conversation per line), each in any layout [`any_dialogue`] reads; or plain text, kept as it
/// is (conversations separated by blank lines, `Name: text` or `U:` / `N:` lines). Returns the
/// text to store (conversations separated by blank lines).
pub fn conversations_from_file(text: &str) -> String {
    let trimmed = text.trim_start();
    // A turn: a string, or an object with a speaker.
    let turn = |x: &Value| x.is_string() || ["role", "from", "speaker"].iter().any(|k| x.get(*k).is_some());
    let one = |v: &Value| -> Vec<String> {
        match v {
            // A bare list of turns is one conversation.
            Value::Array(items) if items.iter().all(turn) => {
                any_dialogue(&serde_json::json!({ "messages": v }))
            }
            _ => any_dialogue(v),
        }
    };
    let mut docs: Vec<String> = Vec::new();
    if trimmed.starts_with('[') || trimmed.starts_with('{') {
        match serde_json::from_str::<Value>(trimmed) {
            Ok(Value::Array(list)) if list.iter().all(turn) => docs.extend(one(&Value::Array(list))),
            Ok(Value::Array(list)) => list.iter().for_each(|v| docs.extend(one(v))),
            Ok(obj) => match ["conversations", "dialogues", "data"]
                .iter()
                .find_map(|k| obj.get(*k).and_then(Value::as_array))
            {
                Some(list)
                    if list.iter().all(|x| !x.is_string()) && list.iter().any(|x| !one(x).is_empty()) =>
                {
                    list.iter().for_each(|v| docs.extend(one(v)))
                }
                _ => docs.extend(one(&obj)),
            },
            // JSON lines.
            Err(_) => {
                for line in trimmed.lines().filter(|l| !l.trim().is_empty()) {
                    if let Ok(v) = serde_json::from_str::<Value>(line) {
                        docs.extend(one(&v));
                    }
                }
            }
        }
        return docs.join("\n\n");
    }
    text.to_string()
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

    /// Every layout the conversation datasets and your own files use becomes two named
    /// speakers that parse as a dialogue; system prompts never come through.
    #[test]
    fn any_layout_becomes_a_two_person_conversation() {
        use yendory_engine::learn::conversation::parse_dialogue;
        let check = |v: Value, first: &str| {
            let d = any_dialogue(&v);
            assert_eq!(d.len(), 1, "{v}");
            let p = parse_dialogue(&d[0]).unwrap_or_else(|| panic!("{}", d[0]));
            assert_eq!(p.turns.len(), 2, "{}", d[0]);
            assert_eq!(p.turns[0], first, "{}", d[0]);
            assert!(!d[0].to_lowercase().contains("system") && !d[0].contains("helpful"), "{}", d[0]);
            d[0].clone()
        };
        // DailyDialog: a list of utterances.
        check(
            json!({"dialog": ["Say , Jim , how about going for a few beers ?", "That's a good idea ."]}),
            "Say, Jim, how about going for a few beers?",
        );
        // EmpatheticDialogues (LLM layout): situation and role/content messages.
        let e = check(
            json!({"situation": "I lost my keys.", "conversations": [{"role": "user", "content": "I lost my keys today."}, {"role": "assistant", "content": "Oh no, where did you see them last?"}]}),
            "I lost my keys today.",
        );
        assert!(e.starts_with("Situation: I lost my keys."));
        // UltraChat: messages, with a system prompt that must be left out.
        check(
            json!({"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is rain?"}, {"role": "assistant", "content": "Water falling from clouds."}]}),
            "What is rain?",
        );
        // OpenAssistant top-1: chat-markup text.
        check(
            json!({"text": "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n<|im_start|>assistant\nBecause air scatters blue light most.<|im_end|>"}),
            "Why is the sky blue?",
        );
        // Guanaco layout.
        check(json!({"text": "### Human: Hi there.### Assistant: Hello! How are you?"}), "Hi there.");
        // ShareGPT layout keeps plain names.
        let named = check(
            json!({"conversations": [{"from": "Rosa", "value": "Morning!"}, {"from": "Theo", "value": "Morning, Rosa."}]}),
            "Morning!",
        );
        assert!(named.starts_with("Rosa: Morning!"));
        // Code or an over-long turn cuts the conversation; three speakers or one turn: none.
        assert!(any_dialogue(&json!({"dialog": ["Hi", "```rust\nfn main() {}\n```"]})).is_empty());
        assert!(any_dialogue(&json!({"messages": [{"role": "a", "content": "x"}, {"role": "b", "content": "y"}, {"role": "c", "content": "z"}]})).is_empty());
        assert!(any_dialogue(&json!({"dialog": ["Only one."]})).is_empty());
        assert!(any_dialogue(&json!({"title": "not a conversation"})).is_empty());
    }

    /// Downloaded conversations in other languages are left out; English ones are kept.
    #[test]
    fn only_english_conversations_are_kept_from_mixed_datasets() {
        let it = json!({"text": "<|im_start|>user\nConsigliami 5 nomi per il mio cucciolo di dobberman<|im_end|>\n<|im_start|>assistant\nEcco 5 nomi per il tuo cucciolo di dobermann: Zeus, Apollo, Thor<|im_end|>"});
        assert!(!any_dialogue(&it).is_empty() && !english(&any_dialogue(&it)[0]));
        let de = json!({"dialog": ["Wie geht es dir heute?", "Mir geht es gut, danke. Und dir?"]});
        assert!(!english(&any_dialogue(&de)[0]));
        let en = json!({"dialog": ["Do you want to go to the park with me?", "Yes, that is a good idea. What time?"]});
        assert!(english(&any_dialogue(&en)[0]));
    }

    /// Files of your own: JSON, JSON lines and plain text all give conversations.
    #[test]
    fn own_files_in_any_form_give_conversations() {
        use yendory_engine::learn::conversation::parse_dialogue;
        let count = |text: &str| {
            conversations_from_file(text).split("\n\n").filter(|d| parse_dialogue(d).is_some()).count()
        };
        assert_eq!(
            count(
                r#"[{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey"}]}, ["Hello", "Hi there"]]"#
            ),
            2
        );
        assert_eq!(
            count(r#"{"conversations": [{"dialog": ["a b", "c d"]}, {"dialog": ["e f", "g h"]}]}"#),
            2
        );
        assert_eq!(count("{\"dialog\": [\"a\", \"b\"]}\n{\"dialog\": [\"c\", \"d\"]}\n"), 2);
        assert_eq!(count("Mia: Hi!\nLeo: Hello.\n\nAnn: Rain again.\nBen: Yes."), 2);
        assert_eq!(count(r#"[{"from": "user", "value": "x"}, {"from": "gpt", "value": "y"}]"#), 1);
    }

    #[test]
    fn every_source_has_a_licence_and_attribution() {
        for s in sources().into_iter().chain(conversation_sources()) {
            assert!(!s.license.is_empty() && !s.attribution.is_empty(), "{}", s.id);
        }
    }
}
