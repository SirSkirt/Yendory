//! Review Training: turns Study Mode memories into review material (worked examples and
//! held-out review questions) as a Lesson Designer draft the teacher reviews and edits, then
//! assesses Nova on the held-out questions without memory before and after an approved neural
//! training pass, and compares the two.
//!
//! The material is generated from the statements with simple sentence patterns ("X is a Y",
//! "X has Y", "X can Y", "X needs Y"): a supporting mechanism, not reasoning. Memories can be
//! incomplete or wrong, so no generated answer is authoritative: every item is marked generated
//! and unverified, with the memory (and the lesson it was read in) it came from, and the lesson
//! cannot be taught until the teacher has checked each answer. Questions combining two memories
//! ("Max is a dog" + "Dogs need water" -> "Does Max need water?") are marked inferred.
//!
//! Held-out review questions are held out from these examples, not necessarily from what Nova
//! read before. They are always asked without memory, so the comparison shows what the network
//! itself can do; with Study Mode the same questions are also asked with memory, reported
//! separately (remembered knowledge).
use super::designer::{AnswerSource, BlockKind, DraftBlock, DraftItem, LessonDraft, Origin, SourceRef};
use super::eval::evaluate;
use super::grading::GradeMode;
use super::lessons::{canonical, parse_lesson, Item, ReadingMode};
use crate::memory::{terms, Memory, MemoryStore};
use crate::model::{Ablation, Bicameral};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
enum Rel {
    /// "X is a Y" / "X are Y".
    Is,
    Has,
    Can,
    /// Any other verb: the verb as written.
    Verb(String),
}

#[derive(Clone, Debug, PartialEq)]
struct Fact {
    subject: String,
    plural: bool,
    rel: Rel,
    object: String,
    negated: bool,
}

const DETERMINERS: [&str; 7] = ["the", "a", "an", "all", "every", "some", "most"];

fn parse(text: &str) -> Option<Fact> {
    let t = text.trim().trim_end_matches(['.', '!']);
    if t.contains(['?', ',', ';', ':']) {
        return None;
    }
    let w: Vec<&str> = t.split_whitespace().collect();
    if w.len() < 3 || w.len() > 12 {
        return None;
    }
    let first = w[0].to_ascii_lowercase();
    let subj_len = if DETERMINERS.contains(&first.as_str()) { 2 } else { 1 };
    if w.len() < subj_len + 2 {
        return None;
    }
    let subject = w[..subj_len].join(" ");
    let verb = w[subj_len].to_ascii_lowercase();
    let mut rest = &w[subj_len + 1..];
    let negated = rest.first().is_some_and(|x| x.eq_ignore_ascii_case("not"));
    if negated {
        rest = &rest[1..];
    }
    if rest.is_empty() {
        return None;
    }
    let object = rest.join(" ");
    let (rel, plural) = match verb.as_str() {
        "is" => (Rel::Is, false),
        "are" => (Rel::Is, true),
        "has" => (Rel::Has, false),
        "have" => (Rel::Has, true),
        "can" => (Rel::Can, subject.ends_with('s')),
        v if v.chars().all(|c| c.is_ascii_alphabetic()) && v.len() > 2 && !negated => {
            (Rel::Verb(v.to_string()), !v.ends_with('s'))
        }
        _ => return None,
    };
    Some(Fact { subject, plural, rel, object, negated })
}

/// The subject as it appears inside a question: a leading determiner is lowercased, names keep
/// their capital.
fn inner(subject: &str) -> String {
    let mut w: Vec<String> = subject.split_whitespace().map(String::from).collect();
    let n = w.len();
    if let Some(f) = w.first_mut() {
        let lower = f.to_ascii_lowercase();
        // A determiner, or a capitalised plural noun at the start ("Dogs"), is lowercased;
        // a name ("Max") keeps its capital.
        let plural_noun =
            n == 1 && f.len() > 3 && f.ends_with('s') && f.chars().skip(1).all(|c| c.is_ascii_lowercase());
        if DETERMINERS.contains(&lower.as_str())
            || plural_noun
            || !f.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        {
            *f = lower;
        }
    }
    w.join(" ")
}

fn cap(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_ascii_uppercase().to_string() + c.as_str(),
        None => String::new(),
    }
}

/// The base form of a verb as written ("needs" -> "need", "watches" -> "watch").
fn base(verb: &str) -> String {
    if verb.len() > 4
        && (verb.ends_with("ches")
            || verb.ends_with("shes")
            || verb.ends_with("sses")
            || verb.ends_with("xes"))
    {
        verb[..verb.len() - 2].to_string()
    } else if verb.len() > 3 && verb.ends_with("ies") {
        format!("{}y", &verb[..verb.len() - 3])
    } else if verb.len() > 2 && verb.ends_with('s') && !verb.ends_with("ss") {
        verb[..verb.len() - 1].to_string()
    } else {
        verb.to_string()
    }
}

/// The verb for a singular subject ("need" -> "needs").
fn third(verb: &str) -> String {
    if verb.ends_with(['s', 'x']) || verb.ends_with("ch") || verb.ends_with("sh") {
        format!("{verb}es")
    } else if verb.ends_with('y') && !verb.ends_with("ay") && !verb.ends_with("ey") && !verb.ends_with("oy") {
        format!("{}ies", &verb[..verb.len() - 1])
    } else {
        format!("{verb}s")
    }
}

fn last_word(s: &str) -> String {
    s.split_whitespace().last().unwrap_or("").to_ascii_lowercase()
}

/// A question, its answer (with accepted forms and rules after `|`) and how it was made.
struct Gen {
    question: String,
    answer: String,
    inferred: bool,
    sources: Vec<u64>,
    /// Direct question about the statement (an example), or a reworded / wh / relationship one.
    direct: bool,
}

fn yes_no(f: &Fact) -> (&'static str, &'static str) {
    if f.negated {
        ("No", "never: yes")
    } else {
        ("Yes", "never: not")
    }
}

/// The questions a single statement gives.
fn questions(f: &Fact, id: u64, difficulty: u8) -> Vec<Gen> {
    let s = inner(&f.subject);
    let o = &f.object;
    let (yn, rule) = yes_no(f);
    let not = if f.negated { " not" } else { "" };
    let key = |full: &str| {
        let short = format!("{yn}.");
        let alt = if full == short { String::new() } else { format!(" | {short}") };
        format!("{full}{alt} | must: {} | {rule}", yn.to_ascii_lowercase())
    };
    let mut out = Vec::new();
    let (direct_q, statement, wh_q, wh_must) = match &f.rel {
        Rel::Is => {
            let be = if f.plural { "are" } else { "is" };
            (
                format!("{} {s} {o}?", cap(be)),
                format!("{} {be}{not} {o}.", cap(&s)),
                (!f.negated).then(|| format!("What {be} {s}?")),
                last_word(o),
            )
        }
        Rel::Has => {
            let (aux, have) = if f.plural { ("Do", "have") } else { ("Does", "has") };
            (
                format!("{aux} {s} have {o}?"),
                format!(
                    "{} {}{o}.",
                    cap(&s),
                    if f.negated { "does not have ".to_string() } else { format!("{have} ") }
                ),
                (!f.negated).then(|| format!("What {} {s} have?", aux.to_ascii_lowercase())),
                last_word(o),
            )
        }
        Rel::Can => (
            format!("Can {s} {o}?"),
            format!("{} can{not} {o}.", cap(&s)),
            (!f.negated).then(|| format!("What can {s} do?")),
            o.split_whitespace().next().unwrap_or("").to_ascii_lowercase(),
        ),
        Rel::Verb(v) => {
            let aux = if f.plural { "Do" } else { "Does" };
            (
                format!("{aux} {s} {} {o}?", base(v)),
                format!("{} {v} {o}.", cap(&s)),
                Some(format!("What {} {s} {}?", aux.to_ascii_lowercase(), base(v))),
                last_word(o),
            )
        }
    };
    out.push(Gen {
        question: direct_q,
        answer: key(&format!("{yn}, {}", lower_first(&statement))),
        inferred: false,
        sources: vec![id],
        direct: true,
    });
    let reworded = format!("Is it true that {}?", lower_first(statement.trim_end_matches('.')));
    out.push(Gen {
        question: reworded,
        answer: key(&format!("{yn}.")),
        inferred: false,
        sources: vec![id],
        direct: false,
    });
    if difficulty >= 2 {
        if let Some(q) = wh_q {
            if !wh_must.is_empty() {
                out.push(Gen {
                    question: q,
                    answer: format!("{statement} | must: {wh_must} | never: not"),
                    inferred: false,
                    sources: vec![id],
                    direct: false,
                });
            }
        }
    }
    out
}

/// "Max is a dog." -> "max is a dog." only for a common noun start; names keep their capital.
fn lower_first(s: &str) -> String {
    let first = s.split_whitespace().next().unwrap_or("");
    if DETERMINERS.contains(&first.to_ascii_lowercase().as_str())
        || (first.ends_with('s') && first.len() > 3 && first.chars().skip(1).all(|c| c.is_ascii_lowercase()))
    {
        let mut c = s.chars();
        match c.next() {
            Some(f) => f.to_ascii_lowercase().to_string() + c.as_str(),
            None => String::new(),
        }
    } else {
        s.to_string()
    }
}

/// "a dog" -> its plural-insensitive concept ("dog"); "animals" -> "animal".
fn concept(s: &str) -> Option<String> {
    terms(s).last().cloned()
}

/// "animals" -> "an animal", "dogs" -> "a dog" (for "Is Max an animal?").
fn singular_with_article(object: &str) -> String {
    let w: Vec<&str> = object.split_whitespace().collect();
    let noun = w.last().copied().unwrap_or("");
    let stem = concept(noun).unwrap_or_else(|| noun.to_string());
    let rest = &w[..w.len().saturating_sub(1)];
    let phrase = if rest.is_empty() { stem } else { format!("{} {stem}", rest.join(" ")) };
    let article = if phrase.starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" };
    format!("{article} {phrase}")
}

/// Questions combining two memories: "S is a C" with "Cs <relation> O" gives "<relation> S O?".
fn relationships(facts: &[(u64, Fact)]) -> Vec<Gen> {
    let mut out = Vec::new();
    for (i1, f1) in facts {
        if f1.rel != Rel::Is || f1.plural || f1.negated {
            continue;
        }
        let Some(c) = concept(&f1.object) else { continue };
        for (i2, f2) in facts {
            if i1 == i2 || concept(&f2.subject).as_deref() != Some(c.as_str()) {
                continue;
            }
            let s = inner(&f1.subject);
            let (yn, rule) = yes_no(f2);
            let q = match &f2.rel {
                Rel::Is => format!(
                    "Is {s} {}?",
                    if f2.plural { singular_with_article(&f2.object) } else { f2.object.clone() }
                ),
                Rel::Has => format!("Does {s} have {}?", f2.object),
                Rel::Can => format!("Can {s} {}?", f2.object),
                Rel::Verb(v) => format!("Does {s} {} {}?", base(v), f2.object),
            };
            let because = format!("{} {} and {}", cap(&s), f1_clause(f1), lower_first(&fact_text(f2)));
            out.push(Gen {
                question: q,
                answer: format!("{yn}. {because}. | {yn}. | must: {} | {rule}", yn.to_ascii_lowercase()),
                inferred: true,
                sources: vec![*i1, *i2],
                direct: false,
            });
        }
    }
    out
}

fn f1_clause(f: &Fact) -> String {
    format!("is {}", f.object)
}

fn fact_text(f: &Fact) -> String {
    let not = if f.negated { " not" } else { "" };
    match &f.rel {
        Rel::Is => format!("{} {}{not} {}", f.subject, if f.plural { "are" } else { "is" }, f.object),
        Rel::Has => format!("{} {}{not} {}", f.subject, if f.plural { "have" } else { "has" }, f.object),
        Rel::Can => format!("{} can{not} {}", f.subject, f.object),
        Rel::Verb(v) => {
            format!("{} {} {}", f.subject, if f.plural { base(v) } else { third(&base(v)) }, f.object)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReviewOptions {
    /// 1: direct and reworded yes/no questions; 2: also "what" questions; 3: also questions
    /// combining two memories.
    pub difficulty: u8,
}

/// A topic (a concept word or a lesson) and the memories in it.
pub type Topic = (String, Vec<u64>);

/// Topics for choosing what to review: concept words and lessons, with their memories.
pub fn topics(store: &MemoryStore) -> (Vec<Topic>, Vec<Topic>) {
    let mut by_term: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut by_lesson: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for m in store.entries.iter().filter(|m| m.enabled) {
        for t in &m.terms {
            by_term.entry(t.clone()).or_default().push(m.id);
        }
        let lesson =
            m.sources.first().map(|s| s.lesson.clone()).unwrap_or_else(|| "written by the teacher".into());
        by_lesson.entry(lesson).or_default().push(m.id);
    }
    let mut terms: Vec<Topic> = by_term.into_iter().filter(|(_, v)| v.len() >= 2).collect();
    terms.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
    (terms, by_lesson.into_iter().collect())
}

fn item(g: &Gen, store: &MemoryStore, n: &mut usize) -> DraftItem {
    *n += 1;
    let describe = |id: &u64| -> String {
        store
            .get(*id)
            .map(|m: &Memory| {
                let read = m
                    .sources
                    .first()
                    .map(|s| format!(", read in {}", s.lesson))
                    .unwrap_or_else(|| ", written by the teacher".into());
                format!("memory #{id} \"{}\"{read}", m.text)
            })
            .unwrap_or_else(|| format!("memory #{id}"))
    };
    let from = g.sources.iter().map(describe).collect::<Vec<_>>().join(" and ");
    let flag = if g.inferred {
        format!("Inferred by combining {from}: check it (memories can be incomplete or wrong)")
    } else {
        format!("Generated from {from}: check the answer")
    };
    DraftItem {
        id: format!("r{n}"),
        question: g.question.clone(),
        answer: g.answer.clone(),
        answer_source: AnswerSource::Generated,
        verified: false,
        origin: Origin::Generated,
        edited: false,
        source: Some(SourceRef {
            file: g.sources.iter().map(|i| format!("memory #{i}")).collect::<Vec<_>>().join(" + "),
            line: 0,
        }),
        flags: vec![flag],
        note: String::new(),
    }
}

/// Review material from the chosen memories: worked examples (one direct question per
/// statement, and every other relationship question with its reason) and held-out review
/// questions (reworded, "what", and the other relationship questions), never the same question
/// in both.
pub fn generate(store: &MemoryStore, ids: &[u64], opts: ReviewOptions, name: &str) -> LessonDraft {
    let chosen: Vec<&Memory> = store.entries.iter().filter(|m| m.enabled && ids.contains(&m.id)).collect();
    let facts: Vec<(u64, Fact)> = chosen.iter().filter_map(|m| parse(&m.text).map(|f| (m.id, f))).collect();
    let mut examples = Vec::new();
    let mut held = Vec::new();
    for (id, f) in &facts {
        for g in questions(f, *id, opts.difficulty) {
            if g.direct {
                examples.push(g);
            } else {
                held.push(g);
            }
        }
    }
    if opts.difficulty >= 3 {
        for (k, g) in relationships(&facts).into_iter().enumerate() {
            if k % 2 == 0 {
                examples.push(g);
            } else {
                held.push(g);
            }
        }
    }
    // A held-out question never repeats an example.
    let taught: BTreeSet<String> = examples.iter().map(|g| canonical(&g.question)).collect();
    held.retain(|g| !taught.contains(&canonical(&g.question)));
    let mut seen = BTreeSet::new();
    held.retain(|g| seen.insert(canonical(&g.question)));
    let mut n = 0;
    let block = |kind: BlockKind, title: &str, items: Vec<DraftItem>| DraftBlock {
        id: format!("b{}", kind as u8),
        kind,
        title: title.to_string(),
        text: String::new(),
        items,
        mode: ReadingMode::Neural,
        use_in_teaching: true,
        origin: Origin::Generated,
        edited: false,
        lines: None,
    };
    let ex_items = examples.iter().map(|g| item(g, store, &mut n)).collect();
    let held_items = held.iter().map(|g| item(g, store, &mut n)).collect();
    let skipped = chosen.len() - facts.len();
    LessonDraft {
        id: String::new(),
        name: name.to_string(),
        source_file: format!("Study Mode memories ({} chosen)", chosen.len()),
        source_sha256: String::new(),
        stage: None,
        finished: false,
        notes: format!(
            "Review Training material generated from Study Mode memories {}. Every answer is generated and must be checked before training.{}",
            chosen.iter().map(|m| format!("#{}", m.id)).collect::<Vec<_>>().join(", "),
            if skipped > 0 { format!(" {skipped} memory(ies) did not fit a sentence pattern and gave no question; add questions for them by hand.") } else { String::new() }
        ),
        blocks: vec![
            block(BlockKind::Examples, "Worked examples from memory", ex_items),
            block(BlockKind::Evaluation, "Held-out review questions", held_items),
        ],
    }
}

/// The lesson units of a review draft, with answer keys (accepted forms and `must:` rules)
/// applied: examples are taught, Evaluation questions held out.
pub fn lesson_items(draft: &LessonDraft) -> Result<Vec<Item>, String> {
    let ex = super::designer::export(draft)?;
    let mut items = parse_lesson(&ex.text, &draft.name)?;
    super::curriculum::apply_keys(&mut items, GradeMode::Sentence)?;
    Ok(items)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Asked {
    pub question: String,
    pub expected: String,
    pub reply: String,
    pub correct: bool,
    /// Study Mode: the reply with memories in front, and whether it was right.
    #[serde(default)]
    pub with_memory: Option<(String, bool)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Assessment {
    pub step: u64,
    pub time: u64,
    pub asked: Vec<Asked>,
    pub correct: usize,
    pub total: usize,
    #[serde(default)]
    pub with_memory: Option<(usize, usize)>,
}

/// Asks the held-out review questions without memory, and (with a store) again with memory.
pub fn assess(
    model: &Bicameral,
    items: &[Item],
    store: Option<&mut MemoryStore>,
    now: u64,
) -> Result<Assessment, String> {
    let held: Vec<&Item> = items.iter().filter(|i| i.held_out && i.prompt.is_some()).collect();
    let report = evaluate(model, &held, Ablation::default())?;
    let mut asked: Vec<Asked> = report
        .outcomes
        .iter()
        .map(|o| Asked {
            question: o.prompt.trim_start_matches("U: ").trim_end_matches("\nN: ").to_string(),
            expected: o.expected.clone(),
            reply: o.reply.clone(),
            correct: o.correct,
            with_memory: None,
        })
        .collect();
    let mut with_memory = None;
    if let Some(store) = store {
        let held_owned: Vec<Item> = held.iter().map(|i| (*i).clone()).collect();
        let eval = crate::memory::evaluation_items(model, store, &held_owned, now);
        let refs: Vec<&Item> = eval.iter().collect();
        let r = evaluate(model, &refs, Ablation::default())?;
        for (a, o) in asked.iter_mut().zip(&r.outcomes) {
            a.with_memory = Some((o.reply.clone(), o.correct));
        }
        let o = r.overall();
        with_memory = Some((o.correct, o.total));
    }
    let correct = asked.iter().filter(|a| a.correct).count();
    let total = asked.len();
    Ok(Assessment { step: model.step, time: now, asked, correct, total, with_memory })
}

/// Records an assessment in the reward ledger: the review questions were never trained, so a
/// right answer counts for generalization (network alone) and, separately, with memories.
pub fn record_assessment(rewards: &mut super::reward::Rewards, items: &[Item], a: &Assessment) {
    use super::reward::{Answer, Evidence, Origin};
    let held: Vec<&Item> = items.iter().filter(|i| i.held_out && i.prompt.is_some()).collect();
    for (it, asked) in held.iter().zip(&a.asked) {
        rewards.record(Evidence {
            item: it,
            answer: Answer::Reply(&asked.reply),
            origin: Origin::Review,
            step: a.step,
            checkpoint: true,
            teacher: None,
        });
        if let Some((reply, _)) = &asked.with_memory {
            let mut with = (*it).clone();
            with.context = Some(format!("{}(retrieved memories)\n", crate::memory::MEMORY));
            rewards.record(Evidence {
                item: &with,
                answer: Answer::Reply(reply),
                origin: Origin::Review,
                step: a.step,
                checkpoint: true,
                teacher: None,
            });
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    pub improved: Vec<String>,
    pub regressed: Vec<String>,
    pub still_right: Vec<String>,
    pub still_wrong: Vec<String>,
    /// Per concept word of the questions: (word, right before, right after, questions).
    pub concepts: Vec<(String, usize, usize, usize)>,
}

/// Question by question and concept by concept, before against after (without memory).
pub fn compare(before: &Assessment, after: &Assessment) -> Comparison {
    let mut c = Comparison::default();
    let prior: BTreeMap<&str, bool> = before.asked.iter().map(|a| (a.question.as_str(), a.correct)).collect();
    let mut per: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    for a in &after.asked {
        let Some(&was) = prior.get(a.question.as_str()) else { continue };
        let list = match (was, a.correct) {
            (false, true) => &mut c.improved,
            (true, false) => &mut c.regressed,
            (true, true) => &mut c.still_right,
            (false, false) => &mut c.still_wrong,
        };
        list.push(a.question.clone());
        // Concept words of the question, not of its wording ("Is it true that ...").
        for t in terms(&a.question).into_iter().filter(|t| t != "true") {
            let e = per.entry(t).or_default();
            e.0 += usize::from(was);
            e.1 += usize::from(a.correct);
            e.2 += 1;
        }
    }
    c.concepts = per.into_iter().map(|(t, (b, a, n))| (t, b, a, n)).collect();
    c.concepts.sort_by(|x, y| y.3.cmp(&x.3).then(x.0.cmp(&y.0)));
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::designer::validate;

    fn store() -> MemoryStore {
        let mut s = MemoryStore::default();
        for t in [
            "Max is a dog.",
            "Dogs are animals.",
            "Animals need water.",
            "Bella can swim.",
            "The barn has a red door.",
            "Cats are not fish.",
        ] {
            s.add_teacher(t, 0, 0);
        }
        s
    }

    #[test]
    fn patterns_give_grounded_questions() {
        assert_eq!(
            parse("Max is a dog."),
            Some(Fact {
                subject: "Max".into(),
                plural: false,
                rel: Rel::Is,
                object: "a dog".into(),
                negated: false
            })
        );
        assert_eq!(parse("Animals need water.").unwrap().rel, Rel::Verb("need".into()));
        assert!(parse("Is Max a dog?").is_none());
        let s = store();
        let ids: Vec<u64> = s.entries.iter().map(|m| m.id).collect();
        let d = generate(&s, &ids, ReviewOptions { difficulty: 3 }, "Review");
        let ex: Vec<&DraftItem> = d.blocks[0].items.iter().collect();
        let held: Vec<&DraftItem> = d.blocks[1].items.iter().collect();
        let q = |v: &[&DraftItem], s: &str| v.iter().any(|i| i.question == s);
        assert!(q(&ex, "Is Max a dog?"), "{:#?}", ex.iter().map(|i| &i.question).collect::<Vec<_>>());
        assert!(q(&ex, "Do animals need water?"));
        assert!(q(&ex, "Can Bella swim?"));
        assert!(q(&ex, "Does the barn have a red door?"));
        assert!(q(&ex, "Are cats fish?"));
        assert!(ex
            .iter()
            .find(|i| i.question == "Are cats fish?")
            .unwrap()
            .answer
            .starts_with("No, cats are not fish."));
        assert!(q(&held, "Is it true that Max is a dog?"));
        assert!(q(&held, "What is Max?"));
        // Relationship questions combine two memories and are marked inferred.
        let all: Vec<&DraftItem> = ex.iter().chain(held.iter()).copied().collect();
        let rel =
            all.iter().find(|i| i.question == "Is Max an animal?").expect("Max is a dog + dogs are animals");
        assert!(rel.flags[0].starts_with("Inferred by combining memory #1"));
        // One step only: "Max is a dog" + "Dogs are animals" + "Animals need water" is two.
        assert!(all.iter().all(|i| i.question != "Does Max need water?"));
        // Nothing generated is taught before the teacher checks it.
        assert!(all.iter().all(|i| i.answer_source == AnswerSource::Generated && !i.verified));
        assert!(validate(&d).iter().any(|i| i.message.contains("must be verified")));
        let ex_q: BTreeSet<String> = ex.iter().map(|i| canonical(&i.question)).collect();
        assert!(
            held.iter().all(|i| !ex_q.contains(&canonical(&i.question))),
            "held-out never repeats an example"
        );
    }

    #[test]
    fn verified_review_becomes_a_lesson_with_keys() {
        let s = store();
        let ids: Vec<u64> = s.entries.iter().map(|m| m.id).collect();
        let mut d = generate(&s, &ids, ReviewOptions { difficulty: 2 }, "Review");
        for b in &mut d.blocks {
            for i in &mut b.items {
                i.verified = true;
            }
        }
        let items = lesson_items(&d).unwrap();
        let held: Vec<&Item> = items.iter().filter(|i| i.held_out).collect();
        assert!(!held.is_empty());
        let max = items.iter().find(|i| i.prompt.as_deref() == Some("Is Max a dog?")).unwrap();
        assert_eq!(
            max.answer.as_deref(),
            Some("Yes, Max is a dog."),
            "the first accepted form is the target"
        );
        let key = max.key.as_ref().unwrap();
        assert!(key.is_correct("Yes.") && key.is_correct("yes, max is a dog") && !key.is_correct("No."));
    }
}
