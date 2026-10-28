//! How many tokens each held-out answer of a built-in course takes with a model's tokenizer
//! (greedy replies stop at 24 tokens, so longer answers can never be exactly right).
//! cargo run --release --example answer_lengths -- <model> <course id>
use yendory_engine::learn::builtin::builtin_courses;
use yendory_engine::learn::eval::MAX_REPLY_TOKENS;
use yendory_engine::learn::lessons::parse_lesson;
use yendory_engine::Bicameral;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut model = Bicameral::load(&a[1]).unwrap();
    // Optional: the vocabulary limit a session would grow to (its lessons' frequent words).
    if let Some(limit) = a.get(3).map(|x| x.parse::<usize>().unwrap()) {
        let c = builtin_courses().into_iter().find(|c| c.id == a[2]).unwrap();
        let items: Vec<_> = c.lessons.iter().flat_map(|f| parse_lesson(&f.text, &f.name).unwrap()).collect();
        let words =
            yendory_engine::learn::lessons::grow_vocabulary(model.tokenizer.words(), &items, limit, 12);
        if words.len() > model.word_capacity() {
            model = model.with_word_capacity(words.len(), 1).unwrap();
        }
        model.tokenizer.set_words(words);
    }
    println!(
        "vocab slots {}, learned words {}, max_tokens {}",
        model.config.vocab,
        model.tokenizer.words().len(),
        model.config.max_tokens
    );
    let c = builtin_courses().into_iter().find(|c| c.id == a[2]).unwrap();
    for f in &c.lessons {
        let items = parse_lesson(&f.text, &f.name).unwrap();
        let held: Vec<_> = items.iter().filter(|i| i.held_out && i.answer.is_some()).collect();
        let lens: Vec<usize> = held
            .iter()
            .map(|i| model.tokenizer.encode(i.answer.as_ref().unwrap()).unwrap().len() + 1)
            .collect();
        let fit = lens.iter().filter(|&&n| n <= MAX_REPLY_TOKENS).count();
        println!(
            "{}: {} held out, {fit} fit in {MAX_REPLY_TOKENS} tokens; median {} tokens",
            f.name,
            held.len(),
            {
                let mut l = lens.clone();
                l.sort();
                l[l.len() / 2]
            }
        );
        let keyed = held.iter().filter(|i| i.key.is_some()).count();
        println!("   {keyed} graded by an answer key");
        if let Some(i) = held.first() {
            println!(
                "   e.g. {:?} -> {:?}",
                i.prompt.as_deref().unwrap_or(""),
                i.answer.as_deref().unwrap_or("")
            );
        }
    }
}
