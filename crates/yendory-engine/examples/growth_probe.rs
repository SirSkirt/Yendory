//! What adding words does to what a model already answers, before any training: the earlier
//! lessons' held-out questions asked before growth, after growing from every lesson trained
//! (earlier lessons included), and after growing from the new course's lessons alone.
//! cargo run --release --example growth_probe -- <model> <lesson files dir> <course id> <word limit>
use yendory_engine::learn::builtin::builtin_courses;
use yendory_engine::learn::eval::evaluate;
use yendory_engine::learn::lessons::{grow_vocabulary, parse_lesson, Item, Kind};
use yendory_engine::{Ablation, Bicameral};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = Bicameral::load(&a[1]).unwrap();
    let mut earlier = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(&a[2]).unwrap().map(|e| e.unwrap().path()).collect();
    files.sort();
    for f in files.iter().filter(|f| f.extension().is_some_and(|x| x == "txt")) {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        earlier.extend(parse_lesson(&std::fs::read_to_string(f).unwrap(), &name).unwrap());
    }
    let c = builtin_courses().into_iter().find(|c| c.id == a[3]).unwrap();
    let lesson: Vec<Item> = c.lessons.iter().flat_map(|f| parse_lesson(&f.text, &f.name).unwrap()).collect();
    let all: Vec<Item> = lesson.iter().chain(&earlier).cloned().collect();
    let limit: usize = a[4].parse().unwrap();
    let held: Vec<&Item> = earlier.iter().filter(|i| i.held_out && i.kind != Kind::Prose).collect();
    let right = |m: &Bicameral| evaluate(m, &held, Ablation::default()).unwrap().overall();
    let before = right(&model);
    let words = grow_vocabulary(model.tokenizer.words(), &all, limit.max(model.tokenizer.words().len()), 12);
    let added = words.len() - model.tokenizer.words().len();
    let mut grown = if words.len() > model.word_capacity() {
        model.with_word_capacity(words.len(), 1 ^ 0x5EED).unwrap()
    } else {
        model.clone()
    };
    grown.tokenizer.set_words(words);
    let random = right(&grown);
    // Only words the new course itself uses often (none taken from the earlier lessons).
    let new_only =
        grow_vocabulary(model.tokenizer.words(), &lesson, limit.max(model.tokenizer.words().len()), 12);
    let mut g2 = if new_only.len() > model.word_capacity() {
        model.with_word_capacity(new_only.len(), 1 ^ 0x5EED).unwrap()
    } else {
        model.clone()
    };
    let added2 = new_only.len() - model.tokenizer.words().len();
    g2.tokenizer.set_words(new_only);
    let only = right(&g2);
    println!(
        "from the new course only: {added2} words added, earlier answers right {}/{}",
        only.correct, only.total
    );
    println!(
        "from every lesson trained: {added} words added. Earlier lessons' held-out answers right: before {}/{}, after {}/{}",
        before.correct, before.total, random.correct, random.total
    );
}
