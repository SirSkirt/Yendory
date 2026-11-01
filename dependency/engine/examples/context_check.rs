//! The context tests (two prompts that differ only in an earlier line; the answer must follow it).
//! cargo run --release --example context_check -- <model> [tests per skill]
use yendory_engine::learn::compare::context;
use yendory_engine::Bicameral;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = Bicameral::load(&a[1]).unwrap();
    let n = a.get(2).map(|x| x.parse().unwrap()).unwrap_or(12);
    println!("{:#?}", context(&m, n));
}
