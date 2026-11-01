//! The perfect player's lessons: turns it plays in real games, written as a Multi-Turn
//! Conversation lesson (the format Studio's Classroom teaches, with Nexus's `Environment:`,
//! `Action:` and `Result:` lines). Held-out positions never appear, so a measurement can tell
//! whether a model plays well where it was never taught.
use crate::games::{opponent_move, Opponent, Rng, Side};
use crate::play::new_game;
use crate::turn::{session, teacher_turn};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LessonSet {
    pub game: String,
    /// Sessions written (one model turn each).
    pub sessions: usize,
    /// Model turns met in held-out positions and left out.
    pub held_out_skipped: usize,
    /// Games played to find them.
    pub games: usize,
    pub seed: u64,
    /// The lesson file.
    #[serde(skip)]
    pub text: String,
    /// The first sessions, as they read.
    pub sample: Vec<String>,
}

pub const HEADER: &str = "[Multi-Turn Conversation Lesson]\n[examples]\n";
pub const FOOTER: &str = "[/examples]\n[/Multi-Turn Conversation Lesson]\n";

/// Up to `count` lesson sessions for a game. The perfect player plays the model's side; the
/// other side is the perfect or a random player (half the games each), so the lessons meet good
/// and bad play. A turn already written with the same opening line is not written twice.
pub fn teacher_lessons(game_id: &str, count: usize, seed: u64) -> Result<LessonSet, String> {
    let mut rng = Rng::new(seed);
    let mut seen = std::collections::HashSet::new();
    let mut sessions = Vec::new();
    let (mut skipped, mut games) = (0, 0);
    let mut fruitless = 0;
    while sessions.len() < count && fruitless < 2_000 {
        games += 1;
        let model_first = rng.next_u64().is_multiple_of(2);
        let mut game = new_game(game_id, &mut rng, model_first, Some(false))?;
        let opponent = if rng.next_u64().is_multiple_of(2) { Opponent::Perfect } else { Opponent::Random };
        let mut pending = vec![game.intro()];
        let before = sessions.len();
        while game.outcome().is_none() && sessions.len() < count {
            if game.to_move() == Side::Opponent {
                let mv =
                    opponent_move(game.as_ref(), opponent, &mut rng).ok_or("The opponent has no move")?;
                pending.push(game.apply(&mv)?);
                continue;
            }
            pending.push(game.your_move());
            let prompt = pending.join(" ");
            pending.clear();
            let held = game.is_held_out();
            let fresh = seen.insert(format!("{}|{prompt}", game.key()));
            let best = rng.pick(&game.best()).cloned().ok_or("The perfect player has no move")?;
            let turn = teacher_turn(game.as_mut(), &prompt, &best);
            if held {
                skipped += 1;
            } else if fresh {
                sessions.push(session(game.as_ref(), &turn));
            }
        }
        fruitless = if sessions.len() == before { fruitless + 1 } else { 0 };
    }
    let text = format!("{HEADER}{}{FOOTER}", sessions.concat());
    Ok(LessonSet {
        game: game_id.into(),
        sessions: sessions.len(),
        held_out_skipped: skipped,
        games,
        seed,
        sample: sessions.iter().take(3).cloned().collect(),
        text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use yendory_engine::learn::parse_lesson;
    use yendory_engine::{Bicameral, Config};

    #[test]
    fn lessons_parse_as_studio_reads_them_and_fit_a_small_context() {
        let model = Bicameral::with_config(Config::default(), 1);
        for id in ["tictactoe", "nim", "maze"] {
            let set = teacher_lessons(id, 120, 7).unwrap();
            assert!(set.sessions > 10, "{id}: {}", set.sessions);
            let items = parse_lesson(&set.text, &format!("{id}.txt")).unwrap();
            assert_eq!(items.len(), set.sessions, "{id}");
            for item in &items {
                assert!(item.has_actions(), "{id}");
                // Every session fits Nova-2's 256-token context, even spelled letter by letter.
                let text: String = item.segments.iter().map(|s| s.text.as_str()).collect();
                let tokens = model.tokenizer.encode(&text).unwrap().len();
                assert!(tokens <= 256, "{id}: {tokens} tokens in {text:?}");
            }
        }
    }

    #[test]
    fn held_out_positions_never_appear_and_lessons_repeat() {
        let a = teacher_lessons("tictactoe", 300, 5).unwrap();
        let b = teacher_lessons("tictactoe", 300, 5).unwrap();
        assert_eq!(a.text, b.text, "the same seed gives the same lessons");
        assert!(a.held_out_skipped > 0);
        let items = parse_lesson(&a.text, "t.txt").unwrap();
        assert_eq!(items.len(), 300);
        // Nim has few positions: it stops when no new turn can be found.
        let n = teacher_lessons("nim", 10_000, 1).unwrap();
        assert!(n.sessions < 10_000 && n.sessions > 20, "{}", n.sessions);
    }
}
