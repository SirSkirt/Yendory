//! The scoreboard: a model plays a game many times and every move is graded. Wins, draws and
//! losses against the perfect and a random player (or, for a puzzle, mazes found), how often it
//! makes a legal move and the perfect player's move, and the same for held-out positions only.
use crate::games::{info, Opponent, Outcome, Rng};
use crate::play::{new_game, Match};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tally {
    pub games: u32,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    /// Losses because the model made no move.
    pub forfeits: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Moves {
    pub turns: u32,
    pub legal: u32,
    pub best: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    pub game: String,
    pub model: String,
    pub model_name: String,
    /// The model's training step when measured.
    pub step: u64,
    pub time: u64,
    pub seed: u64,
    /// Two-player games: against each opponent. A puzzle: `taught` (mazes from the lessons' side)
    /// and `held` (mazes never taught).
    pub tallies: Vec<(String, Tally)>,
    pub moves: Moves,
    pub held_out: Moves,
}

/// What a measurement found: a tally per opponent (or kind of maze), every move, held-out moves.
pub type Measured = (Vec<(String, Tally)>, Moves, Moves);

/// Plays `games` games per opponent (a puzzle: per kind of maze), half with the model first.
/// `line` writes the model's lines; `progress(done, total)` after every game. Stops early (with
/// what was measured) when `stop` is set.
pub fn measure(
    game_id: &str,
    games: u32,
    seed: u64,
    mut line: impl FnMut(&str) -> Result<(String, bool), String>,
    mut progress: impl FnMut(u32, u32),
    stop: &AtomicBool,
) -> Result<Measured, String> {
    let two = info(game_id).ok_or("Unknown game")?.two_player;
    let kinds: Vec<(&str, Opponent, Option<bool>)> = if two {
        vec![("perfect", Opponent::Perfect, None), ("random", Opponent::Random, None)]
    } else {
        vec![("taught", Opponent::Perfect, Some(false)), ("held", Opponent::Perfect, Some(true))]
    };
    let total = games * kinds.len() as u32;
    let mut rng = Rng::new(seed);
    let (mut moves, mut held) = (Moves::default(), Moves::default());
    let mut tallies = Vec::new();
    let mut done = 0;
    for (label, opponent, maze) in kinds {
        let mut t = Tally::default();
        for i in 0..games {
            if stop.load(Ordering::Acquire) {
                break;
            }
            let game = new_game(game_id, &mut rng, i % 2 == 0, maze)?;
            let mut m = Match::new(game, opponent, rng.next_u64());
            m.play_out(&mut line)?;
            for e in &m.log {
                if let Some(g) = e.grade {
                    for into in [Some(&mut moves), g.held_out.then_some(&mut held)].into_iter().flatten() {
                        into.turns += 1;
                        into.legal += g.legal as u32;
                        into.best += g.best as u32;
                    }
                }
            }
            t.games += 1;
            match m.outcome() {
                Some(Outcome::Win) => t.wins += 1,
                Some(Outcome::Draw) => t.draws += 1,
                _ => t.losses += 1,
            }
            t.forfeits += m.forfeit as u32;
            done += 1;
            progress(done, total);
        }
        tallies.push((label.to_string(), t));
    }
    Ok((tallies, moves, held))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_silent_model_forfeits_every_game() {
        let stop = AtomicBool::new(false);
        let (t, moves, _) =
            measure("tictactoe", 4, 1, |_| Ok(("hi".into(), false)), |_, _| {}, &stop).unwrap();
        assert_eq!(t[0].1, Tally { games: 4, wins: 0, draws: 0, losses: 4, forfeits: 4 });
        // 4 games against each of the two opponents, each ended by the first turn.
        assert_eq!((moves.turns, moves.legal, moves.best), (8, 0, 0));
    }

    #[test]
    fn a_model_that_always_takes_one_plays_legally() {
        let stop = AtomicBool::new(false);
        let mut n = 0;
        let lines = move |_: &str| {
            n += 1;
            Ok(if n % 2 == 1 { ("\x05take 1".into(), false) } else { ("ok".into(), false) })
        };
        let (t, moves, _) = measure("nim", 6, 2, lines, |_, _| {}, &stop).unwrap();
        assert_eq!(moves.legal, moves.turns);
        assert!(moves.best < moves.turns);
        assert_eq!(t.iter().map(|x| x.1.forfeits).sum::<u32>(), 0);
        assert_eq!(t[0].1.wins + t[0].1.draws + t[0].1.losses, 6);
    }
}
