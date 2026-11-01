//! A match: one game between the model and an opponent (the perfect player, a random player or
//! a person), turn by turn, with every model move graded against the perfect player.
use crate::games::{self, opponent_move, Game, Opponent, Outcome, Rng, Side};
use crate::turn::{self, TurnRecord};
use serde::Serialize;

/// How one model move compares with the perfect player's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Grade {
    /// It made a legal move this turn.
    pub legal: bool,
    /// The move is one the perfect player would make.
    pub best: bool,
    /// The position is held out: never in lessons.
    pub held_out: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Entry {
    /// game (its opening line), opponent, person or model.
    pub who: &'static str,
    pub text: String,
    pub turn: Option<TurnRecord>,
    pub grade: Option<Grade>,
}

pub struct Match {
    pub game: Box<dyn Game>,
    pub opponent: Opponent,
    rng: Rng,
    /// What the other player said since the model's last turn (it opens the model's next turn).
    pending: Vec<String>,
    pub log: Vec<Entry>,
    /// The model's turn ended without a move: it loses.
    pub forfeit: bool,
}

impl Match {
    pub fn new(game: Box<dyn Game>, opponent: Opponent, seed: u64) -> Match {
        let intro = game.intro();
        Match {
            game,
            opponent,
            rng: Rng::new(seed ^ 0x5eed),
            pending: vec![intro.clone()],
            log: vec![Entry { who: "game", text: intro, turn: None, grade: None }],
            forfeit: false,
        }
    }

    pub fn finished(&self) -> bool {
        self.forfeit || self.game.outcome().is_some()
    }
    pub fn outcome(&self) -> Option<Outcome> {
        if self.forfeit {
            Some(Outcome::Loss)
        } else {
            self.game.outcome()
        }
    }
    /// Waiting for a person's move.
    pub fn needs_person(&self) -> bool {
        !self.finished() && self.game.to_move() == Side::Opponent && self.opponent == Opponent::Person
    }
    pub fn model_to_move(&self) -> bool {
        !self.finished() && self.game.to_move() == Side::Model
    }

    /// The opponent moves: the perfect or random player, or the person's `input`.
    pub fn opponent_turn(&mut self, input: Option<&str>) -> Result<(), String> {
        if self.finished() || self.game.to_move() != Side::Opponent {
            return Err("It is not the opponent's turn".into());
        }
        let mv = match self.opponent {
            Opponent::Person => input.map(str::to_string).ok_or("Make your move")?,
            other => {
                opponent_move(self.game.as_ref(), other, &mut self.rng).ok_or("The opponent has no move")?
            }
        };
        let text = self.game.apply(&mv)?;
        let who = if self.opponent == Opponent::Person { "person" } else { "opponent" };
        self.log.push(Entry { who, text: text.clone(), turn: None, grade: None });
        self.pending.push(text);
        Ok(())
    }

    /// The line that opens the model's next turn.
    pub fn prompt(&self) -> String {
        let mut parts = self.pending.clone();
        parts.push(self.game.your_move());
        parts.join(" ")
    }

    /// The model's turn; `line` writes each of its lines (see `turn::model_turn`).
    pub fn model_turn(
        &mut self,
        line: impl FnMut(&str) -> Result<(String, bool), String>,
    ) -> Result<(), String> {
        if !self.model_to_move() {
            return Err("It is not the model's turn".into());
        }
        let before = self.game.boxed();
        let prompt = self.prompt();
        let record = turn::model_turn(self.game.as_mut(), &prompt, line);
        if let Some(e) = &record.error {
            // The Network could not write the line: the turn is not counted; the game can go on.
            let e = e.clone();
            self.game = before;
            return Err(e);
        }
        self.pending.clear();
        let grade = Grade {
            legal: record.moved.is_some(),
            best: record.moved.as_ref().is_some_and(|m| before.best().iter().any(|b| b == m.trim())),
            held_out: before.is_held_out(),
        };
        if record.moved.is_none() {
            self.forfeit = true;
        }
        let text = record
            .moved
            .as_ref()
            .map(|m| format!("{} {m}", before.info().verb))
            .unwrap_or_else(|| "no move".into());
        self.log.push(Entry { who: "model", text, turn: Some(record), grade: Some(grade) });
        Ok(())
    }

    /// Plays to the end with a computed opponent (not a person).
    pub fn play_out(
        &mut self,
        mut line: impl FnMut(&str) -> Result<(String, bool), String>,
    ) -> Result<(), String> {
        while !self.finished() {
            if self.game.to_move() == Side::Model {
                self.model_turn(&mut line)?;
            } else {
                self.opponent_turn(None)?;
            }
        }
        Ok(())
    }
}

/// A new game, optionally only one whose split is held out (or not): whole mazes are held out,
/// so a measurement can play mazes it was never taught.
pub fn new_game(
    id: &str,
    rng: &mut Rng,
    model_first: bool,
    held: Option<bool>,
) -> Result<Box<dyn Game>, String> {
    for _ in 0..10_000 {
        let g = games::new_game(id, rng, model_first).ok_or("Unknown game")?;
        if held.is_none_or(|h| g.is_held_out() == h) {
            return Ok(g);
        }
    }
    Err("No such game could be made".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in for the model that plays like the perfect player would, through actions.
    fn perfect_lines(game: &dyn Game) -> impl FnMut(&str) -> Result<(String, bool), String> + '_ {
        let mut n = 0;
        move |_| {
            n += 1;
            Ok(match n % 3 {
                1 => ("\x05look".into(), false),
                2 => (format!("\x05{} {}", game.info().verb, game.best()[0]), false),
                _ => ("Your turn.".into(), false),
            })
        }
    }

    #[test]
    fn a_match_runs_turn_by_turn_and_grades_moves() {
        let mut rng = Rng::new(1);
        let g = new_game("nim", &mut rng, true, None).unwrap();
        let mut m = Match::new(g, Opponent::Random, 4);
        assert!(m.prompt().starts_with("New game with ") && m.prompt().ends_with(" Your move."));
        // A model that never acts forfeits at once.
        m.model_turn(|_| Ok(("hello".into(), false))).unwrap();
        assert!(m.finished() && m.forfeit);
        assert_eq!(m.outcome(), Some(Outcome::Loss));
        assert_eq!(
            m.log.last().unwrap().grade.unwrap(),
            Grade { legal: false, best: false, held_out: m.log.last().unwrap().grade.unwrap().held_out }
        );
    }

    #[test]
    fn a_perfect_stand_in_draws_tictactoe_against_the_perfect_player() {
        let mut rng = Rng::new(2);
        for first in [true, false] {
            let g = new_game("tictactoe", &mut rng, first, None).unwrap();
            let mut m = Match::new(g, Opponent::Perfect, 9);
            while !m.finished() {
                if m.model_to_move() {
                    let snapshot = m.game.boxed();
                    m.model_turn(perfect_lines(snapshot.as_ref())).unwrap();
                    assert!(m.log.last().unwrap().grade.unwrap().best);
                } else {
                    m.opponent_turn(None).unwrap();
                }
            }
            assert_eq!(m.outcome(), Some(Outcome::Draw));
            // The opponent's moves open the model's turns.
            if !first {
                assert!(m.log.iter().any(|e| e.who == "opponent"));
            }
        }
    }

    #[test]
    fn a_person_moves_in_turn() {
        let mut rng = Rng::new(3);
        let g = new_game("tictactoe", &mut rng, false, None).unwrap();
        let mut m = Match::new(g, Opponent::Person, 1);
        assert!(m.needs_person());
        assert!(m.opponent_turn(None).is_err());
        assert_eq!(m.opponent_turn(Some("10")).unwrap_err(), "Choose a square from 1 to 9.");
        m.opponent_turn(Some("5")).unwrap();
        assert_eq!(m.prompt(), "New game: I am x and go first. I played 5. Your move.");
        assert!(m.model_to_move());
    }

    #[test]
    fn held_out_mazes_can_be_asked_for() {
        let mut rng = Rng::new(4);
        for _ in 0..5 {
            let g = new_game("maze", &mut rng, true, Some(true)).unwrap();
            assert!(g.is_held_out());
            let g = new_game("maze", &mut rng, true, Some(false)).unwrap();
            assert!(!g.is_held_out());
        }
    }
}
