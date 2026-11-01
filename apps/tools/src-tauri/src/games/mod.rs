//! The games: small text games a Yendory model plays through Nexus (the Engine's actions and
//! results). Each game knows its rules, its perfect player, what the model sees when it looks,
//! and the words of every move, so the same text appears in lessons and in live play.
//!
//! A game is played in turns. On its turn the model sees only the game's announcement and one
//! line from the other player (`U:`), so a turn fits a small model's context; it looks to see
//! the position, makes one move, and replies. Between its turns the opponent moves (a perfect or
//! a random player, or a person).
use serde::Serialize;
use serde_json::Value;
use yendory_engine::nexus::{Access, Kind, Manifest, Tool};

pub mod maze;
pub mod nim;
pub mod tictactoe;

/// Who is to move.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Model,
    Opponent,
}

/// How a game ended, for the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Win,
    Draw,
    Loss,
}

/// A game in progress.
pub trait Game: Send {
    /// The catalogue entry this game belongs to.
    fn info(&self) -> &'static Info;
    /// Who moves next (a one-player game: always the model).
    fn to_move(&self) -> Side;
    /// The result, once the game is over.
    fn outcome(&self) -> Option<Outcome>;
    /// The moves the side to move may make, as written after the move action (`5`, `2`, `east`).
    fn legal(&self) -> Vec<String>;
    /// The perfect player's moves here (every one of them is best).
    fn best(&self) -> Vec<String>;
    /// Makes a move for the side to move. Returns what happened, from the mover's side
    /// ("You played 5" for the model, "I played 5" for the opponent), with the ending when the
    /// move ends the game. An illegal move is refused with the reason and changes nothing.
    fn apply(&mut self, input: &str) -> Result<String, String>;
    /// What the model sees when it looks.
    fn look(&self) -> String;
    /// The first line of a game, before the first move is announced ("New game: you are x.").
    fn intro(&self) -> String;
    /// The line that asks for the model's next move after the opponent's (or at the start).
    fn your_move(&self) -> String {
        "Your move.".into()
    }
    /// The model's short reply that ends its turn in lessons.
    fn reply(&self) -> String {
        match self.outcome() {
            Some(_) => "Good game.".into(),
            None => "Your turn.".into(),
        }
    }
    /// A key for the position and the side to move (duplicates in lessons).
    fn key(&self) -> String;
    /// What decides whether this position is held out (whole mazes, single positions otherwise).
    fn split(&self) -> String {
        self.key()
    }
    /// Whether this position is held out: never in lessons, so measurements on it show what the
    /// model learned rather than memorised. A game keeps its basics always taught.
    fn is_held_out(&self) -> bool {
        held_out(&self.split())
    }
    /// The position for the window to draw.
    fn board(&self) -> Value;
    fn boxed(&self) -> Box<dyn Game>;
}

/// A game in the catalogue.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Info {
    pub id: &'static str,
    pub name: &'static str,
    /// The name the model sees in the announcement (short: it is in every turn's context).
    pub env_name: &'static str,
    /// Shown in the window.
    pub summary: &'static str,
    pub rules: &'static str,
    /// The announcement's description (keep it short: it is in every turn's context).
    pub description: &'static str,
    /// The move action: its name, how to write it, and what it does.
    pub verb: &'static str,
    pub usage: &'static str,
    pub verb_does: &'static str,
    pub look_does: &'static str,
    /// Two players (the model and an opponent) or a puzzle for the model alone.
    pub two_player: bool,
    /// The best the model can do against the perfect player from a fair start.
    pub perfect_note: &'static str,
}

impl Info {
    /// The environment as Nexus announces it.
    pub fn manifest(&self) -> Manifest {
        Manifest {
            name: self.env_name.into(),
            description: self.description.into(),
            kind: Kind::Game,
            tools: vec![
                Tool {
                    name: "look".into(),
                    usage: "look".into(),
                    description: self.look_does.into(),
                    access: Access::Look,
                },
                Tool {
                    name: self.verb.into(),
                    usage: self.usage.into(),
                    description: self.verb_does.into(),
                    access: Access::Act,
                },
            ],
        }
    }
}

pub const CATALOGUE: &[&Info] = &[&tictactoe::INFO, &nim::INFO, &maze::INFO];

pub fn info(id: &str) -> Option<&'static Info> {
    CATALOGUE.iter().copied().find(|i| i.id == id)
}

/// A new game. `model_first` is ignored by one-player games.
pub fn new_game(id: &str, rng: &mut Rng, model_first: bool) -> Option<Box<dyn Game>> {
    match id {
        "tictactoe" => Some(Box::new(tictactoe::TicTacToe::new(model_first))),
        "nim" => Some(Box::new(nim::Nim::new(rng.range(9, 25) as u32, model_first))),
        "maze" => Some(Box::new(maze::Maze::random(rng))),
        _ => None,
    }
}

/// One in ten positions is held out: never in lessons, used to measure whether the model plays
/// well where it was never taught (FNV-1a of the position's key).
pub fn held_out(key: &str) -> bool {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in key.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h.is_multiple_of(10)
}

/// A small, seeded random number generator (xorshift64*), so games and lessons repeat.
#[derive(Clone, Debug)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// A number in `lo..=hi`.
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next_u64() % (hi - lo + 1)
    }
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            None
        } else {
            Some(&items[(self.next_u64() % items.len() as u64) as usize])
        }
    }
}

/// Opponents the model can play against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Opponent {
    Perfect,
    Random,
    /// A person in the window.
    Person,
}

/// The opponent's move (perfect or random); `None` for a person or when no move is left.
pub fn opponent_move(game: &dyn Game, opponent: Opponent, rng: &mut Rng) -> Option<String> {
    match opponent {
        Opponent::Perfect => rng.pick(&game.best()).cloned(),
        Opponent::Random => rng.pick(&game.legal()).cloned(),
        Opponent::Person => None,
    }
}
