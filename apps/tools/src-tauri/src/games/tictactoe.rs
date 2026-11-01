//! Tic-tac-toe: squares 1-9 row by row; x moves first. The perfect player is exact (every
//! position is solved), so every move can be graded.
use super::*;
use serde_json::json;

pub static INFO: Info = Info {
    id: "tictactoe",
    name: "Tic-tac-toe",
    env_name: "Tic-tac-toe",
    summary: "Three in a row on a 3x3 grid.",
    rules: "Players take turns marking a free square, x first. Three marks in a row, column or diagonal win; a full board with no line is a draw. Squares are numbered 1 to 9, left to right, top to bottom.",
    description: "get three in a row. Squares 1-9 row by row.",
    verb: "play",
    usage: "play <1-9>",
    verb_does: "mark a free square",
    look_does: "see the board",
    two_player: true,
    perfect_note: "Perfect play from both sides is a draw: the best result against the perfect player is a draw.",
};

const LINES: [[usize; 3]; 8] =
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];

#[derive(Clone, Debug, PartialEq)]
pub struct TicTacToe {
    /// 'x', 'o' or '.'.
    cells: [char; 9],
    /// The model's mark.
    model: char,
}

fn winner(c: &[char; 9]) -> Option<char> {
    LINES.iter().find(|l| c[l[0]] != '.' && c[l[0]] == c[l[1]] && c[l[1]] == c[l[2]]).map(|l| c[l[0]])
}
fn full(c: &[char; 9]) -> bool {
    c.iter().all(|&x| x != '.')
}
fn turn_mark(c: &[char; 9]) -> char {
    let xs = c.iter().filter(|&&x| x == 'x').count();
    let os = c.iter().filter(|&&x| x == 'o').count();
    if xs > os {
        'o'
    } else {
        'x'
    }
}

/// The value of a position for the player to move: 1 win, 0 draw, -1 loss (perfect play).
fn value(c: &mut [char; 9]) -> i32 {
    if winner(c).is_some() {
        // The previous player made a line.
        return -1;
    }
    if full(c) {
        return 0;
    }
    let me = turn_mark(c);
    let mut best = -2;
    for i in 0..9 {
        if c[i] == '.' {
            c[i] = me;
            best = best.max(-value(c));
            c[i] = '.';
        }
    }
    best
}

impl TicTacToe {
    pub fn new(model_first: bool) -> TicTacToe {
        TicTacToe { cells: ['.'; 9], model: if model_first { 'x' } else { 'o' } }
    }
    /// A position from its cells and the model's mark (tests and held-out positions).
    pub fn from(cells: &str, model: char) -> TicTacToe {
        let mut c = ['.'; 9];
        for (i, ch) in cells.chars().filter(|c| "xo.".contains(*c)).take(9).enumerate() {
            c[i] = ch;
        }
        TicTacToe { cells: c, model }
    }
    fn mover(&self) -> char {
        turn_mark(&self.cells)
    }
    fn rows(&self) -> String {
        let s: String = self.cells.iter().collect();
        format!("{}/{}/{}", &s[0..3], &s[3..6], &s[6..9])
    }
    /// Every position with the model to move that can arise in a game, for either mark.
    pub fn all_model_positions() -> Vec<TicTacToe> {
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::new();
        fn walk(c: [char; 9], seen: &mut std::collections::BTreeSet<String>, out: &mut Vec<TicTacToe>) {
            if winner(&c).is_some() || full(&c) {
                return;
            }
            let key: String = c.iter().collect();
            if !seen.insert(key) {
                return;
            }
            let mover = turn_mark(&c);
            out.push(TicTacToe { cells: c, model: mover });
            for i in 0..9 {
                if c[i] == '.' {
                    let mut n = c;
                    n[i] = mover;
                    walk(n, seen, out);
                }
            }
        }
        walk(['.'; 9], &mut seen, &mut out);
        out
    }
}

impl Game for TicTacToe {
    fn info(&self) -> &'static Info {
        &INFO
    }
    fn to_move(&self) -> Side {
        if self.mover() == self.model {
            Side::Model
        } else {
            Side::Opponent
        }
    }
    fn outcome(&self) -> Option<Outcome> {
        match winner(&self.cells) {
            Some(w) if w == self.model => Some(Outcome::Win),
            Some(_) => Some(Outcome::Loss),
            None if full(&self.cells) => Some(Outcome::Draw),
            None => None,
        }
    }
    fn legal(&self) -> Vec<String> {
        if self.outcome().is_some() {
            return Vec::new();
        }
        (0..9).filter(|&i| self.cells[i] == '.').map(|i| (i + 1).to_string()).collect()
    }
    fn best(&self) -> Vec<String> {
        if self.outcome().is_some() {
            return Vec::new();
        }
        let me = self.mover();
        let mut c = self.cells;
        let mut scored = Vec::new();
        for i in 0..9 {
            if c[i] == '.' {
                c[i] = me;
                scored.push((i, -value(&mut c)));
                c[i] = '.';
            }
        }
        let top = scored.iter().map(|s| s.1).max().unwrap_or(0);
        scored.into_iter().filter(|s| s.1 == top).map(|s| (s.0 + 1).to_string()).collect()
    }
    fn apply(&mut self, input: &str) -> Result<String, String> {
        if self.outcome().is_some() {
            return Err("The game is over.".into());
        }
        let n: usize = input.trim().parse().map_err(|_| "Choose a square from 1 to 9.".to_string())?;
        if !(1..=9).contains(&n) {
            return Err("Choose a square from 1 to 9.".into());
        }
        if self.cells[n - 1] != '.' {
            return Err(format!("Square {n} is taken."));
        }
        let by = self.to_move();
        self.cells[n - 1] = self.mover();
        let who = if by == Side::Model { "You" } else { "I" };
        let mut text = format!("{who} played {n}");
        match self.outcome() {
            Some(Outcome::Win) => text.push_str(": three in a row, you win"),
            Some(Outcome::Loss) => text.push_str(": three in a row, I win"),
            Some(Outcome::Draw) => text.push_str(": the board is full, a draw"),
            None => {}
        }
        text.push('.');
        Ok(text)
    }
    fn look(&self) -> String {
        format!("{}, you are {}", self.rows(), self.model)
    }
    fn intro(&self) -> String {
        if self.model == 'x' {
            "New game: you are x and go first.".into()
        } else {
            "New game: I am x and go first.".into()
        }
    }
    fn key(&self) -> String {
        format!("tictactoe:{}:{}", self.cells.iter().collect::<String>(), self.model)
    }
    /// Openings (fewer than two marks) are always taught; deeper positions may be held out.
    fn is_held_out(&self) -> bool {
        self.cells.iter().filter(|&&c| c != '.').count() >= 2 && held_out(&self.key())
    }
    fn board(&self) -> Value {
        json!({"cells": self.cells.iter().map(|c| c.to_string()).collect::<Vec<_>>(), "model": self.model.to_string()})
    }
    fn boxed(&self) -> Box<dyn Game> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_perfect_player_is_exact() {
        // The empty board is a draw; a blocked line must be blocked; a win is taken.
        assert_eq!(value(&mut ['.'; 9]), 0);
        let g = TicTacToe::from("xx.oo....", 'x');
        assert_eq!(g.best(), ["3"]);
        let g = TicTacToe::from("xx..o....", 'o');
        assert_eq!(g.best(), ["3"]);
        // A lost position: every move is as good (x wins whatever o does).
        assert_eq!(TicTacToe::from("xx.o.....", 'o').best().len(), 6);
        // Perfect against perfect always draws.
        let mut rng = Rng::new(7);
        for first in [true, false] {
            for _ in 0..20 {
                let mut g = TicTacToe::new(first);
                while g.outcome().is_none() {
                    let m = rng.pick(&g.best()).unwrap().clone();
                    g.apply(&m).unwrap();
                }
                assert_eq!(g.outcome(), Some(Outcome::Draw));
            }
        }
    }

    #[test]
    fn moves_are_described_and_checked() {
        let mut g = TicTacToe::new(true);
        assert_eq!(g.apply("5").unwrap(), "You played 5.");
        assert_eq!(g.to_move(), Side::Opponent);
        assert_eq!(g.apply("5").unwrap_err(), "Square 5 is taken.");
        assert_eq!(g.apply("10").unwrap_err(), "Choose a square from 1 to 9.");
        assert_eq!(g.apply("1").unwrap(), "I played 1.");
        assert_eq!(g.look(), "o../.x./..., you are x");
        let mut g = TicTacToe::from("xx.oo....", 'x');
        assert_eq!(g.apply("3").unwrap(), "You played 3: three in a row, you win.");
        assert_eq!(g.outcome(), Some(Outcome::Win));
        // 4,520 positions with a player to move (both marks).
        assert_eq!(TicTacToe::all_model_positions().len(), 4520);
    }
}
