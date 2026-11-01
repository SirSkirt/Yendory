//! Treasure maze: a 5x5 maze for the model alone. It looks to learn where it is, which ways are
//! open and how far the treasure is, and walks one square a turn; it wins by reaching the
//! treasure before its moves run out. The perfect player walks a shortest path.
use super::*;
use serde_json::json;

pub static INFO: Info = Info {
    id: "maze",
    name: "Treasure maze",
    env_name: "Maze",
    summary: "Find the treasure in a 5x5 maze.",
    rules: "A 5x5 maze with walls between some squares. Each turn, walk one square north, south, east or west. Reach the treasure before your moves run out. Looking tells you which ways are open and how far the treasure is.",
    description: "find the treasure.",
    verb: "move",
    usage: "move <north|south|east|west>",
    verb_does: "walk",
    look_does: "look around",
    two_player: false,
    perfect_note: "The perfect player walks a shortest path and always finds the treasure in time.",
};

const N: usize = 5;
const DIRS: [(&str, i32, i32); 4] = [("north", -1, 0), ("south", 1, 0), ("east", 0, 1), ("west", 0, -1)];

#[derive(Clone, Debug, PartialEq)]
pub struct Maze {
    /// open[cell][dir]: no wall that way.
    open: [[bool; 4]; N * N],
    at: usize,
    treasure: usize,
    moves_left: u32,
    budget: u32,
    found: bool,
}

fn step(cell: usize, d: usize) -> Option<usize> {
    let (r, c) = ((cell / N) as i32 + DIRS[d].1, (cell % N) as i32 + DIRS[d].2);
    (r >= 0 && c >= 0 && r < N as i32 && c < N as i32).then(|| r as usize * N + c as usize)
}
fn back(d: usize) -> usize {
    [1, 0, 3, 2][d]
}

impl Maze {
    /// A random maze: a spanning tree (every square reachable) with a few extra openings, the
    /// treasure at least three steps away, and moves for a shortest path plus four.
    pub fn random(rng: &mut Rng) -> Maze {
        let mut open = [[false; 4]; N * N];
        let mut seen = [false; N * N];
        let mut stack = vec![rng.range(0, (N * N - 1) as u64) as usize];
        seen[stack[0]] = true;
        while let Some(&cell) = stack.last() {
            let ways: Vec<usize> = (0..4).filter(|&d| step(cell, d).is_some_and(|n| !seen[n])).collect();
            match rng.pick(&ways) {
                Some(&d) => {
                    let n = step(cell, d).unwrap();
                    open[cell][d] = true;
                    open[n][back(d)] = true;
                    seen[n] = true;
                    stack.push(n);
                }
                None => {
                    stack.pop();
                }
            }
        }
        for _ in 0..3 {
            let cell = rng.range(0, (N * N - 1) as u64) as usize;
            let d = rng.range(0, 3) as usize;
            if let Some(n) = step(cell, d) {
                open[cell][d] = true;
                open[n][back(d)] = true;
            }
        }
        let mut m = Maze { open, at: 0, treasure: 0, moves_left: 0, budget: 0, found: false };
        loop {
            let at = rng.range(0, (N * N - 1) as u64) as usize;
            let treasure = rng.range(0, (N * N - 1) as u64) as usize;
            let d = m.distances(treasure)[at];
            if d >= 3 {
                m.at = at;
                m.treasure = treasure;
                m.budget = d + 4;
                m.moves_left = m.budget;
                return m;
            }
        }
    }

    /// Steps from every square to `to` (breadth first).
    fn distances(&self, to: usize) -> [u32; N * N] {
        let mut dist = [u32::MAX; N * N];
        dist[to] = 0;
        let mut queue = std::collections::VecDeque::from([to]);
        while let Some(c) = queue.pop_front() {
            for d in 0..4 {
                if let (true, Some(n)) = (self.open[c][d], step(c, d)) {
                    if dist[n] == u32::MAX {
                        dist[n] = dist[c] + 1;
                        queue.push_back(n);
                    }
                }
            }
        }
        dist
    }

    fn walls(&self) -> String {
        self.open
            .iter()
            .map(|o| o.iter().fold(0u8, |a, &b| a * 2 + b as u8).to_string())
            .collect::<Vec<_>>()
            .join("")
    }
}

impl Game for Maze {
    fn info(&self) -> &'static Info {
        &INFO
    }
    fn to_move(&self) -> Side {
        Side::Model
    }
    fn outcome(&self) -> Option<Outcome> {
        if self.found {
            Some(Outcome::Win)
        } else if self.moves_left == 0 {
            Some(Outcome::Loss)
        } else {
            None
        }
    }
    fn legal(&self) -> Vec<String> {
        if self.outcome().is_some() {
            return Vec::new();
        }
        (0..4).filter(|&d| self.open[self.at][d]).map(|d| DIRS[d].0.to_string()).collect()
    }
    fn best(&self) -> Vec<String> {
        if self.outcome().is_some() {
            return Vec::new();
        }
        let dist = self.distances(self.treasure);
        (0..4)
            .filter(|&d| {
                self.open[self.at][d] && step(self.at, d).is_some_and(|n| dist[n] + 1 == dist[self.at])
            })
            .map(|d| DIRS[d].0.to_string())
            .collect()
    }
    fn apply(&mut self, input: &str) -> Result<String, String> {
        if self.outcome().is_some() {
            return Err("The game is over.".into());
        }
        let d = DIRS.iter().position(|x| x.0 == input.trim()).ok_or("Move north, south, east or west.")?;
        if !self.open[self.at][d] {
            return Err(format!("A wall blocks {}.", DIRS[d].0));
        }
        self.at = step(self.at, d).expect("an open way leads to a square");
        self.moves_left -= 1;
        let mut text = format!("You moved {}", DIRS[d].0);
        if self.at == self.treasure {
            self.found = true;
            text.push_str(": treasure found, you win");
        } else if self.moves_left == 0 {
            text.push_str(": out of moves, you lose");
        }
        text.push('.');
        Ok(text)
    }
    fn look(&self) -> String {
        let open: Vec<&str> = (0..4).filter(|&d| self.open[self.at][d]).map(|d| DIRS[d].0).collect();
        let (r, c) = ((self.at / N) as i32, (self.at % N) as i32);
        let (tr, tc) = ((self.treasure / N) as i32, (self.treasure % N) as i32);
        let mut away = Vec::new();
        if tr != r {
            away.push(format!("{} {}", (tr - r).abs(), if tr > r { "south" } else { "north" }));
        }
        if tc != c {
            away.push(format!("{} {}", (tc - c).abs(), if tc > c { "east" } else { "west" }));
        }
        format!("open: {}; treasure: {}", open.join(", "), away.join(", "))
    }
    fn intro(&self) -> String {
        format!("New maze, {} moves.", self.budget)
    }
    fn your_move(&self) -> String {
        if self.moves_left == self.budget {
            "Your move.".into()
        } else {
            format!("{} moves left. Your move.", self.moves_left)
        }
    }
    fn reply(&self) -> String {
        match self.outcome() {
            Some(Outcome::Win) => "Found it.".into(),
            Some(_) => "Good game.".into(),
            None => "On my way.".into(),
        }
    }
    fn key(&self) -> String {
        format!("maze:{}:{}:{}:{}", self.walls(), self.treasure, self.at, self.moves_left)
    }
    fn split(&self) -> String {
        split_key(self)
    }
    fn board(&self) -> Value {
        json!({
            "size": N,
            "open": self.open.iter().map(|o| o.to_vec()).collect::<Vec<_>>(),
            "at": self.at, "treasure": self.treasure,
            "movesLeft": self.moves_left, "budget": self.budget,
        })
    }
    fn boxed(&self) -> Box<dyn Game> {
        Box::new(self.clone())
    }
}

/// Whole mazes are held out (not single squares of a maze that is taught).
pub fn split_key(m: &Maze) -> String {
    format!("maze:{}:{}", m.walls(), m.treasure)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mazes_are_solvable_and_the_perfect_player_solves_them() {
        let mut rng = Rng::new(11);
        for _ in 0..200 {
            let mut m = Maze::random(&mut rng);
            // Every square can reach the treasure.
            assert!(m.distances(m.treasure).iter().all(|&d| d != u32::MAX));
            while m.outcome().is_none() {
                let mv = m.best()[0].clone();
                m.apply(&mv).unwrap();
            }
            assert_eq!(m.outcome(), Some(Outcome::Win));
            assert_eq!(m.budget - m.moves_left + 4, m.budget);
        }
    }

    #[test]
    fn walls_block_and_looking_tells_the_way() {
        let mut rng = Rng::new(5);
        let mut m = Maze::random(&mut rng);
        let blocked = (0..4).find(|&d| !m.open[m.at][d]);
        if let Some(d) = blocked {
            assert_eq!(m.apply(DIRS[d].0).unwrap_err(), format!("A wall blocks {}.", DIRS[d].0));
        }
        let look = m.look();
        assert!(look.starts_with("open: ") && look.contains("; treasure: "), "{look}");
        assert_eq!(m.apply("up").unwrap_err(), "Move north, south, east or west.");
    }
}
