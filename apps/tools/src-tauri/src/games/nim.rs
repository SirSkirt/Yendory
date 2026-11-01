//! Nim: take 1, 2 or 3 stones; whoever takes the last stone wins. The perfect player leaves a
//! multiple of four; from a multiple of four every move loses against it.
use super::*;
use serde_json::json;

pub static INFO: Info = Info {
    id: "nim",
    name: "Nim",
    env_name: "Nim",
    summary: "Take 1 to 3 stones; the last stone wins.",
    rules:
        "One pile of stones. Players take turns taking 1, 2 or 3 stones; whoever takes the last stone wins.",
    description: "take 1-3 stones a turn; taking the last stone wins.",
    verb: "take",
    usage: "take <1-3>",
    verb_does: "take stones",
    look_does: "count the stones",
    two_player: true,
    perfect_note: "The player to move wins against perfect play unless the pile is a multiple of four.",
};

#[derive(Clone, Debug, PartialEq)]
pub struct Nim {
    stones: u32,
    start: u32,
    model_turn: bool,
    /// Who took the last stone.
    last: Option<Side>,
}

impl Nim {
    pub fn new(stones: u32, model_first: bool) -> Nim {
        Nim { stones: stones.max(1), start: stones.max(1), model_turn: model_first, last: None }
    }
}

impl Game for Nim {
    fn info(&self) -> &'static Info {
        &INFO
    }
    fn to_move(&self) -> Side {
        if self.model_turn {
            Side::Model
        } else {
            Side::Opponent
        }
    }
    fn outcome(&self) -> Option<Outcome> {
        match self.last {
            Some(Side::Model) => Some(Outcome::Win),
            Some(Side::Opponent) => Some(Outcome::Loss),
            None => None,
        }
    }
    fn legal(&self) -> Vec<String> {
        if self.outcome().is_some() {
            return Vec::new();
        }
        (1..=self.stones.min(3)).map(|k| k.to_string()).collect()
    }
    fn best(&self) -> Vec<String> {
        match self.stones % 4 {
            _ if self.outcome().is_some() => Vec::new(),
            // Every move loses against perfect play: any is as good.
            0 => self.legal(),
            r => vec![r.to_string()],
        }
    }
    fn apply(&mut self, input: &str) -> Result<String, String> {
        if self.outcome().is_some() {
            return Err("The game is over.".into());
        }
        let k: u32 = input.trim().parse().map_err(|_| "Take 1, 2 or 3.".to_string())?;
        if !(1..=3).contains(&k) {
            return Err("Take 1, 2 or 3.".into());
        }
        if k > self.stones {
            return Err(format!("Only {} left.", self.stones));
        }
        let by = self.to_move();
        self.stones -= k;
        self.model_turn = !self.model_turn;
        let who = if by == Side::Model { "You" } else { "I" };
        Ok(if self.stones == 0 {
            self.last = Some(by);
            let ending = if by == Side::Model { "you win" } else { "I win" };
            format!("{who} took {k}: none left, {ending}.")
        } else {
            format!("{who} took {k}, {} left.", self.stones)
        })
    }
    fn look(&self) -> String {
        format!("{} stones", self.stones)
    }
    fn intro(&self) -> String {
        let who = if self.model_turn { "you go" } else { "I go" };
        format!("New game with {} stones: {who} first.", self.start)
    }
    fn key(&self) -> String {
        format!("nim:{}", self.stones)
    }
    /// Piles of 8 or fewer end every game, so they are always taught; larger piles may be held
    /// out (the model must apply the rule to a pile it never saw).
    fn is_held_out(&self) -> bool {
        self.stones > 8 && held_out(&self.key())
    }
    fn board(&self) -> Value {
        json!({"stones": self.stones, "start": self.start})
    }
    fn boxed(&self) -> Box<dyn Game> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_perfect_player_leaves_multiples_of_four() {
        assert_eq!(Nim::new(13, true).best(), ["1"]);
        assert_eq!(Nim::new(10, true).best(), ["2"]);
        assert_eq!(Nim::new(12, true).best().len(), 3);
        // From a winning start, perfect play always wins against any opponent.
        let mut rng = Rng::new(3);
        for n in [9u32, 10, 11, 13, 21] {
            for _ in 0..10 {
                let mut g = Nim::new(n, true);
                while g.outcome().is_none() {
                    let m = if g.to_move() == Side::Model {
                        g.best()[0].clone()
                    } else {
                        rng.pick(&g.legal()).unwrap().clone()
                    };
                    g.apply(&m).unwrap();
                }
                assert_eq!(g.outcome(), Some(Outcome::Win), "{n}");
            }
        }
    }

    #[test]
    fn moves_are_described_and_checked() {
        let mut g = Nim::new(5, true);
        assert_eq!(g.intro(), "New game with 5 stones: you go first.");
        assert_eq!(g.apply("4").unwrap_err(), "Take 1, 2 or 3.");
        assert_eq!(g.apply("3").unwrap(), "You took 3, 2 left.");
        assert_eq!(g.look(), "2 stones");
        assert_eq!(g.apply("3").unwrap_err(), "Only 2 left.");
        assert_eq!(g.apply("2").unwrap(), "I took 2: none left, I win.");
        assert_eq!(g.outcome(), Some(Outcome::Loss));
    }
}
