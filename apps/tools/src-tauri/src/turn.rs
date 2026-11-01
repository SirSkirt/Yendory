//! One turn of the model in a game, through Nexus, and the same turn as the perfect player plays
//! it (a lesson session).
//!
//! The model's context for a turn is the game's announcement and one line from the other player:
//!
//! ```text
//! \x02Tic-tac-toe: get three in a row. … Actions: look - see the board; play <1-9> - …
//! U: I played 5. Your move.
//! N: \x05look              ← the model writes action lines and one reply
//! \x06x../.o./..., you are x
//! \x05play 1
//! \x06You played 1.
//! Your turn.
//! ```
//!
//! Actions run through the Engine's own `Nexus::perform` (permissions, unknown actions, result
//! limits). The model writes each line through the Yendory Network; outside markers in what it
//! writes are removed, and an action counts only at the start of a line, as in the Engine's
//! `Nexus::respond`.
use crate::games::Game;
use serde::Serialize;
use yendory_engine::nexus::{self, Call, Environment, Manifest, Nexus, NoApprover};

/// Limits on one turn (the Engine's Nexus defaults).
pub const ACTIONS: usize = 8;
pub const LINE_TOKENS: usize = 64;

/// The game as the environment of one model turn: looking, and one move.
struct TurnEnv<'a> {
    game: &'a mut dyn Game,
    moved: Option<String>,
}

impl Environment for TurnEnv<'_> {
    fn manifest(&self) -> Manifest {
        self.game.info().manifest()
    }
    fn act(&mut self, call: &Call) -> Result<String, String> {
        if call.tool == "look" {
            return Ok(self.game.look());
        }
        if call.tool != self.game.info().verb {
            return Err(format!("cannot {}", call.tool));
        }
        if self.moved.is_some() {
            return Err("You have already moved this turn.".into());
        }
        let text = self.game.apply(&call.input)?;
        self.moved = Some(call.input.clone());
        Ok(text)
    }
}

/// A line of a turn, as the window and lessons show it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Line {
    /// action, result or reply.
    pub kind: &'static str,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnRecord {
    /// The other player's line that started the turn.
    pub prompt: String,
    pub lines: Vec<Line>,
    /// The move made (the input of the move action), if one was.
    pub moved: Option<String>,
    /// replied, actions (the action limit), line (a line without an end), error.
    pub ending: &'static str,
    pub error: Option<String>,
}

/// The context a turn starts from.
pub fn context(game: &dyn Game, prompt: &str) -> String {
    let announcement = game.info().manifest().announcement();
    format!(
        "{}U: {}\nN: ",
        nexus::environment_line(&announcement),
        nexus::sanitize(prompt).replace('\n', " ")
    )
}

/// The model's turn. `line` writes the model's next line after a context: its text without the
/// newline, and whether it was cut off before ending.
pub fn model_turn(
    game: &mut dyn Game,
    prompt: &str,
    mut line: impl FnMut(&str) -> Result<(String, bool), String>,
) -> TurnRecord {
    let mut context = context(game, prompt);
    let mut nexus = Nexus::new(TurnEnv { game, moved: None });
    let mut lines = Vec::new();
    let mut actions = 0;
    let (ending, error) = loop {
        let (text, cut) = match line(&context) {
            Ok(l) => l,
            Err(e) => break ("error", Some(e)),
        };
        if cut {
            lines.push(Line { kind: "reply", text: nexus::sanitize(&text).trim().to_string() });
            break ("line", None);
        }
        match text.strip_prefix('\x05') {
            Some(action) => {
                if actions == ACTIONS {
                    break ("actions", None);
                }
                actions += 1;
                let action = nexus::sanitize(action).trim().to_string();
                let step = nexus.perform(&action, &mut NoApprover);
                context.push_str(&nexus::action_line(&action));
                context.push_str(&nexus::result_line(&step.result));
                lines.push(Line { kind: "action", text: action });
                lines.push(Line { kind: "result", text: step.result });
            }
            None => {
                let reply = nexus::sanitize(&text).trim().to_string();
                lines.push(Line { kind: "reply", text: reply });
                break ("replied", None);
            }
        }
    };
    TurnRecord { prompt: prompt.to_string(), lines, moved: nexus.environment.moved.take(), ending, error }
}

/// The perfect player's turn: it looks, makes `best`, and replies. Changes the game.
pub fn teacher_turn(game: &mut dyn Game, prompt: &str, best: &str) -> TurnRecord {
    let mut script = vec![format!("\x05look"), format!("\x05{} {best}", game.info().verb)].into_iter();
    let mut turn = model_turn(game, prompt, |_| Ok((script.next().unwrap_or_default(), false)));
    // The reply depends on how the move ended the game.
    if let Some(last) = turn.lines.last_mut() {
        last.text = game.reply();
    }
    turn
}

/// A turn as a lesson session (Studio's Multi-Turn Conversation lesson format).
pub fn session(game: &dyn Game, turn: &TurnRecord) -> String {
    let mut s = format!(
        "[session]\nEnvironment: {}\nUser: {}\n",
        game.info().manifest().announcement(),
        nexus::sanitize(&turn.prompt)
    );
    for l in &turn.lines {
        let label = match l.kind {
            "action" => "Action",
            "result" => "Result",
            _ => "Nova",
        };
        s.push_str(&format!("{label}: {}\n", nexus::sanitize(&l.text).replace('\n', " ")));
    }
    s.push_str("[/session]\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::{nim::Nim, tictactoe::TicTacToe, Side};

    #[test]
    fn a_turn_looks_moves_and_replies() {
        let mut g = TicTacToe::from("x...o....", 'x');
        let mut said = vec!["\x05look", "\x05play 3", "Your turn."].into_iter();
        let mut seen = Vec::new();
        let turn = model_turn(&mut g, "I played 5. Your move.", |c| {
            seen.push(c.to_string());
            Ok((said.next().unwrap().to_string(), false))
        });
        assert_eq!(turn.moved.as_deref(), Some("3"));
        assert_eq!(turn.ending, "replied");
        assert_eq!(turn.lines[1].text, "x../.o./..., you are x");
        // The context grows the way the Engine writes it.
        assert!(seen[0].starts_with('\x02') && seen[0].ends_with("U: I played 5. Your move.\nN: "));
        assert!(seen[2].ends_with("\x05play 3\n\x06You played 3.\n"));
        assert_eq!(g.to_move(), Side::Opponent);
    }

    #[test]
    fn mistakes_are_answered_not_fatal() {
        let mut g = Nim::new(5, true);
        let mut said = vec!["\x05take 9", "\x05dance", "\x05take 1", "\x05take 1", "ok"].into_iter();
        let turn = model_turn(&mut g, "Your move.", |_| Ok((said.next().unwrap().to_string(), false)));
        let results: Vec<&str> =
            turn.lines.iter().filter(|l| l.kind == "result").map(|l| l.text.as_str()).collect();
        // The Engine's Nexus shows a refused move as an error result.
        assert_eq!(results[0], "error: Take 1, 2 or 3.");
        assert!(results[1].contains("dance"), "{}", results[1]);
        assert_eq!(results[2], "You took 1, 4 left.");
        assert_eq!(results[3], "error: You have already moved this turn.");
        assert_eq!(turn.moved.as_deref(), Some("1"));
    }

    #[test]
    fn markers_written_mid_line_and_endless_lines_end_the_turn() {
        let mut g = Nim::new(5, true);
        // A result marker the model writes is removed; a reply without an action makes no move.
        let turn = model_turn(&mut g, "Your move.", |_| Ok(("\x06You took 3.".into(), false)));
        assert_eq!(turn.lines[0].text, "You took 3.");
        assert_eq!(turn.moved, None);
        // The action limit.
        let turn = model_turn(&mut g, "Your move.", |_| Ok(("\x05look".into(), false)));
        assert_eq!((turn.ending, turn.lines.len()), ("actions", 2 * ACTIONS));
        let turn = model_turn(&mut g, "Your move.", |_| Ok(("aaaa".into(), true)));
        assert_eq!(turn.ending, "line");
    }

    #[test]
    fn the_teacher_writes_a_session() {
        let mut g = Nim::new(7, true);
        let turn = teacher_turn(&mut g, "New game with 7 stones: you go first. Your move.", "3");
        let s = session(&g, &turn);
        assert_eq!(
            s,
            "[session]\nEnvironment: Nim: take 1-3 stones a turn; taking the last stone wins. Actions: look - count the stones; take <1-3> - take stones\nUser: New game with 7 stones: you go first. Your move.\nAction: look\nResult: 7 stones\nAction: take 3\nResult: You took 3, 4 left.\nNova: Your turn.\n[/session]\n"
        );
    }
}
