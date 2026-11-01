# Games in Yendory Tools

Yendory Tools is where trained models are put to work. Its first category is games: small text
games a model plays through **Nexus**, the Engine's actions and results (docs/LEARNING.md,
"Yendory Nexus", on yendory-hub). Apps (tools a model uses the same way) come later.

Tools holds no model. It opens one through the Yendory Network, which writes every line the model
writes (`model.reply`). The games, their perfect players, the lessons and the scoreboard live in
Tools (`apps/tools/src-tauri/src`).

## The games

| Game | Actions | Perfect player | Best result against it |
|---|---|---|---|
| Tic-tac-toe | `look`, `play <1-9>` | exact (every position solved) | a draw |
| Nim (take 1-3; last stone wins) | `look`, `take <1-3>` | leaves a multiple of four | a win from a winning start |
| Treasure maze (5x5, alone) | `look`, `move <north\|south\|east\|west>` | a shortest path | always found |

Each game is a Nexus environment of kind *game*: looking is allowed, moving is acting (allowed in
games). Refused moves come back as `error: …` results, so a model can learn from them.

## A turn

The model's context for one turn is only the game's announcement and one line from the other
player, so a turn fits a small model's 256-token context even spelled letter by letter (a test
checks every lesson). The model must look to see the position:

```
Environment: Nim: take 1-3 stones a turn; taking the last stone wins. Actions: look - count the stones; take <1-3> - take stones
User: I took 2, 7 left. Your move.
Action: look
Result: 7 stones
Action: take 3
Result: You took 3, 4 left.
Nova: Your turn.
```

Actions run through the Engine's own `Nexus::perform`, with its limits (8 actions a turn, 64
tokens a line). A turn that ends without a legal move loses the game (a forfeit). Lines come from
the Network's `model.reply` (greedy); markers the model writes are removed and an action counts
only at the start of a line. **The Engine's `Nexus::respond` also never lets the model choose a
result or announcement marker while it writes**; to match it exactly, the Network would offer a
`model.line` method wrapping `nexus::next_line` (a request for the Network's owner).

## Lessons

The perfect player plays the model's side of real games (against the perfect or a random player,
half each) and every turn becomes a session in a **Multi-Turn Conversation lesson**, the format
Studio's Classroom reads. Duplicates (same position and opening line) are written once.

**Held out.** One position in ten is never written, and the scoreboard counts the model's moves
in those positions separately, so it shows what a model learned rather than memorised. Each game
keeps its basics always taught: Tic-tac-toe positions with fewer than two marks, Nim piles of 8
or fewer (they end every game), and in the maze whole mazes are held out.

## The scoreboard

The open model plays a game many times (half the games going first): against the perfect and a
random player, or, for the maze, on mazes from the lessons' side and on held-out mazes. Every
move is graded: legal, and the perfect player's or not. Reports are kept per model in the Tools
data folder (`scores/<model id>.jsonl`) with the model's training step, so progress shows over
training.

## Teaching

Training through the Network waits for its training job. When it arrives, Tools sends a game's
lessons to it and measures again. Until then, a lesson file saved from Tools can be taught in
Studio's Classroom.

**Can a small model learn this?** `examples/learn_game.rs` (a developer experiment, not part of
the app) teaches a new Nova-2-size model (1.2M parameters) with the Engine directly and measures
it with the scoreboard code. Nim, 114 sessions, 300 steps (about 10 minutes on a laptop CPU):

| | Untaught | Taught |
|---|---|---|
| Legal moves | 0 of 20 (every game forfeited) | 93 of 93 |
| The perfect player's move | 0 | 79 of 93 |
| Against the perfect player (10 games) | 0 won | 4 won |
| Against a random player (10 games) | 0 won | 6 won |
| Held-out piles: best move | — | 4 of 12 |

It learns the taught positions and the format quickly; applying Nim's rule to piles it never saw
needs more (more varied lessons, or practice: playing and being taught the perfect player's move
where it erred).

```
cargo run --release --example learn_game -- nim 200 300 [models folder]
```
