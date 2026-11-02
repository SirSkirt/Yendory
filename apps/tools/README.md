# Yendory Tools

Where trained Yendory models are put to work, through Nexus: first **games**, then **apps**.
Same layout as Studio: numbered sections in a sidebar, the open model in the page bar.

| Section | What it does |
|---|---|
| **1.1 Games** | Tic-tac-toe, Nim and Treasure maze: their rules, what the model sees each turn, and how the open model last scored. |
| **1.2 Play** | A game between the open model and the perfect player, a random player, or you. Step through it turn by turn: the model's actions, the results it got, its reply, and each move graded (best, legal, no move; held out). |
| **1.3 Lessons** | The perfect player's turns as a Multi-Turn Conversation lesson file (held-out positions left out). Teaching through the Network waits for its training job; the file can be taught in Studio's Classroom meanwhile. |
| **1.4 Scoreboard** | Measures the open model on a game: wins, draws and losses, legal and best moves, and best moves in held-out positions, kept per model. |
| **2.1 Game profiles** | Real games, played from the screen with a virtual keyboard and mouse: the game's window, the controls a network may use, the picture it sees, the safety limits, and a live view of that picture. *Start the test range* opens Tools' own small first-person game. |
| **2.2 Controls check** | Test actions (`hold w+shift 500; move 15 0; click left`) sent to the game through the safety layer, with every action and refusal logged; on the test range, what the range received. |
| **2.3 Record** | You play; Tools records what you see and the controls you use, in step (only the profile's controls, only while the game is in front), for training or held out for testing. |
| **2.4 Recordings** | Replay a recording picture by picture with the controls used at each moment and a timeline, check it against the raw input, mark it for training or testing, delete it. |
| **2.5 Train** | The open model's game skills (a Nova Neuro model from Yendory Hub): add its skill for a game, train it on your recordings (generation 0, imitation), measured on held-out recordings beside the frequency baseline; continue or start over. The skill is saved inside the model. |
| **2.6 Play** | The open model plays the game by itself with its skill, through the safety layer (after you allow it, each time), optionally recorded; on the test range, measure it against the untrained network and the frequency baseline on unseen layouts. |
| **2.7 Checks** | Each phase checked on its own on this computer: the picture, input and safety (2), recording (3), the network, training and play (4). |
| **3.1 Apps** | Later. |
| **Settings** | The Yendory Network's state, the folders, the versions. |

Tools opens models through the Yendory Network (`yendory-client`, pinned to a commit of
yendory-hub) and takes `--model <id>` from Yendory Hub (a second start hands it to the running
copy). See [docs/GAMES.md](../../docs/GAMES.md) for the text games and
[docs/GAME_TRAINING.md](../../docs/GAME_TRAINING.md) for real games.

**Emergency stop** for real games: Ctrl+Alt+End, in any program, releases every control and stops
everything running (or the Stop button in the page bar).

## Build

```
cd apps/tools
npm ci
npx tauri dev          # run from source; set YENDORY_NETWORK to a built yendory-network
npx tauri build        # this computer's builds
```

`cargo test --release` in `src-tauri` checks the games and their perfect players, turns through
Nexus, matches, lessons (parsed as Studio reads them, and fitting a 256-token context), the
scoreboard, and for real games the controls, the safety layer, scripts, profiles and the test
range. On Linux, run it under an X display (`xvfb-run -a cargo test --release`) to also play the
range with real input and capture (`tests/live_range.rs`).

## Releases

Yendory Hub installs and updates Tools; there is no installer of its own.
`.github/workflows/tools-release.yml` builds a signed package for Windows and Linux and, when
`apps/tools/package.json`'s version is new, publishes `tools-v<version>` and `package.json` on the
`tools-updates` release (it needs Hub 0.1.1 and Network 0.1.2, for game skills).
