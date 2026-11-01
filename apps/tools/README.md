# Yendory Tools

Where trained Yendory models are put to work, through Nexus: first **games**, then **apps**.
Same layout as Studio: numbered sections in a sidebar, the open model in the page bar.

| Section | What it does |
|---|---|
| **1.1 Games** | Tic-tac-toe, Nim and Treasure maze: their rules, what the model sees each turn, and how the open model last scored. |
| **1.2 Play** | A game between the open model and the perfect player, a random player, or you. Step through it turn by turn: the model's actions, the results it got, its reply, and each move graded (best, legal, no move; held out). |
| **1.3 Lessons** | The perfect player's turns as a Multi-Turn Conversation lesson file (held-out positions left out). Teaching through the Network waits for its training job; the file can be taught in Studio's Classroom meanwhile. |
| **1.4 Scoreboard** | Measures the open model on a game: wins, draws and losses, legal and best moves, and best moves in held-out positions, kept per model. |
| **2.1 Apps** | Later. |
| **Settings** | The Yendory Network's state, the folders, the versions. |

Tools opens models through the Yendory Network (`yendory-client`, pinned to a commit of
yendory-hub) and takes `--model <id>` from Yendory Hub (a second start hands it to the running
copy). See [docs/GAMES.md](../../docs/GAMES.md).

## Build

```
cd apps/tools
npm ci
npx tauri dev          # run from source; set YENDORY_NETWORK to a built yendory-network
npx tauri build        # this computer's builds
```

`cargo test --release` in `src-tauri` checks the games and their perfect players, turns through
Nexus, matches, lessons (parsed as Studio reads them, and fitting a 256-token context) and the
scoreboard.

## Releases

Yendory Hub installs and updates Tools; there is no installer of its own.
`.github/workflows/tools-release.yml` builds a signed package for Windows and Linux and, when
`apps/tools/package.json`'s version is new, publishes `tools-v<version>` and `package.json` on the
`tools-updates` release (it needs Hub 0.1.1 and Network 0.1.0).
