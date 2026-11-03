# Notes for development sessions

## Branches and parts

This is the **`yendory-tools`** branch: Yendory Tools (`apps/tools`), where trained models play
games (and later use apps) through Nexus (docs/GAMES.md). The suite's other parts live on their
own branches: Yendory Hub and the shared `dependency/` (the Engine, the Yendory Network, the
client library) on `yendory-hub`, Yendory Studio on `yendory-studio`.

Tools holds no copy of the Engine or the Network: it uses `yendory-client` and `yendory-engine`
from yendory-hub, pinned to one commit in `apps/tools/src-tauri/Cargo.toml`; move the pin on
purpose. Every model is opened and every line it writes comes through the Network. Tools uses the
Engine for what needs no model file (Nexus's announcements and actions, checking lessons) and for
the game-skill network (`yendory_engine::skill`): it trains and plays a model's game skills
(Nova Neuro or Bicameral) itself, but reads and saves them only through the Network (`model.skill`,
`model.skill.save`; docs/FORMAT.md on yendory-hub, format 2). A change Tools needs in the Network
is made on yendory-hub by whoever owns the Network there.

Real games (docs/GAME_TRAINING.md) are played from the screen with virtual keyboard and mouse
input. Every virtual input goes through the safety layer (`live/safety.rs`: permitted controls
only, no system combinations, only while the game is in front, the emergency stop); never add a
path that sends input around it. Gaming Nova's actions come only from its trained weights: no
code chooses actions for a game, the range's truth is never a network input, scripted actions are
never recorded as a person's demonstrations, and results are shown beside baselines that can't
have learned (docs/GAME_TRAINING.md, section 0). The range teacher in `tests/learn_range.rs` is
test-only; never bring it into the app.

Models are handed between programs through Studio's models folder and `--model <id>`
(docs/HANDOFF.md on yendory-hub). Yendory Hub installs and updates Tools from its signed package
(docs/PACKAGES.md on yendory-hub); Tools has no installer and doesn't update itself.

## Version numbers

Tools is at **0.1.9**. A push that changes Tools raises its last digit by one, and the last digit
carries over after 9: 0.1.9 → 0.2.0 → … → 0.9.9 → 1.0.0. The new number goes in the same push,
in every place that holds it:

- `apps/tools/package.json`
- `apps/tools/package-lock.json` (the two entries for `yendory-tools`)
- `apps/tools/src-tauri/tauri.conf.json`
- `apps/tools/src-tauri/Cargo.toml`
- `apps/tools/src-tauri/Cargo.lock` (update with cargo, e.g. `cargo metadata`)

The release workflow publishes Tools only when its version is new (`tools-v<version>` and
`package.json` on `tools-updates`).

## Switches must work

When the owner asks for something to be switchable (a checkbox, an option, a setting), the
switch must change what actually happens. Never build another path that does the same job
regardless of the switch (for example a download that already removes what a practice-time
checkbox is meant to leave out), so that the switch becomes decorative. If one switch only works
together with another, keep them consistent in the page and say so beside them, or disable it
with the reason. When an existing safeguard or rule makes a new switch inert, raise it with the
owner instead of shipping it.
