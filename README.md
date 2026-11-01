# Yendory

**Where artificial minds begin.**

Yendory is an open-source kit for creating, teaching and exporting AI models built on the
Bicameral neural architecture first developed for Neural Nova. A model can have any name; Nova
is the first student, not the engine.

- **Yendory Engine** — the one shared Rust implementation of the Bicameral brain (`dependency/engine`, on the `yendory-hub` branch).
- **Yendory Network** — the background service that runs the Engine for every program (`dependency/network`, on `yendory-hub`).
- **Yendory Hub** — the front door and the suite's installer (`apps/hub`, on `yendory-hub`).
- **Yendory Studio** — the training app (`apps/studio`, on `yendory-studio`).
- **Yendory Tools** — games and apps that put trained models to work, through the Nexus technology (`apps/tools`; this branch, `yendory-tools`; [docs/GAMES.md](docs/GAMES.md)).

*Studio teaches. Engine thinks. Nexus connects.*

## The Bicameral architecture (default configuration, 1,182,144 parameters)

| Part | Parameters | What it is |
|---|---|---|
| Left hemisphere (Transformer) | 755,200 | 4 blocks × 4 heads, d=128, SwiGLU, RoPE, RMSNorm, tied 384×128 embedding; the upper 2 blocks run twice ("thinking") |
| Right hemisphere (Dense) | 336,320 | last 64 characters × 16 fixed features → 256 → 192, hears the left state |
| Interconnect | 90,624 | gated right→left injections, left→right efferent path |

It is deliberately not a Llama-style network. See [docs/ARCHITECTURE.md](https://github.com/SirSkirt/Yendory/blob/yendory-hub/docs/ARCHITECTURE.md) on the `yendory-hub` branch.

## Repository (this branch)

```
apps/tools/              Yendory Tools (Tauri 2: Rust backend + HTML/JS)
docs/GAMES.md            the games, turns, lessons, held-out positions and the scoreboard
.github/                 the release workflow (Tools' signed package for Yendory Hub)
```

## Build and test

See [apps/tools/README.md](apps/tools/README.md): `cargo test --release` in `apps/tools/src-tauri`
checks the games, turns, lessons and the scoreboard. The Engine's own tests (against the numpy
reference) run on the `yendory-hub` branch.

## Principles

- One Engine shared by Studio, Nova's chat app and third parties; no second implementation.
- Every reply comes from the model's distribution. No hardcoded answers, canned fallbacks or invented metrics.
- Existing `.nnova2` checkpoints stay loadable; architecture changes need the owner's approval.
- Training evaluation stays separate from genuinely unseen tests.
- Sharing training material is always an explicit choice.
