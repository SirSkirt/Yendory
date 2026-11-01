# Yendory

**Where artificial minds begin.**

Yendory is an open-source kit for creating, teaching and exporting AI models built on the
Bicameral neural architecture first developed for Neural Nova. A model can have any name; Nova
is the first student, not the engine.

- **Yendory Engine** — the one shared Rust implementation of the Bicameral brain (`dependency/engine` on the `yendory-hub` branch; Studio builds against it, docs/INTEGRATION.md).
- **Yendory Studio** — the training app for Windows, Linux and Android (`apps/studio`, Tauri).
- **Yendory Nexus** — tools, apps and games for trained models (later).

*Studio teaches. Engine thinks. Nexus connects.*

## The Bicameral architecture (default configuration, 1,182,144 parameters)

| Part | Parameters | What it is |
|---|---|---|
| Left hemisphere (Transformer) | 755,200 | 4 blocks × 4 heads, d=128, SwiGLU, RoPE, RMSNorm, tied 384×128 embedding; the upper 2 blocks run twice ("thinking") |
| Right hemisphere (Dense) | 336,320 | last 64 characters × 16 fixed features → 256 → 192, hears the left state |
| Interconnect | 90,624 | gated right→left injections, left→right efferent path |

It is deliberately not a Llama-style network. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository

```
apps/studio/             Yendory Studio, the GUI app (Tauri 2: Rust backend + HTML/JS)
docs/                    architecture, checkpoint format, verification status

The Engine (with its numpy reference and compatibility tests), the Yendory Network and its
client are on the yendory-hub branch, in dependency/; Studio builds against them.
```

## Build and test

```
cd apps/studio/src-tauri
cargo test --release
```

This builds Studio against the Engine and client pinned in `apps/studio/src-tauri/Cargo.toml` and
runs Studio's tests. The Engine's own tests (golden vectors, the numpy reference on a real trained
checkpoint, GPU against CPU) and the Network's run on the yendory-hub branch. See
[docs/STATUS.md](docs/STATUS.md) for what is and is not verified.

## Principles

- One Engine shared by Studio, Nova's chat app and third parties; no second implementation.
- Every reply comes from the model's distribution. No hardcoded answers, canned fallbacks or invented metrics.
- Existing `.nnova2` checkpoints stay loadable; architecture changes need the owner's approval.
- Training evaluation stays separate from genuinely unseen tests.
- Sharing training material is always an explicit choice.
