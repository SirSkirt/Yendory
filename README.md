# Yendory

**Where artificial minds begin.**

Yendory is an open-source kit for creating, teaching and exporting AI models built on the
Bicameral neural architecture first developed for Neural Nova. A model can have any name; Nova
is the first student, not the engine.

- **Yendory Engine** — the one shared Rust implementation of the Bicameral brain (`dependency/engine`).
- **Yendory Network** — the background service that runs the Engine for every program (`dependency/network`; [docs/NETWORK.md](docs/NETWORK.md)).
- **Yendory Hub** — the front door and the suite's installer: models on shelves, and it installs and updates the Network, Studio and itself (`apps/hub`; this branch, `yendory-hub`).
- **Yendory Studio** — the training app (`apps/studio`, on the `yendory-studio` branch).
- **Yendory Tools** — apps and games that put trained models to work, through the Nexus technology (later).

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
dependency/engine/       the Engine library (no UI, no Tauri)
dependency/network/      the Yendory Network service (runs the Engine for every program)
dependency/client/       how a program finds, starts and talks to the Network
apps/hub/                Yendory Hub (Tauri 2: Rust backend + HTML/JS); Studio is on the yendory-studio branch
reference/               numpy reference = the specification, plus golden/compat generators
tests/compat/            golden vectors, a real trained checkpoint, expected reference outputs
docs/                    architecture, checkpoint format, verification status, the hand-off between programs
```

## Build and test

```
cargo test --release
```

This runs the golden-vector tests and checks the Rust Engine against the numpy reference on a
real trained checkpoint (logits, KV-cached and uncached greedy generation, gradients, byte-exact
checkpoint round trip). GPU training is behind the `gpu` feature:
`cargo test --release --features gpu --test gpu` checks it against the CPU (set
`YENDORY_GPU_SOFTWARE=1` to use a software GPU). See [docs/STATUS.md](docs/STATUS.md) for what is and is not verified.

## Principles

- One Engine shared by Studio, Nova's chat app and third parties; no second implementation.
- Every reply comes from the model's distribution. No hardcoded answers, canned fallbacks or invented metrics.
- Existing `.nnova2` checkpoints stay loadable; architecture changes need the owner's approval.
- Training evaluation stays separate from genuinely unseen tests.
- Sharing training material is always an explicit choice.
