# The Yendory Network

The Yendory Network is a small program, without a window, that runs the Yendory Engine for every
program of the suite. Programs don't hold the Engine for model work: they ask the Network. That
keeps one Engine, at one version, for everything, and lets one place guard the models (two
programs can't change the same model at once).

```
Hub ─┐                      ┌─ models folder (Studio's, docs/HANDOFF.md)
Studio ─┼── 127.0.0.1 ── Yendory Network ── Yendory Engine
Tools ─┘                      └─ open models in memory
```

| Part | Where | What |
|---|---|---|
| `dependency/engine` | Rust library | The Engine (unchanged; also used directly for utilities, see below) |
| `dependency/network` | `yendory-network` program | The service: models folder, open models, training (later) |
| `dependency/client` | Rust library | How a program finds, starts and talks to the Network |

The Hub installs the Network as a package (docs/PACKAGES.md) and keeps it updated. In the Hub it
is called the Yendory Network; its version and the Engine's show in the sidebar and Settings.

## How it runs

- **Started on demand.** A program calls `Client::start_and_connect`: if the Network isn't running,
  it starts the copy the Hub installed (`install.json` in the Network's folder names it).
- **One copy.** A lock on `service.lock` keeps a second copy from starting; a second start
  leaves at once (or waits a few seconds while an old copy is stopping during an update).
- **Private.** It listens on 127.0.0.1 only, on a port the system picks. `service.json` (readable
  by this user only) holds the port and a fresh secret token; a connection must send the token
  in its first message (`hello`) or it is closed.
- **Stops by itself** five minutes after the last program disconnects with nothing open
  (`YENDORY_NETWORK_IDLE` seconds, for tests), or when asked (`shutdown`, which is refused while
  another program is connected; the Hub asks before an update).
- **Its folder** (`<data folder>/org.yendory.network/`): `install.json`, `service.json` while
  running, `service.lock`, `network.log` (previous one kept as `network.old.log`).
- `yendory-network --version` prints `{"network", "api", "engine"}` without starting; the Hub
  checks a new install with it.

## Messages

One JSON object per line, over the connection:

```
→ {"id": 1, "method": "models.list", "params": {}}
← {"id": 1, "result": [...]}
← {"id": 2, "error": {"code": "refused", "message": "Yendory Studio has this model open. …"}}
← {"event": "models.changed", "data": {"id": "m18…", "change": "renamed"}}
```

Answers carry the request's `id` and may come in any order (each request runs on its own).
Error codes: `unauthorized`, `invalid`, `refused` (a rule said no; the message says why),
`not_found`, `failed`, `unknown_method`. The client adds `unreachable`, `not_installed`,
`incompatible` and `disconnected`.

## API 1

| Method | Params | Result |
|---|---|---|
| `hello` (first) | `token`, `client` (its name), `clientVersion`, `api` | `network`, `api`, `engine`, `modelsFolder`, `clientId` |
| `ping` | | `"pong"` |
| `status` | | versions, connected programs, open models, uptime |
| `shutdown` | `force` (optional) | `true`; refused while another program is connected |
| `models.list` | | `[{meta, fileBytes, openBy}]`, newest first (`meta` is meta.json as written) |
| `models.options` | | sizes and Neuro designs, with parameters, per-word cost, bytes per parameter |
| `models.create` | `name`, `architecture`, `preset`, `wordLimit`, `context`; for a Neuro brain chosen network by network, `preset` `"custom"` and `brain` (`BrainChoice`: `width`, and for each network wanted about how many weights: `transformer`, `memory`, `associative`, `relational`, `experts`, `consolidation`, `controller`) (0.1.3) | the new model's entry |
| `models.plan` | the same as `models.create` | `{parameters, fileBytes, networks: [{name, parameters}], config, design}`: what it would make (each Neuro network's exact weights), nothing made (0.1.3) |
| `models.import` | `path`, `name` (optional) | the new model's entry |
| `models.rename` | `id`, `name` | the entry (every other meta.json field kept) |
| `models.delete` | `id` | `true`; refused while any program has it open |
| `models.export` | `id`, `path` | `{path, bytes}` (the copy is read back and compared) |
| `models.open` | `id` | the entry; the model is held in memory for this program |
| `models.close` | `id` | `true` |
| `model.reply` | `id`, `prompt`, `maxTokens` | `{text, cut}`; the model must be open by this program |
| `model.chat` | `id`, `prompt`, `maxTokens`, `sampling` (optional: `temperature`, `topK`, `topP`, `repetitionPenalty`, `seed`; greedy without), `stream` (optional) | `{text, tokens, finished, stopped, step, sampling}`: Studio's chat reply, the same the Engine writes with those settings. With `stream`, each token's text arrives first as a `model.token` event to this program only (Network 0.1.1) |
| `model.stop` | `stream` | `true` when that reply of this program was being written (it stops after the current token), else `false` (0.1.1) |
| `models.reload` | `id` | the entry; reads the model's file again for every program that has it open, after the caller wrote it itself (Studio, while it still trains on its own) (0.1.1) |
| `model.skills` | `id` | the open model's game skills: each one's description (`SkillSpec`: game, name, size, arch, controls, lineage, steps, notes), no weights (0.1.2) |
| `model.skill` | `id`, `game`, `parts` (optional) | `{spec, accepted, latest, m, v, t}`: one skill with its accepted weights, latest weights and optimizer state; floats as base64 little-endian f32 (`yendory_client::floats`) (0.1.2). With `parts`: `{spec, t, bytes}` only, the weights to be read with `model.skill.read` (0.1.3) |
| `model.skill.read` | `id`, `game`, `offset`, `length` (at most 16 MB, whole values) | `{data, bytes}`: those bytes (base64) of the skill's accepted weights, latest weights, Adam m and v, one after another as little-endian f32 (0.1.3) |
| `model.skill.save` | `id`, `spec`, `t`, and the weights: `accepted`, `latest`, `m`, `v` (base64), or `upload` (an upload of the four arrays one after another) | the entry; adds or replaces the skill for `spec.game` and saves the model's file (docs/FORMAT.md: Neuro format 2, or a Bicameral file with skills, 0.1.3) and a `skills` summary in meta.json. Refused for weights that don't fit the skill's network and while another program has the model open (0.1.2) |
| `upload.begin` | `bytes` (at most 1,100 MB) | `{upload}`: a transfer in parts for a method that takes an upload; it belongs to this program and is dropped when it disconnects (0.1.3) |
| `upload.part` | `upload`, `data` (base64, at most 16 MB) | `{received}` (0.1.3) |
| `model.skill.delete` | `id`, `game` | the entry; the file goes back to format 1 when no skill is left (0.1.2) |

Events: `models.changed` (`id`, `change`: created, imported, renamed, deleted),
`network.stopping` (`reason`), and `model.token` (`stream`, `text`; only to the program whose
`model.chat` is being written). A model a program opened is closed when it disconnects; another
program can't rename it, and no one can delete it, while it is open.

**Large messages.** A message is one line of at most 64 MB. Anything larger goes in parts:
`upload.begin` and `upload.part` toward the Network, `model.skill.read` from it
(`yendory_client`: `Client::upload`, `Client::read_parts`, 16 MB a part). A skill's weights
in one message are four arrays as base64, about 21 bytes a weight, so a skill over about 3
million weights must go in parts.

## Studio on the Network (the port)

Studio calls the Engine directly today (19 files, about 100 functions and types). Moving it onto
the Network is a large job, done in steps, each shipping a working Studio. What Studio needs, as
the Studio branch listed it:

1. **Model files**: load and save; create from a size or a Neuro design (presets, size and memory
   estimates); grow; change the word limit and context; export; checkpoints and recovery points.
2. **Training runs**: Reading, Classroom (with rewards and checkpoints), Conversation,
   Planner/courses, Study, Practice, Review and Reflection. Each lasts minutes to hours, streams
   progress, stops on request at a safe point, and writes recovery points that resume exactly.
   Also: the device choice, GPU and CPU-thread settings, the benchmark and memory limits. One
   training at a time owns the GPU.
3. **Generating text**: chat replies with sampling, streamed token by token, with stop; exams
   and grading; the Brain page's measurements, the thinking tree and the per-network trace (live
   steps and influence passes; traces can be several MB).
4. **Utilities that need no model**: the tokenizer, lesson parsing and the Lesson Designer's
   checks, corpus and conversation parsing, dataset filters, practice generators, fingerprints,
   the chat's text clean-up.

How each maps onto the Network:

- **4 stays a library.** Studio keeps using `dependency/engine` directly for work that needs no
  model; the Network handles only model work. So Studio still builds against the Engine source,
  from this one folder, never a copy.
- **1 and 3 become methods** on open models (`model.*`), like `model.reply`. Long answers
  (traces) are sent whole: lines may be up to 64 MiB.
- **2 becomes jobs.** `train.start` returns a job id at once; the Network runs the run and sends
  `job.progress` events (to every program, so the Hub can show a ribbon live), `job.stop` asks
  for a safe stop, `job.finished` / `job.failed` end it. Recovery points are written by the
  Network, so a run survives Studio closing. One job at a time owns the GPU; a second waits or is
  refused with the reason.
- Studio's own planning (which course next, remediation) can stay in Studio at first and move
  into the Network later, so a plan runs on with Studio closed.

Suggested order: (1) model files and replies, so Studio opens, saves and chats through the
Network; (2) one training run end to end as a job (Reading), proving progress, stop and recovery;
(3) the other runs; (4) the Brain page. New methods raise API to 2 only when an existing one
changes; adding methods keeps API 1.
