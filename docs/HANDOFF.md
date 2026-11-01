# Handing a model between programs

The programs of the Yendory suite (Hub, Studio, Tools) share one models folder and pass a
model to each other by its id. This page is the contract every program follows.

## The models folder

Every model lives in **Studio's models folder**, the `models` folder inside Studio's data folder
(identifier `org.yendory.studio`):

| System | Folder |
|---|---|
| Windows | `%APPDATA%\org.yendory.studio\models\` |
| Linux | `~/.local/share/org.yendory.studio/models/` |
| macOS | `~/Library/Application Support/org.yendory.studio/models/` |

Each model is a folder named by its id (`m` followed by hex digits, e.g. `m18e3ef0178564504`):

| File | What it is | Written by |
|---|---|---|
| `model.nnova2` | The checkpoint (Bicameral `.nnova2` or Nova Neuro layout; the header says which) | Studio, the Network |
| `meta.json` | Name, size, steps, what it read and was taught (Studio's `Meta`, camelCase) | Studio, the Network |
| `archive.json` | The lessons taught, kept for rehearsal | Studio, the Network (`[]` for a new model) |
| `history.jsonl` | Studio's training log | Studio only |
| `checkpoints/`, recovery points | Unfinished training | Studio only |

Rules for every program:

- **Never rewrite a field you don't know.** Change a `meta.json` by reading it as JSON, changing
  only your field, and writing all the rest back as it was. The Hub's rename does exactly this.
- **Write atomically**: a temporary file, flushed, then renamed over the old one.
- **Don't change a model Studio has open.** Studio keeps the open model in memory and writes it
  back when it saves, so a change made underneath it is undone. The Network refuses a change to
  a model another program opened through it, and the Hub refuses renames and deletes while a
  Studio that doesn't use the Network yet is running.
- Each program keeps its own settings and logs in its own data folder (the Hub: `org.yendory.hub`,
  with `shelves.json`, `settings.json`, `programs.json` and `log.jsonl`). Shelves belong to the
  Hub and never change the models folder.

## Opening a program with a model

A program is started with the model's id on its command line:

```
yendory-studio --model m18e3ef0178564504
```

The receiving program, when it starts:

1. reads `--model <id>`;
2. checks that `<models folder>/<id>/meta.json` exists;
3. opens that model as if the owner had chosen it, and shows its usual page for an open model;
4. if the model is missing or can't be read, starts normally and says why.

A program started without `--model` starts as it always has.

**Already running.** A second start of Studio (0.9.3 and later) hands its `--model` to the
running copy and closes, so two copies never write the same files. The Hub starts a program
that is already running only when it does this; otherwise it asks the owner to switch to it.

**Through the Yendory Network.** Programs make and change models through the Yendory Network
(docs/NETWORK.md), which also refuses a change to a model another program has open. Until Studio
works through the Network, the Hub refuses renames and deletes while Studio runs.

## Who supports what

| Program | Reads `--model` | Hands it to a running copy | Through the Network |
|---|---|---|---|
| Hub 0.1.2 | — (starts programs with it) | — | yes |
| Studio 0.9.3 | yes | yes | not yet (docs/NETWORK.md, "Studio on the Network") |
| Tools | not released | | |
