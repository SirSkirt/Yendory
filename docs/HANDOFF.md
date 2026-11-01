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
| `model.nnova2` | The checkpoint (Bicameral `.nnova2` or Nova Neuro layout; the header says which) | Studio, Hub |
| `meta.json` | Name, size, steps, what it read and was taught (Studio's `Meta`, camelCase) | Studio, Hub |
| `archive.json` | The lessons taught, kept for rehearsal | Studio, Hub (`[]` for a new model) |
| `history.jsonl` | Studio's training log | Studio only |
| `checkpoints/`, recovery points | Unfinished training | Studio only |

Rules for every program:

- **Never rewrite a field you don't know.** Change a `meta.json` by reading it as JSON, changing
  only your field, and writing all the rest back as it was. The Hub's rename does exactly this.
- **Write atomically**: a temporary file, flushed, then renamed over the old one.
- **Don't change a model Studio has open.** Studio keeps the open model in memory and writes it
  back when it saves, so a change made underneath it is undone. The Hub refuses rename and
  delete while Studio is running.
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

**Already running.** The Hub does not start a second copy of a program that is running (two
copies would write the same files). It tells the owner to switch to the open program instead.
A later version can forward the id to the running program (for example with Tauri's
single-instance plugin), and then the Hub can hand over a model at any time.

## Who supports what

| Program | Reads `--model` | Notes |
|---|---|---|
| Hub 0.1.0 | — | Starts Studio with `--model`; creates, imports, renames, deletes and exports models |
| Studio 0.9.1 | not yet | Starts on § 1 Models; the owner opens the model there. When Studio supports the hand-off, set `STUDIO_OPENS_MODELS_FROM` in `apps/hub/web/js/app.js` to that version so the Hub stops saying so. |
| Tools | not released | |
