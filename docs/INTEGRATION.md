# Studio, the Engine and the Yendory Network

Yendory Studio lives on the `yendory-studio` branch (`apps/studio`). Yendory Hub lives on
`yendory-hub` (the default branch), together with the parts every program shares, once:

| Path on `yendory-hub` | What |
|---|---|
| `dependency/engine` | The Yendory Engine (Rust library) |
| `dependency/network` | The Yendory Network: the background service that runs the Engine for every program (docs/NETWORK.md) |
| `dependency/client` | How a program finds, starts and talks to the Network |
| `apps/hub` | Yendory Hub |

## Building Studio

Studio takes the Engine (for work that needs no model: tokenizer, lessons, parsing, filters) and
the client from `yendory-hub`, pinned to one commit in `apps/studio/src-tauri/Cargo.toml`:

```toml
yendory-engine = { git = "https://github.com/SirSkirt/Yendory", rev = "<commit>", features = ["gpu"] }
yendory-client = { git = "https://github.com/SirSkirt/Yendory", rev = "<commit>" }
```

To move to a newer Engine or Network, change `rev` (all three places) and run
`cargo update -p yendory-engine -p yendory-client` in `apps/studio/src-tauri`. Engine changes are
made on `yendory-hub`, in `dependency/engine`; there is no copy here.

## At run time

Yendory Hub installs the Network (a package, docs/PACKAGES.md). Studio connects to it when it
starts (`Client::start_and_connect`, which starts the copy the Hub installed), and shows how in
Global Settings ▸ Yendory Network. Without it, Studio does everything itself.

What goes through the Network so far (step 1 of docs/NETWORK.md, "Studio on the Network"):

- the model Studio has open is opened there too (no other program renames or deletes it);
- chat replies (`model.chat`, streamed; Stop reaches it with `model.stop`), except replies that
  record a brain trace, which Studio writes itself;
- after Studio saves a model (it still trains on its own), the Network reads it again
  (`models.reload`).

## Game skills

A Nova Neuro model can carry game skills in its own file (Engine 0.9.4, docs/FORMAT.md on the Hub
branch). Yendory Tools trains them and saves them into the model through the Network; Studio
opens, trains and saves such models (Studio 0.9.6 and later) and never changes their skills.
Whenever Studio writes a model, the file's own skills go with it: a skill added or removed in Tools
while Studio had the model open is kept (or stays removed), and `meta.json` lists the skills the
file holds (`skills`), with any field other programs wrote kept as written
(`apps/studio/src-tauri/src/skills.rs`). Studio needs Network 0.1.2 or later, whose Engine reads
these files.

Studio shows them (0.9.7): the page bar says how many the open model has (a link to them), the
model card lists them, and the Brain page draws them. On the Nova Neuro plate they are a region
outside the brain, joined to the bus by a dashed fibre that carries nothing yet (a skill's goal
input is zeros until Tools' Command Mode gives it a command; text never passes through a skill).
Below the plate, each skill's network is drawn layer by layer (pictures, convolutions, encoder,
goal input, working memory, controls), one dot per unit; every figure is measured from the saved
weights (the size of each unit's accepted weights, and how far training has moved them since).

## Builds and installing

Every push to `yendory-studio` that changes Studio builds it (`studio-v<version>-dev.<run>`): its
Hub packages, signed zips of the program itself (`yendory-studio_<version>_windows-x86_64.zip`
with `yendory-studio.exe`, `yendory-studio_<version>_linux-x86_64.zip` with
`Yendory-Studio.AppImage`), and `package.json` with `icon.png` on the fixed `studio-updates`
release (docs/PACKAGES.md on the Hub branch; `.github/scripts/package.py` is the Hub branch's,
unchanged). `package.json` names the Network version Studio needs (`requires.network`), so the
Hub installs that Network first: a Network change is pushed and published before the Studio that
needs it. The installers (`-setup.exe`, `.deb`, `.AppImage`) and `latest.json` are still
published for anyone without the Hub. Studio does not update itself.

## Handing a model to Studio

`yendory-studio --model <id>` opens that model (Studio 0.9.3 and later); see `docs/HANDOFF.md` on
the Hub branch. Started again while it runs, Studio hands the id to the running copy instead of
starting a second one.
