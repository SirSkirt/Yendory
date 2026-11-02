# Game training in Yendory Tools (plan)

Status: **phases 2, 3 and 4 built** (Tools 0.1.2: game profiles, the safety layer, virtual
input, capture, the emergency stop and the test range; 0.1.3: recording a person's play, and
inspecting, replaying and checking recordings; 0.1.4: Gaming Nova's network, generation 0 by
imitation, live play and measuring on the range, and a Checks page that verifies each phase on
its own; see "Built so far" at the end). The rest is the plan. The goal
(from the owner): Gaming Nova learns to
play real games from the screen, through virtual keyboard and mouse, first by imitating the
owner's play, then by its own experience and by generations of candidates, without any
game-specific bot code. Conversational Nova later talks with the owner and gives Gaming Nova its
goals.

This document records what the current suite offers for that, what it does not, and the order of
work. Each phase ends with something tested on a real window, not just a page.

## 0. Ground rules (from the owner)

1. **Learning is real.** Every action Gaming Nova takes in play comes from its network's weights,
   and those weights change only through training on demonstrations and experience. No code
   chooses actions for a game, the test range included. In practice:
   - the network's only inputs are the captured pictures, its own previous action and the
     command (section 4); the range's truth report (position, angle, targets) is never one of
     them; it is used only to check inputs and to score evaluations, as a game's score would be;
   - demonstrations are recorded from a person playing; scripted test actions (2.2 Controls
     check) are never recorded as demonstrations, and every recording says who played (a person,
     or which model);
   - every result is shown beside baselines that cannot have learned anything: the untrained
     network and random play, on the same scenarios; and generalization is measured on layouts
     never seen in training (the range's held-out seeds).
2. **One family of networks in the end.** Gaming Nova starts in Tools but is built to join the
   Engine's Neuro design rather than stay a separate architecture (section 4, "Built to join
   Neuro"), so the two Novas can later share an interface and be trained together.
3. **Windows first** (the owner's gaming desktop); Linux (X11) stays supported and is where the
   automatic tests run.
4. **The range first**, Ravenfield once recording, training, playing and evaluation work there.
5. **Kept in view**: generations of competing candidates with checkpoint inheritance and a
   protected champion (section 5), and Conversational Nova working with Gaming Nova (section 6).

## 1. What exists and what carries over

| Part | What it has | Use for game training |
|---|---|---|
| **Tools** (`apps/tools`) | Studio's notebook shell, the text games (Tic-tac-toe, Nim, maze) played through Nexus, lessons, the scoreboard (held-out positions, reports kept per model), `--model` hand-off | The shell, the scoreboard's ideas (held-out evaluation, reports per model, history) and the release path. The text games stay as they are. |
| **Engine** (`dependency/engine`) | Bicameral and Neuro models: transformers over **text tokens**, its own reverse-mode autodiff (`model/tape.rs`), AdamW, `.nnova2` checkpoints, GPU training through wgpu, teaching sessions with candidates, guards, probation and held-out checks | The *rules* for protecting the accepted model (candidate vs accepted, held-out guards, probation branches) carry over as concepts. The model itself cannot: it has no image input, no multi-label or continuous outputs, and its autodiff is private to the Engine (`pub(crate)`). |
| **Nexus** (`engine/src/nexus.rs`) | Text actions and results, one turn at a time | Not for real-time play (a game doesn't wait for a line of text). Still the right channel for Conversational Nova's side. |
| **Network** (`dependency/network`) | Opens models, `model.reply` / `model.chat`, one service on 127.0.0.1 | Conversational Nova (later). Training jobs are planned there but not built; it listens on this computer only. |
| **Studio** | Classroom, acceptance and probation, recovery points | The acceptance/probation behaviour to copy for champions. |
| **Hub** | Signed packages, installs and updates Tools | Unchanged: Tools ships the new features in its own package. |

**The main finding.** The Engine's models read text. A game frame is not text: turning pixels into
characters would give a 256-token model a picture of a few hundred dots with no sense of two
dimensions, and every decision would cost a full pass over that sequence. Gaming Nova therefore
needs a network made for frames and controls: a small convolutional encoder over the last few
frames and heads that output the state of each permitted control. This is a second kind of model
next to Bicameral/Neuro, not a replacement; the plan keeps it in the Engine's style (plain Rust,
no ML framework, gradient-checked, a documented file format) so it can move into
`dependency/engine` once it has proven itself, where the Network could then run it and its
training, and where Neuro's bus can be connected to it (the neural interconnect).

## 2. How a game is played by a network

**Ticks.** Play runs in fixed ticks (20 per second by default, set per profile). Every tick: grab
the game window, shrink it to the profile's size, give the network the last few frames and its
previous action, and get back the *state* of every permitted control for this tick:

- each permitted key: held or not (one yes/no output per key);
- each permitted mouse button: held or not;
- mouse movement x and y: one of a set of steps (for example 0, ±1, ±3, ±8, ±20, ±50 units),
  chosen as classes rather than a single number, so "a small or a large turn" can both be likely.

Holding is how long a control stays on across ticks: "hold W for 250 ms" is W on for 5 ticks, W
and Shift together are two keys on at once, "hold the right button while aiming" is the button on
while mouse steps change. The input layer turns changes of state into press and release events,
so timing is exact to the tick, and the raw events (with microsecond times) are kept in
recordings too.

**Safety layer (between the network and the system).** Every action passes through it:

- only the profile's permitted keys and buttons exist; everything else (system keys, Alt+Tab,
  the Windows key, Escape unless allowed) cannot be expressed at all;
- input is sent only while the chosen game window is in the foreground; when focus leaves it,
  everything is released and play pauses;
- a maximum hold time, a maximum action rate and mouse step limits per profile;
- an **emergency stop** hotkey (global, works while the game has focus) and a stop button: both
  release every control and end autonomous play at once; the same happens when Tools closes,
  crashes, or (later) loses its connection to a remote inference machine;
- optional take-over: any physical keyboard or mouse use during autonomous play pauses it;
- autonomous play starts only after an explicit confirmation each session, and everything sent
  is logged.

**Limits worth knowing.** On Windows, virtual input (`SendInput`) goes to whichever window has
focus, so keeping it to the game is done by the focus rule above, not by the operating system.
Real isolation (a second desktop session or a virtual machine running the game) is possible later
but not in the first version. Windows marks injected input as injected; Tools does not hide this,
and it is not for games with anti-cheat or for online play. Games in exclusive fullscreen may not
be capturable; borderless windowed works.

**Platforms.** Windows: capture by copying what the screen shows in the game's drawing area (GDI,
shrunk as it is copied), input through `SendInput` with scan codes and relative mouse movement,
recording (phase 3) through Raw Input, which still sees mouse movement when the game locks the
cursor. The screen copy means the game must be visible (borderless or windowed; an exclusive
full-screen game may come out black); Windows Graphics Capture, which sees a covered window too,
can replace it if that matters. Linux: X11 only (capture with GetImage, input with XTEST, the stop
keys with a key grab; recording with XInput2), through x11rb; Wayland does not allow either.

## 3. Recordings (demonstrations and episodes)

A recording is a folder in Tools' data (`games/<profile>/recordings/<id>/`; the code and the exact
layout are in `live/recording.rs`):

- `recording.json`: the game, who played (`person`, later the model), training or testing (held
  out), notes, the picture's size and channels, the rate, the controls in bit order, the length,
  the time paused, missed pictures, the size, and whether it ended cleanly (a recording cut short by
  a crash stays readable);
- `frames.bin`: the pictures one after another, each compressed on its own (deflate), so any
  picture can be read directly;
- `ticks.bin`: one 40-byte record per picture: its time (microseconds), where its picture is,
  the keys (64 bits) and buttons **active at any moment since the previous picture** (so a click
  shorter than a tick still counts), the mouse movement since the previous picture, and whether
  it is the first after a pause;
- `events.jsonl`: the raw presses, releases and movements with their times, and the controls held
  whenever recording (re)starts.

**What a tick means.** The picture of tick k is taken at its time; the controls of tick k are
those used before it (since tick k-1). What the player did after seeing picture k is in tick k+1
and later (people react in a few hundred milliseconds); training chooses that offset.

**What is recorded, and what never is.** Input is read as the devices report it (Windows Raw
Input, X11 XInput2 raw events), like a game reads it. Only the profile's controls are written down
(other keys are never kept, so typing elsewhere is not recorded), and only while the game is in
front: switching away pauses the recording and those moments are left out. Virtual input (from
Tools or any other program) is never recorded as the person's: on Windows it has no device, on
X11 it comes from XTEST, and both are skipped. Recording sends no input to the game.

**Checking a recording** reads every picture and rebuilds every tick's controls from the raw
events; any disagreement, unreadable picture or time out of order is reported. The recorder ends
on Stop, the emergency stop keys (the recording is kept), the profile's storage limit, or the
game's window closing.

Size: the range at 128x72 in colour is about 1 MB a minute; a real game compresses less. Each
profile has a storage limit; recording stops at it and nothing is deleted without the owner
deleting it.

## 4. Generation 0 (imitation)

Training learns the person's control state for each tick from the frames before it
(behavioural cloning: cross-entropy for the mouse steps, a yes/no loss per key and button), with
Adam and the network's own backward pass. Held-out data is **whole recordings** (neighbouring
frames are nearly identical, so a random split would only measure memory). It reports per-control
accuracy on the held-out recordings, keeps the best checkpoint by held-out loss, and can be
stopped and resumed.

Then the model plays live (same tick loop, same safety layer), and its episodes are saved as
recordings marked as model play, so they can be watched and reused.

**Built to join Neuro.** Neuro already is a set of specialised networks exchanging information
over a learned bus; Gaming Nova is designed as more of the same, so that moving it into
`dependency/engine` is a move, not a rewrite:

- the same container: every weight in one flat array with named parameter slots, the same AdamW,
  and parameter groups that map onto the Engine's (`Focus`: the vision and control networks as
  modules, the goal and bus parts as interconnect), so Studio's candidate, acceptance and
  probation machinery can apply to it later;
- the same building blocks as the Engine's autodiff (`linear`, `silu`, `norm`, `sigmoid`,
  `softmax_rows`, and Neuro's minGRU `scan` as working memory across ticks), plus one new block,
  a convolution for pictures, written so it can be added to the Engine's tape;
- a bus-shaped interface from the start: the network's state is projected to a fixed-width
  vector each tick, and its goal input (the command now) has the same width, so Conversational
  Nova's Neuro bus can later write goals into it and read Gaming Nova's state back (the neural
  interconnect, with its own training);
- its file layout follows the Engine's checkpoint conventions, with the lineage (parent, method,
  generation) in every file from the first one, which generations and inheritance need.

The files: `players/<id>/player.json` (architecture, profile and control order, frame size,
mouse steps, training history, lineage) and the weights, documented here once built. Players live
in Tools' data until the Engine can open them; Studio and the Network read `.nnova2` models only
today.

**A command input from the start.** The network also gets the current command (none, follow me,
move to the objective, take cover, …) as an input, so Command Mode does not need a new
architecture later. Recordings can be tagged with the command being demonstrated.

## 5. Evaluation, generations and continuous learning (later phases)

**Rewards without a game API.** Per profile, rewards are built from things on the screen and from
the person: screen-region detectors (a colour or a template appears: a hit marker, a kill notice,
a death screen), how long an episode lasts, how much the view changes (stuck or moving), and the
owner's rating of an episode. Optional integrations per game (for Ravenfield, a mod reporting
kills, deaths and flags) can make them exact, but are never required. Several measures together,
never one number.

**Making candidates better (how weights change).** Selecting a winner changes nothing by itself,
so each method that makes a candidate changes weights on purpose:

1. more imitation on new demonstrations;
2. **corrections**: the person takes over when the model goes wrong; those ticks become training
   data in the situations the model actually reaches (DAgger), probably the strongest method here;
3. reward-weighted imitation of the model's own episodes (better episodes count more, worse ones
   are pushed away from), which needs no new machinery beyond the imitation trainer;
4. policy-gradient fine-tuning on the model's own episodes, once rewards are trustworthy;
5. variation between candidates: different data mixes, learning rates and seeds, and small
   weight noise; parents are combined by training a child on both parents' best episodes
   (distillation), not by averaging weights.

**Speed is the hard limit.** A real game runs in real time, one copy at a time on one PC: about
1 hour of play is 72,000 ticks. Reinforcement learning from pixels usually needs millions. So
demonstrations and corrections do most of the teaching, and generations compare a handful of
candidates over repeated, comparable episodes (the same scenarios for every candidate), not
thousands.

**Champions.** The accepted model is never replaced automatically by a weaker one: a candidate
becomes champion only by beating it on the evaluation episodes (never trained on) without losing
on a fixed benchmark of earlier skills, as Studio accepts checkpoints. Candidates that lose
can continue on probation for a limited number of rounds, as in Studio. Every champion stays
restorable, and every generation is recorded (parents, method, results, accepted or not).

**Continuous learning.** Play collects experience; training happens separately and periodically
on a candidate, mixed with earlier data (replay) so old skills are kept, and the candidate is
accepted only through the same checks.

## 6. Later: two machines, Conversational Nova

The tick loop is split from the start into the **rig** (capture, input, safety; where the game
runs) and the **brain** (the network), talking through messages with tick numbers and deadlines.
In the first version both run in Tools on one PC; later the brain can run on another computer
(over the Network, once it can listen beyond this computer with a key). A late action is dropped
and the controls released after a timeout; a lost connection releases everything.

Conversational Nova (a Bicameral or Neuro model through the Network) comes after Gaming Nova can
play: first in Command Mode (it chooses a command, which is Gaming Nova's command input), then
the neural interconnect (a learned vector in place of the command, with its own training).
Discord is part of that later work.

## 7. A test range before Ravenfield

To prove the whole chain (capture, input, recording, training, live play, scoring) with a known
answer before a real game, Tools gets a small original first-person **test range**: its own window,
mouse to turn, WASD to move, click to hit targets, read only through the screen and driven only
through virtual input like any game; it reports its score to Tools as an optional integration,
so evaluation can be checked against the truth. It also lets these features be tested
automatically on Linux in CI. Ravenfield follows with its own profile.

## 8. Order of work

| Phase | Delivers | Checked by |
|---|---|---|
| 2 Input and capture | Window list and choice, profiles with permitted controls, the safety layer, emergency stop, capture at a set size and rate | The test range receives exactly the inputs sent (automatic test on Linux); the owner's check on Windows with the range and Ravenfield |
| 3 Demonstrations | Recording the person's play in sync, the recordings list, inspection and replay, storage limits | Replay matches what was played; tick times and inputs agree with the raw events |
| 4 Generation 0 | The network, gradient checks, imitation training with held-out recordings, stop and resume, live play, saving and reloading | On the test range: a model trained on demonstrations scores clearly above random play |
| 5 Evaluation | Reward definitions per profile, evaluation episodes, reports, champion acceptance | The range's true score agrees with the screen-based rewards |
| 6 Generations | Candidates, methods 1-5, selection, history, champions | A later generation beats generation 0 on held-out scenarios of the range |
| 7 Continuous learning | Experience datasets, periodic candidates, replay | Earlier benchmarks hold while new skills improve |
| 8 Advanced | Controller, two machines, Conversational Nova, Command Mode, interconnect, Discord, transfer between games | — |

The first milestone the owner asked for (choose a window, permitted controls, capture, record,
save, inspect and replay, train a small model, let it play, evaluate, save and reload) is phases
2 to 4 plus the first part of 5.

## Built so far (phase 2, Tools 0.1.1)

Code: `apps/tools/src-tauri/src/live/` (controls, the safety layer, scripts, profiles, the system
layer for Windows and X11) and `src/range/` (the test range).

- **Real games, 2.1 Game profiles**: choose the game's window from the open ones (it is found
  again by title and program), permit keys (from a fixed table: nothing outside it can ever be
  pressed) and mouse buttons and movement, set the mouse sensitivity, the picture (size, colour
  or grey, pictures a second) and the safety limits. *What the network sees* shows the window's
  picture at that size and rate, and the rate actually reached.
- **2.2 Controls check**: test actions written as text (`hold w+shift 500; move 15 0; click
  left`) run through the safety layer, with a log of every action and every refusal; on the test
  range, the range's own report of what it received.
- **The safety layer** (`live/safety.rs`): only permitted controls; no combinations that reach the
  system (Alt+Tab, Alt+F4, Alt+Esc, Alt+Space, Alt+Enter, Ctrl+Esc, Ctrl with Alt); input only
  while the game is in front (when it leaves, everything is released and input pauses); the
  longest hold; actions a second (releases never limited); the largest mouse step; everything
  released when actions end, on the emergency stop, and when Tools closes.
- **The emergency stop**: Ctrl+Alt+End in any program (a low-level keyboard hook on Windows, a key
  grab on X11; the game never sees those keys), or the Stop button in the page bar.
- **The test range** (`yendory-tools --range`): Tools' own first-person game in its own process
  and window. It reads raw keys and mouse movement as games do and reports the truth on standard
  output (presses and mouse units received, position, angle, shots, hits, score).

Checked: unit tests (controls, combinations, the safety layer with a fake desktop, scripts,
profiles, picture shrinking, the range's game and drawing), and `tests/live_range.rs` on a real X
display (CI runs it under Xvfb): the range receives exactly the presses and mouse units sent, the
view turns by the expected angle, refused keys never arrive, the picture changes when the view
turns, another window coming to the front releases everything and pauses input, and the stop keys
pressed on the keyboard release everything. The window itself was driven by hand under Xvfb
(profiles, the preview at 20 pictures a second, the controls check, the stop keys).

**Not yet checked on Windows**: the Windows code is compiled and linted for Windows but has not
run there. To check it: start Tools, *Start the test range* (2.1), *Watch*, then 2.2 *Run on the
game*; the range's report should match the actions. Then make a profile for Ravenfield (windowed or
borderless) and run the same checks on it.

## Built so far (phase 3, Tools 0.1.3)

Code: `live/sys` (the listeners: Raw Input on Windows, XInput2 on X11), `live/record.rs` (the
recorder) and `live/recording.rs` (the format, reading, checking).

- **2.3 Record**: choose the game and whether the recording is for training or testing (held
  out), add notes, start; Tools brings the game to the front and records until Stop or
  Ctrl+Alt+End, showing the time, pictures, rate, size and the controls held. The profile's
  storage limit (2.1) is shown and enforced.
- **2.4 Recordings**: every recording of a game; open one to replay it (play at ¼× to 4×, step
  picture by picture, scrub), see the controls active at each picture and a timeline of every
  control and the mouse (click to jump), mark it for training or testing, edit its notes, check
  it, or delete it.

Checked: unit tests (the tick format, writing, reading, checking, catching a tampered tick, a
recording cut short, the recorder with a simulated person including a pause and a key outside
the profile, the storage limit), and on a real X display (CI under Xvfb): a person (played
through XTEST, which only tests may listen to) walking, running, turning and clicking on the
range is recorded with W active for the right number of ticks, exactly the mouse movement sent,
the click, pictures that change, and every tick matching its raw events; and Tools' own
listener does not hear XTEST input at all. The window was driven by hand under Xvfb: recording
the range at 20 pictures a second, ending with the stop keys, the list, replay, the timeline and
checking.

**Not yet checked on Windows** (compiled and linted only): record a minute on the range, open it
in 2.4 and play it back; the controls should follow what you did, and *Check the recording*
should pass.

## Built so far (phase 4, Tools 0.1.4)

Code: `live/policy.rs` (the network), `live/player.rs` (players and their files),
`live/train.rs` (imitation), `live/autoplay.rs` (live play), `live/evaluate.rs` (measuring on
the range), `live/checks.rs` (the Checks page), `live/nova.rs` (the window's commands).

**The network** (Small: about 0.3M weights at 128×72 in colour; Medium: about 1M): the last two
pictures stacked, three convolutions (8×8 stride 4, 4×4 stride 2, 3×3; SiLU), a linear encoder
(SiLU) to 128, Neuro's minGRU working memory (128) carried from tick to tick, and from the memory
and the encoding together: one yes/no per key and per button, and one class over mouse steps per
axis. The goal input (32 wide, zero until Command Mode) feeds the encoder. All weights in one
flat array with named slots and groups (vision, memory, heads: the Engine's modules; goal: its
interconnect), AdamW, gradient clipping at 1. The backward pass is checked against finite
differences for every kind of layer (a test, and a check in the window). Training a tick takes
about 3 ms on one core (the batch is spread over every core); a decision in play about 1 ms.

**Players** (`games/<profile>/players/<id>/`): `player.json` (the architecture with the controls
in order and the mouse steps, the lineage: generation, parent, method; every training run with
its settings, the recordings used and held out, the measurement before and the best),
`weights.bin` (the accepted weights), `last.bin` and `adam.bin` (to continue). The mouse steps are
chosen from the person's own movements (zero and five sizes each way at the 20/45/70/90/98%
quantiles of a tick's movement). A player fits the profile it was made from; if the picture or
the controls change, a new player is needed (the window says so).

**Training (2.5 Train)**: only recordings by a person, marked for training; each picture is
paired with the controls `reaction` ticks later (1 by default), in stretches of 8 ticks with the
memory carried through, 16 stretches a step. Every 100 steps it is measured on the held-out
(testing) recordings, the memory carried through each one as in play, beside the frequency
baseline (each control's training frequency, no picture): loss a tick (keys and buttons, mouse),
the mouse step right, and for each control how often it was used, how often the network would,
and F1. The accepted weights are the best held-out measurement (they are replaced only by a
better one); with nothing held out the latest are accepted and the window says so. Training can
stop and continue.

**Play (2.6)**: the player plays by itself at the profile's rate, through the safety layer,
after the person allows it each time; decisions are the most likely or drawn from the
probabilities. Its play can be recorded (as the player's, never as a person's demonstration).
**Measuring on the range**: the player, the untrained network (same architecture) and the
frequency baseline play the same episodes on layouts from seed 100,000 (never reached by playing),
episode by episode; the range counts shots and hits.

**Checks (2.7)**: each phase verified on its own, on the owner's computer. Phase 2 (automatic,
on the range): the picture, presses and mouse units arriving exactly, a click firing, nothing
left held, a key outside the profile refused, the picture changing, another window coming
forward releasing everything and pausing, the stop keys watched; and an interactive check of the
stop keys. Phase 3: Tools' own input never heard by the recorder (automatic), and the person's
latest recording read back, matching its raw input, at the right rate, with controls and mouse
movement in it. Phase 4: the gradients, saving and loading, a short training on a copy lowering
the loss on the person's recordings, and a network playing the range through the safety layer.

**Does it learn?** `tests/learn_range.rs` (run on purpose; about twelve minutes): a test-only
teacher plays the range from the range's truth, through the safety layer, and is recorded by the
ordinary recorder; a new network learns from those recordings alone and is measured on layouts it
never saw against the untrained network and the frequency baseline. The teacher exists only in
that test. Result (on this
sandbox's virtual display, CPU only): the teacher recorded 9 training layouts and 1 held-out
layout, 30 s each (5 minutes of play; it hits about 12 targets a minute). Training 1,200 steps took
163 s; the held-out loss fell from 7.40 (untrained) to 0.256 at step 600, against the frequency
baseline's 0.926, and then rose again (over-fitting five minutes of play; the accepted weights
are the best held-out measurement, step 600). On 4 unseen layouts of 20 s each:

| Who played | Hits a minute | Shots a minute | Hits in each episode |
|---|---|---|---|
| The trained network | 4.5 | 8.2 | 1, 1, 1, 3 |
| The untrained network | 0.8 | 18.8 | 0, 1, 0, 0 |
| The frequency baseline | 1.5 | 21.0 | 0, 2, 0, 0 |

So the network learned from pixels to aim before firing (fewer shots, more hits), on layouts it
never saw; but it is far from the teacher (about 12 a minute), and 4 short episodes are few:
the numbers are small and noisy. Much more play, and later practice (phases 5 to 7), are needed
for good play.

**Not yet checked on Windows** (compiled and linted only): run the checks on 2.7, then record a
few minutes on the range (some marked for testing), train a player on 2.5 and measure it on 2.6.

Next: phase 5, evaluation beyond the range (rewards from the screen, evaluation episodes,
champions), then generations.
