# Nova Neuro: feasibility review and implementation plan

Status: **Neuro-Core implemented (Stages 1-3, review point 1).** Nova Bicameral stays the default
and is not changed by any of this. Nova Messaging integration is out of scope (handled elsewhere).
What exists now is listed in section 12; the rest of this document is the approved plan.

## 1. What the engine already gives us

- **One network definition, two devices.** The bicameral forward pass is written once against
  a small set of operations (`model::tape::Graph`: embed, linear, add, mul, silu, sigmoid,
  norm, rope, causal attention, fused output + cross-entropy). The CPU tape and the Vulkan
  graph (`gpu::GpuGraph`) both implement it, with hand-written backward passes and tests that
  check gradients by finite differences and CPU against GPU. Nova Neuro can be written the same
  way: every *network* is defined once; only new *operations* need CPU and GPU versions.
- **Per-module training masks.** Parameters carry a group (`Focus`), and the optimizer can train
  one group while freezing the others. Bicameral education stages and Foundation candidates
  already use this; Nova Neuro's training phases A-E map onto the same mechanism.
- **Ablations and measurements.** `Ablation` cuts a pathway at evaluation time without touching
  the checkpoint, and "Dense reliance" (held-out loss rise with Dense cut) is exactly the kind
  of contribution measure the proposal asks for. The same idea extends to every Neuro module.
- **Everything around the model.** Reading (corpus pre-training), lessons, courses, the
  Planner, Study Mode, Practice, Reflection, Review Training, the reward ledger, protected
  exams, the comparison harnesses (`examples/guard_compare.rs`, `reward_compare.rs`) and the
  on-device practice generator (`learn::generate`, including "who is where" stories with
  random names and places) all work on token sequences, so they can serve a second architecture.

What does **not** exist yet: operations the new networks need (row softmax, per-row gating,
a linear recurrence scan, non-causal attention among a few nodes, chunk pooling, an auxiliary
reconstruction loss), a second model type in the learning pipeline (the code uses the concrete
`Bicameral` type in about 55 files), and a checkpoint format for a modular model.

## 2. Verdict

Feasible, as a large multi-stage project. The proposal's principles (genuinely neural, learned
coordination, no keyword rules, protected evaluation, bicameral preserved, every module judged
by ablation) fit the engine well. Seven adjustments are recommended; they keep the intent and
make it trainable, causal and fast enough on an RTX 2070 Super.

## 3. Recommended adjustments

1. **Working memory: start with a minGRU, not a classic GRU.** A classic GRU must process
   token after token (256 sequential steps per sequence, forward and backward), which is slow on
   any GPU. A minGRU keeps the GRU's update gate but computes its gates from the current input
   only, so the whole sequence is one parallel *scan* (`h_t = (1 - z_t) * h_{t-1} + z_t * c_t`).
   It is still a gated recurrent working memory, trains in parallel, keeps an O(1) state during
   generation, and is a short step from a State Space Model, which makes the later GRU / LSTM /
   SSM comparison natural. A classic GRU can be added later as a CPU comparison option.
2. **Relations must be causal: the graph is built from text already read.** A language model may
   never use future tokens. A graph built from a whole training sequence and fed back to every
   token would leak the answer during training and fail at generation. So the relational network
   works on **chunks** (e.g. 16 tokens): after each chunk it updates a small set of learned
   **entity slots** (e.g. 16) and the relations between them; tokens in the next chunk read the
   graph as it was. This is also the cheap design the proposal asks for (no graph rebuild per
   token). The slots are distributed vectors, not text triples: "Daniel - current location -
   hallway" is something we *test for* with probes and unseen-name evaluations, never something
   written into the network.
3. **Associative memory = learned patterns read by attention.** A modern Hopfield retrieval step
   (softmax(beta * query . patterns) . values) is mathematically attention over learned pattern
   vectors. Patterns are parameters learned in training, separate from the working memory
   (minGRU state, cleared per conversation) and from Study Mode's external text memory (which
   stays exactly as it is, optional, and shown to the model only as marked memory lines).
4. **Give the autoencoder a concrete job.** A reconstruction network on its own does not
   consolidate anything. Proposed role: it compresses bus representations into a short code
   (with a reconstruction loss so the code stays informative), and that code is what the
   associative memory stores and matches. Consolidation itself stays measured the way Yendory
   already measures it: Study Mode material learned, then tested **without** memory, retention
   across checkpoints, and the reward ledger's neural-only track. If ablation shows the
   autoencoder adds nothing, its budget goes to the other modules.
5. **Build a smaller core first, then add modules that earn their place.** Milestone 1,
   **Neuro-Core**: Transformer + minGRU working memory + associative memory + Executive
   Controller + communication bus. Compare it with Bicameral at equal size before adding the
   relational network, the experts and consolidation. Each later module must show an effect in
   ablation; otherwise the experiment says so.
6. **Computation rounds cost something.** Start with a fixed 2 rounds (like Bicameral's thinking
   rounds). Adaptive rounds (up to 4) come later with a ponder cost added to the loss, so the
   controller cannot buy accuracy by always using every round.
7. **Stay one model type behind one interface.** A `Brain` type (Bicameral or Neuro) gives the
   learning pipeline and Studio one interface (tokenizer, loss and gradients, optimizer step,
   generation with a cache, save/load, measurements). Bicameral code paths stay as they are;
   tests check its weights and replies are bit-identical before and after.

## 4. Nova Neuro 15M: a concrete allocation

Vocabulary 7,500 learned words + 128 characters; model width 352, bus width 256; context 256.
Counts below are hand estimates to be replaced by the engine's exact parameter calculator.

| Module | Structure (proposed) | Parameters | Share |
|---|---|---|---|
| Transformer | 4 blocks, width 352, 8 heads, SwiGLU 704; 2 lower blocks, 2 upper blocks run once per round | ~4.96M | 33% |
| Token embeddings | 7,628 x 352, **tied** with the output layer (as Bicameral already does) | ~2.69M | 18% |
| Relational network (GNN) | 16 entity slots x 256, chunk writes by attention, slot update by minGRU, 2 rounds of relation-aware attention between slots, token read-back | ~1.4-1.65M | 10% |
| Working memory | 2-layer minGRU (640 wide), output projection to the bus | ~1.43M | 9.5% |
| Experts (MoE) | 4 SwiGLU experts on the bus (hidden 448), learned router | ~1.38M | 9% |
| Associative memory | ~1,280 learned patterns (key 256, value 256), query and output projections, learned sharpness | ~0.79M | 5% |
| Consolidation autoencoder | 352 -> 512 -> 128 code -> 512 -> 352 | ~0.5-0.8M | 4-5% |
| Executive Controller | MLP over the bus: module gates, write gates, (later) halting | ~0.4-0.65M | 3-4% |
| Bus and shared parts | read/write projections Transformer <-> bus, norms, output bias | ~0.4-0.5M | 3% |
| **Total** | | **~14.4-15.0M** | |

The embedding share depends on the vocabulary: with Nova-2's 256-word layout the table is about
0.14M, so presets are computed from the chosen vocabulary, not fixed percentages. The engine
reports the exact count of every tensor; if a configuration exceeds the target, creation is
refused with the real numbers rather than silently growing.

**Smaller presets** (Small ~4M, 8M) keep the order of priority from the proposal: transformer and
tied embeddings first, then working memory, then a small relational network (8 slots) and a light
controller; associative memory, experts and the autoencoder shrink first or are left out.
**Larger presets** add transformer depth, more slots and relation rounds, more patterns and
larger experts. Every preset is created, trained briefly and checked by tests before it is
offered.

**Hardware.** A Bicameral 15M already trains on the RTX 2070 Super (about 9,750 tokens/s in the
screenshot from a reading session). Nova Neuro 15M runs the upper transformer blocks twice (as
Bicameral does), all four experts while routing is soft, and the relational network once per
chunk, so expect roughly 0.5-0.7x Bicameral's speed and 1.5-2x its activation memory: an
estimated 3-4 GB at batch 16 x 256 tokens, inside 8 GB. Real numbers come from the existing
benchmark page once the GPU kernels exist.

## 5. How information flows (one forward pass)

1. Tokens -> embeddings (width 352).
2. Lower transformer blocks; in parallel the minGRU reads the embeddings (working memory).
3. For each computation round (2 to start):
   - the bus collects the transformer state and the working-memory state (projections to 256);
   - the **Executive Controller** reads the bus and gives each module a weight (soft at first);
   - the associative memory, the experts and the relational network (graph state from earlier
     chunks) read the bus and write their outputs back, scaled by their weights and by learned
     write gates;
   - the combined bus is injected into the transformer's residual stream (gated, like
     Bicameral's interconnect), the upper blocks run, and the result feeds the next round, so
     information moves both ways (transformer -> modules -> transformer, working memory ->
     associative memory -> controller -> transformer).
4. After the last round: final norm -> tied output layer -> next-token probabilities.
5. At chunk boundaries the relational network updates its slots from the chunk just read.

Every path is differentiable, so all interfaces receive gradients. "Does it matter?" is measured,
not assumed: per-module ablation (held-out loss and answers with the module cut), the size of
each module's write into the bus, and the controller's weights per token.

**Routing safeguards** (training health, not answers): a load-balancing term so no expert or
module is starved, module dropout during training, an entropy floor on the controller early on,
and logged activation statistics. Sparse routing (at most 2 supplementary modules per round) is
a later, configurable switch, tested against soft routing.

## 6. Training plan

The proposal's phases map onto per-module training masks (the bicameral education machinery):

- **A, foundation**: embeddings, transformer, working memory, bus. Reading as today.
- **B, relations**: the relational network joins; generated "who is where / who has what"
  stories with random names and places (`learn::generate`), judged on unseen names and
  sequences.
- **C, associative memory** joins; it must lower held-out loss, not just be active.
- **D, routing and experts** join; specialization is observed, never assigned.
- **E, joint training** of everything with a balanced objective. This phase should get most of
  the tokens; A-D only prepare modules so they do not start as noise.
- **F**: lessons, Practice, Reflection, Review Training, Study Mode and the reward ledger work
  unchanged on Nova Neuro.

The objective stays next-token prediction. Optional auxiliary losses where labels are reliable
(e.g. which entity is where, from generated stories; reconstruction for the autoencoder; load
balance; ponder cost), each with a weight you can set to zero. Never trained: protected exam
questions and answer keys.

## 7. The custom brain builder (create-model screen)

Choose **Nova Bicameral** (unchanged) or **Nova Neuro (experimental)**, then build the brain from
menus, with the engine computing the exact parameter count, memory and hardware support live:

| Menu | Choices (first version) |
|---|---|
| Size preset | Small, 8M, 15M, or Custom |
| Language core (Transformer) | width, blocks, heads, feed-forward width, context |
| Vocabulary | Nova-2 layout or a word limit (embeddings tied) |
| Working memory | minGRU (layers, width), or none; later: classic GRU, LSTM, State Space Model |
| Relational network | entity slots, relation rounds, chunk length, or none |
| Associative memory | patterns, code width, or none |
| Experts | none, 2, 4 or 8 experts, expert width, soft or top-k routing |
| Consolidation | autoencoder on/off, code width |
| Controller and rounds | rounds (2 to start), adaptive rounds (later), routing safeguards |
| Communication bus | width |
| Future modules | CNN, Liquid, Spiking, Diffusion shown as "not available yet" |

Each choice shows its parameters and share; impossible combinations are explained and refused;
modules not yet supported on the GPU are marked. The chosen design is saved with the model.

## 8. Visualizer and thinking tree

Bicameral keeps its brain plate. Nova Neuro gets its own view drawn from measured values of a
real forward pass: each module with its parameter count, activity (size of its write into the
bus) and how often the controller uses it; controller weights per round; working-memory change
per token; associative patterns retrieved (strength); relational slots (which tokens wrote to
which slot, slot-to-slot attention); expert usage; rounds. Selecting a module shows its
contribution to a chosen reply (the reply's probability with and without it). The thinking tree
gains these per-token values next to the existing token probabilities and alternatives; nothing
is written as "thoughts" the model did not compute.

## 9. Checkpoints

A new, separately versioned format (`.nneuro`): magic and version, the full architecture design
(JSON: every module, every dimension, rounds, routing settings), tokenizer words, every tensor by
name and shape, optional optimizer state in a separate section. `.nnova2` is untouched, and no
conversion from Bicameral is offered (a Bicameral checkpoint would need retraining anyway).
Study Mode memory, the reward ledger and practice records stay beside the model as today.

## 10. Evaluation

Built on the existing comparison harnesses: Nova Bicameral 15M against Nova Neuro 15M (and
Neuro-Core), same data, seeds and settings; parameter-matched and compute-matched (estimated
training FLOPs per token from the design); held-out loss per corpus, lesson and course results,
protected exams, generated relational and state-change tests with unseen names and places,
retention across checkpoints, with/without Study Mode memory; training time, tokens/s, memory,
generation latency; ablations of every module at evaluation time. Internal activity or reward
totals are reported, never used as evidence on their own. Expect an honest outcome either way: at
15M parameters a simpler model may well win, and that is a valid result.

## 11. Stages and review points

| Stage | Contents | Ends with |
|---|---|---|
| 1 | Design format, exact parameter / compute / memory calculator, presets, `Brain` interface with Bicameral unchanged (bit-identical tests), `.nneuro` format | calculator tested; Bicameral unaffected |
| 2 | New operations on the CPU (softmax, row gating, scan, node attention, chunk pooling, reconstruction loss) with finite-difference gradient tests; bus and controller | every operation's gradients verified |
| 3 | **Neuro-Core** (transformer, minGRU, associative memory, controller, bus) end to end on the CPU: training, generation with a cache, save/load; Reading, lessons, courses, Study Mode and rewards working with it; builder screen with the core menus | **review point 1**: a small Neuro-Core trains and talks; quick comparison with Bicameral |
| 4 | Relational network (causal chunk slots), experts (soft, then top-k with load balance), consolidation autoencoder; full builder menus | every module tested and ablatable |
| 5 | Vulkan kernels for the new operations, CPU-GPU agreement tests, memory and speed of the presets on the GPU | presets measured |
| 6 | Neuro visualizer, thinking-tree trace, training phases A-E in Studio | |
| 7 | Comparisons and ablations as in section 10, written up | **review point 2**: results and what to do next |

Messaging integration (the proposal's Stage 8) is left out as requested.

## 12. Implemented so far (Neuro-Core)

- **Engine** (`model/neuro.rs`): `NeuroSpec` (design, versioned) stored on the model; the
  transformer core, embeddings (tied with the output layer), optimizer and parameter groups are
  shared with Bicameral. Modules: minGRU working memory (layers x width), associative memory
  (patterns x key width, softmax retrieval of learned patterns), Executive Controller (one sigmoid
  gate per module, per token and round), communication bus (transformer -> bus -> modules ->
  gated write-back into the residual stream; working memory joins the bus). Groups: language core
  = `Left`, modules = `Right`, bus and controller = `Interconnect`, so Foundation candidates and
  the education stages work unchanged.
- **New operations** (`model/tape.rs`): row softmax, per-row gating by a gate column, the minGRU
  scan. CPU forward and backward; finite-difference gradient tests through every module.
- **Generation cache** with the working memory's state per context; equal to the full pass under
  every ablation and across a full-window rebuild. A test checks no later token changes an earlier
  position's output (no leakage from the future).
- **Ablations** (evaluation only, checkpoints untouched): `no_memory`, `no_associative`,
  `no_controller` (all gates open), `one_round`, `no_afferent` (no write-back into the
  transformer), `no_efferent` (no transformer state on the bus).
- **Checkpoints**: `.nneuro` (magic `NNEURO1`, JSON header with architecture, format, config and
  design, then the same body as `.nnova2`). `.nnova2` is byte-identical for Bicameral.
- **Presets** at the Bicameral sizes (1.2M, 4M, 8M, 15M), each within 0.1% of its twin's
  parameter count; the exact calculator is `Bicameral::tensors_for`.
- **Trace**: controller gates, module output sizes, recalled patterns, bus and write-back sizes,
  working-memory change and update gate, for measurements and the thinking tree.
- **Studio**: architecture selector and brain builder (presets, every module's settings, the
  Engine's exact parameter count per part, memory estimate, hardware support; later-stage and
  future networks listed), `.nneuro` import/export, Neuro activity on the Brain page.
- **Not yet**: GPU kernels (Neuro trains on the CPU), the relational network, experts,
  consolidation, auxiliary objectives (planned with the relational network, where story labels
  feed entity tracking), adaptive rounds, the full Neuro visualizer.
