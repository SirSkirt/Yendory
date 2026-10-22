# Growing networks and using system RAM: design

Status: Stage 1 (inspection), Stage 2 (manual growth) and Stage 3 (the training memory
manager) complete; later stages are recorded in the sections below as they land.

## 1. What exists today

### Parameters and checkpoints

- A model is one flat array of parameters cut into named *slots* (`model::spec`), each a
  `rows x cols` matrix with a group (transformer, Dense/modules, interconnect/communication)
  and an initialisation. AdamW keeps two moments per parameter and a global step.
- A checkpoint stores the architecture as JSON (`Config`: width `d`, heads, feed-forward width
  `ff`, `blocks`, lower blocks `split`, rounds `loops`, Dense widths `r1`/`r2`, vocabulary,
  context; for Nova Neuro also the `NeuroSpec`: bus, working memory, associative memory,
  controller), then the step, the weights, both Adam moments and the learned words. Loading
  rebuilds the slots from those dimensions. **Any grown shape is therefore already a valid
  checkpoint of the same format**; no format change is needed to store expanded dimensions.
- The Engine accepts up to 1e9 parameters. Studio offers presets of 1.2M to 15M (Neuro designs
  capped at 16M by the builder).

### Where training memory goes (GPU)

Weights, gradients and both Adam moments: 4 x parameters x 4 bytes. Activations of a batch:
the per-token activation count x rows x tokens x 4 bytes (plus working-buffer rounding).

| Preset | Parameters | Weights + gradients + Adam | Activations per token | Batch 16 x 128 | Batch 16 x 256 |
|---|---|---|---|---|---|
| Bicameral 1.2M | 1.18M | 19 MB | 22,016 floats | 180 MB | 361 MB |
| Bicameral 4M | 3.94M | 63 MB | 47,552 | 390 MB | 779 MB |
| Bicameral 8M | 7.95M | 127 MB | 69,248 | 567 MB | 1,135 MB |
| Bicameral 15M | 14.62M | 234 MB | 104,320 | 855 MB | 1,709 MB |
| Neuro-Core 15M | 14.62M | 234 MB | 129,160 | 1,058 MB | 2,116 MB |

At these sizes **activations dominate** (7-9x the model state at 16 x 256). Growing a model
raises both; activations grow with width and depth, the model state with the parameter count.

### The GPU backend

wgpu 30 (DirectX 12 on Windows, Vulkan on Linux/Android). A training step records every kernel
of the forward pass, the loss, the backward pass and AdamW, then submits them at once; every
kernel binds the whole weight and gradient buffers. Working buffers come from a pool (size
classes, released when unused). Hybrid training already exists: the GPU computes part of a
batch's gradient, the CPU the rest, and **the CPU applies AdamW**, then the new weights are sent
to the GPU.

## 2. Growing a trained network without losing what it learned

The approach is *function-preserving growth* (Net2Net / network morphism): new units are added
so that, at the moment of growth, the network computes exactly (or within float rounding) what
it computed before. Knowledge is kept because no existing weight is changed except where
noted; the new capacity is then trained by ordinary reading and lessons.

The rule that makes it work: **a new unit's outgoing weights start at zero, its incoming
weights start small and random.** The unit adds nothing at first, but gradients reach its
outgoing weights at once (its activity is not zero), so training can start using it.

| Component | Growth | Preserved? | How |
|---|---|---|---|
| Transformer feed-forward width (`ff`) | more hidden units | exact | new `w_gate`/`w_up` rows random; new `w_down` columns zero |
| Transformer depth | a new lower or upper block | exact | new block random except its attention output `wo` and `w_down` zero: the block is an identity on the residual stream (it runs in every round if upper) |
| Transformer width (`d`) | whole attention heads (32 dims each) | within float rounding | new residual dimensions start at zero (embedding columns, `wo`/`w_down`/write-back rows zero); readers of the residual stream get random new input columns; RMS-norm gains are rescaled by sqrt(d/d') because the mean square is taken over more (zero) dimensions; the norm's tiny epsilon makes this near-exact rather than exact |
| Bicameral Dense (`r1`, `r2`) | wider Dense layers | exact | new input rows random; new rows' outgoing columns (`c.a1`, `r.w2`, `c.a2`) zero |
| Bicameral interconnect | follows `d`, `r1`, `r2` | exact | as above |
| Neuro working memory width | more memory units per layer | exact | new update-gate rows random (closed: bias -1); new candidate rows zero, so new memory units stay at zero until trained; the next layer's norm gain is rescaled |
| Neuro controller width | more hidden units | exact | new rows random; new output columns zero |
| Neuro bus width | wider bus | within float rounding | new bus dimensions start at zero (`n.read`, `wm_out` new rows zero); bus norms rescaled; module outputs into new dimensions are ignored by the write-back (new columns zero) |
| Neuro associative key width | wider keys | exact | new query rows zero, new key columns random |
| Neuro associative patterns | more patterns | **not offered** | the softmax over patterns renormalises: a new pattern takes weight from the old ones, so outputs change; no exact construction exists without a bias |
| Neuro working-memory layers | more layers | **not offered** | a minGRU layer is not an identity; a new layer would change outputs |
| Thinking rounds (`loops`) | more rounds | **not offered** | another round adds another write-back and upper pass; not an identity |
| Vocabulary | more word slots | exact (exists) | `with_word_capacity` |

Nothing above changes the architecture's kind: a Bicameral model stays Bicameral, a Neuro model
keeps its networks. What cannot be grown safely is reported, never approximated by resetting.

### Optimizer state

Each existing parameter keeps its two Adam moments; new parameters start with zero moments.
The global step is kept (bias correction is already ~1 at trained step counts), so the first
updates of new parameters have the usual Adam step size.

### Safety around an expansion

1. Save a recovery checkpoint of the model as it is (never overwritten).
2. Build the expanded model in memory.
3. Verify: on probe text, the largest change of any next-token logit and the change in loss
   between the original and the expanded model must be within tolerance (exact growths: float
   rounding; width growths: the epsilon effect). On a failure nothing is saved.
4. Save the expanded model; keep the recovery checkpoint and the growth record (dimensions and
   parameter counts before and after) so the expansion can be rolled back later.

### Stage 2 as built

- Engine: `Bicameral::grow(&Growth, seed)`, `growth_options()` (the steps offered for the
  model's architecture, with exact parameter counts, and what is not offered and why),
  `compare_outputs` and the probe texts (`model/grow.rs`).
- Studio (model card, "Network growth"): the growth setting (Off, Manual, Automatic; also
  chosen when creating a model), original and current size, the offered steps with the
  parameters and training memory they add, an Expand button per step, the growth history and
  "Roll back the latest growth". Growth works on any model in the library, including models
  created and trained before this feature.
- Expanding: the grown network is checked against the original on the probe texts; only if
  its outputs match is a recovery checkpoint written to the model's `growth/` folder (and read
  back to check it), and then the grown model replaces `model.nnova2`. The original file is
  never overwritten before the recovery copy is verified.
- Rolling back restores the recovery checkpoint and keeps the grown model (with any training
  since) as another checkpoint in `growth/`.

Measured (`tests/growth.rs`, both architectures, trained weights): the exact growths change no
output at all (largest logit change 0); width growths change outputs by 1e-5 to 3e-5; the real
step-15,000 checkpoint grown from 1.18M to 2.04M parameters gives the same replies; new units'
outgoing weights get gradient at once; grown models save, load, train (loss falls) and match
the CPU on the GPU.

## 3. Using system RAM: what fits this backend

What is realistic with wgpu, in the order the memory manager should use it:

1. **Keep everything on the GPU** when it fits (fastest).
2. **Micro-batches with gradient accumulation** (exact): split a batch into smaller parts on the
   GPU, add their gradients, take one AdamW step. Activation memory falls with the part size,
   which is where most of the memory is. Costs only some GPU efficiency.
3. **Optimizer state in system RAM**: keep both Adam moments (half the model state) in RAM; the
   GPU computes the gradient, the CPU applies AdamW, the new weights go back to the GPU. Hybrid
   training already works this way. Transfer per step: the gradient down and the weights up
   (2 x parameters x 4 bytes, e.g. 117 MB for 15M parameters).
4. **Weights and gradients in system RAM, streamed per layer**: not possible with the current
   design, where every kernel binds the whole weight and gradient buffers. It needs per-layer
   buffers and split submissions: a larger change, only worthwhile for models far beyond 8 GB of
   model state. Not in the first implementation; reported as unavailable.
5. **Activation checkpointing** (recompute instead of store): needs recomputation inside the
   recorded graph; possible later, not first.

What the memory manager can measure:

- Its own allocations exactly (model buffers and the working-buffer pool).
- wgpu's allocator report on DirectX 12 and Vulkan: bytes allocated and reserved on the device.
- wgpu's **memory budget threshold**: buffer creation fails with an out-of-memory error at a
  chosen percentage of the budget the operating system reports for the GPU (DirectX 12; Vulkan
  with the memory-budget extension). That is the GPU safety margin: a step that would not fit
  fails cleanly instead of paging into shared memory or losing the device.
- The GPU's total memory is not exposed by wgpu; Studio asks for it (with the card's name shown)
  or uses the budget threshold alone.
- System RAM (total and available) from the operating system, to honour the RAM limits.

What is avoided: relying on the driver silently placing buffers in "shared GPU memory" (Windows
does this when VRAM runs out). It works but is slow and uncontrolled; the budget threshold stops
allocation before that point so the manager can choose a plan instead.

### Stage 3 as built

- **Settings** (`learn::memory::MemorySettings`, Studio "Training memory", saved with the other
  settings): memory mode (GPU only, Automatic offloading, Manual), the GPU's memory (wgpu cannot
  read it; empty means unknown), the GPU safety margin (default 10%), reduce memory
  automatically (micro-batches), the most system RAM offloading may use (default half the RAM),
  RAM always kept free (default 4 GB), warnings about slow offloading, and for Manual the rows
  per micro-batch and the optimizer state in RAM.
- **Budget**: training may use the GPU's memory less the margin. The working-buffer pool counts
  every buffer it creates; a buffer that would pass the budget is refused (after releasing free
  buffers the step does not use), the step is marked and refused *before anything is
  submitted*, and nothing changes. Independently, the margin is given to wgpu as the memory
  budget threshold when the GPU is opened (DirectX 12; Vulkan with the budget extension), so the
  driver refuses allocations past it even when the GPU's size was not given.
- **Micro-batches** (`GpuModel::fit_step`): the batch's gradient is computed in parts, each
  divided by the whole batch's weight and added into the gradient buffer (the first part
  clears it; dropout masks follow each row's place in the batch); AdamW runs once, in its own
  submission, only after every part succeeded.
- **Optimizer state in system RAM** (`GpuModel::upload_with`, `move_optimizer_to_ram`): the GPU
  holds weights, gradients and parameter flags; the gradient is read back in 16 MB parts, the
  CPU applies AdamW to the model's own moments (the same code as CPU training) and the new
  weights are written back. GPU model state falls from 5 to 3 model-sized buffers; each step
  moves 2 x parameters x 4 bytes.
- **Planning** (`learn::memory::Fitter`): each step's plan comes from the model's activation
  count per token, the batch's rows, its longest row and the vocabulary (the estimate is
  1.3x the measured use). In order: whole batches on the GPU; the largest micro-batch that fits;
  the optimizer state in RAM (Automatic only, and only within the RAM limit and reserve). When
  a step does not fit anyway (a GPU out-of-memory error or the budget), nothing was applied:
  the micro-batch is halved (remembered as rows x tokens, so longer batches get fewer rows), then
  the optimizer state moves to RAM, then training continues on the CPU with a notice that says
  why. Manual settings are used as set and never changed behind your back.
- **Report** (`MemoryReport`, every step): the strategy in words, GPU memory counted by Yendory
  and by the driver's allocator, the budget, RAM used for offloading, bytes and time of the
  GPU <-> RAM transfers, steps, tokens and time (tokens per second), steps that had to be
  refitted and why. Studio shows it under the Reading, Classroom and Planner progress cards while
  training runs and in the Training memory panel, with warnings when offloading takes more than
  a fifth of the time, when micro-batches are down to one or two rows, or when free RAM is below
  the reserve.
- Hybrid training keeps the optimizer state in RAM from the start (the CPU applies AdamW
  there anyway), freeing the GPU's copy.
- Not done (section 3, items 4-5): weights and gradients streamed per layer, and activation
  recomputation. Models whose weights and gradients alone exceed the GPU are reported as not
  fitting.

## 4. Stages

1. Inspection and this design (done).
2. Manual expansion with verification, recovery checkpoint and rollback, for new and existing
   models of both architectures (done).
3. The memory manager: budgets, micro-batching, optimizer state in RAM, settings and statistics
   (done).
4. Growth and memory together: expansion estimates and refuses plans beyond the limits; expanded
   models train under the memory plan.
5. Automatic growth from training evidence (plateaus, held-out results), within the limits, at
   safe boundaries.
