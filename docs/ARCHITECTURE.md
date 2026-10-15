# Nova-2 "Bicameral" architecture

A proposal for the next Neural Nova brain. It keeps the three-piece design (left hemisphere,
right hemisphere, interconnect), stays under the current 1,200,144-parameter budget, and
needs roughly half the compute per generated character/word of the current brain.

| | Nova v1 (v0.15) | Nova-2 |
|---|---|---|
| Total parameters | 1,200,144 | **1,182,144** (98.5%) |
| Left hemisphere (Transformer) | 347,024 incl. 234,128 in expansion blocks | 755,200 |
| Right hemisphere (dense) | 853,120 | 336,320 |
| Interconnect | ~74,000 (bridge, queries, responses) inside the Transformer | 90,624, as its own piece |
| Attention | 1 head, 1 layer, query from last position only | 4 heads x 4 layers, every position |
| Width (D) | 32 | 128 |
| Hemisphere exchanges per prediction | 2, through a single 32-number vector | 1 + 2 "thinking" rounds, at every position |
| Learning signal per training pass | 1 target character | every position in the sequence |
| Compute per prediction | ~2.9M multiply-adds | ~1.6M (+0.4M attention at 256 tokens) |
| Tokenizer, lesson formats, ASCII rules | as today | unchanged (exact port) |

## What the step-970 checkpoint showed

I parsed `neural-nova-step-970.nhyb` and reproduced each region's initial random weights from
the seeds in the code, so how far each part has moved from initialization can be measured
directly. (relative drift = |W - W_init| / |W_init|)

| Region | Params | Drift from init | Reading |
|---|---|---|---|
| Attention q / k | 2,048 | 1.6x / 2.0x | Where most learning happened |
| Feed-forward, output head | ~8,200 | 0.8-1.1x | Learning |
| Token / position embeddings | 12,288 | 0.52 / 0.35 | Learning |
| **Dense hemisphere (all 3 layers)** | **853,120** | **0.10-0.13, corr 0.99 with init** | **Still almost its random initialization** |
| Interconnect queries (feedback/context) | 65,536 | 0.25-0.30 | Moderate |
| Interconnect / memory / evidence / reasoning outputs (started at 0) | ~7,200 | rms 0.006-0.015 | Barely opened |
| Word embeddings | 8,192 | 0.15 | Only 143 of 256 slots used |

Other facts from the file: Transformer Adam step 970, dense Adam step 444, 143 learned word
units. The file stores 112,896 Transformer parameters, i.e. it predates the four expansion
blocks (those 234,128 parameters are appended at zero when it loads).

Interpretation: **71% of the brain (the dense hemisphere) has barely changed**, and the paths
that let the hemispheres influence each other are still nearly closed. The model's learning
went into roughly 30K attention/embedding weights. This is not proof the dense pathway is
useless (the run is short), but it says the current wiring makes it hard for the dense side to
earn a role. Nova-2 is designed around fixing that.

## Diagnosis of v1, piece by piece

**Left hemisphere.** One attention layer reads raw (token + position) embeddings, and only
the last position forms a query. Tokens never interact before being read, so it can match
"what's here" but not build "who has what" relations. The reasoning loop and the four
expansion blocks add depth, but only to one 32-number vector after reading is over.

**Right hemisphere.** A 1024 -> 512 -> 512 -> 128 network over 64 characters x 16 fixed
features. Its first layer alone is 524,288 weights (44% of the brain), mostly devoted to
learning one separate weight per (character position, feature, neuron). It answers in 128
"next ASCII character" logits, so everything it knows must be squeezed into character
predictions before the left side can use it.

**Interconnect.** Two 1024x32 query matrices (65,536 weights) map the left state back into
the dense input space, the dense network is re-run in full, and its 128 outputs are projected
back to 32 numbers. Three full dense passes per prediction make this the most expensive part,
and the responses still sit near zero in the trained checkpoint.

**Training.** Each step computes the loss on one next character for one context. A sequence
of 200 characters costs 200 full passes for 200 learning signals, when one causal pass could
give all 200.

## Nova-2 design

```
                 tokens (same tokenizer, ids 0-383)                  last 64 chars x 16 features
                         |                                                     |
                  [tied embedding 384x128]                           RIGHT HEMISPHERE
                         |                                             trunk: 1024 -> 256 (SiLU)  ---- z1
             LEFT: blocks 0-1 (lower stage)                                     |
                         |                                                     |
                         |<--- INTERCONNECT: afferent 1 (gate x A1 z1) --------|
                         |                                                     |
                         |---- INTERCONNECT: efferent (left state -> right) -->| upper: 256 + 128 -> 192 (SiLU) -- z2
                         |                                                     |
       +--------------> |<--- INTERCONNECT: afferent 2 (gate x A2 z2) --------|
       |                 |                                                     |
       |      LEFT: blocks 2-3 (upper stage, shared weights)                   |
       |                 |---- efferent again (left state -> right) ---------->| z2 recomputed
       +---- loop 2x ----+                                                     |
                         |
                 [final RMSNorm] -> tied output head -> next token (char or word)
```

Each block is a modern pre-norm Transformer block: RMSNorm, 4-head causal attention with
rotary positions (RoPE), residual, RMSNorm, SwiGLU feed-forward (128 -> 288 -> 128), residual.

### Left hemisphere (755,200 params): language and long-range reading
- **4 layers x 4 heads, width 128.** Tokens now interact before being read. Two layers is
  the known minimum for "find the earlier mention and copy what followed" (induction heads),
  which is what answering "Who has more apples?" from facts in the prompt needs.
- **RoPE instead of 256 learned position vectors.** No position parameters at all, and
  it generalizes to relative distances; the window can grow later without new weights.
- **Tied input/output embeddings, and the output head predicts words too.** v1 could
  read word tokens but only ever output characters. Nova-2 can emit "apples" in one step,
  and it is a lossless superset: the 128 character outputs are still there.
- **RMSNorm everywhere.** Replaces hand-tuned scale constants (1/32, /REASONING_STEPS) and
  keeps training stable as depth grows.

### Right hemisphere (336,320 params): fast association over raw characters
- Keeps exactly the v1 input (64 characters x 16 features, same encoder) so it still
  contributes spelling and local character patterns the word tokenizer hides.
- Trunk 1024 -> 256, then an upper layer 256 -> 192 that also **receives the left
  hemisphere's state** (efferent path). So the right hemisphere is no longer blind to meaning:
  its upper layer computes "these characters, given what the left side currently thinks."
- It outputs **features, not character predictions**. Nothing forces its knowledge through
  a 128-way softmax-shaped bottleneck first.
- Smaller (853K -> 336K) because the checkpoint shows the large version wasn't being used;
  the freed budget went to the left hemisphere.

### Interconnect (90,624 params): the corpus callosum
- **Afferent (right -> left):** after the lower stage, and again before every upper-stage
  round, right-hemisphere features are projected to width 128 and added to the left stream
  **through a learned gate** (sigmoid, per channel, decided by the left side's current state).
  Gates start mostly closed (bias -2, about 12% open), keeping the "start neutral, let training
  open it" principle from v1 but without starting at exactly zero, so gradients flow from step 1.
- **Efferent (left -> right):** the left state feeds the right hemisphere's upper layer.
- **Reciprocal rounds = thinking.** The upper stage runs twice with shared weights
  (`loops = 2`): exchange, think, exchange again, think again. This replaces v1's 3-step
  reasoning loop + 2 dense round trips, but now happens **at every position**, with full
  attention each round. Depth without parameters: 6 block evaluations from 4 unique blocks.
- **Ablations are built in** (`no_afferent`, `no_efferent`, `no_right`) so the app's existing
  "Full / No Dense / No Exchange" diagnostics map directly onto the new model.

### Training changes
- **Every position is a training target** (weighted mask: e.g. prompt tokens 0.2, answer
  tokens 1.0). This is the largest practical speedup.
- **AdamW** (decoupled weight decay 0.01 on matrices only), betas (0.9, 0.95), global-norm
  clip 1.0, warmup then cosine learning rate (`lr_schedule`).
- **Training focus kept:** `adam_step(..., focus="left"|"right"|"inter"|"joint")` updates
  only that piece, matching v1's TrainingFocus (Dense / Transformer / Interconnect / Joint).
- One flat parameter array + Adam moments, like v1, so append-only growth still works.

### What stays the same
Tokenizer (byte-for-byte port, same 143-word vocabulary can be loaded:
`data/vocab_from_step970.txt`), ASCII-only rule, the 64-char x 16-feature encoder, lesson
formats, prompt markers (`U:` / `N:`), the no-canned-answers rule (all output is greedy or
sampled from the model's distribution), and the checkpoint layout style (magic, step,
weights, Adam m, Adam v, `WORDUNIT` trailer).

## Things that change for the app

- **Checkpoints are not compatible.** Shapes differ everywhere, so Nova-2 starts from fresh
  weights. The vocabulary carries over; lessons and ZIP curricula can be re-taught as-is.
  Keep v1 checkpoints; this should ship as a separate model, not a migration.
- **Visualizer:** the existing two-hemisphere picture still applies. Natural signals to show:
  per-head attention maps (4 layers x 4 heads), the two gate openings (how much each
  hemisphere is listening), and the right-hemisphere activations.
- **KV cache** for generation is straightforward (standard causal attention) and is the next
  inference speedup after the port.

## Known limits (unchanged by any architecture at this size)
1.2M parameters trained on hand-written lessons will still not give general conversation or
reliable reasoning. This design makes the parameters work much harder; it does not replace
data and training time. See `EXPERIMENTS.md` for what was actually measured.
