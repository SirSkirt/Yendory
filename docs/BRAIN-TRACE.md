# Per-network trace (§ 11 Brain)

What each of Nova's networks did while she wrote a reply, step by step: the Causal Transformer,
the working memory (minGRU), the associative memory (modern Hopfield), the Executive
Controller, the communication bus, and the other Nova Neuro networks when a design has them.
Engine: `model::inspect` (built on the KV cache's trace in `model/cache.rs`). Studio:
`trace.rs`, page `js/trace.js`.

## How the networks are actually built

From the code (`model/neuro.rs` and the cached forward pass in `model/cache.rs`), not from the
names:

- **Causal Transformer**: `split` lower blocks run once per token; the upper blocks run once
  per round (`loops`), each round with its own key/value cache. It reads at most the model's
  context (`max_tokens`); past that the window is rebuilt from the most recent tokens.
- **Working memory (minGRU)**: `m_t = (1 − z_t)·m_(t−1) + z_t·c_t` per channel, the gate `z` and
  candidate `c` from the current input only. It reads each token **once, after the lower
  blocks**; its output is the same in every round. Its state belongs to the conversation
  (never saved with the model) and restarts when the context window is rebuilt.
- **Associative memory (modern Hopfield)**: softmax over **learned patterns**, a weighted sum of
  their values. The patterns are parameters fixed after training: nothing said in a conversation
  is written into it. What changes during a conversation is which patterns the current state
  recalls.
- **Executive Controller**: reads the bus and sets one gate (0 to 1) per network per round.
- **Communication bus**: a connection, not a network that computes on its own. Each round the
  transformer's state is projected onto it, the working memory's output is added, the controller
  and networks read it, their gated outputs are summed and written back into the transformer's
  state through a learned gate, and the upper blocks run again.

So a specific fact said earlier in a conversation can reach a reply by only two routes:
attention over the window (the transformer) and the working memory's state. The associative
memory can only respond to it indirectly, through the state that queries it.

## Modes

- **Recorded** (Chat setting *Record a brain trace of each reply*, off by default): the reply is
  written with tracing on and the trace saved with the chat (`models/<id>/traces/<chat>/`).
  While it is written, its steps stream to § 11 (live mode; *Follow the newest step* can be
  turned off to look at earlier steps while it continues). Taken-back replies and deleted chats
  lose their traces; the page shows the space used and can delete them all.
- **Replay**: any earlier reply is read through the current model as if it were writing it.
  The page says where this can differ from what happened: the weights may have trained since,
  Study Mode memories that were in front of the message are not rebuilt, and the reply's tokens
  are its saved text encoded again.
- **Question**: the question box on § 11, answered greedily.
- **Influence** (optional, any mode): the whole turn is read again once per network with that
  network switched off; recorded traces can get it afterwards, with the weights that wrote them.

## What is measured, what is inferred, what cannot be read

**Measured directly** (values the computation itself produced, copied out):

| Network | Measures |
|---|---|
| Transformer | Next-token probabilities (chosen, alternatives, entropy); every block execution's attention for the newest token (per head: entropy, strongest positions; per line: attention mass); length of the state after each block and of the block's change; what the output layer would predict after the lower blocks and after each round (read-outs) |
| Working memory | Per layer: mean update gate, share of channels with gate above ½, change and length of the state; the state **split exactly by line** (`m = Σ_s z_s·c_s·Π_(u>s)(1 − z_u)`, tested to add up to the state); the update gate at every token of the conversation |
| Associative memory | Per round: retrieval entropy over all patterns, the strongest patterns and weights, the controller's gate and the output length; for each strong pattern, the tokens of this conversation where it was recalled most strongly |
| Executive Controller | Each gate per network and round, the networks' output lengths, the hidden activations; gates at every token |
| Bus | Per round: transformer → bus and working memory → bus lengths, the bus values and length, the networks' summed output, the write-back's length relative to the transformer's state and its gate's mean |
| Influence | Total-variation change of the next-token distribution with each network switched off, the probability then of the token actually written, and whether the most likely token changes |

**Inferred** (marked on the page): the share of the working memory's contribution lengths that
comes from lines before the message (vectors can cancel, so lengths do not add exactly); each
network's share of the gated output lengths; *similar recall*: cosine similarity between the
patterns recalled at this step and those recalled at a line's tokens (from the strongest 32 and
16 kept). Similar recall says the memory is in a similar state, not that it retrieved that line.

**Not interpretable from these numbers** (and not attempted): what a pattern, a bus value, a
controller activation or a memory channel *means*. No concept labels are given, no pattern is
projected to words, and no text is written about intentions. Attention to a position in a later
block is attention to that position's mixed state, not only to its word. Influence measures what
changes when a network is removed for the whole turn, which is not the same as what it
contributed with everything else present. The Dense pathway of Nova Bicameral is traced only at
its exchanges with the transformer.

## Verification

- `model::inspect` tests: a traced reply is the untraced one (greedy and sampled chat replies,
  several seeds), every step's probability equal bit for bit; a replay gives the same trace as
  the generation; steps line up with token positions and attention covers only what was read;
  the working memory's split adds up to its state; influence is measured per network; the
  networks listed are those built.
- Studio: chats saved before this feature open with recording off.
- Cost on a Nova Neuro model of about 19M parameters (CPU, 4 threads): a 236-token prompt traced
  in 1.18 times the untraced time, 0.8 MB of trace; a 2,189-token prompt 1.07 times, 2.9 MB.
  Influence (five networks) took about as long as the reply itself (the passes run in parallel);
  peak memory 1.1 GB, mostly those passes' caches. With recording off nothing is traced.
- The page was checked with real traces in light and dark mode and at phone width.
