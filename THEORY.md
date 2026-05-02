# Theory & Code Walkthrough — mini-chat

> Step 5 of the mini-LLM series. Prerequisite: [mini-gpt](../mini-gpt).

---

## Table of Contents

1. [What this step adds](#1-what-this-step-adds)
2. [The instruction fine-tuning idea](#2-the-instruction-fine-tuning-idea)
3. [Loss masking — the core mechanism](#3-loss-masking--the-core-mechanism)
4. [The conversation format](#4-the-conversation-format)
5. [Why the architecture does not change](#5-why-the-architecture-does-not-change)
6. [Masked cross-entropy — the math](#6-masked-cross-entropy--the-math)
7. [How this relates to InstructGPT and ChatGPT](#7-how-this-relates-to-instructgpt-and-chatgpt)
8. [Code walkthrough](#8-code-walkthrough)
9. [Full data flow](#9-full-data-flow)
10. [What mini-cross-attention will add](#10-what-mini-cross-attention-will-add)

---

## 1. What this step adds

`mini-gpt` was a language model: given any text, predict the next token.
`mini-chat` is an instruction-following model: given a question, produce a useful answer.

The architecture is **identical**. The dataset format changes.
The loss computation changes in one specific way.

That is the entire difference between a base language model and an
instruction-following assistant.

| | mini-gpt | mini-chat |
|---|---|---|
| Data | plain sentences | (question, answer) pairs |
| Format | raw text | `[HUMAN]: ... [AI]: ...` |
| Loss | all tokens | AI response tokens only |
| Goal | predict next token | respond to questions |
| Architecture | MiniGPT | MiniChat (identical) |

---

## 2. The instruction fine-tuning idea

A base language model trained on text will complete any sequence:

```
Input:  "What is imagination?"
Output: "What is imagination? is a question asked by..."
        (continues the text, does not answer it)
```

To make it answer questions instead, we fine-tune it on examples that show
the desired behaviour: here is a question, here is a good answer.

This is called **supervised fine-tuning (SFT)** or **instruction fine-tuning**.
The model sees thousands of (question, answer) examples and learns to produce
answers in the style of the training data.

The key insight: you do not need a different architecture. You need
different training data and a modified loss function.

---

## 3. Loss masking — the core mechanism

In `mini-gpt`, the loss was computed over every token in every sequence:

```
Sequence:  <BOS>  I  like  cats  <EOS>
Loss:        ✓    ✓    ✓     ✓     ✓     (all positions)
```

In `mini-chat`, the sequence includes both the question and the answer:

```
Sequence:  <BOS> [HUMAN] What is imagination ? [AI] Imagination is ... <EOS>
Loss:        ✗      ✗      ✗    ✗      ✗      ✗    ✗      ✓   ✓  ...   ✓
```

The prompt tokens (`<BOS>`, `[HUMAN]`, the question words, `[AI]`) are
**masked out** — their loss contribution is zero, they receive no gradient,
they do not update any weights.

Only the AI response tokens (and `<EOS>`) contribute to the loss.

**Why this matters:**

If we computed loss on the prompt too, the model would spend capacity
learning to predict the human's words. But the human's words are given
at inference time — the model never needs to generate them.
Loss masking focuses 100% of the training signal on what the model
actually produces.

---

## 4. The conversation format

Each training example is encoded as a single flat token sequence:

```
<BOS> [HUMAN] w_0 w_1 ... w_n [AI] r_0 r_1 ... r_m <EOS>
```

Where `w_i` are the question words and `r_j` are the response words.

Two new special tokens are added to the vocabulary:
- `[HUMAN]` — marks the start of the human turn
- `[AI]` — marks the start of the AI turn

These act as role markers. The model learns that text following `[AI]`
should be a coherent, informative response to the question following `[HUMAN]`.

At inference time, we construct a prompt that stops at `[AI]`:

```
<BOS> [HUMAN] What is imagination ? [AI]
                                         ↑
                                     generation starts here
```

The model then generates tokens until it produces `<EOS>`.
Everything between `[AI]` and `<EOS>` is the response.

---

## 5. Why the architecture does not change

The transformer is a general sequence-to-sequence function.
It takes a token sequence and produces a probability distribution
over the next token at each position.

Nothing in the architecture specifies whether those tokens are from
a base language model, an instruction-following model, or a chat system.
The architecture is the same. What changes is:

1. **The vocabulary** — two new tokens `[HUMAN]` and `[AI]`
2. **The training data** — (question, answer) format instead of raw text
3. **The loss** — masked to AI response tokens only

This is also why GPT-3 and InstructGPT have the same architecture.
InstructGPT is GPT-3 fine-tuned on instruction data with a modified loss
(and further refined with reinforcement learning from human feedback, which
is beyond the scope of this series).

---

## 6. Masked cross-entropy — the math

Standard cross-entropy for a sequence of length T:

```
L = (1/T) × Σ_t  -log P(target_t | context_{0..t-1})
```

Masked cross-entropy with a binary mask `m_t ∈ {0, 1}`:

```
L = Σ_t  m_t × -log P(target_t | context_{0..t-1})
    ──────────────────────────────────────────────────
                    Σ_t  m_t
```

The denominator is the number of active (unmasked) positions.
This ensures the loss scale is the same regardless of how long
the prompt is relative to the response.

In code:

```python
def masked_cross_entropy(logits, targets, mask, pad_idx=0):
    B, T, V = logits.shape

    # Per-token loss, no reduction
    loss_per_token = F.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
        ignore_index=pad_idx,
        reduction="none",
    ).view(B, T)

    # Zero out prompt positions
    masked_loss = loss_per_token * mask

    # Normalise by number of active positions
    return masked_loss.sum() / mask.sum().clamp(min=1)
```

The `ignore_index=pad_idx` handles padding independently of the mask —
padded positions are excluded before the mask is applied.

---

## 7. How this relates to InstructGPT and ChatGPT

The full pipeline from base model to ChatGPT has three stages:

```
Stage 1: Supervised Fine-Tuning (SFT)          ← this is mini-chat
  Train on (prompt, response) pairs
  Loss only on response tokens
  Result: model that follows instructions

Stage 2: Reward Model Training
  Humans rank multiple model responses
  Train a separate model to predict human preference scores
  (Not covered in this series)

Stage 3: Reinforcement Learning from Human Feedback (RLHF)
  Use the reward model as a signal
  Fine-tune the SFT model to produce responses the reward model scores highly
  (Not covered in this series)
```

`mini-chat` implements Stage 1 exactly.
The resulting model follows instructions because it was trained on
(instruction, response) pairs with masked loss.

The difference between `mini-chat` and InstructGPT Stage 1 is:
- Scale: 200 examples vs ~13,000 for InstructGPT
- Base model: MiniGPT (~40K params) vs GPT-3 (175B params)
- The mechanism is identical

---

## 8. Code walkthrough

### `tokenizer.py`

**`_parse_file()`** reads the conversation file line by line:

```python
if line.startswith("[HUMAN]:"):
    human = line[len("[HUMAN]:"):].strip().split()
elif line.startswith("[AI]:"):
    ai = line[len("[AI]:"):].strip().split()
```

Each complete (human, ai) pair is stored as a dict.

**`encode_conversation()`** is the core new method:

```python
tokens = (
    [bos_idx, human_idx]
    + human_ids
    + [ai_idx]
    + ai_ids
    + [eos_idx]
)

prompt_len = 2 + len(human_ids) + 1   # BOS + [HUMAN] + words + [AI]
loss_mask  = [0] * prompt_len + [1] * (len(ai_ids) + 1)
```

The mask is aligned with the tokens: index `i` in the mask corresponds
to index `i` in the token sequence.

**`decode_response()`** extracts only the AI portion after generation:

```python
ai_pos = indices.index(self.ai_idx)
rest   = indices[ai_pos + 1:]
if self.eos_idx in rest:
    rest = rest[:rest.index(self.eos_idx)]
return self.decode(rest)
```

---

### `dataset.py`

**`ChatDataset`** applies the standard next-token shift AND aligns the mask:

```python
# tokens:  [t0, t1, t2, t3, t4]
# input:   [t0, t1, t2, t3]       (tokens[:-1])
# target:  [t1, t2, t3, t4]       (tokens[1:])
# mask:    [m1, m2, m3, m4]       (mask[1:])  ← shifted to align with target
```

The mask is shifted by 1 because the target is shifted by 1.
At position `i` in the output, the model predicts `tokens[i+1]`,
so the mask at position `i` should correspond to whether `tokens[i+1]`
is an AI token.

---

### `train.py`

The training loop is identical to `mini-gpt` except for the loss call:

```python
# mini-gpt:
loss = nn.CrossEntropyLoss(ignore_index=pad_idx)(logits.transpose(1,2), tgt)

# mini-chat:
loss = masked_cross_entropy(logits, tgt, msk, pad_idx)
```

The `msk` tensor comes directly from the DataLoader via `collate_fn`.

**Validation** uses the same `masked_cross_entropy` function:

```python
@torch.no_grad()
def evaluate(model, dataset, pad_idx, batch_size):
    for src, tgt, msk in loader:
        logits, _ = model(src)
        loss = masked_cross_entropy(logits, tgt, msk, pad_idx)
```

This means validation perplexity is also computed only on AI response tokens —
a much more honest metric than measuring perplexity on the full sequence
including the question.

---

### `utils.py`

**`chat_response()`** constructs the inference prompt:

```python
prompt_ids = (
    [bos_idx, human_idx]
    + [word2idx.get(w, unk_idx) for w in question.split()]
    + [ai_idx]
)
src = torch.tensor([prompt_ids])
out = model.generate(src, ..., eos_idx=eos_idx)
return tokenizer.decode_response(out[0].tolist())
```

The prompt ends at `[AI]`. Generation starts from the position after
`[AI]` and continues until `<EOS>` or `max_new_tokens`.

**`show_loss_masking()`** prints the mask visually:

```
░ [0]  <BOS>
░ [0]  [HUMAN]
░ [0]  What
░ [0]  is
░ [0]  imagination
░ [0]  ?
░ [0]  [AI]
▓ [1]  Imagination
▓ [1]  is
▓ [1]  the
...
```

`░` = excluded from loss, `▓` = included in loss.

---

## 9. Full data flow

Tracing one training example end-to-end:

```
science_qa.txt
  [HUMAN]: What is imagination?
  [AI]: Imagination is the ability to form ideas ...
        │
        ▼  tokenizer.py  encode_conversation()
  tokens: [1, 4, 47, 55, 102, 35, 5, 102, 55, 3, 89, ...]
  mask:   [0, 0,  0,  0,   0,  0, 0,   1,  1, 1,  1, ...]
        │
        ▼  dataset.py  ChatDataset
  src:    tokens[:-1]     target: tokens[1:]
  mask:   mask[1:]        (shifted to align with target)
        │
        ▼  model.py  MiniChat.forward(src)
  tok_emb + pos_emb → dropout
  4 × TransformerBlock (causal mask)
  LayerNorm → head
  logits: (1, seq_len, vocab_size)
        │
        ▼  train.py  masked_cross_entropy(logits, target, mask)
  per-token loss: (1, seq_len)
  × mask          zero out prompt positions
  / mask.sum()    normalise by AI token count
  = scalar loss
        │
        ▼  backward + AdamW + cosine LR step
  only [AI] response tokens drove the gradient
  the model learned to produce better responses
```

---

## 10. What mini-cross-attention will add

`mini-chat` uses self-attention: each token attends to other tokens
in the **same sequence**. The question and answer are concatenated
into one sequence and the model attends within it.

`mini-cross-attention` introduces a fundamentally different operation:
a token in one sequence attends to tokens in a **different sequence**.

```
Self-attention (mini-chat):
    [HUMAN] What is imagination? [AI] Imagination is...
    ←────────────── all tokens attend within one sequence ──────────────→

Cross-attention (mini-cross-attention):
    Encoder sequence:  "What is imagination"     (source)
    Decoder sequence:  "Imagination is the..."   (target)
    Each decoder token queries the encoder's representations
    ←── decoder queries ──→  ←── encoder keys/values ──→
```

This is the building block for translation, summarisation, and
any task where the input and output are fundamentally separate sequences.

The cross-attention weight matrix will show you:
- Which source words each target word attended to
- Alignment emerging from training without any supervision
- Why "imagination" in the output attended most to "imagination" in the input

---

*Next: `mini-cross-attention` — isolating cross-attention as a standalone module
with full visualisation of source↔target alignment.*
