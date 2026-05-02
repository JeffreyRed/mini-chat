# mini-chat

> A minimal instruction-following language model trained on science Q&A pairs.
> Step 5 of the mini-LLM series.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## Series

| Step | Repository | What it builds |
|------|-----------|----------------|
| 1 | [mini-embedding](https://github.com/JeffreyRed/mini-embedding) | Word vectors — Skip-gram Word2Vec |
| 2 | [mini-self-attention](https://github.com/JeffreyRed/mini-self-attention) | Multi-head self-attention encoder block |
| 3 | [mini-transformer]((https://github.com/JeffreyRed/mini-transformer) | Positional encoding + stacked causal decoder |
| 4 | [mini-gpt](https://github.com/JeffreyRed/mini-gpt) | Real corpus, overfitting, beam search, evaluation |
| **5** | **mini-chat** ← you are here | Instruction format, loss masking, chat interface |
| 6 | [mini-cross-attention](https://github.com/JeffreyRed/mini-cross-attention) | Cross-attention module, source↔target alignment |
| 7 | [mini-translator](https://github.com/JeffreyRed/mini-translator) | English→Spanish encoder-decoder with cross-attention |

---

## The one new idea

`mini-gpt` generated text by continuing any sequence.
`mini-chat` responds to questions.

The architecture is identical. The single change is **how the loss is computed:**

```
mini-gpt:   loss = mean( cross_entropy(token_t) for ALL t )

mini-chat:  loss = sum( mask_t × cross_entropy(token_t) for all t )
                   / sum(mask_t)

            where mask_t = 1 only on [AI] response tokens
```

This is called **instruction fine-tuning** or **supervised fine-tuning (SFT)**.
It is exactly how OpenAI turned GPT-3 into InstructGPT:
same base model, same architecture, trained on (prompt, response) pairs
with loss only on the response tokens.

---

## Data format

Each training example is a conversation turn:

```
[HUMAN]: What is imagination?
[AI]: Imagination is the ability to form ideas and mental images beyond what
we directly observe. It lets us explore possibilities that do not yet exist.
```

Encoded as a flat token sequence:

```
<BOS> [HUMAN] What is imagination ? [AI] Imagination is the ability ... <EOS>
 ░       ░      ░   ░      ░    ░    ░       ▓   ▓   ▓    ▓    ▓  ...   ▓

░ = mask 0 (excluded from loss)
▓ = mask 1 (gradient flows here)
```

The model sees the full context but only updates weights to predict the AI response.

---

## What you will observe

### Loss mask visualisation
`outputs/loss_mask.png` shows the token strip for one example, colour-coded:
- Blue = prompt tokens (no gradient)
- Orange = AI response tokens (gradient flows)

### Training curves
`outputs/training_curves.png` shows train and val perplexity computed
**only on AI response tokens** — a more honest metric than full-sequence perplexity.

### Chat responses
After training the model responds to science and AI questions:

```
You: What is imagination?
AI:  Imagination is the ability to form ideas beyond what we directly observe.

You: What is a neural network?
AI:  A neural network is a computational model consisting of layers of
     interconnected nodes that transform input data through learned weights.

You: What is overfitting?
AI:  Overfitting occurs when a model learns the training data too well
     including its noise and fails to generalize to new data.
```

---

## Project structure

```
mini-chat/
│
├── data/
│   └── science_qa.txt          # ~200 science and AI Q&A pairs
│
├── src/
│   ├── tokenizer.py            # ChatTokenizer: [HUMAN]/[AI] tokens + loss mask
│   ├── attention.py            # multi-head self-attention (unchanged)
│   ├── model.py                # MiniChat (identical architecture to MiniGPT)
│   ├── dataset.py              # ChatDataset with per-token loss mask
│   ├── train.py                # masked_cross_entropy + cosine LR + checkpointing
│   ├── utils.py                # chat_response, interactive_chat, demo_responses
│   └── visualize.py            # loss mask plot, training curves, attention
│
├── outputs/
├── main.py
├── environment.yml
├── requirements.txt
├── THEORY.md
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/your-username/mini-chat.git
cd mini-chat
conda env create -f environment.yml
conda activate mini-chat
python main.py
```

Training: **3–6 minutes on CPU**, under 1 minute on GPU.

---

## Configuration

| Parameter | Default | Notes |
|---|---|---|
| `EMB_DIM` | `64` | Same as mini-gpt |
| `N_HEADS` | `4` | |
| `N_LAYERS` | `4` | |
| `MAX_LEN` | `128` | Longer to fit full Q&A pairs |
| `EPOCHS` | `300` | |
| `LR` | `3e-3` | Peak cosine LR |
| `VAL_SPLIT` | `0.15` | 15% held out for validation |

---

## Outputs

| File | Description |
|---|---|
| `minichat_best.pt` | Best checkpoint (lowest val perplexity on AI tokens) |
| `loss_mask.png` | **Colour-coded token strip showing which positions train** |
| `training_curves.png` | Train vs val perplexity + loss (masked) |
| `attention.png` | All layers × heads for one conversation turn |

---

## Deep dive

See [`THEORY.md`](./THEORY.md) for:
- Why loss masking is the key idea in instruction fine-tuning
- The exact math of masked cross-entropy
- How this relates to InstructGPT and ChatGPT
- The `[HUMAN]` / `[AI]` token format and why it works
- Line-by-line code walkthrough of every new piece
- What mini-cross-attention will add next

---

## References

- Ouyang et al. (2022) — [InstructGPT: Training language models to follow instructions](https://arxiv.org/abs/2203.02155)
- Wei et al. (2022) — [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
- Taori et al. (2023) — [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)

---

## License

MIT
