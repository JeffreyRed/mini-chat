"""
utils.py — Chat interface and response generation.

The key new function is chat_response():
  - Takes a human question as a plain string
  - Wraps it in the [HUMAN] / [AI] format the model was trained on
  - Feeds the formatted prompt (without the AI response) to the model
  - Decodes and returns only the generated AI tokens

This is the same pattern used by all instruction-following LLMs:
the prompt is formatted to match the training format,
and only the completion (after [AI]) is returned to the user.
"""

import torch
from typing import List

from src.model     import MiniChat
from src.tokenizer import ChatTokenizer


def chat_response(
    model       : MiniChat,
    tokenizer   : ChatTokenizer,
    question    : str,
    max_new     : int   = 50,
    temperature : float = 0.8,
    top_k       : int   = 40,
    top_p       : float = 0.9,
    greedy      : bool  = False,
) -> str:
    """
    Generates a response to a question.

    Constructs the prompt in training format:
        <BOS> [HUMAN] w0 w1 ... [AI]
    then generates tokens until <EOS> or max_new_tokens.

    Args:
        model:       trained MiniChat
        tokenizer:   ChatTokenizer used during training
        question:    human question as plain text
        max_new:     max tokens to generate
        temperature: sampling temperature
        top_k:       top-k sampling (0 = off)
        top_p:       nucleus sampling threshold (0 = off)
        greedy:      deterministic argmax

    Returns:
        Generated AI response as a plain string.
    """
    model.eval()
    words = question.strip().split()

    # Build prompt: <BOS> [HUMAN] words [AI]
    prompt_ids = (
        [tokenizer.bos_idx, tokenizer.human_idx]
        + [tokenizer.word2idx.get(w, tokenizer.unk_idx) for w in words]
        + [tokenizer.ai_idx]
    )
    src = torch.tensor([prompt_ids], dtype=torch.long)

    out_ids = model.generate(
        src,
        max_new_tokens = max_new,
        temperature    = temperature,
        top_k          = top_k,
        top_p          = top_p,
        greedy         = greedy,
        eos_idx        = tokenizer.eos_idx,
    )

    return tokenizer.decode_response(out_ids[0].tolist())


def interactive_chat(model: MiniChat, tokenizer: ChatTokenizer) -> None:
    """
    Interactive chat loop.

    Type a question → model responds.
    Flags: --greedy  --temp=0.8  --topk=40  --topp=0.9  --n=50
    Type 'quit' to exit.
    """
    print("── Mini Chat ───────────────────────────────────────")
    print("  Ask any science or AI question.")
    print("  Flags: --greedy  --temp=0.8  --topk=40  --topp=0.9  --n=50")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            raw = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye.")
            break

        if raw.lower() in ("quit", "exit", "q"):
            break
        if not raw:
            continue

        # Parse flags
        parts  = raw.split()
        flags  = {p for p in parts if p.startswith("--")}
        words  = [p for p in parts if not p.startswith("--")]
        question = " ".join(words)

        greedy  = "--greedy" in flags
        temp    = 0.8
        top_k   = 40
        top_p   = 0.9
        n_new   = 50

        for f in flags:
            if f.startswith("--temp="): temp  = float(f.split("=")[1])
            if f.startswith("--topk="): top_k = int(f.split("=")[1])
            if f.startswith("--topp="): top_p = float(f.split("=")[1])
            if f.startswith("--n="):    n_new = int(f.split("=")[1])

        response = chat_response(
            model, tokenizer, question,
            max_new=n_new, temperature=temp,
            top_k=top_k, top_p=top_p, greedy=greedy,
        )
        print(f"\n  AI: {response}\n")

    print("────────────────────────────────────────────────────\n")


def demo_responses(model: MiniChat, tokenizer: ChatTokenizer) -> None:
    """Prints model responses to a fixed set of questions."""
    questions = [
        ("What is imagination?",          dict(greedy=True)),
        ("What is a neural network?",      dict(temperature=0.8, top_k=40)),
        ("Why is curiosity important?",    dict(top_p=0.9)),
        ("What is overfitting?",           dict(greedy=True)),
        ("What is cross-attention?",       dict(temperature=0.7, top_k=20)),
        ("What makes a good question?",    dict(top_p=0.85)),
    ]
    print("── Demo responses (best checkpoint) ────────────────")
    for question, kwargs in questions:
        response = chat_response(model, tokenizer, question,
                                 max_new=40, **kwargs)
        print(f"  Q: {question}")
        print(f"  A: {response}\n")


def show_loss_masking(tokenizer: ChatTokenizer, n_examples: int = 3) -> None:
    """
    Prints the loss mask for a few training examples so the masking
    mechanism is fully visible.

    Shows: token sequence + which positions contribute to the loss.
    """
    print("── Loss masking examples ────────────────────────────")
    print("  0 = no gradient (prompt tokens)")
    print("  1 = gradient flows (AI response tokens)\n")

    for pair in tokenizer.train_pairs[:n_examples]:
        tokens, mask = tokenizer.encode_conversation(pair)
        print(f"  Q: {' '.join(pair['human'][:8])} ...")
        print(f"  A: {' '.join(pair['ai'][:8])} ...")
        print()
        for i, (tid, m) in enumerate(zip(tokens, mask)):
            word = tokenizer.idx2word.get(tid, "?")
            flag = "▓" if m == 1 else "░"
            print(f"    {flag} [{m}]  {word}")
        print()
    print("  ░ = excluded from loss  ▓ = included in loss")
    print("────────────────────────────────────────────────────\n")
